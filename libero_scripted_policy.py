"""Scripted top-down pick-and-place policy for LIBERO.

This is the scalable demo generator for the RoboCasa-objects-in-LIBERO experiment:
the motion is identical across objects and only the manipulated object changes, which
cleanly isolates *object* generalization and gives a consistent retrieval structure.

Embodiment assumptions (LIBERO): fixed-base Franka, robosuite OSC_POSE controller, so
the action is 7-D ``[dx, dy, dz, drx, dry, drz, gripper]`` where the position/rotation
entries are *deltas* in the world frame (action in [-1, 1] is scaled by the controller's
output limits, ~0.05 m / ~0.5 rad at full deflection) and the gripper uses the sign
convention ``+1 = close`` / ``-1 = open``.

Phase machine (holds the reset, top-down orientation throughout):

    APPROACH -> DESCEND -> GRASP -> LIFT -> MOVE -> LOWER -> DROP -> SETTLE -> done

The object is carried at a fixed high ``carry_z`` across the whole traverse so it clears
the receptacle rim and distractor objects; only once aligned above the target does it
descend (LOWER) and open the gripper (DROP). It needs only object/receptacle positions
(``obs[f"{name}_pos"]``) and ``obs["robot0_eef_pos"]``.

Grasp height: rather than a per-object hand-tuned ``grasp_z_offset`` (a metres offset from
the object *origin*, which conflates where the mesh origin sits with the object's size and
so does not transfer across geometries), the policy prefers a geometry-relative target. If
the collector injects the manipuland's live world-frame bottom/top z
(``obs[f"{name}_bottom_z"]`` / ``obs[f"{name}_top_z"]`` -- derived each step from the
object's ``bottom_offset``/``top_offset`` sites and its *current* pose), the grasp closes at
``bottom_z + grasp_frac * (top_z - bottom_z)`` -- a single dimensionless fraction that is
correct across objects and tracks the actual resting height. When those keys are absent
(e.g. stock objects, no injection) it falls back to the old ``obj_z + grasp_z_offset``.

Injected grasp pose (all optional, same best-effort contract as the extents):

* ``obs[f"{name}_grasp_xyz"]`` / ``obs[f"{name}_grasp_yaw"]`` -- a specific grasp
  candidate from ``libero_grasp_synthesis`` (e.g. a bowl-rim or box-short-side pinch).
  When present, APPROACH/DESCEND/GRASP target the candidate *midpoint* instead of the
  object origin, and the yaw servo drives the gripper's finger-closing axis (its local
  y-axis) onto ``grasp_yaw`` in **every** phase, so the alignment holds through the
  carry. ``a[5]`` is a world-frame z rotation (robosuite composes
  ``R_goal = R_delta @ R_current``), scaled by the controller's 0.5 rad output limit.
* Placement is origin-centred: at GRASP the policy latches the object-origin-to-eef
  offset and steers ``target_xy - offset`` during MOVE/LOWER/DROP, so the *object
  origin* -- which is what LIBERO's point-like ``In`` check tests (+-6.1 cm window) --
  lands on the basket centre even for deliberately off-centre grasps.
* Release: when the live ``bottom_z`` is injected, LOWER descends until the object's
  bottom clears the basket floor by ``basket_clear`` instead of the fixed
  ``place_drop`` above the basket origin. That target is then **clamped so the hand
  itself never goes below the receptacle rim**: if the collector injects
  ``obs[f"{target}_rim_z"]``, the descent stops at ``rim_z + release_above_rim`` and the
  object is let go from there, hanging below the fingers.

  This matches how the stock (human-teleop) LIBERO-Object demos place: measured over
  those demos the end-effector at release sits **+0.4 to +4.6 cm above the rim** (median
  +2.1) with *zero* arm-basket contact, whereas targeting the object's floor clearance
  alone puts the hand ~3.3 cm *below* the rim and rakes the basket on every episode.
  Tall objects are unaffected (they already release high); compact ones are raised ~5 cm.
"""

from __future__ import annotations

import numpy as np

OPEN_GRIPPER = -1.0
CLOSE_GRIPPER = +1.0
ROT_SCALE = 0.5  # OSC_POSE output_max for rotation (rad at full action deflection)
MAX_JAW = 0.08   # Panda parallel-jaw opening (mirrors libero_grasp_synthesis.MAX_JAW)


def wrap_half_pi(a):
    """Wrap to (-pi/2, pi/2] -- a parallel jaw is symmetric under a 180 deg flip."""
    return (a + np.pi / 2) % np.pi - np.pi / 2


def gripper_opening(obs):
    """Current jaw opening (m) from the two Panda finger joints, or None if absent."""
    q = obs.get("robot0_gripper_qpos")
    if q is None:
        return None
    q = np.asarray(q, dtype=float)
    return float(abs(q[0]) + abs(q[1]))


def finger_axis_yaw(obs):
    """World heading of the gripper's local y-axis (the finger-closing direction).

    Pure-numpy quat (xyzw) -> rotation-matrix middle column, so the module stays
    obs-only (no robosuite import).
    """
    x, y, z, w = np.asarray(obs["robot0_eef_quat"], dtype=float)
    # R[:, 1] = (2(xy - wz), 1 - 2(x^2 + z^2), ...): the world-frame finger axis
    return float(np.arctan2(1.0 - 2.0 * (x * x + z * z), 2.0 * (x * y - w * z)))


class ScriptedPickPlacePolicy:
    """Top-down pick-and-place controller.

    Args:
        obj_name: BDDL instance name of the manipuland (e.g. ``"akita_black_bowl_1"``).
        target_name: BDDL instance name of the place-target receptacle (e.g. ``"basket_1"``).
        The remaining kwargs are geometric/timing knobs that may need per-geometry tuning.
    """

    def __init__(
        self,
        obj_name: str,
        target_name: str,
        hover_height: float = 0.12,
        grasp_z_offset: float = 0.005,
        grasp_frac: float = 0.55,
        lift_height: float = 0.20,
        place_drop: float = 0.06,
        basket_clear: float = 0.02,
        release_above_rim: float = 0.02,   # None -> no rim clamp at all
        pos_gain: float = 20.0,
        max_step: float = 1.0,
        xy_tol: float = 0.012,
        z_tol: float = 0.015,
        grasp_steps: int = 14,
        drop_steps: int = 10,
        settle_steps: int = 12,
        max_phase_steps: int = 90,
        align_yaw: bool = True,
        yaw_gain: float = 2.0,
        center_placement: bool = True,
        pre_grasp_clearance: float = 0.012,  # None -> always descend fully open
    ):
        self.obj_name = obj_name
        self.target_name = target_name
        self.hover_height = hover_height
        self.grasp_z_offset = grasp_z_offset
        self.grasp_frac = grasp_frac
        self.lift_height = lift_height
        self.place_drop = place_drop
        self.basket_clear = basket_clear
        self.release_above_rim = release_above_rim
        self.pos_gain = pos_gain
        self.max_step = max_step
        self.xy_tol = xy_tol
        self.z_tol = z_tol
        self.grasp_steps = grasp_steps
        self.drop_steps = drop_steps
        self.settle_steps = settle_steps
        self.max_phase_steps = max_phase_steps
        self.align_yaw = align_yaw
        self.yaw_gain = yaw_gain
        self.center_placement = center_placement
        self.pre_grasp_clearance = pre_grasp_clearance
        self.open_tol = 0.004
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self):
        self.phase = "APPROACH"
        self.t = 0
        self.phase_t = 0
        self._grasp_xy = None
        self._carry_z = None  # absolute carry height, set at grasp (= grasp eef_z + lift_height)
        self._grasp_offset = None  # object-origin - eef xy offset, latched when grasping
        return self

    def _advance(self, nxt: str):
        self.phase = nxt
        self.phase_t = 0

    def _grasp_z(self, obs, obj):
        """World-z at which to close the gripper.

        Prefer the geometry-relative target from the injected live extents
        (``bottom_z + grasp_frac * height``); fall back to the origin-relative offset
        when the collector did not inject them.
        """
        bz = obs.get(f"{self.obj_name}_bottom_z")
        tz = obs.get(f"{self.obj_name}_top_z")
        if bz is not None and tz is not None:
            return float(bz) + self.grasp_frac * (float(tz) - float(bz))
        return float(obj[2]) + self.grasp_z_offset

    def _grasp_point(self, obs, obj):
        """Where to close the fingers: the injected candidate midpoint, else the
        object origin at the calibrated ``grasp_frac`` height."""
        gxyz = obs.get(f"{self.obj_name}_grasp_xyz")
        if gxyz is not None:
            return np.asarray(gxyz, dtype=float)
        return np.array([obj[0], obj[1], self._grasp_z(obs, obj)])

    def _place_xy(self, tgt):
        """xy that puts the OBJECT ORIGIN (what the point-like ``In`` check tests)
        over the basket centre, compensating an off-centre grasp."""
        if self._grasp_offset is None:
            return tgt[:2]
        return tgt[:2] - self._grasp_offset

    def _open_action(self, obs):
        """Gripper command while approaching: pre-shape to the grasp width if we know it.

        The Panda gripper's action is a sign-based accumulator (robosuite
        ``PandaGripper.format_action``, speed 0.01/step), so an intermediate opening cannot
        be commanded directly -- but it CAN be servo'd: drive toward the target and send 0
        (``sign(0) == 0``) to freeze the accumulator once there.

        Descending pre-shaped rather than fully open is what makes narrow grasps inside
        hollow shapes reachable (a donut's ring, a kettle's handle); the sampler's clearance
        test models exactly this opening via ``pre_grasp_half_for``. Falls back to fully
        open when the grasp width was not injected."""
        if self.pre_grasp_clearance is None:
            return OPEN_GRIPPER
        w = obs.get(f"{self.obj_name}_grasp_width")
        cur = gripper_opening(obs)
        if w is None or cur is None:
            return OPEN_GRIPPER
        target = float(np.clip(float(w) + 2.0 * self.pre_grasp_clearance,
                               float(w) + 0.004, MAX_JAW))
        if cur < target - self.open_tol:
            return OPEN_GRIPPER          # keep opening
        if cur > target + self.open_tol:
            return CLOSE_GRIPPER         # eased back toward the target
        return 0.0                       # hold this opening

    def _yaw_action(self, obs):
        """World-z yaw command driving the finger axis onto the injected grasp yaw."""
        gyaw = obs.get(f"{self.obj_name}_grasp_yaw")
        if not self.align_yaw or gyaw is None:
            return 0.0
        err = wrap_half_pi(float(gyaw) - finger_axis_yaw(obs))
        return float(np.clip(err * self.yaw_gain / ROT_SCALE, -self.max_step,
                             self.max_step))

    def _goto(self, obs, target_xyz, grip):
        eef = np.asarray(obs["robot0_eef_pos"], dtype=float)
        err = np.asarray(target_xyz, dtype=float) - eef
        a = np.zeros(7, dtype=float)
        a[:3] = np.clip(err * self.pos_gain, -self.max_step, self.max_step)
        # a[3:5] = 0 -> hold top-down; a[5] is set once per act() by the yaw servo
        a[6] = grip
        return a, err

    # ------------------------------------------------------------------ #
    def act(self, obs):
        """Return ``(action(7,), done)`` for the current observation."""
        obj = np.asarray(obs[f"{self.obj_name}_pos"], dtype=float)
        tgt = np.asarray(obs[f"{self.target_name}_pos"], dtype=float)
        eef = np.asarray(obs["robot0_eef_pos"], dtype=float)
        self.t += 1
        self.phase_t += 1
        done = False
        timed_out = self.phase_t > self.max_phase_steps

        if self.phase == "APPROACH":
            gp = self._grasp_point(obs, obj)
            target = np.array([gp[0], gp[1], obj[2] + self.hover_height])
            a, err = self._goto(obs, target, self._open_action(obs))
            if (np.linalg.norm(err[:2]) < self.xy_tol and abs(err[2]) < self.z_tol) or timed_out:
                self._advance("DESCEND")

        elif self.phase == "DESCEND":
            a, err = self._goto(obs, self._grasp_point(obs, obj), self._open_action(obs))
            if abs(err[2]) < self.z_tol or timed_out:
                self._grasp_xy = eef[:2].copy()
                self._advance("GRASP")

        elif self.phase == "GRASP":
            a, _ = self._goto(obs, self._grasp_point(obs, obj), CLOSE_GRIPPER)
            if self.phase_t >= self.grasp_steps:
                self._carry_z = eef[2] + self.lift_height  # clearance above the grasp
                if self.center_placement:
                    self._grasp_offset = obj[:2] - eef[:2]  # origin-centred placement
                self._advance("LIFT")

        elif self.phase == "LIFT":
            xy = self._grasp_xy if self._grasp_xy is not None else eef[:2]
            target = np.array([xy[0], xy[1], self._carry_z])  # absolute high carry
            a, err = self._goto(obs, target, CLOSE_GRIPPER)
            if abs(err[2]) < self.z_tol or timed_out:
                self._advance("MOVE")

        elif self.phase == "MOVE":
            xy = self._place_xy(tgt)
            target = np.array([xy[0], xy[1], self._carry_z])  # stay high while traversing
            a, err = self._goto(obs, target, CLOSE_GRIPPER)
            if np.linalg.norm(err[:2]) < self.xy_tol or timed_out:
                self._advance("LOWER")

        elif self.phase == "LOWER":
            xy = self._place_xy(tgt)
            bz = obs.get(f"{self.obj_name}_bottom_z")
            if bz is not None:
                # descend until the object's bottom clears the basket floor
                z_err = (tgt[2] + self.basket_clear) - float(bz)
                target_z = eef[2] + z_err
            else:
                target_z = tgt[2] + self.place_drop
            # ...but never put the HAND below the rim (see module docstring).
            # release_above_rim=None disables the clamp entirely (old behaviour).
            rim_z = obs.get(f"{self.target_name}_rim_z")
            if rim_z is not None and self.release_above_rim is not None:
                target_z = max(target_z, float(rim_z) + self.release_above_rim)
            target = np.array([xy[0], xy[1], target_z])
            a, err = self._goto(obs, target, CLOSE_GRIPPER)
            if abs(err[2]) < 1.5 * self.z_tol or timed_out:
                self._advance("DROP")

        elif self.phase == "DROP":
            xy = self._place_xy(tgt)
            target = np.array([xy[0], xy[1], eef[2]])  # hold pose, open the jaw
            a, _ = self._goto(obs, target, OPEN_GRIPPER)
            if self.phase_t >= self.drop_steps:
                self._advance("SETTLE")

        elif self.phase == "SETTLE":
            target = np.array([tgt[0], tgt[1], self._carry_z])  # retreat up, stay open
            a, _ = self._goto(obs, target, OPEN_GRIPPER)
            if self.phase_t >= self.settle_steps:
                a = np.zeros(7)
                a[6] = OPEN_GRIPPER
                done = True

        else:  # safety
            a = np.zeros(7)
            a[6] = OPEN_GRIPPER
            done = True

        a[5] = self._yaw_action(obs)  # hold the grasp yaw through every phase
        return a, done


def infer_obj_and_target(env):
    """Best-effort: derive (manipuland, receptacle) from the env's obj_of_interest.

    LIBERO BDDL lists ``:obj_of_interest`` as [manipuland, receptacle] for pick-place.
    """
    ooi = list(env.obj_of_interest)
    if len(ooi) >= 2:
        return ooi[0], ooi[1]
    raise ValueError(f"Cannot infer obj/target from obj_of_interest={ooi}")
