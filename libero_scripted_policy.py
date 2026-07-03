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
"""

from __future__ import annotations

import numpy as np

OPEN_GRIPPER = -1.0
CLOSE_GRIPPER = +1.0


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
        pos_gain: float = 20.0,
        max_step: float = 1.0,
        xy_tol: float = 0.012,
        z_tol: float = 0.015,
        grasp_steps: int = 14,
        drop_steps: int = 10,
        settle_steps: int = 12,
        max_phase_steps: int = 90,
    ):
        self.obj_name = obj_name
        self.target_name = target_name
        self.hover_height = hover_height
        self.grasp_z_offset = grasp_z_offset
        self.grasp_frac = grasp_frac
        self.lift_height = lift_height
        self.place_drop = place_drop
        self.pos_gain = pos_gain
        self.max_step = max_step
        self.xy_tol = xy_tol
        self.z_tol = z_tol
        self.grasp_steps = grasp_steps
        self.drop_steps = drop_steps
        self.settle_steps = settle_steps
        self.max_phase_steps = max_phase_steps
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self):
        self.phase = "APPROACH"
        self.t = 0
        self.phase_t = 0
        self._grasp_xy = None
        self._carry_z = None  # absolute carry height, set at grasp (= grasp eef_z + lift_height)
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

    def _goto(self, obs, target_xyz, grip):
        eef = np.asarray(obs["robot0_eef_pos"], dtype=float)
        err = np.asarray(target_xyz, dtype=float) - eef
        a = np.zeros(7, dtype=float)
        a[:3] = np.clip(err * self.pos_gain, -self.max_step, self.max_step)
        # a[3:6] = 0 -> hold the (top-down) orientation setpoint
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
            target = obj + np.array([0.0, 0.0, self.hover_height])
            a, err = self._goto(obs, target, OPEN_GRIPPER)
            if (np.linalg.norm(err[:2]) < self.xy_tol and abs(err[2]) < self.z_tol) or timed_out:
                self._advance("DESCEND")

        elif self.phase == "DESCEND":
            target = np.array([obj[0], obj[1], self._grasp_z(obs, obj)])
            a, err = self._goto(obs, target, OPEN_GRIPPER)
            if abs(err[2]) < self.z_tol or timed_out:
                self._grasp_xy = eef[:2].copy()
                self._advance("GRASP")

        elif self.phase == "GRASP":
            target = np.array([obj[0], obj[1], self._grasp_z(obs, obj)])
            a, _ = self._goto(obs, target, CLOSE_GRIPPER)
            if self.phase_t >= self.grasp_steps:
                self._carry_z = eef[2] + self.lift_height  # clearance above the grasp
                self._advance("LIFT")

        elif self.phase == "LIFT":
            xy = self._grasp_xy if self._grasp_xy is not None else eef[:2]
            target = np.array([xy[0], xy[1], self._carry_z])  # absolute high carry
            a, err = self._goto(obs, target, CLOSE_GRIPPER)
            if abs(err[2]) < self.z_tol or timed_out:
                self._advance("MOVE")

        elif self.phase == "MOVE":
            target = np.array([tgt[0], tgt[1], self._carry_z])  # stay high while traversing
            a, err = self._goto(obs, target, CLOSE_GRIPPER)
            if np.linalg.norm(err[:2]) < 1.5 * self.xy_tol or timed_out:
                self._advance("LOWER")

        elif self.phase == "LOWER":
            target = np.array([tgt[0], tgt[1], tgt[2] + self.place_drop])
            a, err = self._goto(obs, target, CLOSE_GRIPPER)
            if abs(err[2]) < 1.5 * self.z_tol or timed_out:
                self._advance("DROP")

        elif self.phase == "DROP":
            target = np.array([tgt[0], tgt[1], tgt[2] + self.place_drop])
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

        return a, done


def infer_obj_and_target(env):
    """Best-effort: derive (manipuland, receptacle) from the env's obj_of_interest.

    LIBERO BDDL lists ``:obj_of_interest`` as [manipuland, receptacle] for pick-place.
    """
    ooi = list(env.obj_of_interest)
    if len(ooi) >= 2:
        return ooi[0], ooi[1]
    raise ValueError(f"Cannot infer obj/target from obj_of_interest={ooi}")
