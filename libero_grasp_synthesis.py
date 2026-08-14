"""Antipodal grasp synthesis for the LIBERO scripted demo generator.

Given a live simulator, propose top-down parallel-jaw grasp candidates for one object:
where to put the fingers (midpoint), which way they close (yaw), and how wide the grip
is. The scripted policy stays obs-only; the collector injects the chosen candidate as
``<obj>_grasp_xyz`` / ``<obj>_grasp_yaw`` and retries down a ranked ladder.

Pure NumPy + the MuJoCo arrays already exposed through ``env.sim`` (mesh vertices and
faces in ``model.mesh_*``) -- no new dependency.

Geometry conventions (all world frame):
  * A candidate is a pair of opposed surface contacts. The *closing direction* ``u`` is
    the unit vector from contact 1 to contact 2; the Panda's fingers close along the
    gripper's local y-axis, so the policy servos that axis onto ``yaw = atan2(u_y, u_x)``
    (mod pi -- a parallel jaw is symmetric under a 180 deg flip).
  * The antipodal test MUST be winding-agnostic: these converted meshes have ~97%
    *outward* face normals but no guarantee, and ``u`` points into the object at contact
    1. The robust form is ``|N1.u| > cos(atan mu)`` and ``|N2.u| > cos(atan mu)`` and
    ``(N1.u)(N2.u) < 0`` -- the naive ``N.u > cos(theta)`` form silently returns zero
    candidates on every object here.
  * Clearance means a *real overhang*: material further outward than the contact, above
    it, in the column a descending open finger sweeps -- NOT gentle curvature (a strict
    outward-projection test rejects an apple, a known 100%-yield object).

The mesh sampler can false-negative (measured: low-poly cereal/boxed_food meshes whose
3-5 cm short side trivially fits the jaw came back with 0 candidates, yet the plain
minor-axis pinch scores 4/4 at every height). ``rank_grasps`` therefore ALWAYS appends
the ``minor_axis_pinch`` fallback, so no object ever has an empty ladder.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MAX_JAW = 0.08          # Panda parallel-jaw opening (2 x 0.04 m finger travel)
PRE_GRASP_CLEARANCE = 0.012  # per-side gap the jaw pre-shapes to before descending
MAX_GRASP_WIDTH = 0.075  # leave 5 mm so the fingers close with margin
MIN_GRASP_WIDTH = 0.004  # thinner than this and the jaw bottoms out before contact
FINGER_LEN = 0.048      # fingertip-to-palm clearance below the eef
MU = 0.5                # friction coefficient for the antipodal cone
COS_CONE = 1.0 / np.sqrt(1.0 + MU * MU)   # cos(atan(mu))


def wrap_half_pi(a):
    """Wrap to (-pi/2, pi/2] -- parallel jaws are symmetric under a 180 deg flip."""
    return (a + np.pi / 2) % np.pi - np.pi / 2


def pre_grasp_half_for(half_width, clearance=PRE_GRASP_CLEARANCE):
    """Half-opening the jaw descends at for a grasp of ``2 * half_width``.

    Single source of truth shared by the clearance test here and the policy's jaw-width
    servo, so the two cannot drift apart. Mirrors
    ``pre_grasp_opening(width) = 2 * pre_grasp_half_for(width / 2)``.
    """
    return float(min(MAX_JAW / 2.0, half_width + clearance))


def pre_grasp_opening(width, clearance=PRE_GRASP_CLEARANCE):
    """Full jaw opening (m) to pre-shape to for a grasp of ``width``."""
    return float(np.clip(width + 2.0 * clearance, width + 0.004, MAX_JAW))


def object_com(sim, obj_name):
    """Mass-weighted world centre of mass of the object's body subtree.

    Grasp stability under the carry depends on where the MASS is, not where the mesh
    centroid is; MuJoCo gives the exact value via ``body_mass`` + ``body_ipos``. (Measured
    within 0.4 cm of the mesh centroid on these assets, so this is a refinement, not a
    correction -- but it costs nothing and is right by construction.)
    """
    m, d = sim.model, sim.data
    bids = _object_body_ids(sim, obj_name)
    tot, acc = 0.0, np.zeros(3)
    for i in bids:
        mass = float(m.body_mass[i])
        if mass <= 0:
            continue
        world = d.body_xpos[i] + d.body_xmat[i].reshape(3, 3) @ m.body_ipos[i]
        acc += mass * world
        tot += mass
    if tot <= 0:  # massless (visual-only) object -> fall back to the mesh centroid
        return object_world_points(sim, obj_name).mean(0)
    return acc / tot


@dataclass
class Grasp:
    """One executable top-down grasp candidate (world frame, at sampling time)."""

    midpoint: np.ndarray          # (3,) between the two contacts
    yaw: float                    # closing-direction heading, in (-pi/2, pi/2]
    width: float                  # contact-to-contact distance (m)
    height_frac: float            # (mid_z - bottom_z) / object height
    radial_offset: float          # xy distance to centroid / (max xy half-extent)
    score: float = 0.0
    kind: str = "central"         # "central" | "side" | "fallback"
    straddles_com: bool = True    # is the COM between the fingers? (holds under carry)
    lever_m: float = 0.0          # absolute xy distance midpoint -> COM, metres
    meta: dict = field(default_factory=dict)

    def summary(self):
        return (f"{self.kind:8s} w={self.width * 100:4.1f}cm h={self.height_frac:.2f} "
                f"r={self.radial_offset:.2f} lever={self.lever_m * 100:4.1f}cm "
                f"{'straddle' if self.straddles_com else 'OFFSET  '} "
                f"yaw={np.degrees(self.yaw):6.1f}deg score={self.score:.3f}")


# --------------------------------------------------------------------------- #
# raw geometry out of env.sim
# --------------------------------------------------------------------------- #
def body_subtree(model, root_name):
    """All body ids under ``root_name`` (inclusive)."""
    rid = model.body_name2id(root_name)
    ids, frontier = [rid], [rid]
    while frontier:
        p = frontier.pop()
        for i in range(model.nbody):
            if model.body_parentid[i] == p and i != p:
                ids.append(i)
                frontier.append(i)
    return ids


def _object_body_ids(sim, obj_name):
    m = sim.model
    root = f"{obj_name}_main"
    try:
        return set(body_subtree(m, root))
    except Exception:
        names = [m.body_id2name(i) for i in range(m.nbody)]
        cand = [n for n in names if n and n.startswith(obj_name)]
        if not cand:
            raise KeyError(f"no body found for object {obj_name!r}")
        return set(body_subtree(m, cand[0]))


def object_world_points(sim, obj_name):
    """World-frame mesh vertices (plus primitive-geom AABB corners) of the object."""
    m, d = sim.model, sim.data
    bids = _object_body_ids(sim, obj_name)
    pts = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] not in bids:
            continue
        gpos, gmat = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3)
        if m.geom_type[g] == 7:  # mjGEOM_MESH
            mid = m.geom_dataid[g]
            v0, nv = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            V = m.mesh_vert[v0:v0 + nv].reshape(-1, 3)
        else:
            s = m.geom_size[g]
            V = np.array([[sx * s[0], sy * s[1], sz * s[2]]
                          for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
        pts.append(gpos + V @ gmat.T)
    if not pts:
        raise ValueError(f"object {obj_name!r} has no geoms")
    return np.vstack(pts)


def object_xy_axes(sim, obj_name, points=None):
    """Horizontal principal axes: (major, minor, ext_major, ext_minor), world xy."""
    P = (points if points is not None else object_world_points(sim, obj_name))[:, :2]
    P = P - P.mean(0)
    _, _, Vt = np.linalg.svd(P, full_matrices=False)
    major, minor = Vt[0], Vt[1]
    span = lambda ax: float((P @ ax).max() - (P @ ax).min())  # noqa: E731
    return major, minor, span(major), span(minor)


def _subdivide_triangles(tri, max_edge=0.02, max_tris=60000):
    """Midpoint-split triangles until every edge is <= ``max_edge`` (bounded).

    Low-poly meshes (a cereal box is ~12 triangles) otherwise contribute a handful of
    surface samples and starve the pair sampling -- the measured box false-negative.
    """
    while len(tri) < max_tris:
        e = np.stack([np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1),
                      np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1),
                      np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1)], axis=1)
        big = e.max(axis=1) > max_edge
        if not big.any():
            break
        keep, split = tri[~big], tri[big]
        a, b, c = split[:, 0], split[:, 1], split[:, 2]
        ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
        tri = np.concatenate([
            keep,
            np.stack([a, ab, ca], axis=1), np.stack([ab, b, bc], axis=1),
            np.stack([ca, bc, c], axis=1), np.stack([ab, bc, ca], axis=1)])
    return tri


def face_samples(sim, obj_name, max_edge=0.02):
    """World-frame surface samples of the object's mesh geoms.

    Returns ``(centroids, unit_normals, areas)`` of the (subdivided) triangles --
    dense enough to double as the clearance point cloud.
    """
    m, d = sim.model, sim.data
    bids = _object_body_ids(sim, obj_name)
    C, N, A = [], [], []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] not in bids or m.geom_type[g] != 7:
            continue
        gpos, gmat = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3)
        mid = m.geom_dataid[g]
        v0 = m.mesh_vertadr[mid]
        V = m.mesh_vert[v0:v0 + m.mesh_vertnum[mid]].reshape(-1, 3)
        f0 = m.mesh_faceadr[mid]
        F = m.mesh_face[f0:f0 + m.mesh_facenum[mid]].reshape(-1, 3)
        tri = _subdivide_triangles(V[F], max_edge=max_edge)
        e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
        n = np.cross(e1, e2)
        area2 = np.linalg.norm(n, axis=1)
        ok = area2 > 1e-14
        tri, n, area2 = tri[ok], n[ok], area2[ok]
        n = n / area2[:, None]
        C.append(tri.mean(axis=1) @ gmat.T + gpos)
        N.append(n @ gmat.T)
        A.append(area2 / 2.0)
    if not C:
        return (np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0,)))
    return np.vstack(C), np.vstack(N), np.concatenate(A)


# --------------------------------------------------------------------------- #
# candidate generation
# --------------------------------------------------------------------------- #
def _finger_path_blocked(P, midpoint, u_out, half_width, pre_grasp_half=None,
                         finger_thick=0.010, lat_half=0.012, z_eps=0.003,
                         finger_len=FINGER_LEN):
    """True if the object blocks one finger's actual path to its contact.

    The finger comes straight down at ``pre_grasp_half`` outward of the grasp midpoint --
    material above the grasp height in that column is a real overhang -- and then closes
    horizontally, its ``finger_len``-tall body sweeping the band from there in to the
    contact. Gentle curvature inside the contact width blocks nothing (an apple's equator
    passes); a bowl's flared wall over a base grasp correctly does.

    ``pre_grasp_half`` is the half-opening the jaw actually descends at, NOT the full 4 cm
    stroke. **This is only valid because the policy pre-shapes the jaw to
    ``grasp_width + 2 * pre_grasp_clearance`` before descending** (see
    ``libero_scripted_policy``'s jaw-width servo). Keep the two in sync: widening the
    policy's pre-shape without widening this makes the sampler optimistic, and vice versa
    makes it reject reachable grasps -- the latter is what returned 0 candidates on every
    donut when this modelled the fully-open jaw.
    """
    if pre_grasp_half is None:
        pre_grasp_half = pre_grasp_half_for(half_width)
    d = P - midpoint
    t = np.array([-u_out[1], u_out[0]])
    out = d[:, 0] * u_out[0] + d[:, 1] * u_out[1]
    lat = np.abs(d[:, 0] * t[0] + d[:, 1] * t[1])
    z = d[:, 2]
    descent = (np.abs(out - pre_grasp_half) < finger_thick) & (lat < lat_half) & (z > z_eps)
    sweep = ((out > half_width + 0.002) & (out < pre_grasp_half + finger_thick)
             & (lat < lat_half) & (z > z_eps) & (z < finger_len))
    return bool((descent | sweep).any())


def _palm_blocked(P, midpoint, u, half_width, z_palm, lat_half=0.015):
    """True if material between the fingers rises above where the palm will sit."""
    d = P - midpoint
    t = np.array([-u[1], u[0], 0.0])
    out = d[:, 0] * u[0] + d[:, 1] * u[1]
    lat = d[:, 0] * t[0] + d[:, 1] * t[1]
    blocked = ((np.abs(out) <= half_width + 0.005) & (np.abs(lat) < lat_half)
               & (P[:, 2] > z_palm))
    return bool(blocked.any())


def sample_grasps(sim, obj_name, max_pairs_samples=3000, max_candidates=1500,
                  max_width=MAX_GRASP_WIDTH, cos_cone=COS_CONE, max_tilt=0.35,
                  central_r=0.30, min_floor_clear=0.010, floor_clear_frac=0.35,
                  rng=None):
    """Propose antipodal top-down grasps for ``obj_name`` from the live sim state.

    Returns an unranked ``list[Grasp]`` (empty is possible -- callers must go through
    ``rank_grasps`` so the fallback pinch is appended).
    """
    rng = np.random.default_rng(rng)
    C_all, N_all, A = face_samples(sim, obj_name)
    if len(C_all) == 0:
        return []
    P = object_world_points(sim, obj_name)
    all_pts = np.vstack([P, C_all])       # dense clearance cloud
    centroid = P.mean(0)
    com = object_com(sim, obj_name)
    bottom_z, top_z = float(P[:, 2].min()), float(P[:, 2].max())
    height = max(top_z - bottom_z, 1e-6)
    _, _, ext_maj, _ = object_xy_axes(sim, obj_name, points=P)
    half_extent = max(ext_maj / 2.0, 1e-6)

    # How many surface samples enter the O(n^2) pair search. This is the single most
    # important recall knob and it was far too low: at 300 samples a thin handle or a narrow
    # ring is a tiny fraction of area-weighted surface, so the chance of drawing two samples
    # on OPPOSITE faces of it is negligible. Measured going 300 -> 3000: kettle_6 0 -> 5,
    # kettle_1 0 -> 28, teapot 0 -> 37, jug 0 -> 11, pizza_cutter 0 -> 91, steak_2 0 -> 5,
    # and the recovered grasps are mostly CENTRAL with ~0.1 cm levers. Cost is 2-16 s once
    # per task, against minutes of rollouts.
    #
    # NOTE: `max_candidates` subsamples surviving pairs BEFORE the (expensive, per-candidate)
    # clearance loop, so shrinking it can throw away the few candidates that would have
    # survived -- measured pitcher 3 -> 0 at a 500 cap. Do not lower it to buy speed.
    if len(C_all) > max_pairs_samples:
        idx = rng.choice(len(C_all), size=max_pairs_samples, replace=False,
                         p=A / A.sum())
        C, N = C_all[idx], N_all[idx]
    else:
        C, N = C_all, N_all

    n = len(C)
    ii, jj = np.triu_indices(n, k=1)
    w = C[jj] - C[ii]
    width = np.linalg.norm(w, axis=1)
    ok = (width > MIN_GRASP_WIDTH) & (width < max_width)
    u = np.zeros_like(w)
    u[ok] = w[ok] / width[ok, None]
    # near-horizontal closing direction (the wrist only yaws, it never tilts)
    ok &= np.abs(u[:, 2]) < max_tilt
    # winding-agnostic antipodal test
    d1 = np.einsum("ij,ij->i", N[ii], u)
    d2 = np.einsum("ij,ij->i", N[jj], u)
    ok &= (np.abs(d1) > cos_cone) & (np.abs(d2) > cos_cone) & (d1 * d2 < 0)
    # contacts at compatible heights, midpoint not scraping the floor
    mid = (C[ii] + C[jj]) / 2.0
    ok &= np.abs(C[ii][:, 2] - C[jj][:, 2]) < 0.02
    # Floor clearance must scale with the object: a flat object (pizza cutter 0.9cm, tongs
    # 1.1cm) has NO grasp 1cm above its own base, so a fixed 1cm silently rejected every
    # candidate on it -- measured 444 -> 0 on pizza_cutter, 393 -> 0 on tongs. Tall objects
    # are unaffected (0.35 * height exceeds 1cm above ~2.9cm).
    floor_clear = min(min_floor_clear, floor_clear_frac * height)
    ok &= mid[:, 2] > bottom_z + floor_clear

    cand_idx = np.flatnonzero(ok)
    if len(cand_idx) > max_candidates:
        cand_idx = rng.choice(cand_idx, size=max_candidates, replace=False)

    grasps = []
    for k in cand_idx:
        i, j = ii[k], jj[k]
        uk = u[k].copy()
        uk[2] = 0.0
        nrm = np.linalg.norm(uk)
        if nrm < 1e-6:
            continue
        uk /= nrm
        midpoint = mid[k]
        # both fingers must reach their contacts: open descent + closing sweep clear
        if _finger_path_blocked(all_pts, midpoint, -uk, width[k] / 2.0):
            continue
        if _finger_path_blocked(all_pts, midpoint, +uk, width[k] / 2.0):
            continue
        if _palm_blocked(all_pts, midpoint, uk, width[k] / 2.0,
                         midpoint[2] + FINGER_LEN):
            continue
        radial = float(np.linalg.norm((midpoint - centroid)[:2])) / half_extent
        hfrac = float((midpoint[2] - bottom_z) / height)
        align = (abs(d1[k]) + abs(d2[k])) / 2.0
        lever = float(np.linalg.norm((midpoint - com)[:2]))
        s1 = float((C[i] - com)[:2] @ uk[:2])
        s2 = float((C[j] - com)[:2] @ uk[:2])
        straddles = bool(s1 * s2 < 0)
        # SCORE IS FEASIBILITY ONLY -- how squarely opposed the contacts are and how level
        # they are -- NOT a reliability predictor. Whether a grasp actually holds (a handle
        # grasp is great on a mug but slips on a pitcher; a central grasp is stable on an
        # apple but rolls off a banana) is DYNAMIC and only the simulator knows it. Earlier
        # a `-4*lever + 0.5*straddle` reliability prior was baked in here; it was internally
        # contradictory (it must both favour and disfavour handle grasps depending on the
        # object) and regressed ~270 tasks. Reliability is now decided by calibration over a
        # DIVERSE ladder (rank_grasps); lever/straddles_com survive only as diversity axes.
        score = align - 0.3 * abs(hfrac - 0.4) - 5.0 * abs(C[i][2] - C[j][2])
        grasps.append(Grasp(
            midpoint=midpoint.copy(),
            yaw=wrap_half_pi(float(np.arctan2(uk[1], uk[0]))),
            width=float(width[k]),
            height_frac=hfrac,
            radial_offset=radial,
            score=float(score),
            kind="central" if radial <= central_r else "side",
            straddles_com=straddles,
            lever_m=lever,
        ))
    return grasps


def minor_axis_pinch(sim, obj_name, grasp_frac=0.55):
    """The guaranteed fallback: pinch at the object origin along the minor xy axis.

    This is exactly the measured-good yaw-servo grasp (took all 7 sampler-false-negative
    boxes to 4/4 at every height), so it is always appended to the ladder.
    """
    P = object_world_points(sim, obj_name)
    _, minor, _, ext_min = object_xy_axes(sim, obj_name, points=P)
    bottom_z, top_z = float(P[:, 2].min()), float(P[:, 2].max())
    centroid_xy = P[:, :2].mean(0)
    z = bottom_z + grasp_frac * (top_z - bottom_z)
    com = object_com(sim, obj_name)
    return Grasp(
        midpoint=np.array([centroid_xy[0], centroid_xy[1], z]),
        yaw=wrap_half_pi(float(np.arctan2(minor[1], minor[0]))),
        width=float(min(ext_min, MAX_JAW)),
        height_frac=float(grasp_frac),
        radial_offset=0.0,
        score=-1.0,
        kind="fallback",
        # a centroid pinch straddles the COM by construction
        straddles_com=True,
        lever_m=float(np.linalg.norm(centroid_xy - com[:2])),
    )


def rank_grasps(grasps, sim=None, obj_name=None, k=6, fallback_grasp_frac=0.55):
    """Assemble a DIVERSE retry ladder for the simulator to choose from.

    The score is feasibility-only (see ``sample_grasps``), so ranking must NOT just take
    the top-k by score -- with a dense candidate set that yields k near-duplicates of the
    single most-antipodal grasp, and if that grasp is unreliable (a banana's centre, a
    mug's body) the whole ladder is unreliable. Instead the ladder SPANS the strategy
    space so the robust grasp -- whatever kind it is -- is present for calibration to find:
    the object is bucketed by ``kind`` (central/side, i.e. body vs rim/handle) x height
    band (low/mid/high), and the best-scoring representative of each occupied bucket is
    taken, filling any remaining slots with the next-best-scoring candidates. Calibration
    (in the collector) then selects by MEASURED yield -- so a handle grasp wins on a mug
    and loses on a pitcher, decided by physics rather than a geometric guess.

    The minor-axis centroid pinch is always appended (when ``sim`` given) as a guaranteed
    fallback.
    """
    def hband(h):
        return "low" if h < 0.45 else ("mid" if h < 0.70 else "high")

    buckets = {}
    for g in sorted(grasps, key=lambda g: -g.score):
        buckets.setdefault((g.kind, hband(g.height_frac)), []).append(g)
    # one representative (best-scored) per occupied bucket, richest buckets first so a
    # near-empty run still yields a spread
    ladder, used = [], set()
    for cell in sorted(buckets, key=lambda c: -len(buckets[c])):
        ladder.append(buckets[cell][0]); used.add(id(buckets[cell][0]))
        if len(ladder) >= k:
            break
    # fill remaining slots with the next best-scoring candidates not already chosen
    if len(ladder) < k:
        for g in sorted(grasps, key=lambda g: -g.score):
            if id(g) not in used:
                ladder.append(g)
                if len(ladder) >= k:
                    break
    if sim is not None and obj_name is not None:
        ladder.append(minor_axis_pinch(sim, obj_name, grasp_frac=fallback_grasp_frac))
    return ladder


def grasp_to_local(grasp, obj_pos, obj_mat):
    """Express a world-frame grasp relative to the object's current pose.

    Returns ``(local_offset(3,), yaw_rel)`` such that the collector can re-derive the
    live world grasp each step even if the object shifts before pickup.
    """
    off = obj_mat.T @ (np.asarray(grasp.midpoint, float) - np.asarray(obj_pos, float))
    obj_yaw = float(np.arctan2(obj_mat[1, 0], obj_mat[0, 0]))
    return off, wrap_half_pi(grasp.yaw - obj_yaw)


def grasp_from_local(local_offset, yaw_rel, obj_pos, obj_mat):
    """Inverse of :func:`grasp_to_local` at the object's current pose."""
    xyz = np.asarray(obj_pos, float) + obj_mat @ np.asarray(local_offset, float)
    obj_yaw = float(np.arctan2(obj_mat[1, 0], obj_mat[0, 0]))
    return xyz, wrap_half_pi(yaw_rel + obj_yaw)


# --------------------------------------------------------------------------- #
# CLI: inspect one task's candidate ladder (diagnosis + unit checks)
# --------------------------------------------------------------------------- #
def _main():
    import argparse
    import json as _json
    import os
    import sys

    repo = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(repo, "scripts"))
    import init_path  # noqa: F401
    from robosuite import load_controller_config
    from robosuite.utils.errors import RandomizationError

    import libero.libero.envs.bddl_utils as BDDLUtils
    from libero.libero.envs import TASK_MAPPING

    ap = argparse.ArgumentParser()
    ap.add_argument("--bddl-file", required=True)
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    problem_info = BDDLUtils.get_problem_info(args.bddl_file)
    env = None
    for attempt in range(10):
        try:
            env = TASK_MAPPING[problem_info["problem_name"]](
                bddl_file_name=args.bddl_file, robots=["Panda"],
                controller_configs=load_controller_config(default_controller="OSC_POSE"),
                has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
                ignore_done=True, initialization_noise=None, control_freq=20)
            break
        except RandomizationError:
            continue
    if env is None:
        raise SystemExit("env build failed 10x (RandomizationError)")
    for _ in range(20):
        try:
            env.reset()
            break
        except Exception:
            continue
    obj_name = list(env.obj_of_interest)[0]

    grasps = sample_grasps(env.sim, obj_name, rng=args.seed)
    ladder = rank_grasps(grasps, sim=env.sim, obj_name=obj_name, k=args.top_k)
    n_central = sum(g.kind == "central" for g in grasps)
    n_side = sum(g.kind == "side" for g in grasps)
    print(f"[{obj_name}] {len(grasps)} candidates ({n_central} central / {n_side} side)")
    for g in ladder:
        print("  " + g.summary())
    if args.out_json:
        with open(args.out_json, "w") as f:
            _json.dump({"obj": obj_name, "n": len(grasps), "n_central": n_central,
                        "n_side": n_side,
                        "ladder": [dict(width=g.width, yaw=g.yaw, kind=g.kind,
                                        height_frac=g.height_frac,
                                        radial_offset=g.radial_offset, score=g.score,
                                        midpoint=[float(x) for x in g.midpoint])
                                   for g in ladder]}, f, indent=2)
    env.close()


if __name__ == "__main__":
    _main()
