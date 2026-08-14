"""Collect scripted top-down pick-and-place demonstrations for a LIBERO task.

This is the autonomous analogue of ``scripts/collect_demonstration.py``: instead of a
human teleop device driving the arm via ``input2action``, the 7-D OSC actions come from
``ScriptedPickPlacePolicy``. Only episodes that pass ``env._check_success()`` are kept.

Grasp selection: at task start an antipodal sampler (``libero_grasp_synthesis``)
proposes a ranked ladder of grasp candidates (central pinches, side/rim pinches, plus a
guaranteed minor-axis fallback at several heights). A short warmup calibrates over the
ladder -- the first candidate with a perfect warmup wins immediately, otherwise the
best-yielding one -- and the batch is collected at that setting. Candidates are stored
relative to the object's pose and injected into ``obs`` live each step
(``<obj>_grasp_xyz`` / ``<obj>_grasp_yaw``), keeping the policy obs-only.

Placement reliability: environment construction retries ``RandomizationError`` with a
fresh numpy seed per attempt (the constructor runs one placement sample itself), and
``--init-states`` seeds episodes from a stored init-state pool for tasks whose random
placement sampling is too flaky to reset reliably.

The output ``demo.hdf5`` matches robosuite's collection format (low-dim states + actions
+ per-demo ``model_file`` xml + the ``data`` attrs ``env``/``env_info``/``problem_info``/
``bddl_file_name``/``bddl_file_content``), so it feeds straight into
``scripts/create_dataset.py --use-camera-obs`` which renders the images later. The env is
built fully headless (no on-screen / no offscreen renderer / no camera obs), so collection
needs no GPU and only the cheap low-dim observables the policy reads.

Example:
    python scripts/collect_scripted_demonstrations.py \
        --bddl-file /path/to/task.bddl --num-success 20 --out-dir /tmp/demos/task
"""

import argparse
import datetime
import json
import os
import shutil
import sys
import time
from glob import glob

import h5py
import numpy as np
import robosuite as suite
from robosuite import load_controller_config
from robosuite.utils import transform_utils as T
from robosuite.utils.errors import RandomizationError
from robosuite.wrappers import DataCollectionWrapper

import init_path  # noqa: F401  (adds repo to sys.path, mirrors other scripts)
import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero.envs import TASK_MAPPING

# make the repo-root policy importable regardless of CWD
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from libero_scripted_policy import ScriptedPickPlacePolicy, infer_obj_and_target  # noqa: E402
import libero_grasp_synthesis as GS  # noqa: E402


def gather_successful_demos_as_hdf5(tmp_dir, out_dir, env_info, problem_info,
                                    bddl_file, keep_dirs, out_name="demo.hdf5"):
    """Aggregate only the ``keep_dirs`` episodes from ``tmp_dir`` into out_dir/<out_name>.

    Mirrors collect_demonstration.gather_demonstrations_as_hdf5 but (a) filters to the
    successful episode directories, (b) takes problem_info/bddl_file explicitly rather
    than via script globals, and (c) stores the real bddl file contents.
    """
    os.makedirs(out_dir, exist_ok=True)
    hdf5_path = os.path.join(out_dir, out_name)
    keep = set(keep_dirs)
    f = h5py.File(hdf5_path, "w")
    grp = f.create_group("data")

    num_eps = 0
    env_name = None
    for ep_directory in sorted(os.listdir(tmp_dir)):
        if ep_directory not in keep:
            continue
        states, actions = [], []
        for state_file in sorted(glob(os.path.join(tmp_dir, ep_directory, "state_*.npz"))):
            dic = np.load(state_file, allow_pickle=True)
            env_name = str(dic["env"])
            states.extend(dic["states"])
            for ai in dic["action_infos"]:
                actions.append(ai["actions"])
        if len(states) == 0:
            continue
        # states were recorded AFTER each action -> drop the trailing state
        del states[-1]
        assert len(states) == len(actions), (len(states), len(actions))

        num_eps += 1
        ep_grp = grp.create_group("demo_{}".format(num_eps))
        with open(os.path.join(tmp_dir, ep_directory, "model.xml"), "r") as mf:
            ep_grp.attrs["model_file"] = mf.read()
        ep_grp.create_dataset("states", data=np.array(states))
        ep_grp.create_dataset("actions", data=np.array(actions))

    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["repository_version"] = suite.__version__
    grp.attrs["env"] = env_name
    grp.attrs["env_info"] = env_info
    grp.attrs["problem_info"] = json.dumps(problem_info)
    grp.attrs["bddl_file_name"] = bddl_file
    with open(bddl_file, "r", encoding="utf-8") as bf:
        grp.attrs["bddl_file_content"] = bf.read()
    f.close()
    return hdf5_path, num_eps


def inject_obj_extents(env, obs, obj_name):
    """Add the manipuland's live world-frame bottom/top z to ``obs`` in place.

    The object's ``bottom_offset``/``top_offset`` (from the synthesized bottom/top sites)
    are local to its root body; combined with the *current* body pose from obs
    (``<obj>_pos`` world position + ``<obj>_quat`` in xyzw) this yields the true resting
    extents, which the policy turns into a geometry-relative grasp height. Best-effort:
    if the object or its sites are unavailable, leave obs untouched (policy falls back to
    ``--grasp-z-offset``)."""
    try:
        obj = env.objects_dict[obj_name]
        bottom = np.asarray(obj.bottom_offset, dtype=float)
        top = np.asarray(obj.top_offset, dtype=float)
        pos = np.asarray(obs[f"{obj_name}_pos"], dtype=float)
        R = T.quat2mat(np.asarray(obs[f"{obj_name}_quat"], dtype=float))  # obs quat is xyzw
    except (KeyError, AttributeError):
        return obs
    obs[f"{obj_name}_bottom_z"] = float(pos[2] + (R @ bottom)[2])
    obs[f"{obj_name}_top_z"] = float(pos[2] + (R @ top)[2])
    return obs


RIM_REFRESH_EVERY = 20  # steps; the basket settles ~4cm in the first two steps


def receptacle_rim_z(env, target_name):
    """World z of the receptacle's highest point (its rim).

    Walks the target's mesh vertices, so this is refreshed every ``RIM_REFRESH_EVERY``
    steps rather than every step. It must NOT be cached from the reset pose alone: the
    basket drops ~4cm into the floor within the first two steps of every episode, so a
    rim measured at reset is 4cm too high and would release the object from that much
    further up. Returns None if the geometry is unavailable."""
    try:
        return float(GS.object_world_points(env.sim, target_name)[:, 2].max())
    except Exception:
        return None


def inject_rim(obs, target_name, rim_z):
    """Add the cached receptacle rim height to ``obs`` in place (best-effort)."""
    if rim_z is not None:
        obs[f"{target_name}_rim_z"] = rim_z
    return obs


def inject_grasp(obs, obj_name, candidate):
    """Add the live world-frame grasp pose of an object-local ``candidate`` to ``obs``.

    ``candidate`` holds ``local_off``/``yaw_rel`` (from ``GS.grasp_to_local``); composing
    with the object's *current* pose keeps the target valid even if the object shifted.
    Best-effort, same contract as ``inject_obj_extents``."""
    if candidate is None:
        return obs
    try:
        pos = np.asarray(obs[f"{obj_name}_pos"], dtype=float)
        R = T.quat2mat(np.asarray(obs[f"{obj_name}_quat"], dtype=float))
    except KeyError:
        return obs
    xyz, yaw = GS.grasp_from_local(candidate["local_off"], candidate["yaw_rel"], pos, R)
    obs[f"{obj_name}_grasp_xyz"] = xyz
    obs[f"{obj_name}_grasp_yaw"] = yaw
    if candidate.get("width") is not None:
        # lets the policy pre-shape the jaw instead of descending fully open, which is what
        # the sampler's clearance test assumes (GS.pre_grasp_half_for)
        obs[f"{obj_name}_grasp_width"] = float(candidate["width"])
    return obs


def build_env(bddl_file, problem_name, config, build_tries=15, seed=0):
    """Construct the headless env, retrying RandomizationError with fresh numpy seeds.

    The constructor runs one placement sample itself; a task with a tight region can
    need several draws, and reseeding matters because an unlucky RNG stream can stay
    unlucky. Returns ``(env, n_attempts)``."""
    last = None
    for attempt in range(build_tries):
        np.random.seed(seed + 7919 * attempt)
        try:
            env = TASK_MAPPING[problem_name](
                bddl_file_name=bddl_file,
                **config,
                has_renderer=False,
                has_offscreen_renderer=False,
                use_camera_obs=False,
                ignore_done=True,
                initialization_noise=None,
                control_freq=20,
            )
            return env, attempt + 1
        except RandomizationError as e:
            last = e
    raise RuntimeError(f"env build failed {build_tries}x (RandomizationError: {last})")


def safe_reset(env, max_tries=20, init_states=None, rng=None):
    """Reset, retrying RandomizationError; optionally seed from a stored state pool."""
    for _ in range(max_tries):
        try:
            obs = env.reset()
        except Exception:
            continue
        if init_states is not None:
            state = init_states[rng.integers(len(init_states))]
            env.sim.set_state_from_flattened(np.asarray(state, dtype=np.float64))
            env.sim.forward()
            obs = env._get_observations(force_update=True)
        return obs
    raise RuntimeError("env.reset() failed repeatedly (RandomizationError?)")


def load_init_states(path):
    """Load a ``.pruned_init``-format pool (torch-saved (N, state_dim) float64)."""
    import torch
    arr = torch.load(path, map_location="cpu")
    return np.asarray(arr, dtype=np.float64)


def load_grasp_entry(path):
    """Load a previously chosen ladder entry (see ``--grasp-json``).

    Accepts either a bare entry dict or a whole ``result.json`` (the entry is pulled from
    ``chosen_entry``/``chosen``). The entry is object-LOCAL -- ``inject_grasp`` composes
    ``local_off``/``yaw_rel`` with the object's live pose -- so it stays valid for the same
    object in any scene layout, which is the point: re-collecting a known-good object in a
    new layout needs no sampling and no calibration.
    """
    with open(path) as f:
        blob = json.load(f)
    entry = blob
    for key in ("chosen_entry", "chosen"):
        if isinstance(blob, dict) and isinstance(blob.get(key), dict):
            entry = blob[key]
            break
    if not isinstance(entry, dict):
        raise SystemExit(f"{path}: expected a grasp entry dict")
    missing = [k for k in ("kind", "local_off", "yaw_rel") if entry.get(k) is None]
    if missing:
        raise SystemExit(f"{path}: grasp entry is missing {missing} "
                         f"(got keys {sorted(entry)})")
    entry = dict(entry)
    entry["local_off"] = np.asarray(entry["local_off"], dtype=float)
    entry.setdefault("width", None)
    entry.setdefault("height_frac", None)
    entry.setdefault("radial_offset", 0.0)
    entry.setdefault("score", None)
    entry.setdefault("grasp_frac", None)
    if entry["kind"] == "grasp_frac":
        raise SystemExit(f"{path}: kind='grasp_frac' carries no pose to inject; "
                         f"use --no-grasp-sampler --grasp-frac instead")
    return entry


def build_candidate_ladder(env, obj_name, top_k, fallback_fracs, mu, seed,
                           n_sample_resets=3, reset_fn=None):
    """Sample + rank grasps over a few resets; return object-local ladder entries.

    Objects spawn at a fixed yaw with only xy jitter, so object-local candidates are
    valid across resets -- but the candidate COUNT swings with the settled pose (a slight
    tilt changes which grasps pass the near-horizontal test), so sampling once at a single
    reset is a lottery that occasionally starves an otherwise-easy object. Sampling over
    ``n_sample_resets`` resets and unioning the object-local candidates (dedup at ~5 mm /
    ~0.1 rad) removes that variance. The ladder always ends with the minor-axis fallback
    pinch at each ``fallback_fracs`` height, so it is never empty.
    """
    cos_cone = 1.0 / np.sqrt(1.0 + mu * mu)
    seen, pooled = set(), []          # dedup key -> object-local candidate dicts
    n_raw = 0
    for i in range(max(1, n_sample_resets)):
        if i > 0 and reset_fn is not None:
            try:
                reset_fn()
            except Exception:
                break                  # can't reset -> stop, use what we have
        obj_pos = env.sim.data.get_body_xpos(f"{obj_name}_main")
        obj_mat = env.sim.data.get_body_xmat(f"{obj_name}_main").reshape(3, 3)
        try:
            grasps = GS.sample_grasps(env.sim, obj_name, cos_cone=cos_cone, rng=seed + i)
        except Exception as e:         # sampler crash must never kill collection
            print(f"[sampler] failed ({type(e).__name__}: {e})")
            grasps = []
        n_raw += len(grasps)
        for g in grasps:
            off, yaw_rel = GS.grasp_to_local(g, obj_pos, obj_mat)
            key = (round(float(off[0]), 3), round(float(off[1]), 3),
                   round(float(off[2]), 3), round(float(yaw_rel), 1))
            if key in seen:
                continue
            seen.add(key)
            pooled.append(dict(off=off, yaw_rel=yaw_rel, width=g.width,
                               height_frac=g.height_frac, radial_offset=g.radial_offset,
                               score=g.score, kind=g.kind,
                               straddles_com=g.straddles_com, lever_m=g.lever_m))

    # reconstruct Grasp objects at the CURRENT pose so rank_grasps (interleave + fallback)
    # can be reused unchanged
    obj_pos = env.sim.data.get_body_xpos(f"{obj_name}_main")
    obj_mat = env.sim.data.get_body_xmat(f"{obj_name}_main").reshape(3, 3)
    reconstructed = []
    for c in pooled:
        xyz, yaw = GS.grasp_from_local(c["off"], c["yaw_rel"], obj_pos, obj_mat)
        reconstructed.append(GS.Grasp(
            midpoint=xyz, yaw=yaw, width=c["width"], height_frac=c["height_frac"],
            radial_offset=c["radial_offset"], score=c["score"], kind=c["kind"],
            straddles_com=c["straddles_com"], lever_m=c["lever_m"]))
    ranked = GS.rank_grasps(reconstructed, sim=env.sim, obj_name=obj_name, k=top_k)

    entries = []
    for g in ranked:
        off, yaw_rel = GS.grasp_to_local(g, obj_pos, obj_mat)
        entries.append({"kind": g.kind, "local_off": off, "yaw_rel": yaw_rel,
                        "width": g.width, "height_frac": g.height_frac,
                        "radial_offset": g.radial_offset, "score": g.score,
                        "grasp_frac": None})
    for frac in fallback_fracs:
        try:
            g = GS.minor_axis_pinch(env.sim, obj_name, grasp_frac=frac)
        except Exception:
            continue
        off, yaw_rel = GS.grasp_to_local(g, obj_pos, obj_mat)
        entries.append({"kind": "fallback", "local_off": off, "yaw_rel": yaw_rel,
                        "width": g.width, "height_frac": frac,
                        "radial_offset": 0.0, "score": g.score, "grasp_frac": frac})
    stats = {"n_candidates": len(pooled), "n_raw_samples": n_raw,
             "n_resets": max(1, n_sample_resets),
             "n_central": sum(c["kind"] == "central" for c in pooled),
             "n_side": sum(c["kind"] == "side" for c in pooled)}
    return entries, stats


def entry_desc(e):
    if e is None:
        return "grasp_frac-only"
    tag = f"{e['kind']} w={e['width'] * 100:.1f}cm h={e['height_frac']:.2f}"
    return tag + (f" r={e['radial_offset']:.2f}" if e["kind"] != "fallback" else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bddl-file", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True,
                    help="where demo.hdf5 is written")
    ap.add_argument("--tmp-dir", type=str, default=None,
                    help="scratch dir for per-episode state npz (default: under out-dir)")
    ap.add_argument("--obj-name", type=str, default=None,
                    help="manipuland instance (default: infer from obj_of_interest[0])")
    ap.add_argument("--target-name", type=str, default=None,
                    help="receptacle instance (default: infer from obj_of_interest[1])")
    ap.add_argument("--num-success", type=int, default=20)
    ap.add_argument("--max-attempts", type=int, default=80)
    ap.add_argument("--horizon", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--build-tries", type=int, default=15,
                    help="constructor retries on RandomizationError (fresh seed each)")
    ap.add_argument("--init-states", type=str, default=None,
                    help=".pruned_init-format state pool; episodes are seeded from it "
                         "instead of trusting the flaky random placement sampler")
    # policy knobs (override per geometry if needed)
    ap.add_argument("--hover-height", type=float, default=0.12)
    ap.add_argument("--grasp-z-offset", type=float, default=0.005,
                    help="fallback origin-relative grasp offset (used only if live "
                         "extents can't be injected)")
    ap.add_argument("--grasp-frac", type=float, default=0.65,
                    help="grasp height fraction for the no-sampler path")
    ap.add_argument("--grasp-frac-grid", type=str, default="0.5,0.65,0.8",
                    help="fallback-pinch heights appended to the candidate ladder (and "
                         "the whole ladder when --no-grasp-sampler)")
    ap.add_argument("--calib-attempts", type=int, default=8,
                    help="warmup rollouts per ladder entry; the ladder is evaluated by "
                         "measured yield and the best kept (8 so a ~67%% grasp can't win "
                         "on a small-sample fluke, as 5/5 allowed)")
    ap.add_argument("--calib-early-accept", type=int, default=8,
                    help="stop calibration early once an entry warms up perfect over at "
                         "least this many attempts (strong evidence -> skip the rest)")
    ap.add_argument("--no-calibrate", action="store_true",
                    help="skip calibration; use the ladder's first entry (or "
                         "--grasp-frac with --no-grasp-sampler)")
    ap.add_argument("--grasp-json", type=str, default=None,
                    help="path to a JSON grasp entry (a `chosen_entry` from an earlier "
                         "run's result.json) to use as the ONLY ladder entry. Skips both "
                         "the sampler and calibration: the entry is object-local, so it "
                         "stays valid wherever the object spawns, which makes this the "
                         "cheap way to re-collect a known-good object in new scene layouts "
                         "-- and keeps the grasp identical across them")
    ap.add_argument("--top-k-candidates", type=int, default=8,
                    help="sampler candidates on the ladder (before the fallbacks); 8 so a "
                         "working grasp is reached even when hold-quality scoring mis-ranks "
                         "it below the top few (measured: cucumber/steak recovered at 8-10)")
    ap.add_argument("--sample-resets", type=int, default=3,
                    help="resets over which grasp candidates are sampled and unioned; >1 "
                         "removes the single-reset lottery that starves easy objects")
    ap.add_argument("--mu", type=float, default=0.5,
                    help="friction coefficient for the antipodal cone (escalation may "
                         "relax it upward)")
    ap.add_argument("--lift-height", type=float, default=0.20)
    ap.add_argument("--place-drop", type=float, default=0.06)
    # old-behaviour switches
    ap.add_argument("--no-grasp-sampler", action="store_true",
                    help="no candidate ladder: calibrate grasp_frac exactly as before")
    ap.add_argument("--no-align-yaw", action="store_true",
                    help="disable the wrist yaw servo (reproduces the fixed-yaw policy)")
    ap.add_argument("--no-center-placement", action="store_true",
                    help="steer the eef (not the object origin) onto the basket centre")
    ap.add_argument("--release-above-rim", type=float, default=0.02,
                    help="stop the descent this far above the receptacle rim and release "
                         "there, so the hand never enters the basket (stock LIBERO-Object "
                         "demos release a median 2.1cm above the rim)")
    ap.add_argument("--no-rim-release", action="store_true",
                    help="disable the rim clamp (reproduces the hand-inside-the-basket "
                         "behaviour of the first collection run)")
    ap.add_argument("--pre-grasp-clearance", type=float, default=0.012,
                    help="per-side gap the jaw pre-shapes to before descending; must match "
                         "libero_grasp_synthesis.PRE_GRASP_CLEARANCE, which the sampler's "
                         "clearance test assumes")
    ap.add_argument("--no-preshape", action="store_true",
                    help="always descend with the jaw fully open (pre-fix behaviour); note "
                         "this makes the sampler's clearance test optimistic")
    ap.add_argument("--result-json", type=str, default=None,
                    help="write a machine-readable gate result (chosen candidate, "
                         "calib/collect yields, reached_target, n_demos) here")
    ap.add_argument("--keep-tmp-states", action="store_true",
                    help="keep the per-episode npz scratch after demo.hdf5 is written "
                         "(default: delete -- at suite scale it would eat the disk)")
    args = ap.parse_args()

    assert os.path.exists(args.bddl_file), args.bddl_file
    problem_info = BDDLUtils.get_problem_info(args.bddl_file)
    problem_name = problem_info["problem_name"]
    print("[task]", problem_info["language_instruction"])

    controller_config = load_controller_config(default_controller="OSC_POSE")
    config = {"robots": ["Panda"], "controller_configs": controller_config}
    env_info = json.dumps(config)

    env, build_attempts = build_env(args.bddl_file, problem_name, config,
                                    build_tries=args.build_tries, seed=args.seed)
    if build_attempts > 1:
        print(f"[build] constructor needed {build_attempts} attempts")

    tmp_dir = args.tmp_dir or os.path.join(args.out_dir, "_tmp_states",
                                           str(time.time()).replace(".", "_"))
    os.makedirs(tmp_dir, exist_ok=True)
    env = DataCollectionWrapper(env, tmp_dir)
    env.seed(args.seed)

    init_states = load_init_states(args.init_states) if args.init_states else None
    rng = np.random.default_rng(args.seed)
    if init_states is not None:
        print(f"[init] seeding episodes from {len(init_states)} stored states "
              f"({args.init_states})")

    # resolve manipuland/receptacle names once (obj_of_interest is static across resets)
    obj_name = args.obj_name
    target_name = args.target_name
    safe_reset(env, init_states=init_states, rng=rng)
    if obj_name is None or target_name is None:
        obj_name, target_name = infer_obj_and_target(env)

    fallback_fracs = [float(x) for x in args.grasp_frac_grid.split(",") if x.strip()]
    if args.grasp_json:
        ladder = [load_grasp_entry(args.grasp_json)]
        sampler_stats = {"n_candidates": 0, "n_central": 0, "n_side": 0,
                         "sampler": "preset", "grasp_json": args.grasp_json}
        print(f"[grasp] preset from {args.grasp_json}: {entry_desc(ladder[0])} "
              f"(sampler + calibration skipped)")
    elif args.no_grasp_sampler:
        ladder = [None] if args.no_calibrate else [
            {"kind": "grasp_frac", "local_off": None, "yaw_rel": None, "width": None,
             "height_frac": f, "radial_offset": None, "score": None, "grasp_frac": f}
            for f in fallback_fracs]
        sampler_stats = {"n_candidates": 0, "n_central": 0, "n_side": 0,
                         "sampler": "disabled"}
    else:
        ladder, sampler_stats = build_candidate_ladder(
            env, obj_name, args.top_k_candidates, fallback_fracs, args.mu, args.seed,
            n_sample_resets=args.sample_resets,
            reset_fn=lambda: safe_reset(env, init_states=init_states, rng=rng))
        print(f"[sampler] {sampler_stats['n_candidates']} candidates over "
              f"{sampler_stats['n_resets']} resets "
              f"({sampler_stats['n_central']} central / {sampler_stats['n_side']} side); "
              f"ladder: {[entry_desc(e) for e in ladder]}")

    def make_policy(entry):
        frac = args.grasp_frac
        if entry is not None and entry.get("grasp_frac") is not None:
            frac = entry["grasp_frac"]
        return ScriptedPickPlacePolicy(
            obj_name, target_name,
            hover_height=args.hover_height, grasp_z_offset=args.grasp_z_offset,
            grasp_frac=frac,
            lift_height=args.lift_height, place_drop=args.place_drop,
            release_above_rim=(None if args.no_rim_release else args.release_above_rim),
            pre_grasp_clearance=(None if args.no_preshape else args.pre_grasp_clearance),
            align_yaw=not args.no_align_yaw,
            center_placement=not args.no_center_placement)

    def run_episodes(entry, max_attempts, want_successes, tag,
                     flush_cb=None, flush_every=10, base_keep=()):
        """Roll out up to ``max_attempts`` episodes at one ladder entry; return the
        kept (successful) ep-dir names, #successes and #attempts. Stops early once
        ``want_successes`` successes are collected.

        If ``flush_cb`` is given it is called with ``list(base_keep) + keep`` every
        ``flush_every`` new successes, so a task that is later KILLED (e.g. the orchestrator
        timeout on a slow low-yield object) keeps the demos gathered so far instead of
        losing everything -- the collector otherwise only writes the HDF5 at the very end."""
        # entries whose kind is grasp_frac/fallback drive height via grasp_frac; only
        # real sampler candidates inject a full grasp pose (fallback injects yaw only
        # through its local candidate as well -- it has one, pointing down the minor
        # axis at the origin)
        inject_candidate = None
        if entry is not None and entry["kind"] != "grasp_frac":
            inject_candidate = entry
        keep, ns, nt = [], 0, 0
        while nt < max_attempts and ns < want_successes:
            obs = safe_reset(env, init_states=init_states, rng=rng)
            rim_z = None
            pol = make_policy(entry)
            for step in range(args.horizon):
                if step % RIM_REFRESH_EVERY == 0:
                    rim_z = receptacle_rim_z(env, target_name)
                inject_obj_extents(env, obs, obj_name)
                inject_rim(obs, target_name, rim_z)
                if inject_candidate is not None:
                    inject_grasp(obs, obj_name, inject_candidate)
                a, done = pol.act(obs)
                obs, r, d, info = env.step(a)
                if done:
                    break
            ep_dir = os.path.basename(env.ep_directory)  # created on first step()
            success = bool(env._check_success())
            nt += 1
            if success:
                ns += 1
                keep.append(ep_dir)
                if flush_cb is not None and ns % flush_every == 0:
                    flush_cb(list(base_keep) + keep)
            print(f"  [{tag}] attempt {nt:3d}: success={success}  (kept {ns})")
        return keep, ns, nt

    calibrate = (not args.no_calibrate) and len(ladder) >= 2

    def flush_partial(all_keep):
        """Write the demos gathered so far to demo.hdf5 (atomic via tmp+rename) so a later
        kill keeps them, plus a ``partial`` result.json so the orchestrator credits them
        (it counts n_demos from result.json, which is otherwise only written at the end).
        Cheap: re-reads the persisted per-episode npz. Best-effort."""
        if not all_keep:
            return
        try:
            _, n = gather_successful_demos_as_hdf5(
                tmp_dir, args.out_dir, env_info, problem_info, args.bddl_file, all_keep,
                out_name="demo.hdf5.tmp")
            os.replace(os.path.join(args.out_dir, "demo.hdf5.tmp"),
                       os.path.join(args.out_dir, "demo.hdf5"))
            if args.result_json:
                tmp = args.result_json + ".tmp"
                with open(tmp, "w") as rf:
                    json.dump({"obj_name": obj_name, "target_name": target_name,
                               "n_demos": n, "partial": True, "reached_target": False},
                              rf)
                os.replace(tmp, args.result_json)
        except Exception as e:
            print(f"[flush] partial flush failed ({type(e).__name__}: {e})")

    keep_dirs, n_succ, n_try = [], 0, 0
    calib_yields = []          # [{entry, successes, attempts}] for the result json
    collect_ns, collect_nt = 0, 0
    if calibrate:
        print(f"[calibrate] {len(ladder)} ladder entries x {args.calib_attempts} warmups "
              f"(best-measured; early-accept at {args.calib_early_accept}/"
              f"{args.calib_attempts} perfect)")
        # Select the grasp by MEASURED warmup yield across the DIVERSE ladder, not by the
        # first to clear a low bar. A grasp that is really ~67% reliable passes 5/5 ~13% of
        # the time, so the old "first 5/5 wins" locked onto lucky-but-fragile grasps (banana,
        # baguette, mug). We now evaluate every ladder entry and keep the best, early-
        # accepting only on a STRONG perfect warmup (>= calib_early_accept, default 8; a 67%
        # grasp clears 8/8 only ~6% of the time). Because the ladder is diverse, the entries
        # have genuinely different true yields, so max-selection is not noise-dominated.
        best_entry, best_ns, best_keep = ladder[0], -1, []
        for e_i, entry in enumerate(ladder):
            keep, ns, nt = run_episodes(entry, args.calib_attempts, args.calib_attempts,
                                        tag=f"cal {e_i}:{entry_desc(entry)}")
            n_try += nt
            calib_yields.append({"entry": entry_desc(entry), "kind": entry["kind"],
                                 "successes": ns, "attempts": nt})
            print(f"[calibrate] {entry_desc(entry)}: {ns}/{nt}")
            if ns > best_ns:
                best_entry, best_ns, best_keep = entry, ns, keep
            if ns >= nt and nt >= args.calib_early_accept:   # strong perfect warmup -> stop
                break
        keep_dirs, n_succ = list(best_keep), len(best_keep)
        print(f"[calibrate] chosen: {entry_desc(best_entry)} "
              f"(best warmup yield {best_ns}/{args.calib_attempts})")
    else:
        best_entry = ladder[0] if ladder else None
        print(f"[collect] calibration off; using {entry_desc(best_entry)}")

    remaining = args.num_success - n_succ
    if remaining > 0:
        if keep_dirs:
            flush_partial(keep_dirs)     # persist calibration successes before collecting
        keep, ns, nt = run_episodes(best_entry, args.max_attempts, remaining,
                                    tag="collect", flush_cb=flush_partial,
                                    flush_every=10, base_keep=keep_dirs)
        keep_dirs += keep
        n_succ += ns
        n_try += nt
        collect_ns, collect_nt = ns, nt

    env.close()  # flush the final episode's states to disk

    yield_pct = 100.0 * n_succ / max(n_try, 1)
    n_written = 0
    if keep_dirs:
        hdf5_path, n_written = gather_successful_demos_as_hdf5(
            tmp_dir, args.out_dir, env_info, problem_info, args.bddl_file, keep_dirs)
        print(f"[done] obj={obj_name} target={target_name} chosen={entry_desc(best_entry)}")
        print(f"[yield] {n_succ}/{n_try} successes ({yield_pct:.0f}%); wrote {n_written} demos -> {hdf5_path}")
    else:
        print(f"[done] obj={obj_name} target={target_name} chosen={entry_desc(best_entry)}")
        print(f"[yield] 0/{n_try} successes (0%); no demos written (scripted policy never succeeded)")
    if not args.keep_tmp_states:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if args.result_json:
        chosen = None
        if best_entry is not None:
            chosen = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                      for k, v in best_entry.items()}
        result = {
            "bddl_file": args.bddl_file,
            "obj_name": obj_name, "target_name": target_name,
            "build_attempts": build_attempts,
            "init_state_seeding": bool(init_states is not None),
            "sampler": sampler_stats,
            "chosen_entry": chosen,
            "chosen_grasp_frac": (best_entry or {}).get("grasp_frac"),
            "calib_yields": calib_yields,
            "collect_successes": collect_ns, "collect_attempts": collect_nt,
            "total_successes": n_succ, "total_attempts": n_try,
            "num_success_target": args.num_success, "max_attempts": args.max_attempts,
            # gate: did the chosen entry reach the target within the collect budget?
            "reached_target": bool(n_succ >= args.num_success),
            "gate_yield": round(yield_pct, 1),
            "collect_yield": round(100.0 * collect_ns / max(collect_nt, 1), 1),
            "n_demos": n_written,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.result_json)), exist_ok=True)
        with open(args.result_json, "w") as rf:
            json.dump(result, rf, indent=2)
        print(f"[result] wrote {args.result_json}  reached_target={result['reached_target']} "
              f"n_demos={n_written}")


if __name__ == "__main__":
    main()
