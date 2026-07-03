"""Collect scripted top-down pick-and-place demonstrations for a LIBERO task.

This is the autonomous analogue of ``scripts/collect_demonstration.py``: instead of a
human teleop device driving the arm via ``input2action``, the 7-D OSC actions come from
``ScriptedPickPlacePolicy``. Only episodes that pass ``env._check_success()`` are kept.

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
import sys
import time
from glob import glob

import h5py
import numpy as np
import robosuite as suite
from robosuite import load_controller_config
from robosuite.utils import transform_utils as T
from robosuite.wrappers import DataCollectionWrapper

import init_path  # noqa: F401  (adds repo to sys.path, mirrors other scripts)
import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero.envs import TASK_MAPPING

# make the repo-root policy importable regardless of CWD
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from libero_scripted_policy import ScriptedPickPlacePolicy, infer_obj_and_target  # noqa: E402


def gather_successful_demos_as_hdf5(tmp_dir, out_dir, env_info, problem_info,
                                    bddl_file, keep_dirs):
    """Aggregate only the ``keep_dirs`` episodes from ``tmp_dir`` into out_dir/demo.hdf5.

    Mirrors collect_demonstration.gather_demonstrations_as_hdf5 but (a) filters to the
    successful episode directories, (b) takes problem_info/bddl_file explicitly rather
    than via script globals, and (c) stores the real bddl file contents.
    """
    os.makedirs(out_dir, exist_ok=True)
    hdf5_path = os.path.join(out_dir, "demo.hdf5")
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


def safe_reset(env, max_tries=20):
    for _ in range(max_tries):
        try:
            return env.reset()
        except Exception:
            continue
    raise RuntimeError("env.reset() failed repeatedly (RandomizationError?)")


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
    # policy knobs (override per geometry if needed)
    ap.add_argument("--hover-height", type=float, default=0.12)
    ap.add_argument("--grasp-z-offset", type=float, default=0.005,
                    help="fallback origin-relative grasp offset (used only if live "
                         "extents can't be injected)")
    ap.add_argument("--grasp-frac", type=float, default=0.65,
                    help="grasp height as a fraction of object height above its base "
                         "(0=base, 1=top); used when calibration is off")
    ap.add_argument("--grasp-frac-grid", type=str, default="0.5,0.65,0.8",
                    help="comma-separated grasp_frac values to auto-calibrate over; the "
                         "best-yielding value on a short warmup is used for the batch")
    ap.add_argument("--calib-attempts", type=int, default=6,
                    help="warmup rollouts per grid value during grasp_frac calibration")
    ap.add_argument("--no-calibrate", action="store_true",
                    help="skip calibration and use --grasp-frac as a fixed value")
    ap.add_argument("--lift-height", type=float, default=0.20)
    ap.add_argument("--place-drop", type=float, default=0.06)
    ap.add_argument("--result-json", type=str, default=None,
                    help="write a machine-readable gate result (grasp_frac, calib/collect "
                         "yields, reached_target, n_demos) here for the scale orchestrator")
    args = ap.parse_args()

    assert os.path.exists(args.bddl_file), args.bddl_file
    problem_info = BDDLUtils.get_problem_info(args.bddl_file)
    problem_name = problem_info["problem_name"]
    print("[task]", problem_info["language_instruction"])

    controller_config = load_controller_config(default_controller="OSC_POSE")
    config = {"robots": ["Panda"], "controller_configs": controller_config}
    env_info = json.dumps(config)

    env = TASK_MAPPING[problem_name](
        bddl_file_name=args.bddl_file,
        **config,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        ignore_done=True,
        initialization_noise=None,
        control_freq=20,
    )

    tmp_dir = args.tmp_dir or os.path.join(args.out_dir, "_tmp_states",
                                           str(time.time()).replace(".", "_"))
    os.makedirs(tmp_dir, exist_ok=True)
    env = DataCollectionWrapper(env, tmp_dir)
    env.seed(args.seed)

    # resolve manipuland/receptacle names once (obj_of_interest is static across resets)
    obj_name = args.obj_name
    target_name = args.target_name
    safe_reset(env)
    if obj_name is None or target_name is None:
        obj_name, target_name = infer_obj_and_target(env)

    def run_episodes(grasp_frac, max_attempts, want_successes, tag):
        """Roll out up to ``max_attempts`` episodes at a fixed ``grasp_frac``; return the
        kept (successful) ep-dir names, #successes and #attempts. Stops early once
        ``want_successes`` successes are collected."""
        keep, ns, nt = [], 0, 0
        while nt < max_attempts and ns < want_successes:
            obs = safe_reset(env)
            pol = ScriptedPickPlacePolicy(
                obj_name, target_name,
                hover_height=args.hover_height, grasp_z_offset=args.grasp_z_offset,
                grasp_frac=grasp_frac,
                lift_height=args.lift_height, place_drop=args.place_drop)
            for _ in range(args.horizon):
                inject_obj_extents(env, obs, obj_name)
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
            print(f"  [{tag}] attempt {nt:3d}: success={success}  (kept {ns})")
        return keep, ns, nt

    grid = [float(x) for x in args.grasp_frac_grid.split(",") if x.strip() != ""]
    calibrate = (not args.no_calibrate) and len(grid) >= 2

    keep_dirs, n_succ, n_try = [], 0, 0
    calib_yields = {}          # {grasp_frac: [successes, attempts]} for the result json
    collect_ns, collect_nt = 0, 0
    if calibrate:
        print(f"[calibrate] sweeping grasp_frac over {grid} "
              f"({args.calib_attempts} attempts each)")
        best_frac, best_ns, best_keep = grid[len(grid) // 2], -1, []
        for f in grid:
            keep, ns, nt = run_episodes(f, args.calib_attempts, args.calib_attempts,
                                        tag=f"cal f={f:g}")
            n_try += nt
            calib_yields[f"{f:g}"] = [ns, nt]
            print(f"[calibrate] grasp_frac={f:g}: {ns}/{nt}")
            if ns > best_ns:  # first (lowest) frac wins ties
                best_frac, best_ns, best_keep = f, ns, keep
        # keep the winning grid value's successes toward the batch (same frac -> valid)
        keep_dirs, n_succ = list(best_keep), len(best_keep)
        note = "no grasp succeeded during calibration; using grid midpoint" if best_ns <= 0 else ""
        print(f"[calibrate] chosen grasp_frac={best_frac:g} "
              f"(warmup yield {best_ns}/{args.calib_attempts}) {note}")
    else:
        best_frac = args.grasp_frac
        print(f"[collect] calibration off; using grasp_frac={best_frac:g}")

    remaining = args.num_success - n_succ
    if remaining > 0:
        keep, ns, nt = run_episodes(best_frac, args.max_attempts, remaining, tag="collect")
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
        print(f"[done] obj={obj_name} target={target_name} grasp_frac={best_frac:g}")
        print(f"[yield] {n_succ}/{n_try} successes ({yield_pct:.0f}%); wrote {n_written} demos -> {hdf5_path}")
    else:
        print(f"[done] obj={obj_name} target={target_name} grasp_frac={best_frac:g}")
        print(f"[yield] 0/{n_try} successes (0%); no demos written (scripted policy never succeeded)")

    if args.result_json:
        result = {
            "bddl_file": args.bddl_file,
            "obj_name": obj_name, "target_name": target_name,
            "chosen_grasp_frac": best_frac,
            "calib_yields": calib_yields,
            "collect_successes": collect_ns, "collect_attempts": collect_nt,
            "total_successes": n_succ, "total_attempts": n_try,
            "num_success_target": args.num_success, "max_attempts": args.max_attempts,
            # gate: did the chosen frac reach the target within the collect budget?
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
