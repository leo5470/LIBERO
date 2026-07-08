"""Feasibility smoke check for a manifest-generated suite (libero_object_unseen).

Per task, on a fully headless raw env (never ControlEnv.reset — that retries forever):
  1. env builds from the BDDL (all object MJCFs merge),
  2. placement sampling succeeds for --num-resets resets within --max-reset-tries,
  3. the goal is False after every reset,
  4. teleporting the target to the basket's contain_region fires _check_success(),
     and a settle variant (drop from 3 cm above, step the sim) stays successful —
     this catches objects that satisfy the center-in-box predicate but don't
     physically fit/stay in the basket,
  5. (when the .pruned_init exists) init states round-trip via
     sim.set_state_from_flattened without firing the goal.

Writes a JSON report and exits non-zero if any task fails any check.

Example:
    python scripts/smoke_check_suite.py \
        --manifest libero/libero/bddl_files/libero_object_unseen/manifest.json
"""

import argparse
import json
import os
import sys
import traceback

import numpy as np
import torch
from robosuite import load_controller_config
from robosuite.utils.errors import RandomizationError

import init_path  # noqa: F401
import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero.envs import TASK_MAPPING

RECEPTACLE_SITE = "basket_1_contain_region"


def build_env(bddl_file, construct_tries=10):
    problem_info = BDDLUtils.get_problem_info(bddl_file)
    controller_config = load_controller_config(default_controller="OSC_POSE")
    # The constructor runs one placement sample itself; retry the rare unlucky draw.
    for attempt in range(construct_tries):
        try:
            return TASK_MAPPING[problem_info["problem_name"]](
                bddl_file_name=bddl_file,
                robots=["Panda"],
                controller_configs=controller_config,
                has_renderer=False,
                has_offscreen_renderer=False,
                use_camera_obs=False,
                ignore_done=True,
                initialization_noise=None,
                control_freq=20,
            )
        except RandomizationError:
            if attempt == construct_tries - 1:
                raise


def bounded_reset(env, max_tries):
    """-> tries used; raises RuntimeError if placement keeps failing."""
    for i in range(1, max_tries + 1):
        try:
            env.reset()
            return i
        except Exception:
            continue
    raise RuntimeError(f"reset failed {max_tries} times (placement crowding?)")


def teleport_target(env, target_name, offset_z=0.0):
    obj = env.objects_dict[target_name]
    pos = env.sim.data.get_site_xpos(RECEPTACLE_SITE).copy()
    pos[2] += offset_z
    env.sim.data.set_joint_qpos(
        obj.joints[-1], np.concatenate([pos, np.array([1.0, 0.0, 0.0, 0.0])])
    )
    env.sim.forward()


def check_task(task, bddl_dir, init_dir, args):
    checks = {}
    bddl_file = os.path.join(bddl_dir, task["name"] + ".bddl")
    target_name = task["target_key"] + "_1"
    env = None
    try:
        # inside the try: for the ungated pool a construction failure (placement never
        # converges, broken MJCF) is a per-task verdict, not a fatal error
        env = build_env(bddl_file)
        np.random.seed(args.seed)

        # placement + goal-false-at-reset
        total_tries = 0
        for _ in range(args.num_resets):
            total_tries += bounded_reset(env, args.max_reset_tries)
            if env._check_success():
                raise AssertionError("goal already True right after reset")
        checks["placement"] = {
            "resets": args.num_resets, "tries": total_tries, "ok": True,
        }

        # teleport target to the contain-region center -> goal must fire
        teleport_target(env, target_name)
        checks["teleport_goal_fires"] = bool(env._check_success())
        if not checks["teleport_goal_fires"]:
            raise AssertionError("goal did not fire with target at contain_region")

        # settle variant: drop from above, let physics run, must stay successful
        bounded_reset(env, args.max_reset_tries)
        teleport_target(env, target_name, offset_z=0.03)
        for _ in range(args.settle_steps):
            env.sim.step()
        checks["settle_still_success"] = bool(env._check_success())
        if not checks["settle_still_success"]:
            raise AssertionError("target did not stay in basket after settling")

        # init-state round trip (only once the .pruned_init exists)
        init_file = os.path.join(init_dir, task["name"] + ".pruned_init")
        if os.path.isfile(init_file):
            try:
                states = torch.load(init_file, weights_only=False)
            except TypeError:  # torch<1.13 has no weights_only kwarg
                states = torch.load(init_file)
            for k in {0, len(states) - 1}:
                env.sim.set_state_from_flattened(states[k])
                env.sim.forward()
                if env._check_success():
                    raise AssertionError(f"goal True after loading init state {k}")
            checks["init_roundtrip"] = {"n_states": int(len(states)), "ok": True}
        else:
            checks["init_roundtrip"] = "skipped (no .pruned_init)"

        return {"name": task["name"], "ok": True, "checks": checks}
    except Exception as e:
        return {
            "name": task["name"], "ok": False, "checks": checks,
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(limit=3),
        }
    finally:
        if env is not None:
            env.close()


def main():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(
        repo, "libero/libero/bddl_files/libero_object_unseen/manifest.json"))
    ap.add_argument("--init-dir", default=None,
                    help="default: libero/libero/init_files/<suite>")
    ap.add_argument("--report", default=None,
                    help="default: smoke_report.json next to the manifest")
    ap.add_argument("--tasks", nargs="*", default=None,
                    help="subset of task names (default: all in manifest)")
    ap.add_argument("--num-resets", type=int, default=5)
    ap.add_argument("--max-reset-tries", type=int, default=25)
    ap.add_argument("--settle-steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", nargs=2, type=int, default=(0, 1), metavar=("I", "N"),
                    help="process only manifest tasks with index %% N == I (parallel workers)")
    args = ap.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)
    bddl_dir = os.path.dirname(os.path.abspath(args.manifest))
    init_dir = args.init_dir or os.path.join(
        repo, "libero/libero/init_files", manifest["suite"])
    report_path = args.report or os.path.join(bddl_dir, "smoke_report.json")

    tasks = manifest["tasks"]
    if args.tasks:
        tasks = [t for t in tasks if t["name"] in set(args.tasks)]
        assert tasks, "no manifest tasks match --tasks"
    shard_i, shard_n = args.shard
    if shard_n > 1:
        tasks = [t for idx, t in enumerate(tasks) if idx % shard_n == shard_i]
        report_path = report_path.replace(".json", f"_shard{shard_i}of{shard_n}.json")

    results = []
    for i, task in enumerate(tasks, 1):
        result = check_task(task, bddl_dir, init_dir, args)
        status = "ok" if result["ok"] else f"FAIL ({result['error']})"
        print(f"[{i}/{len(tasks)}] {task['name']}: {status}", flush=True)
        results.append(result)

    n_fail = sum(not r["ok"] for r in results)
    report = {"suite": manifest["suite"], "n_tasks": len(results),
              "n_failed": n_fail, "results": results}
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[report] {report_path}  ({len(results) - n_fail}/{len(results)} passed)")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
