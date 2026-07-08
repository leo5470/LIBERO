"""Generate .pruned_init eval init-state files for a manifest-generated suite.

LIBERO benchmarks load fixed initial sim states per task
(``Benchmark.get_task_init_states`` -> ``torch.load(<task>.pruned_init)`` ->
``env.set_init_state`` -> ``sim.set_state_from_flattened``), but the repo ships no
generator for them. This script produces files byte-compatible with the stock ones
(inspected: numpy float64 array of shape ``(num_states, sim_state_dim)``): per task it
builds the raw headless env, draws seeded placement resets (bounded retries, discarding
any draw where the goal is somehow already satisfied), and saves the flattened states.

Example:
    python scripts/create_suite_init_states.py \
        --manifest libero/libero/bddl_files/libero_object_unseen/manifest.json
"""

import argparse
import json
import os

import numpy as np
import torch
from robosuite import load_controller_config
from robosuite.utils.errors import RandomizationError

import init_path  # noqa: F401
import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero.envs import TASK_MAPPING


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


def collect_states(env, num_states, max_tries):
    states, tries = [], 0
    while len(states) < num_states and tries < max_tries:
        tries += 1
        try:
            env.reset()
        except Exception:
            continue  # RandomizationError etc. -> resample
        if env._check_success():
            continue  # never bake an already-solved episode into eval states
        states.append(env.sim.get_state().flatten())
    return states, tries


def main():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(
        repo, "libero/libero/bddl_files/libero_object_unseen/manifest.json"))
    ap.add_argument("--out-dir", default=None,
                    help="default: libero/libero/init_files/<suite>")
    ap.add_argument("--num-states", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tries-per-state", type=int, default=20)
    ap.add_argument("--overwrite", action="store_true",
                    help="regenerate files that already exist")
    ap.add_argument("--shard", nargs=2, type=int, default=(0, 1), metavar=("I", "N"),
                    help="process only manifest tasks with index %% N == I (parallel workers)")
    args = ap.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)
    bddl_dir = os.path.dirname(os.path.abspath(args.manifest))
    out_dir = args.out_dir or os.path.join(
        repo, "libero/libero/init_files", manifest["suite"])
    os.makedirs(out_dir, exist_ok=True)

    shard_i, shard_n = args.shard
    tasks = [t for idx, t in enumerate(manifest["tasks"]) if idx % shard_n == shard_i]

    # A task that cannot produce init states (e.g. an oversize object whose placement
    # sampling never converges) is an expected outcome for the ungated full pool: record
    # it for the exclusion flow and keep going -- never kill the shard.
    failures = []
    for i, task in enumerate(tasks, 1):
        out_file = os.path.join(out_dir, task["name"] + ".pruned_init")
        if os.path.isfile(out_file) and not args.overwrite:
            print(f"[{i}/{len(tasks)}] {task['name']}: exists, skipping")
            continue
        env = None
        try:
            env = build_env(os.path.join(bddl_dir, task["name"] + ".bddl"))
            np.random.seed(args.seed)
            states, tries = collect_states(
                env, args.num_states, args.num_states * args.max_tries_per_state)
            if len(states) < args.num_states:
                failures.append({"name": task["name"], "n_states": len(states),
                                 "tries": tries, "error": "too_few_states"})
                print(f"[{i}/{len(tasks)}] {task['name']}: FAIL "
                      f"({len(states)}/{args.num_states} states in {tries} tries)",
                      flush=True)
                continue
            arr = np.stack(states).astype(np.float64)
            torch.save(arr, out_file)
            print(f"[{i}/{len(tasks)}] {task['name']}: "
                  f"{arr.shape} in {tries} resets -> {out_file}", flush=True)
        except Exception as e:  # noqa: BLE001 - env build/reset crash = task infeasible
            failures.append({"name": task["name"], "n_states": 0, "tries": 0,
                             "error": f"{type(e).__name__}: {e}"})
            print(f"[{i}/{len(tasks)}] {task['name']}: CRASH ({type(e).__name__}: {e})",
                  flush=True)
        finally:
            if env is not None:
                env.close()

    fail_file = os.path.join(out_dir, f"init_failures_shard{shard_i}of{shard_n}.json")
    with open(fail_file, "w") as f:
        json.dump(failures, f, indent=2)
    print(f"[done] shard {shard_i}/{shard_n}: {len(tasks) - len(failures)}/{len(tasks)} ok; "
          f"failures -> {fail_file}")


if __name__ == "__main__":
    main()
