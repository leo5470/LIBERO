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
import sys

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
    args = ap.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)
    bddl_dir = os.path.dirname(os.path.abspath(args.manifest))
    out_dir = args.out_dir or os.path.join(
        repo, "libero/libero/init_files", manifest["suite"])
    os.makedirs(out_dir, exist_ok=True)

    failures = []
    for i, task in enumerate(manifest["tasks"], 1):
        out_file = os.path.join(out_dir, task["name"] + ".pruned_init")
        if os.path.isfile(out_file) and not args.overwrite:
            print(f"[{i}/{len(manifest['tasks'])}] {task['name']}: exists, skipping")
            continue
        env = build_env(os.path.join(bddl_dir, task["name"] + ".bddl"))
        try:
            np.random.seed(args.seed)
            states, tries = collect_states(
                env, args.num_states, args.num_states * args.max_tries_per_state)
            if len(states) < args.num_states:
                failures.append((task["name"], len(states), tries))
                print(f"[{i}/{len(manifest['tasks'])}] {task['name']}: FAIL "
                      f"({len(states)}/{args.num_states} states in {tries} tries)",
                      flush=True)
                continue
            arr = np.stack(states).astype(np.float64)
            torch.save(arr, out_file)
            print(f"[{i}/{len(manifest['tasks'])}] {task['name']}: "
                  f"{arr.shape} in {tries} resets -> {out_file}", flush=True)
        finally:
            env.close()

    if failures:
        print(f"[error] {len(failures)} task(s) could not produce "
              f"{args.num_states} states: {failures}")
        sys.exit(1)
    print(f"[done] init states in {out_dir}")


if __name__ == "__main__":
    main()
