"""Eval-physics reproducibility check -- CPU only, no rendering, no GPU.

For one task: build the env exactly as eval does (plain bddl build, no round-trip), replay
the task's stored ``chosen_entry`` grasp with the scripted policy, and report how often it
succeeds. Also reports the manipuland's mass in both physics, so the near-zero-mass scan
comes free from the same build.

Nothing here creates a GL context: has_offscreen_renderer=False and use_camera_obs=False,
so no renderer is instantiated and MUJOCO_GL is never exercised.
"""
import os
os.environ.setdefault("MUJOCO_GL", "osmesa")   # CPU fallback; never used, nothing renders

import argparse, json, sys, time
import numpy as np

SP = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [SP, "/tmp2/leocheng/forks/LIBERO", "/tmp2/leocheng/forks/LIBERO/scripts"]

from robosuite import load_controller_config
from libero.libero.envs.bddl_utils import get_problem_info
from collect_scripted_demonstrations import (
    build_env, safe_reset, inject_obj_extents, inject_rim, inject_grasp,
    receptacle_rim_z, load_grasp_entry, RIM_REFRESH_EVERY)
from libero_scripted_policy import ScriptedPickPlacePolicy, infer_obj_and_target

BDDL_DIR = "/tmp2/leocheng/forks/LIBERO/libero/libero/bddl_files/libero_object_unseen_stockbg"
V2 = "/tmp2/leocheng/ricl_scratch/stockbg_v2/collect"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--attempts", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    t0 = time.time()
    out = {"task": a.task, "attempts": a.attempts}
    try:
        bddl = os.path.join(BDDL_DIR, a.task + ".bddl")
        entry = load_grasp_entry(os.path.join(V2, a.task, "result.json"))
        cfg = {"robots": ["Panda"],
               "controller_configs": load_controller_config(default_controller="OSC_POSE")}
        env, _ = build_env(bddl, get_problem_info(bddl)["problem_name"], cfg,
                           build_tries=60, seed=a.seed)
        env.seed(a.seed)
        on, tn = infer_obj_and_target(env)
        env.reset()

        mid = env.sim.model.body_name2id(on + "_main")
        out["eval_mass_g"] = float(env.sim.model.body_mass[mid]) * 1000
        xml = env.sim.model.get_xml()
        st = np.array(env.sim.get_state().flatten())
        env.reset_from_xml_string(xml); env.sim.reset()
        env.sim.set_state_from_flattened(st); env.sim.forward()
        out["demo_mass_g"] = float(
            env.sim.model.body_mass[env.sim.model.body_name2id(on + "_main")]) * 1000
        out["mass_ratio"] = out["demo_mass_g"] / max(out["eval_mass_g"], 1e-12)

        ok = 0
        for _ in range(a.attempts):                    # plain build == eval physics
            obs = safe_reset(env)
            pol = ScriptedPickPlacePolicy(on, tn, grasp_frac=(entry.get("grasp_frac") or 0.65),
                                          transit_height=None)
            rim = None
            for step in range(a.horizon):
                if step % RIM_REFRESH_EVERY == 0:
                    rim = receptacle_rim_z(env, tn)
                inject_obj_extents(env, obs, on)
                inject_rim(obs, tn, rim)
                inject_grasp(obs, on, entry)
                act, done = pol.act(obs)
                obs, _, _, _ = env.step(act)
                if done:
                    break
            ok += bool(env._check_success())
        env.close()
        out["eval_successes"] = ok
        out["eval_yield"] = 100.0 * ok / a.attempts
        out["v2_collect_yield"] = json.load(
            open(os.path.join(V2, a.task, "result.json"))).get("collect_yield")
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    out["secs"] = round(time.time() - t0, 1)
    json.dump(out, open(a.out, "w"))
    if "error" in out:
        print(f"{a.task[:52]:54s} ERROR {out['error'][:60]}", flush=True)
    else:
        print(f"{a.task[:52]:54s} eval {out['eval_yield']:5.1f}%  "
              f"(v2 collect {out['v2_collect_yield']:5.1f}%)  "
              f"mass {out['eval_mass_g']:7.2f}g x{out['mass_ratio']:.2f}  "
              f"{out['secs']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
