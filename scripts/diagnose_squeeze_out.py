"""Why do only some objects break in the lighter eval physics?

Hypothesis: a lighter object is displaced by the approaching/closing fingers before the jaw
shuts, so the grasp lands off-target. Whether that matters depends on how much positional
margin the grasp has -- which is why no static property predicts it.

Measures, per rollout, how far the manipuland moves between settling and the moment the
gripper starts closing, in both physics, for objects known to break and known to survive.
CPU only: no renderer, no camera obs.
"""
import os
os.environ.setdefault("MUJOCO_GL", "osmesa")

import argparse, json, sys
import numpy as np

SP = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [SP, "/tmp2/leocheng/forks/LIBERO", "/tmp2/leocheng/forks/LIBERO/scripts"]

from robosuite import load_controller_config
from libero.libero.envs.bddl_utils import get_problem_info
from collect_scripted_demonstrations import (
    build_env, safe_reset, inject_obj_extents, inject_rim, inject_grasp,
    receptacle_rim_z, load_grasp_entry, RIM_REFRESH_EVERY)
from libero_scripted_policy import ScriptedPickPlacePolicy, infer_obj_and_target, gripper_opening

BDDL_DIR = "/tmp2/leocheng/forks/LIBERO/libero/libero/bddl_files/libero_object_unseen_stockbg"
V2 = "/tmp2/leocheng/ricl_scratch/stockbg_v2/collect"


def roundtrip(env):
    xml = env.sim.model.get_xml()
    st = np.array(env.sim.get_state().flatten())
    env.reset_from_xml_string(xml)
    env.sim.reset()
    env.sim.set_state_from_flattened(st)
    env.sim.forward()


def run(env, on, tn, entry, n, heavy, horizon=400):   # entry needed for grasp width
    out = []
    body = on + "_main"
    for _ in range(n):
        obs = safe_reset(env)
        if heavy:
            roundtrip(env)
            obs = env._get_observations(force_update=True)
        pol = ScriptedPickPlacePolicy(on, tn, grasp_frac=(entry.get("grasp_frac") or 0.65),
                                      transit_height=None)
        rim, settled, at_grasp, jaw_end = None, None, None, None
        lifted_z, at_release, tgt_xy = None, None, None
        jaw_move, obj_z_move = None, None
        for step in range(horizon):
            if step % RIM_REFRESH_EVERY == 0:
                rim = receptacle_rim_z(env, tn)
            inject_obj_extents(env, obs, on)
            inject_rim(obs, tn, rim)
            inject_grasp(obs, on, entry)
            if step == 5:                       # after the scene settles
                settled = env.sim.data.get_body_xpos(body).copy()
            if at_grasp is None and pol.phase == "GRASP":
                at_grasp = env.sim.data.get_body_xpos(body).copy()
            was = pol.phase
            a, done = pol.act(obs)
            obs, _, _, _ = env.step(a)
            if was == "GRASP" and pol.phase == "LIFT":
                jaw_end = gripper_opening(obs)
            if was == "LIFT" and pol.phase == "MOVE":
                jaw_move = gripper_opening(obs)
                obj_z_move = float(env.sim.data.get_body_xpos(body)[2]) * 100
            if was == "MOVE" and pol.phase == "LOWER":
                lifted_z = float(env.sim.data.get_body_xpos(body)[2])
            if was == "LOWER" and pol.phase == "DROP":
                at_release = env.sim.data.get_body_xpos(body).copy()
                tgt_xy = env.sim.data.get_body_xpos(tn + "_main").copy()
            if done:
                break
        end = env.sim.data.get_body_xpos(body).copy()
        rec = {"success": bool(env._check_success()), "jaw_after_close": jaw_end,
               "carried_z_cm": None if lifted_z is None else lifted_z * 100,
               "jaw_at_move": jaw_move, "obj_z_at_move_cm": obj_z_move,
               "grasp_width_cm": (entry.get("width") or 0) * 100}
        if at_release is not None and tgt_xy is not None:
            rec["release_offset_cm"] = float(np.linalg.norm(at_release[:2] - tgt_xy[:2])) * 100
            rec["final_offset_cm"] = float(np.linalg.norm(end[:2] - tgt_xy[:2])) * 100
            rec["moved_after_release_cm"] = float(np.linalg.norm(end[:2] - at_release[:2])) * 100
        if settled is not None and at_grasp is not None:
            rec["pre_grasp_disp_cm"] = float(np.linalg.norm(at_grasp[:2] - settled[:2])) * 100
        out.append(rec)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--group", required=True)     # "affected" | "survivor"
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {"task": a.task, "group": a.group}
    try:
        bddl = os.path.join(BDDL_DIR, a.task + ".bddl")
        entry = load_grasp_entry(os.path.join(V2, a.task, "result.json"))
        cfg = {"robots": ["Panda"],
               "controller_configs": load_controller_config(default_controller="OSC_POSE")}
        env, _ = build_env(bddl, get_problem_info(bddl)["problem_name"], cfg,
                           build_tries=60, seed=0)
        env.seed(0)
        on, tn = infer_obj_and_target(env)
        res["light"] = run(env, on, tn, entry, a.n, heavy=False)
        res["heavy"] = run(env, on, tn, entry, a.n, heavy=True)
        env.close()
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {e}"
    json.dump(res, open(a.out, "w"))
    if "error" in res:
        print(f"{a.task[:46]:48s} ERROR {res['error'][:50]}", flush=True)
    else:
        f = lambda k, g: np.nanmean([x.get(k, np.nan) for x in res[g]])
        print(f"{a.task[:46]:48s} [{a.group:9s}] "
              f"disp light {f('pre_grasp_disp_cm','light'):5.2f}cm heavy "
              f"{f('pre_grasp_disp_cm','heavy'):5.2f}cm | "
              f"succ {sum(x['success'] for x in res['light'])}/{a.n} vs "
              f"{sum(x['success'] for x in res['heavy'])}/{a.n}", flush=True)
