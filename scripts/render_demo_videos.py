"""Render mp4 videos from a scripted-demo ``demo.hdf5`` (collection format).

The demo files produced by ``scripts/collect_scripted_demonstrations.py`` store only
low-dim states + actions + the per-episode ``model_file`` xml (no images), so to *see*
the rollouts we replay them exactly the way ``scripts/create_dataset.py`` does -- reset
the env from each demo's model xml, seed it with the recorded initial state, step the
recorded actions, and grab the offscreen camera images -- then encode one mp4 per demo.

Runs in the LIBERO env and needs the GPU (offscreen rendering), so ``MUJOCO_GL=egl`` is
forced below. By default the agentview and wrist cameras are shown side by side.

Example:
    python scripts/render_demo_videos.py \
        --demo-file /tmp/demos/apple/demo.hdf5 --height 256 --fps 20
"""

import os

# offscreen rendering needs a GL backend; set before mujoco/robosuite import
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import json

import h5py
import imageio
import numpy as np

import init_path  # noqa: F401  (adds repo to sys.path, mirrors other scripts)
import libero.libero.utils.utils as libero_utils
from libero.libero.envs import TASK_MAPPING


def _cam_key(cam):
    return f"{cam}_image"


def render_demo(env, f, ep, cameras, flip):
    """Replay one demo episode and return a list of (stacked) RGB frames."""
    model_xml = f[f"data/{ep}"].attrs["model_file"]
    states = f[f"data/{ep}/states"][()]
    actions = np.array(f[f"data/{ep}/actions"][()])

    reset_ok = False
    while not reset_ok:
        try:
            env.reset()
            reset_ok = True
        except Exception:
            continue

    model_xml = libero_utils.postprocess_model_xml(model_xml, {})
    env.reset_from_xml_string(model_xml)
    env.sim.reset()
    env.sim.set_state_from_flattened(states[0])
    env.sim.forward()

    frames = []

    def grab():
        obs = env._get_observations(force_update=True)
        views = []
        for cam in cameras:
            img = obs[_cam_key(cam)]
            if flip:
                img = img[::-1]
            views.append(np.ascontiguousarray(img))
        # pad-to-tallest then hstack so mixed camera sizes still concatenate
        h = max(v.shape[0] for v in views)
        views = [
            np.pad(v, ((0, h - v.shape[0]), (0, 0), (0, 0))) if v.shape[0] < h else v
            for v in views
        ]
        return np.concatenate(views, axis=1)

    frames.append(grab())  # initial pose
    for action in actions:
        env.step(action)
        frames.append(grab())
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo-file", required=True)
    ap.add_argument("--out-dir", default=None,
                    help="where mp4s are written (default: <demo-dir>/videos)")
    ap.add_argument("--cameras", default="agentview,robot0_eye_in_hand",
                    help="comma-separated robosuite camera names, shown side by side")
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--max-demos", type=int, default=None,
                    help="cap number of episodes rendered (default: all)")
    ap.add_argument("--no-flip", action="store_true",
                    help="do not vertically flip camera images")
    args = ap.parse_args()

    cameras = [c for c in args.cameras.split(",") if c.strip()]
    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(args.demo_file)),
                                           "videos")
    os.makedirs(out_dir, exist_ok=True)

    f = h5py.File(args.demo_file, "r")
    env_kwargs = json.loads(f["data"].attrs["env_info"])
    problem_info = json.loads(f["data"].attrs["problem_info"])
    problem_name = problem_info["problem_name"]
    bddl_file_name = f["data"].attrs["bddl_file_name"]
    print("[task]", problem_info["language_instruction"])

    libero_utils.update_env_kwargs(
        env_kwargs,
        bddl_file_name=bddl_file_name,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        camera_names=cameras,
        reward_shaping=True,
        control_freq=20,
        camera_heights=args.height,
        camera_widths=args.width,
        camera_depths=False,
        camera_segmentations=None,
    )
    env = TASK_MAPPING[problem_name](**env_kwargs)

    demos = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))
    if args.max_demos is not None:
        demos = demos[: args.max_demos]

    for ep in demos:
        frames = render_demo(env, f, ep, cameras, flip=not args.no_flip)
        out = os.path.join(out_dir, f"{ep}.mp4")
        imageio.mimwrite(out, frames, fps=args.fps, quality=8, macro_block_size=1)
        print(f"  wrote {out}  ({len(frames)} frames)")

    env.close()
    f.close()
    print(f"[done] {len(demos)} videos -> {out_dir}")


if __name__ == "__main__":
    main()
