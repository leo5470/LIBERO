"""Batch-render mp4 spot-check videos for a whole suite collection run.

``scripts/render_demo_videos.py`` renders the episodes of ONE ``demo.hdf5``; this script
walks a collection output root (``<root>/<task>/demo.hdf5``, the layout written by
``scripts/collect_scripted_demonstrations.py`` runs), renders the first ``--per-task``
episodes of every task, and writes flat, task-prefixed mp4s so a whole run can be
eyeballed from a single directory.

Rendering is EGL offscreen on a shared GPU, so tasks are rendered serially (the same
reason ``run_scale_pipeline.py`` serializes its render stage). Re-running skips mp4s
that already exist, and a task whose replay env cannot be built is reported and skipped
rather than aborting the batch. Use ``--select``/``--limit`` for spot checks instead of
rendering everything at a high ``--per-task``.

Example:
    python scripts/render_suite_videos.py \
        --collect-root /tmp2/leocheng/ricl_scratch/scale/collect \
        --per-task 1 --select bowl,donut,rolling_pin
"""

import os
import sys

# GL backend (and optional GPU pin) must be set before any mujoco import; argparse only
# runs after imports, so peek at argv here.
os.environ.setdefault("MUJOCO_GL", "egl")
if "--gpu" in sys.argv:
    _gpu = sys.argv[sys.argv.index("--gpu") + 1]
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", _gpu)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", _gpu)

import argparse
import glob
import traceback

import h5py
import imageio

import init_path  # noqa: F401  (adds repo to sys.path, mirrors other scripts)
from render_demo_videos import build_env, render_demo


def find_demo_files(collect_root, select, limit):
    """(task_name, demo_file) pairs under <collect_root>/<task>/demo.hdf5, filtered."""
    files = sorted(glob.glob(os.path.join(collect_root, "*", "demo.hdf5")))
    pairs = [(os.path.basename(os.path.dirname(p)), p) for p in files]
    if select:
        subs = [s for s in select.split(",") if s.strip()]
        pairs = [(t, p) for t, p in pairs if any(s in t for s in subs)]
    if limit is not None:
        pairs = pairs[:limit]
    return pairs


def render_task(task, demo_file, out_dir, cameras, args):
    """Render the requested episodes of one task file. Returns (n_rendered, n_skipped)."""
    with h5py.File(demo_file, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))
        if args.per_task > 0:
            demos = demos[: args.per_task]
        out = {ep: os.path.join(out_dir, f"{task}__{ep}.mp4") for ep in demos}
        todo = [ep for ep in demos if args.overwrite or not os.path.isfile(out[ep])]
        if not todo:
            return 0, len(demos)

        env, _ = build_env(f, cameras, args.height, args.width)
        try:
            for ep in todo:
                frames = render_demo(env, f, ep, cameras, flip=not args.no_flip)
                imageio.mimwrite(out[ep], frames, fps=args.fps, quality=8,
                                 macro_block_size=1)
        finally:
            env.close()
    return len(todo), len(demos) - len(todo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect-root", required=True,
                    help="collection output root containing <task>/demo.hdf5 subdirs")
    ap.add_argument("--out-dir", default=None,
                    help="where mp4s are written (default: <collect-root>/../videos)")
    ap.add_argument("--per-task", type=int, default=1,
                    help="episodes rendered per task, taken in demo order (0 = all)")
    ap.add_argument("--select", default=None,
                    help="comma-separated substrings; keep tasks whose dir name matches any")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap number of tasks (after --select)")
    ap.add_argument("--cameras", default="agentview,robot0_eye_in_hand",
                    help="comma-separated robosuite camera names, shown side by side")
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--overwrite", action="store_true",
                    help="re-render mp4s that already exist")
    ap.add_argument("--no-flip", action="store_true",
                    help="do not vertically flip camera images")
    ap.add_argument("--gpu", type=int, default=None,
                    help="pin EGL rendering to this GPU id")
    args = ap.parse_args()

    cameras = [c for c in args.cameras.split(",") if c.strip()]
    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(args.collect_root)), "videos")
    os.makedirs(out_dir, exist_ok=True)

    pairs = find_demo_files(args.collect_root, args.select, args.limit)
    if not pairs:
        sys.exit(f"no <task>/demo.hdf5 found under {args.collect_root}"
                 + (f" matching --select {args.select}" if args.select else ""))
    print(f"[plan] {len(pairs)} tasks x {args.per_task or 'all'} episodes -> {out_dir}")

    n_rendered = n_skipped = 0
    failures = []
    for i, (task, demo_file) in enumerate(pairs, 1):
        try:
            r, s = render_task(task, demo_file, out_dir, cameras, args)
            n_rendered += r
            n_skipped += s
            print(f"  [{i}/{len(pairs)}] {task}: {r} rendered, {s} skipped", flush=True)
        except Exception as e:
            failures.append((task, f"{type(e).__name__}: {e}"))
            print(f"  [{i}/{len(pairs)}] {task}: FAILED ({type(e).__name__})", flush=True)
            traceback.print_exc(file=sys.stderr)

    print(f"[done] {n_rendered} rendered, {n_skipped} skipped, "
          f"{len(failures)} tasks failed -> {out_dir}")
    for task, err in failures:
        print(f"  [failed] {task}: {err}")
    sys.exit(1 if failures and not n_rendered else 0)


if __name__ == "__main__":
    main()
