"""Integrity sweep over a scripted-collection run's HDF5 files.

``scripts/check_dataset_integrity.py`` is the stock LIBERO checker: it walks
``get_libero_path("datasets")`` and hard-codes 50 demos per file. Our runs live at
``<out-root>/collect/<task>/demo.hdf5`` with a variable demo count (a task that fell short
of ``--num-success`` still produced usable demos), so this sweep takes the run root and
checks what actually matters for training:

  * the file opens and every ``data/demo_*`` group is readable (a killed worker can leave a
    truncated HDF5 that only fails when something reads it);
  * ``states`` and ``actions`` have the same length -- they are written by separate code
    paths in the collector, and a mismatch silently misaligns the whole episode;
  * actions stay inside ``[-1, 1]``, the OSC_POSE controller's valid range;
  * no empty or absurdly short episodes;
  * episode-length distribution, so a run can be compared against a known-good one
    (the 65k stockbg run: mean 198 +- 39 steps).

Exits non-zero if anything is corrupt or malformed.

Example:
    python scripts/sweep_collect_integrity.py --root /tmp2/leocheng/ricl_scratch/stockbg_layoutext
"""

import argparse
import glob
import json
import os
import sys

import h5py
import numpy as np


def sweep_file(path, min_steps):
    """Return (n_demos, [lengths], [problems])."""
    problems, lengths = [], []
    try:
        f = h5py.File(path, "r")
    except Exception as e:
        return 0, [], [f"unreadable: {type(e).__name__}: {e}"]
    with f:
        if "data" not in f:
            return 0, [], ["no /data group"]
        names = [k for k in f["data"].keys() if k.startswith("demo")]
        for name in names:
            g = f["data"][name]
            try:
                if "actions" not in g or "states" not in g:
                    problems.append(f"{name}: missing actions/states")
                    continue
                acts = g["actions"][()]
                n_states = g["states"].shape[0]
            except Exception as e:
                problems.append(f"{name}: read failed ({type(e).__name__}: {e})")
                continue
            n = acts.shape[0]
            lengths.append(n)
            if n != n_states:
                problems.append(f"{name}: {n} actions vs {n_states} states")
            if n < min_steps:
                problems.append(f"{name}: only {n} steps")
            if not np.isfinite(acts).all():
                problems.append(f"{name}: non-finite actions")
            elif acts.min() < -1.0 - 1e-6 or acts.max() > 1.0 + 1e-6:
                problems.append(f"{name}: actions outside [-1,1] "
                                f"({acts.min():.3f}, {acts.max():.3f})")
        return len(names), lengths, problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="collection out-root (contains collect/<task>/demo.hdf5)")
    ap.add_argument("--glob", default="collect/*/demo.hdf5",
                    help="demo-file pattern under --root; the per-layout datasets from "
                         "make_layout_grid_datasets.py use '*/layout_*/demo.hdf5'")
    ap.add_argument("--expect-demos", type=int, default=None,
                    help="flag tasks whose demo count is below this")
    ap.add_argument("--min-steps", type=int, default=20)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.root, args.glob)))
    if not files:
        raise SystemExit(f"no files matching {args.glob!r} under {args.root}")

    all_lengths, bad, short, total = [], {}, [], 0
    for path in files:
        task = os.path.basename(os.path.dirname(path))
        n, lengths, problems = sweep_file(path, args.min_steps)
        total += n
        all_lengths += lengths
        if problems:
            bad[task] = problems
        if args.expect_demos and n < args.expect_demos:
            short.append((task, n))

    L = np.array(all_lengths) if all_lengths else np.array([0])
    print(f"root      : {args.root}")
    print(f"files     : {len(files)}")
    print(f"demos     : {total}")
    print(f"episode   : mean {L.mean():.0f} +- {L.std():.0f} steps "
          f"(min {L.min()}, median {int(np.median(L))}, max {L.max()})")
    print(f"corrupt   : {len(bad)} files with problems")
    if args.expect_demos:
        print(f"short     : {len(short)} files below {args.expect_demos} demos")

    if bad:
        print("\nproblems:", file=sys.stderr)
        for task, problems in list(bad.items())[:20]:
            print(f"  {task}: {problems[0]}"
                  + (f" (+{len(problems) - 1} more)" if len(problems) > 1 else ""),
                  file=sys.stderr)
        if len(bad) > 20:
            print(f"  ... and {len(bad) - 20} more files", file=sys.stderr)

    if args.out_json:
        json.dump({"root": os.path.abspath(args.root), "n_files": len(files),
                   "n_demos": total,
                   "episode_mean": float(L.mean()), "episode_std": float(L.std()),
                   "episode_min": int(L.min()), "episode_max": int(L.max()),
                   "problems": bad, "short": short},
                  open(args.out_json, "w"), indent=2)
        print(f"\n[json] -> {args.out_json}")

    if bad:
        sys.exit(1)
    print("\nOK: no corrupt or malformed demos")


if __name__ == "__main__":
    main()
