"""How much do a set of demos actually differ? Trajectory-space diversity of demo sets.

Built to answer "is N scripted demos at one location reasonable?", and kept because the
answer drives how collection budget is spent. Each demo's action sequence is resampled to a
fixed length and flattened, then the set is summarised by:

  * **PCs for 95% var** -- effective dimensionality of the demo cloud;
  * **mean pairwise distance / scale** -- how spread the cloud is, normalised by mean
    trajectory magnitude so sets of different lengths compare;
  * **coverage(n)** -- mean distance from every demo to its nearest neighbour in a random
    n-subset, i.e. what a set loses if only n of its demos are kept. This is the curve that
    says where extra demos stop buying anything.

Reference numbers measured on this project (2026-08-02):

    scripted, one layout, 50 demos        8 PCs   spread 0.149
    scripted, across 6 layouts, 50 demos  9 PCs   spread 0.595
    human LIBERO-Object, one task         22 PCs  spread 0.595

The scripted policy is deterministic given state, so repeats within a layout differ only by
the reset jitter -- which is why the one-layout cloud is ~4x tighter than the human one and
~4x tighter than the gap between two layouts.

Examples:
    # one task's demo file vs the stock human demos
    python scripts/measure_demo_diversity.py \
        --set 'one-layout=/tmp2/.../collect/pick_up_the_egg__aigen_0_..._var1/demo.hdf5' \
        --set 'human=/tmp2/leocheng/libero/libero_object/pick_up_the_milk_..._demo.hdf5'

    # a whole per-layout dataset tree (one demo per layout) as a single set
    python scripts/measure_demo_diversity.py --set 'grid=/tmp2/.../grid/mushroom__aigen_3/layout_*/demo.hdf5'
"""

import argparse
import glob
import sys

import h5py
import numpy as np


def load_actions(pattern, cap=None):
    """All demos matching a glob (or a single file), as a list of (T, 7) arrays."""
    paths = sorted(glob.glob(pattern)) or ([pattern] if "*" not in pattern else [])
    acts = []
    for p in paths:
        try:
            with h5py.File(p, "r") as f:
                for k in sorted(k for k in f["data"] if k.startswith("demo")):
                    acts.append(f["data"][k]["actions"][()])
                    if cap and len(acts) >= cap:
                        return acts
        except Exception as e:
            print(f"[warn] {p}: {type(e).__name__}: {e}", file=sys.stderr)
    return acts


def featurise(acts, n=100):
    def rs(a):
        idx = np.linspace(0, len(a) - 1, n)
        return np.stack([np.interp(idx, np.arange(len(a)), a[:, j])
                         for j in range(a.shape[1])], 1).ravel()
    return np.stack([rs(a) for a in acts])


def summarise(X, rng, trials=200, ns=(1, 2, 3, 5, 10, 20, 50)):
    scale = np.linalg.norm(X, axis=1).mean()
    Xc = X - X.mean(0)
    s = np.linalg.svd(Xc, compute_uv=False)
    var = s ** 2 / max((s ** 2).sum(), 1e-12)
    k95 = int(np.searchsorted(np.cumsum(var), 0.95) + 1)
    D = np.linalg.norm(X[:, None] - X[None], axis=-1)
    spread = D[np.triu_indices(len(X), 1)].mean() / scale if len(X) > 1 else 0.0
    cov = {}
    for k in ns:
        if k > len(X):
            continue
        vals = [D[:, rng.choice(len(X), k, replace=False)].min(1).mean()
                for _ in range(trials)]
        cov[k] = float(np.mean(vals) / scale)
    return {"n": len(X), "pcs_95": k95, "top1_pc": float(var[0]),
            "spread": float(spread), "coverage": cov}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", required=True, metavar="NAME=GLOB",
                    help="named demo set; repeat for each set to compare")
    ap.add_argument("--cap", type=int, default=None, help="max demos per set")
    ap.add_argument("--resample", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    ns = (1, 2, 3, 5, 10, 20, 50)
    rows = []
    for spec in args.set:
        if "=" not in spec:
            raise SystemExit(f"--set needs NAME=GLOB, got {spec!r}")
        name, pattern = spec.split("=", 1)
        acts = load_actions(pattern, args.cap)
        if len(acts) < 2:
            print(f"[warn] {name}: {len(acts)} demos, skipping", file=sys.stderr)
            continue
        rows.append((name, summarise(featurise(acts, args.resample), rng, ns=ns)))

    if not rows:
        raise SystemExit("no sets with >=2 demos")
    print(f"{'set':28s} {'n':>4s} {'PCs95':>6s} {'top1%':>6s} {'spread':>7s}")
    for name, r in rows:
        print(f"{name:28s} {r['n']:4d} {r['pcs_95']:6d} {100 * r['top1_pc']:6.1f} "
              f"{r['spread']:7.3f}")
    print(f"\ncoverage(n) -- mean nearest-neighbour distance to a random n-subset, /scale")
    print(f"{'set':28s} " + " ".join(f"{n:>6d}" for n in ns))
    for name, r in rows:
        print(f"{name:28s} " + " ".join(
            f"{r['coverage'][n]:6.3f}" if n in r["coverage"] else "     -" for n in ns))


if __name__ == "__main__":
    main()
