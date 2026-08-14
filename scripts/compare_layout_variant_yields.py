"""Compare a trainvar layout run's scripted yields against the parent eval-suite run.

The whole point of ``libero_object_unseen_stockbg_trainvar`` is that a variant differs from
its parent *only* in where the target sits and which distractors surround it (proved by
``scripts/verify_stockbg_trainvar.py``). So any yield gap between a variant and its parent
is attributable to the layout -- which is exactly the quantity this run exists to measure,
and the pilot's go/no-go criterion.

Reports the yield delta broken down by target slot and by category, flags variants that
regress by more than ``--tolerance`` points, and separates "collected fewer demos" from
"never built" (a RandomizationError-style failure in a swapped slot looks very different
from a grasp that just got harder).

Example:
    python scripts/compare_layout_variant_yields.py \
        --variant-report /tmp2/leocheng/ricl_scratch/stockbg_layoutext_pilot/suite_collect_report.json \
        --parent-report  /tmp2/leocheng/ricl_scratch/stockbg_rim/suite_collect_report.json
"""

import argparse
import collections
import json
import os
import statistics
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VAR_MANIFEST_DEFAULT = os.path.join(
    REPO, "libero/libero/bddl_files/libero_object_unseen_stockbg_trainvar/manifest.json")


def load_tasks(path):
    return {t["name"]: t for t in json.load(open(path))["tasks"]}


def fmt(v):
    return "  n/a" if v is None else f"{v:5.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant-report", required=True)
    ap.add_argument("--parent-report", required=True)
    ap.add_argument("--manifest", default=VAR_MANIFEST_DEFAULT)
    ap.add_argument("--tolerance", type=float, default=10.0,
                    help="points of yield a variant may lose before it is flagged")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    meta = load_tasks(args.manifest)
    var = load_tasks(args.variant_report)
    par = load_tasks(args.parent_report)

    recs = []
    for name, v in var.items():
        m = meta.get(name)
        if m is None:
            print(f"[warn] {name} not in {args.manifest}", file=sys.stderr)
            continue
        p = par.get(m["parent_task"])
        if p is None:
            print(f"[warn] parent {m['parent_task']} not in the parent report", file=sys.stderr)
            continue
        vy = v.get("collect_yield")
        py = p.get("collect_yield")
        recs.append({
            "name": name, "parent": m["parent_task"], "category": m["target_category"],
            "slot": m["target_slot"], "variant": m["variant"],
            "parent_yield": py, "variant_yield": vy,
            "delta": None if (vy is None or py is None) else round(vy - py, 1),
            "n_demos": v.get("n_demos", 0), "status": v.get("status"),
        })

    if not recs:
        raise SystemExit("no variant/parent pairs matched")

    def summarize(rows):
        d = [r["delta"] for r in rows if r["delta"] is not None]
        return {
            "n": len(rows),
            "demos": sum(r["n_demos"] for r in rows),
            "median_delta": round(statistics.median(d), 1) if d else None,
            "mean_variant_yield": round(
                statistics.mean([r["variant_yield"] for r in rows
                                 if r["variant_yield"] is not None]), 1) if d else None,
            "n_regressed": sum(1 for x in d if x < -args.tolerance),
            "n_zero": sum(1 for r in rows if r["n_demos"] == 0),
            "n_crashed": sum(1 for r in rows if r["status"] == "crashed"),
        }

    print(f"variants : {len(recs)}   parent run: {args.parent_report}")
    print(f"tolerance: {args.tolerance:.0f} points\n")

    hdr = (f"{'slot':>4s} {'n':>4s} {'demos':>6s} {'med d':>6s} {'mean y':>6s} "
           f"{'regr':>5s} {'zero':>5s} {'crash':>5s}")
    print("by target slot (slot 0 = original position, distractors only):")
    print(hdr)
    by_slot = collections.defaultdict(list)
    for r in recs:
        by_slot[r["slot"]].append(r)
    for slot in sorted(by_slot):
        s = summarize(by_slot[slot])
        print(f"{slot:4d} {s['n']:4d} {s['demos']:6d} {fmt(s['median_delta'])} "
              f"{fmt(s['mean_variant_yield'])} {s['n_regressed']:5d} {s['n_zero']:5d} "
              f"{s['n_crashed']:5d}")

    print("\nby category:")
    print(f"{'category':16s} {'n':>4s} {'demos':>6s} {'med d':>6s} {'mean y':>6s} "
          f"{'regr':>5s} {'zero':>5s} {'crash':>5s}")
    by_cat = collections.defaultdict(list)
    for r in recs:
        by_cat[r["category"]].append(r)
    for cat in sorted(by_cat):
        s = summarize(by_cat[cat])
        print(f"{cat:16s} {s['n']:4d} {s['demos']:6d} {fmt(s['median_delta'])} "
              f"{fmt(s['mean_variant_yield'])} {s['n_regressed']:5d} {s['n_zero']:5d} "
              f"{s['n_crashed']:5d}")

    overall = summarize(recs)
    print(f"\noverall: {overall['n']} variants, {overall['demos']} demos, "
          f"median delta {fmt(overall['median_delta']).strip()} pts, "
          f"{overall['n_regressed']} regressed >{args.tolerance:.0f} pts, "
          f"{overall['n_zero']} with zero demos, {overall['n_crashed']} crashed")

    flagged = sorted((r for r in recs if r["delta"] is not None and r["delta"] < -args.tolerance),
                     key=lambda r: r["delta"])
    if flagged:
        print(f"\nregressed by more than {args.tolerance:.0f} points ({len(flagged)}):")
        for r in flagged[:25]:
            print(f"  slot {r['slot']}  {r['category']:15s} {r['parent_yield']:5.1f} -> "
                  f"{r['variant_yield']:5.1f}  ({r['delta']:+.1f})  {r['name']}")
        if len(flagged) > 25:
            print(f"  ... and {len(flagged) - 25} more")

    zero = [r for r in recs if r["n_demos"] == 0]
    if zero:
        print(f"\nzero demos ({len(zero)}):")
        for r in zero[:25]:
            print(f"  slot {r['slot']}  {r['category']:15s} status={r['status']:10s} {r['name']}")

    if args.out_json:
        json.dump({"tolerance": args.tolerance,
                   "overall": overall,
                   "by_slot": {str(k): summarize(v) for k, v in sorted(by_slot.items())},
                   "by_category": {k: summarize(v) for k, v in sorted(by_cat.items())},
                   "variants": recs},
                  open(args.out_json, "w"), indent=2)
        print(f"\n[json] -> {args.out_json}")


if __name__ == "__main__":
    main()
