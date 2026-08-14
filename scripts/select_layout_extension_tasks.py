"""Select ``libero_object_unseen_stockbg`` tasks worth collecting in extra layouts.

The eval suite gives each object exactly ONE task: one of two stock layouts
(``layout_variant`` 0 = milk base, 1 = bbq_sauce base), a fixed target slot, fixed
distractors, and only a 5x5 cm jitter per reset. Cosmos-Predict2.5 scores cluster
all-or-nothing per (category, layout_variant) cell, so a category whose rate sits on the
layout split -- e.g. ``shaker`` 0.545 vs f=0.545, ``jam`` 0.508 vs 0.500 -- is one where
the object is graspable and the *placement* is what breaks. Those are the categories where
more layouts should pay off, provided our scripted policy already solves the object.

This script joins the two evidence sources and emits the target-key list consumed by
``scripts/create_stockbg_train_variants.py``:

  * per-category policy success, from ``eval/cosmos25_stockbg_category_sr.csv``
    (the only surviving copy of that eval -- the per-task result JSONs are gone);
  * per-task scripted ``collect_yield``, from a run's ``suite_collect_report.json``.

A task is selected when its category's success rate is inside ``[--sr-min, --sr-max]`` and
its own scripted yield is strictly above ``--min-yield``. ``--categories`` narrows to a
hand-picked subset afterwards; ``--min-cat-frac`` / ``--min-cat-tasks`` drop categories too
thin or too flaky to be worth the collection time.

Example (the 5-category / 102-task selection):
    python scripts/select_layout_extension_tasks.py \
        --report /tmp2/leocheng/ricl_scratch/stockbg_rim/suite_collect_report.json \
        --categories egg boxed_drink brussel_sprout honey_bottle mushroom \
        --out-keys eval/layout_extension_keys.txt \
        --out-report eval/layout_extension_selection.json
"""

import argparse
import collections
import csv
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SR_CSV_DEFAULT = os.path.join(REPO, "eval/cosmos25_stockbg_category_sr.csv")
MANIFEST_DEFAULT = os.path.join(
    REPO, "libero/libero/bddl_files/libero_object_unseen_stockbg/manifest.json")
SR_COLUMN_DEFAULT = "cosmos-predict2.5-2b"


def load_success_rates(path, column):
    """{category: success_rate} from the aggregator's per-category CSV."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{path}: no rows")
    if column not in rows[0]:
        raise SystemExit(f"{path}: no column {column!r}; have {sorted(rows[0])}")
    return {r["category"]: float(r[column]) for r in rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True,
                    help="suite_collect_report.json from a scripted collection run")
    ap.add_argument("--sr-csv", default=SR_CSV_DEFAULT)
    ap.add_argument("--sr-column", default=SR_COLUMN_DEFAULT,
                    help="which policy's column to read from --sr-csv")
    ap.add_argument("--manifest", default=MANIFEST_DEFAULT,
                    help="eval suite manifest, for target_key / layout_variant")
    ap.add_argument("--sr-min", type=float, default=0.30)
    ap.add_argument("--sr-max", type=float, default=0.60)
    ap.add_argument("--min-yield", type=float, default=80.0,
                    help="task's scripted collect_yield must be STRICTLY above this")
    ap.add_argument("--categories", nargs="*", default=None,
                    help="restrict to these categories (applied after the band filter)")
    ap.add_argument("--min-cat-frac", type=float, default=0.0,
                    help="drop a category unless this fraction of its tasks clears --min-yield")
    ap.add_argument("--min-cat-tasks", type=int, default=0,
                    help="drop a category with fewer than this many qualifying tasks")
    ap.add_argument("--out-keys", default=None,
                    help="write the selected target keys, one per line")
    ap.add_argument("--out-report", default=None,
                    help="write the full selection report as JSON")
    args = ap.parse_args()

    sr = load_success_rates(args.sr_csv, args.sr_column)
    report = json.load(open(args.report))
    manifest = json.load(open(args.manifest))
    meta = {t["name"]: t for t in manifest["tasks"]}

    band = {c for c, v in sr.items() if args.sr_min <= v <= args.sr_max}
    if args.categories:
        unknown = [c for c in args.categories if c not in sr]
        if unknown:
            raise SystemExit(f"unknown categories (not in {args.sr_csv}): {unknown}")
        outside = [c for c in args.categories if c not in band]
        if outside:
            print(f"[warn] outside the [{args.sr_min}, {args.sr_max}] band, keeping anyway: "
                  f"{', '.join(f'{c}={sr[c]:.3f}' for c in outside)}", file=sys.stderr)
        band = set(args.categories)

    per_cat = collections.defaultdict(list)
    for t in report["tasks"]:
        if t["category"] in band:
            per_cat[t["category"]].append(t)

    selected, dropped = {}, {}
    for cat, tasks in per_cat.items():
        good = [t for t in tasks
                if t.get("status") == "ok" and t.get("collect_yield", 0.0) > args.min_yield]
        frac = len(good) / len(tasks) if tasks else 0.0
        if len(good) < args.min_cat_tasks or frac < args.min_cat_frac:
            dropped[cat] = {"n_tasks": len(tasks), "n_qualifying": len(good),
                            "frac": round(frac, 3)}
            continue
        selected[cat] = sorted(good, key=lambda t: -t["collect_yield"])

    rows, keys = [], []
    for cat in sorted(selected, key=lambda c: sr[c]):
        good = selected[cat]
        layouts = collections.Counter(meta[t["name"]]["layout_variant"] for t in good)
        rows.append({
            "category": cat,
            "success_rate": sr[cat],
            "n_tasks": len(per_cat[cat]),
            "n_selected": len(good),
            "n_perfect_yield": sum(t["collect_yield"] == 100.0 for t in good),
            "layout_0": layouts[0],
            "layout_1": layouts[1],
        })
        keys += [meta[t["name"]]["target_key"] for t in good]

    hdr = f"{'category':18s} {args.sr_column[:8]:>8s} {'tasks':>5s} {'sel':>5s} {'100%':>5s} {'lay0/1':>7s}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['category']:18s} {r['success_rate']:8.3f} {r['n_tasks']:5d} "
              f"{r['n_selected']:5d} {r['n_perfect_yield']:5d} "
              f"{str(r['layout_0']) + '/' + str(r['layout_1']):>7s}")
    print("-" * len(hdr))
    print(f"{'TOTAL':18s} {'':8s} {sum(r['n_tasks'] for r in rows):5d} "
          f"{len(keys):5d} {sum(r['n_perfect_yield'] for r in rows):5d}")
    if dropped:
        print(f"\n[dropped] {len(dropped)} categories below --min-cat-frac/--min-cat-tasks: "
              f"{', '.join(sorted(dropped))}", file=sys.stderr)

    assert len(keys) == len(set(keys)), "duplicate target keys in selection"

    if args.out_keys:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_keys)), exist_ok=True)
        with open(args.out_keys, "w") as f:
            f.write("\n".join(keys) + "\n")
        print(f"\n[keys] {len(keys)} -> {args.out_keys}")
    if args.out_report:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_report)), exist_ok=True)
        payload = {
            "report": os.path.abspath(args.report),
            "sr_csv": os.path.abspath(args.sr_csv),
            "sr_column": args.sr_column,
            "filters": {"sr_min": args.sr_min, "sr_max": args.sr_max,
                        "min_yield": args.min_yield,
                        "categories": args.categories,
                        "min_cat_frac": args.min_cat_frac,
                        "min_cat_tasks": args.min_cat_tasks},
            "n_categories": len(rows),
            "n_selected": len(keys),
            "per_category": rows,
            "dropped_categories": dropped,
            "target_keys": keys,
        }
        with open(args.out_report, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[report] -> {args.out_report}")


if __name__ == "__main__":
    main()
