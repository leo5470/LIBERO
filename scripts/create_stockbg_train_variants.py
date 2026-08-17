"""Per-object train-time variants for ``libero_object_unseen_stockbg``.

The eval suite gives each object exactly ONE task: fixed layout, fixed distractor
identities, and only a 5x5 cm placement jitter per reset. For more per-object demo
diversity this script emits extra train-only BDDLs into a SEPARATE suite dir
(default ``libero_object_unseen_stockbg_trainvar``) -- the eval suite is never touched.

Each variant changes, relative to the object's eval task:
  * **target slot** -- the ``(:ranges ...)`` of ``target_object_region`` are swapped
    with one of the five ``other_object_region_*``, moving the target 10-20 cm across
    the table while keeping the exact stock region geometry (placement feasibility is
    inherited, not re-risked: RandomizationError is this suite's #1 blocker, so no new
    regions are invented);
  * **distractor identities** -- the 5 distractor categories are re-drawn (seeded) from
    the union of the two stock bases' distractor pools, excluding the target category.

Variant ``k`` uses slot ``k % 6`` (0 = the original target region) and an independent
distractor draw, so ``--variants-per-object 6`` covers every slot once.

The emitted manifest is directly consumable by ``collect_suite_demonstrations.py``:
    python scripts/collect_suite_demonstrations.py \
        --manifest libero/libero/bddl_files/libero_object_unseen_stockbg_trainvar/manifest.json \
        --out-root /tmp2/.../trainvar_demos --select cereal__objaverse_1

Example:
    python scripts/create_stockbg_train_variants.py \
        --target-key cereal__objaverse_1 --variants-per-object 6 --seed 0
"""

import argparse
import json
import os
import re
import sys
import zlib

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from create_libero_object_stockbg_tasks import parse_base  # noqa: E402

SUITE_DIR_DEFAULT = os.path.join(
    REPO, "libero/libero/bddl_files/libero_object_unseen_stockbg")
STOCK_DIR_DEFAULT = os.path.join(REPO, "libero/libero/bddl_files/libero_object")

_REGION_RE = re.compile(
    r"\((target_object_region|other_object_region_\d)\s*"
    r"\(:target floor\)\s*\(:ranges \(\s*\(([^()]+)\)", re.S)


def region_ranges(raw):
    """{region_name: 'x1 y1 x2 y2' coordinate string} from a suite BDDL."""
    return {m.group(1): m.group(2).strip() for m in _REGION_RE.finditer(raw)}


def swap_target_slot(raw, slot):
    """Swap the target region's range with other_object_region_<slot-1> (slot 0 = keep)."""
    if slot == 0:
        return raw
    other = f"other_object_region_{slot - 1}"
    ranges = region_ranges(raw)
    assert "target_object_region" in ranges and other in ranges, sorted(ranges)

    def repl(m):
        name, coords = m.group(1), m.group(2).strip()
        if name == "target_object_region":
            coords = ranges[other]
        elif name == other:
            coords = ranges["target_object_region"]
        return m.group(0).replace(m.group(2), coords)

    return _REGION_RE.sub(repl, raw)


def swap_distractors(raw, old_cats, new_cats):
    """Rewrite the 5 distractor identities (objects block + init lines), in order."""
    for old, new in zip(old_cats, new_cats):
        if old == new:
            continue
        pat_obj = re.compile(rf"^(\s*){re.escape(old)}_1 - {re.escape(old)}$", re.M)
        raw, n1 = pat_obj.subn(rf"\g<1>{new}_1 - {new}", raw, count=1)
        pat_init = re.compile(
            rf"^(\s*)\(On {re.escape(old)}_1 (floor_other_object_region_\d)\)$", re.M)
        raw, n2 = pat_init.subn(rf"\g<1>(On {new}_1 \g<2>)", raw, count=1)
        assert n1 == 1 and n2 == 1, (old, new, n1, n2)
    return raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite-dir", default=SUITE_DIR_DEFAULT,
                    help="the eval suite whose tasks are being varied (read-only)")
    ap.add_argument("--stock-dir", default=STOCK_DIR_DEFAULT)
    ap.add_argument("--out-dir", default=None,
                    help="default: <suite-dir>_trainvar")
    ap.add_argument("--target-key", nargs="*", default=None,
                    help="object keys to generate variants for")
    ap.add_argument("--all", action="store_true",
                    help="generate variants for every task in the suite manifest")
    ap.add_argument("--variants-per-object", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not args.all and not args.target_key:
        ap.error("give --target-key KEY [KEY ...] or --all")

    manifest = json.load(open(os.path.join(args.suite_dir, "manifest.json")))
    by_key = {t["target_key"]: t for t in manifest["tasks"]}
    keys = list(by_key) if args.all else list(args.target_key)
    missing = [k for k in keys if k not in by_key]
    if missing:
        raise SystemExit(f"keys not in suite manifest: {missing}")

    # distractor pool = union of the two stock bases' distractor categories
    pool = set()
    for fname in manifest["base_bddls"].values():
        _, dcats = parse_base(open(os.path.join(args.stock_dir, fname)).read())
        pool.update(dcats)
    pool = sorted(pool)

    out_dir = args.out_dir or (args.suite_dir.rstrip("/") + "_trainvar")
    os.makedirs(out_dir, exist_ok=True)

    tasks = []
    for key in keys:
        parent = by_key[key]
        raw_parent = open(os.path.join(args.suite_dir, parent["name"] + ".bddl")).read()
        cat = parent["target_category"]
        cand = [c for c in pool if c != cat]
        rng = np.random.default_rng([args.seed, zlib.crc32(key.encode())])
        for k in range(args.variants_per_object):
            slot = k % 6
            new_d = list(rng.choice(cand, size=5, replace=False))
            raw = swap_target_slot(raw_parent, slot)
            raw = swap_distractors(raw, parent["distractors"], new_d)
            name = f"{parent['name']}_var{k}"
            with open(os.path.join(out_dir, name + ".bddl"), "w") as f:
                f.write(raw)
            tasks.append({
                "name": name,
                "language": parent["language"],
                "target_key": key,
                "target_category": cat,
                "layout_variant": parent["layout_variant"],
                "parent_task": parent["name"],
                "variant": k,
                "target_slot": slot,
                "distractors": new_d,
            })

    var_manifest = {
        "suite": manifest["suite"] + "_trainvar",
        "parent_suite": manifest["suite"],
        "seed": args.seed,
        "variants_per_object": args.variants_per_object,
        "distractor_pool": pool,
        "num_tasks": len(tasks),
        "tasks": tasks,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(var_manifest, f, indent=2)
    print(f"[generated] {len(tasks)} variant tasks for {len(keys)} objects "
          f"-> {out_dir}")


if __name__ == "__main__":
    main()
