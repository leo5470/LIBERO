"""Emit a converter manifest covering the FULL strict-unseen pool (no scripted gate).

Unlike tools/select_scale_pool.py (which budgeted instances per category because every
key had to pay for scripted-demo collection), this selector has no gate to feed: it lists
EVERY eligible instance (graspable, not excluded) of every strict-pool category. Physical
feasibility is decided later, empirically, by scripts/smoke_check_suite.py.

Outputs:
  full_pool_manifest.json  -- input for tools/convert_robocasa_xml.py --manifest
                              (keys: [{key, category, instance, source, applied_scale,
                              model_xml}]; already-converted keys are skipped there)
  full_pool_meta.json      -- per-key tags for suite-manifest enrichment: category,
                              libero_overlap, prior_status (prime/tier_a/tier_b survivor,
                              dropped, not_collected, unattempted, on_disk_extra)

Usage:
    python scripts/make_full_pool_manifest.py \
        --meta robocasa_object_meta.json --pool-md robocasa_object_pool.md \
        --tier-split /tmp2/leocheng/ricl_scratch/scale/tier_split.json \
        --out-manifest full_pool_manifest.json --out-meta full_pool_meta.json
"""

import argparse
import ast
import json
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(REPO, "libero", "libero", "assets", "robocasa_objects")

# mirrors tools/select_scale_pool.py LIBERO_OVERLAP (categories semantically present in
# LIBERO training scenes; outside the strict pool but their on-disk instances stay
# eval-eligible, tagged, so the user can filter when choosing train/eval splits)
LIBERO_OVERLAP = {
    "bowl", "plate", "basket", "pan", "mug", "ketchup", "milk", "corn", "wine", "tray",
    "mayonnaise", "butter_stick", "cream_cheese_stick", "cherry", "colander", "strainer",
    "can", "canned_food", "boxed_food", "cup", "coffee_cup", "glass_cup",
    "condiment_bottle", "tupperware", "juice",
}

# converted by hand in the Phase-0 pilot (--map) with no instance provenance in the key;
# their categories are fully covered by properly-keyed imports, so never target these
BARE_PILOT_KEYS = {"apple", "bell_pepper", "lemon", "lime", "orange", "rc_tray"}


def make_key(category, source, instance):
    # same rule as tools/select_scale_pool.py: strip the category prefix when present
    suffix = instance[len(category) + 1:] if instance.startswith(category + "_") else instance
    return f"{category}__{source}_{suffix}"


def load_strict_categories(pool_md):
    md = open(pool_md).read()
    m = re.search(r"\[([^\]]*)\]\s*#\s*135 strict", md, re.S)
    if not m:
        raise SystemExit(f"[error] could not find the '# 135 strict' list in {pool_md}")
    cats = ast.literal_eval("[" + m.group(1) + "]")
    assert len(cats) == 135, len(cats)
    return cats


def load_prior_status(tier_split_path):
    """key -> prior scale-run status (informational only; nothing is filtered on it)."""
    status = {}
    if not tier_split_path or not os.path.isfile(tier_split_path):
        return status
    with open(tier_split_path) as f:
        split = json.load(f)
    for tier in ("seen", "tier_a", "tier_b"):
        for rec in split.get(tier, []):
            status[rec["key"]] = f"gate_survivor_{tier}"
    for rec in split.get("dropped", []):
        status[rec["key"]] = "gate_dropped"
    for key in split.get("not_collected", []):
        status[key] = "collect_crashed"
    return status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default=os.path.join(REPO, "robocasa_object_meta.json"))
    ap.add_argument("--pool-md", default=os.path.join(REPO, "robocasa_object_pool.md"))
    ap.add_argument("--tier-split", default="/tmp2/leocheng/ricl_scratch/scale/tier_split.json")
    ap.add_argument("--out-manifest", default=os.path.join(REPO, "full_pool_manifest.json"))
    ap.add_argument("--out-meta", default=os.path.join(REPO, "full_pool_meta.json"))
    args = ap.parse_args()

    meta = json.load(open(args.meta))
    strict = set(load_strict_categories(args.pool_md))
    prior = load_prior_status(args.tier_split)

    keys, key_meta = [], {}
    n_inelig, cats_seen = 0, set()
    for rec in meta["instances"]:
        cat = rec["category"]
        if cat not in strict:
            continue
        if not rec.get("graspable") or rec.get("excluded"):
            n_inelig += 1
            continue
        key = make_key(cat, rec["source"], rec["instance"])
        cats_seen.add(cat)
        keys.append({
            "key": key,
            "category": cat,
            "instance": rec["instance"],
            "source": rec["source"],
            "applied_scale": rec.get("scale", 1.0),
            "model_xml": rec["model_xml"],
        })
        key_meta[key] = {
            "category": cat,
            "libero_overlap": False,
            "prior_status": prior.get(key, "unattempted"),
        }

    # on-disk keys outside the strict pool (LIBERO-overlap categories) stay eval-eligible
    n_extra = 0
    for name in sorted(os.listdir(ASSETS)):
        if name in key_meta or name in BARE_PILOT_KEYS:
            continue
        if not os.path.isfile(os.path.join(ASSETS, name, "model.xml")):
            continue
        cat = name.split("__")[0]
        key_meta[name] = {
            "category": cat,
            "libero_overlap": cat in LIBERO_OVERLAP,
            "prior_status": prior.get(name, "on_disk_extra"),
        }
        n_extra += 1

    strict_missing = sorted(strict - cats_seen)
    json.dump({"params": {"pool": "strict135_all_eligible_instances", "gate": "none"},
               "keys": keys}, open(args.out_manifest, "w"), indent=1)
    json.dump({"keys": key_meta,
               "strict_categories_no_eligible_instances": strict_missing,
               "bare_pilot_keys_skipped": sorted(BARE_PILOT_KEYS)},
              open(args.out_meta, "w"), indent=1)
    print(f"[manifest] {len(keys)} strict instances across {len(cats_seen)} categories "
          f"-> {args.out_manifest}")
    print(f"[meta] +{n_extra} on-disk non-strict keys tagged; {n_inelig} ineligible "
          f"instances skipped; {len(strict_missing)} strict categories have no eligible "
          f"instances -> {args.out_meta}")


if __name__ == "__main__":
    main()
