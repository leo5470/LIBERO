"""Select the Seen / Tier-A / Tier-B scale pool from robocasa_object_meta.json.

Runs in the LIBERO env (pure JSON + XML parse, no robocasa import). Deterministic and
seeded so the tier assignment is *pre-registered* before any collection: categories are
designated Seen-candidate vs Tier-B-candidate here, and only gate survivors are assigned
to a final tier later by ``make_tier_split.py``. We over-provision both candidate slices
comfortably above the post-gate targets (>=30 Seen / >=15 Tier B) so gate attrition never
forces a post-hoc rebalance.

Design (see plan think-about-whether-there-composed-feather.md):
  * Seen-candidate categories = the 13 LIBERO-overlap categories (base-familiar semantics)
    + a type-group-stratified slice of RoboCasa-only categories, up to --seen-target.
  * Per Seen category: up to --n-prime OBJAVERSE priming instances + up to --n-tiera AIGEN
    held-out instances (Tier-A, objaverse->aigen source shift). If a category has no aigen,
    Tier A falls back to held-out OBJAVERSE instances disjoint from priming (logged).
  * Tier-B-candidate categories = remaining RoboCasa-only categories (prefer the aigen-only
    NEW categories + produce/protein/bakery groups), up to --tierb-target; up to --n-tierb
    instances each, held out entirely.

Per-instance registry key = ``<category>__<source>_<suffix>`` (e.g. ``bowl__objaverse_7``);
one converted object dir per key. Emits scale_pool_manifest.json consumed by
``convert_robocasa_xml.py --manifest`` and ``run_scale_pipeline.py``.

Usage:
    python tools/select_scale_pool.py --meta robocasa_object_meta.json \
        --out scale_pool_manifest.json --seed 0
"""

import argparse
import json
import os
import random
import xml.etree.ElementTree as ET

import numpy as np

# LIBERO-overlap categories (semantic match to the base's own objects). These anchor Seen
# with base-familiar semantics; they can never be Tier B. Kept as Seen *candidates* only --
# most are hollow/handled shapes expected to drop at the top-down grasp gate, so they supply
# semantics, not category count (see plan P2).
LIBERO_OVERLAP = [
    "bowl", "plate", "basket", "pan", "mug", "ketchup", "milk", "corn", "wine", "tray",
    "mayonnaise", "butter_stick", "cream_cheese_stick", "cherry", "colander", "strainer",
    "can", "canned_food", "boxed_food", "cup", "coffee_cup", "glass_cup",
    "condiment_bottle", "tupperware", "juice",
]

# Tier-B is richest from raw produce / protein / bakery / prepared (LIBERO has ~none of
# these, so they are unambiguously novel categories). Used to bias the Tier-B slice.
TIERB_PREFERRED_GROUPS = {
    "fruit", "vegetable", "meat", "cooked_food", "bread_food", "pastry", "sweets", "dairy",
}


def group_of(cat_meta):
    t = cat_meta.get("types") or ["untyped"]
    return t[0] if t else "untyped"


def _bbox_half(model_xml, scale):
    """Approx baked reg_bbox half-extents [hx,hy,hz] (raw half * scale), or None.

    Mirrors convert_robocasa_xml._find_reg_bbox but reads the RAW (pre-bake) size and
    multiplies by the category scale so we can prefer graspable-sized instances and log a
    size covariate without importing the converter."""
    try:
        root = ET.parse(model_xml).getroot()
    except (ET.ParseError, OSError):
        return None
    wb = root.find("worldbody")
    if wb is None:
        return None
    s = np.array(scale, dtype=float).reshape(-1)
    if s.size == 1:
        s = np.repeat(s, 3)
    for geom in wb.iter("geom"):
        name = geom.get("name") or ""
        if "reg_bbox" in name:
            size = geom.get("size")
            if not size:
                return None
            half = np.array([float(x) for x in size.split()], dtype=float)
            return list(np.round(half * s[:3], 5))
    return None


def make_key(category, source, instance):
    suffix = instance[len(category) + 1:] if instance.startswith(category + "_") else instance
    return f"{category}__{source}_{suffix}"


def pick(rng, items, n):
    """Deterministically pick up to n items (shuffle a copy so callers can chain picks)."""
    pool = list(items)
    rng.shuffle(pool)
    return pool[:n], pool[n:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", default="scale_pool_manifest.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seen-target", type=int, default=40,
                    help="# Seen-candidate categories to designate (over-provisioned above "
                         "the >=30 post-gate target)")
    ap.add_argument("--tierb-target", type=int, default=25,
                    help="# Tier-B-candidate categories (over-provisioned above >=15)")
    ap.add_argument("--n-prime", type=int, default=5, help="objaverse priming instances / Seen cat")
    ap.add_argument("--n-tiera", type=int, default=3, help="aigen (or objaverse-fallback) Tier-A / Seen cat")
    ap.add_argument("--n-tierb", type=int, default=3, help="instances / Tier-B cat")
    ap.add_argument("--max-xy-half", type=float, default=0.075,
                    help="prefer instances whose baked horizontal half-extent is below this "
                         "(metres); larger ones are deprioritized, not excluded")
    ap.add_argument("--receptacle", default="rc_tray", help="excluded from manipuland selection")
    args = ap.parse_args()

    meta = json.load(open(args.meta))
    cats = meta["categories"]
    rng = random.Random(args.seed)

    # index: category -> source -> [instance records], graspable & non-excluded only
    idx = {}
    for i in meta["instances"]:
        if not i.get("graspable") or i.get("excluded"):
            continue
        idx.setdefault(i["category"], {}).setdefault(i["source"], []).append(i)

    ondisk_cats = sorted(idx)
    overlap_ondisk = [c for c in LIBERO_OVERLAP if c in idx]
    robocasa_only = [c for c in ondisk_cats if c not in LIBERO_OVERLAP]

    # --- designate Seen-candidate categories: overlap + stratified RoboCasa-only slice ---
    seen_cats = list(overlap_ondisk)
    need = max(0, args.seen_target - len(seen_cats))
    # stratify the RoboCasa-only pool by type-group, round-robin so Seen spans groups.
    # Only categories WITH objaverse instances can be Seen (priming needs a uniform objaverse
    # distribution); aigen-only categories are novel-category Tier-B fodder instead.
    by_group = {}
    for c in robocasa_only:
        if "objaverse" not in idx.get(c, {}):
            continue
        by_group.setdefault(group_of(cats[c]), []).append(c)
    for g in by_group:
        rng.shuffle(by_group[g])
    groups = sorted(by_group)
    rng.shuffle(groups)
    seen_extra = []
    while need > 0 and any(by_group.values()):
        for g in groups:
            if by_group[g] and need > 0:
                seen_extra.append(by_group[g].pop())
                need -= 1
    seen_cats += seen_extra
    seen_set = set(seen_cats)

    # --- Tier-B-candidate categories: remaining RoboCasa-only, prefer novel produce groups ---
    tierb_pool = [c for c in robocasa_only if c not in seen_set]
    tierb_pool.sort(key=lambda c: (group_of(cats[c]) not in TIERB_PREFERRED_GROUPS, c))
    tierb_cats = tierb_pool[:args.tierb_target]

    # --- per-category instance selection -> manifest keys ---
    def size_sorted(recs):
        """graspable-sized first: smaller baked horizontal half-extent ranks earlier."""
        def sz(r):
            h = _bbox_half(r["model_xml"], r["scale"])
            return max(h[0], h[1]) if h else 1e9
        return sorted(recs, key=sz)

    keys = []

    def emit(rec, role, is_overlap):
        key = make_key(rec["category"], rec["source"], rec["instance"])
        keys.append({
            "key": key,
            "category": rec["category"],
            "instance": rec["instance"],
            "source": rec["source"],
            "role_candidate": role,             # prime | tier_a | tier_b
            "is_overlap": is_overlap,
            "group": group_of(cats[rec["category"]]),
            "applied_scale": rec["scale"],
            "model_xml": rec["model_xml"],
            "bbox_half": _bbox_half(rec["model_xml"], rec["scale"]),
        })

    seen_report = {}
    for c in seen_cats:
        is_ov = c in LIBERO_OVERLAP
        obj = size_sorted(idx[c].get("objaverse", []))
        aig = size_sorted(idx[c].get("aigen", []))
        if obj:  # normal case: objaverse priming + aigen (or objaverse-fallback) Tier-A holdout
            prime, prime_rest, prime_src = obj[:args.n_prime], obj[args.n_prime:], "objaverse"
            if aig:
                tier_a, ta_src = aig[:args.n_tiera], "aigen"
            else:
                tier_a, ta_src = prime_rest[:args.n_tiera], "objaverse_fallback"
        else:  # overlap category with no objaverse (e.g. cherry) -> prime on aigen instead
            prime, prime_rest, prime_src = aig[:args.n_prime], aig[args.n_prime:], "aigen"
            tier_a, ta_src = prime_rest[:args.n_tiera], "aigen_instance_only"
        for r in prime:
            emit(r, "prime", is_ov)
        for r in tier_a:
            emit(r, "tier_a", is_ov)
        seen_report[c] = {"group": group_of(cats[c]), "is_overlap": is_ov,
                          "n_prime": len(prime), "n_tier_a": len(tier_a),
                          "prime_source": prime_src, "tier_a_source": ta_src}

    tierb_report = {}
    for c in tierb_cats:
        recs = size_sorted(idx[c].get("objaverse", []) or idx[c].get("aigen", []))
        chosen = recs[:args.n_tierb]
        for r in chosen:
            emit(r, "tier_b", False)
        tierb_report[c] = {"group": group_of(cats[c]), "n": len(chosen),
                           "source": chosen[0]["source"] if chosen else None}

    manifest = {
        "seed": args.seed,
        "params": {k: getattr(args, k) for k in
                   ("seen_target", "tierb_target", "n_prime", "n_tiera", "n_tierb", "max_xy_half")},
        "seen_candidate_categories": seen_cats,
        "tierb_candidate_categories": tierb_cats,
        "seen_report": seen_report,
        "tierb_report": tierb_report,
        "keys": keys,
    }
    with open(args.out, "w") as f:
        json.dump(manifest, f, indent=2)

    # --- summary ---
    n_prime = sum(1 for k in keys if k["role_candidate"] == "prime")
    n_ta = sum(1 for k in keys if k["role_candidate"] == "tier_a")
    n_tb = sum(1 for k in keys if k["role_candidate"] == "tier_b")
    n_ta_aigen = sum(1 for c, r in seen_report.items() if r["tier_a_source"] == "aigen")
    print(f"[select] Seen-candidate categories: {len(seen_cats)} "
          f"({len(overlap_ondisk)} overlap + {len(seen_extra)} RoboCasa-only)")
    print(f"[select] Tier-B-candidate categories: {len(tierb_cats)}")
    print(f"[select] keys: {len(keys)} total  (prime={n_prime}, tier_a={n_ta}, tier_b={n_tb})")
    print(f"[select] Tier-A source: {n_ta_aigen}/{len(seen_cats)} Seen cats use real aigen "
          f"(rest objaverse-fallback)")
    print(f"[select] wrote {args.out}")


if __name__ == "__main__":
    main()
