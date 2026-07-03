"""Emit the Seen / Tier-A / Tier-B split over gate SURVIVORS -> tier_split.json.

A key survives the empirical gate iff its collect ``result.json`` has
``reached_target == True`` (the chosen grasp_frac reached the demo target within the collect
budget). Tiers follow the pre-registered per-key ``role_candidate`` from the manifest:
prime -> Seen, tier_a -> Tier A, tier_b -> Tier B. Only survivors are assigned (the gate
never re-decides which category is Seen vs Tier B -- that was fixed in select_scale_pool).

Asserts the disjointness the downstream RICL experiment relies on:
  * Seen instances  ∩  Tier-A instances      == ∅  (instance-level novelty)
  * Seen categories ∩  Tier-B categories     == ∅  (category-level novelty)

Usage:
    python tools/make_tier_split.py --manifest scale_pool_manifest.json \
        --collect-dir /tmp2/leocheng/ricl_scratch/scale/collect --out tier_split.json
"""

import argparse
import collections
import json
import os

ROLE_TO_TIER = {"prime": "seen", "tier_a": "tier_a", "tier_b": "tier_b"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--collect-dir", required=True,
                    help="dir with <key>/result.json from the collector")
    ap.add_argument("--out", default="tier_split.json")
    ap.add_argument("--min-seen-cats", type=int, default=30)
    ap.add_argument("--min-tierb-cats", type=int, default=15)
    args = ap.parse_args()

    manifest = json.load(open(args.manifest))
    tiers = {"seen": [], "tier_a": [], "tier_b": []}
    dropped, not_collected = [], []

    for rec in manifest["keys"]:
        key = rec["key"]
        rj = os.path.join(args.collect_dir, key, "result.json")
        if not os.path.isfile(rj):
            not_collected.append(key)
            continue
        r = json.load(open(rj))
        member = {
            "key": key, "category": rec["category"], "source": rec["source"],
            "is_overlap": rec.get("is_overlap"), "group": rec.get("group"),
            "bbox_half": rec.get("bbox_half"), "applied_scale": rec.get("applied_scale"),
            "chosen_grasp_frac": r.get("chosen_grasp_frac"),
            "gate_yield": r.get("gate_yield"), "n_demos": r.get("n_demos"),
        }
        if not r.get("reached_target"):
            dropped.append({**member, "reason": "gate_yield_below_target"})
            continue
        tiers[ROLE_TO_TIER[rec["role_candidate"]]].append(member)

    def cats(tier):
        return sorted({m["category"] for m in tiers[tier]})

    seen_inst = {m["key"] for m in tiers["seen"]}
    tiera_inst = {m["key"] for m in tiers["tier_a"]}
    seen_cats, tiera_cats, tierb_cats = cats("seen"), cats("tier_a"), cats("tier_b")

    # --- disjointness the RICL experiment depends on ---
    assert seen_inst.isdisjoint(tiera_inst), "Seen and Tier-A share instances!"
    assert set(seen_cats).isdisjoint(tierb_cats), \
        f"Seen and Tier-B share categories: {set(seen_cats) & set(tierb_cats)}"

    split = {
        "counts": {
            "seen_categories": len(seen_cats), "tier_a_categories": len(tiera_cats),
            "tier_b_categories": len(tierb_cats),
            "seen_instances": len(tiers["seen"]), "tier_a_instances": len(tiers["tier_a"]),
            "tier_b_instances": len(tiers["tier_b"]),
            "dropped": len(dropped), "not_collected": len(not_collected),
        },
        "seen": tiers["seen"], "tier_a": tiers["tier_a"], "tier_b": tiers["tier_b"],
        "dropped": dropped, "not_collected": not_collected,
    }
    with open(args.out, "w") as f:
        json.dump(split, f, indent=2)

    c = split["counts"]
    print(f"[tier-split] Seen: {c['seen_categories']} cats / {c['seen_instances']} inst | "
          f"Tier A: {c['tier_a_categories']} cats / {c['tier_a_instances']} inst | "
          f"Tier B: {c['tier_b_categories']} cats / {c['tier_b_instances']} inst")
    print(f"[tier-split] dropped(gated)={c['dropped']} not_collected={c['not_collected']}")
    # source-shift + overlap covariate summary for Tier A
    ta_src = collections.Counter(m["source"] for m in tiers["tier_a"])
    ta_ov = collections.Counter(m["is_overlap"] for m in tiers["tier_a"])
    print(f"[tier-split] Tier-A source mix: {dict(ta_src)} | overlap mix: {dict(ta_ov)}")
    if c["seen_categories"] < args.min_seen_cats:
        print(f"[warn] Seen categories {c['seen_categories']} < target {args.min_seen_cats}")
    if c["tier_b_categories"] < args.min_tierb_cats:
        print(f"[warn] Tier-B categories {c['tier_b_categories']} < target {args.min_tierb_cats}")
    print(f"[tier-split] wrote {args.out}")


if __name__ == "__main__":
    main()
