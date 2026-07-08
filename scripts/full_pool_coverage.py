"""Coverage audit: every strict-pool category lands in exactly one bucket.

Buckets (per category, decided by its best-covered instance):
  evaluated            -- >=1 feasible task in the suite
  physics_excluded     -- instances converted + tasks generated, but every task failed
                          the policy-independent feasibility smoke / init generation
  quarantined          -- instances existed but every conversion was quarantined
  no_eligible_instances-- category has zero (graspable, not excluded) instances in
                          robocasa_object_meta.json

Also emits per-key coverage for the user's later train/eval split decision.

Usage:
    python scripts/full_pool_coverage.py \
        --manifest libero/libero/bddl_files/libero_object_unseen_full/manifest.json \
        --out coverage_report.json
"""

import argparse
import ast
import collections
import json
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_strict_categories(pool_md):
    md = open(pool_md).read()
    m = re.search(r"\[([^\]]*)\]\s*#\s*135 strict", md, re.S)
    return ast.literal_eval("[" + m.group(1) + "]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(
        REPO, "libero/libero/bddl_files/libero_object_unseen_full/manifest.json"))
    ap.add_argument("--pool-md", default=os.path.join(REPO, "robocasa_object_pool.md"))
    ap.add_argument("--key-meta", default=os.path.join(REPO, "full_pool_meta.json"))
    ap.add_argument("--import-report", default=os.path.join(REPO, "import_report.json"))
    ap.add_argument("--out", default=os.path.join(REPO, "coverage_report.json"))
    args = ap.parse_args()

    strict = load_strict_categories(args.pool_md)
    key_meta = json.load(open(args.key_meta))
    imp = {e["key"]: e["status"] for e in json.load(open(args.import_report))["keys"]}
    manifest = json.load(open(args.manifest))

    task_of = {}  # key -> task record
    for t in manifest["tasks"]:
        task_of[t["target_key"]] = t

    keys_by_cat = collections.defaultdict(list)
    for key, meta in key_meta["keys"].items():
        keys_by_cat[meta["category"]].append(key)

    def key_state(key):
        conv = imp.get(key, "exists")  # on-disk extras aren't in the import manifest
        if conv == "quarantined":
            return "quarantined"
        t = task_of.get(key)
        if t is None:
            return "no_task_generated"
        if t.get("excluded"):
            return "physics_excluded"
        return "evaluated"

    per_key = {k: key_state(k) for keys in keys_by_cat.values() for k in keys}

    order = ["evaluated", "physics_excluded", "no_task_generated", "quarantined"]
    per_cat, bucket_counts = {}, collections.Counter()
    for cat in strict:
        keys = keys_by_cat.get(cat, [])
        if not keys:
            bucket = "no_eligible_instances"
            detail = {}
        else:
            states = collections.Counter(per_key[k] for k in keys)
            bucket = next(s for s in order if states.get(s))
            detail = dict(states)
        per_cat[cat] = {"bucket": bucket, "instances": detail}
        bucket_counts[bucket] += 1

    # non-strict (LIBERO-overlap) categories ride along, reported separately
    overlap_cats = sorted(c for c in keys_by_cat if c not in set(strict))
    overlap = {c: dict(collections.Counter(per_key[k] for k in keys_by_cat[c]))
               for c in overlap_cats}

    out = {
        "strict_bucket_counts": dict(bucket_counts),
        "strict_categories": per_cat,
        "libero_overlap_categories": overlap,
        "per_key": per_key,
    }
    json.dump(out, open(args.out, "w"), indent=1)
    print("[coverage] strict categories:", dict(bucket_counts))
    print(f"[coverage] overlap categories riding along: {len(overlap_cats)}")
    print(f"[out] {args.out}")
    assert sum(bucket_counts.values()) == len(strict), "a category fell out of the audit"


if __name__ == "__main__":
    main()
