"""Exhaustive constraint verifier for ``libero_object_unseen_stockbg`` (pure text, no robosuite).

Proves, over **every** emitted BDDL, that each task is "one of the 2 stock files with ONLY the target
changed." Per task it hard-checks:

  1. minimal-diff / 2-file provenance -- byte-identical to its declared stock base except exactly the 5
     allowed target lines (:language, target :objects line, :obj_of_interest, :init On, :goal In);
  2. distractors untouched -- the 5 distractor :objects lines + their (On ... other_object_region_i);
  3. regions untouched -- the whole (:regions ...) block equals the base's;
  4. base in {milk, bbq_sauce} and the target-slot coords match that base's layout;
  5. no category collision -- target category not among the base's 5 distractor categories;
  6. target consistency -- obj_of_interest == {key_1, basket_1}, goal/objects reference key_1;
  7. coverage/bijection -- one file per targets-file instance, 1:1 with the manifest.

Exits non-zero on any violation.

Example:
    python scripts/verify_stockbg_suite.py \
        --manifest libero/libero/bddl_files/libero_object_unseen_stockbg/manifest.json
"""

import argparse
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STOCK_DIR_DEFAULT = os.path.join(REPO, "libero/libero/bddl_files/libero_object")
# expected target_object_region center per base file (slot A vs slot B).
BASE_SLOT = {
    "pick_up_the_milk_and_place_it_in_the_basket.bddl": (-0.12, -0.24),       # A
    "pick_up_the_bbq_sauce_and_place_it_in_the_basket.bddl": (0.05, -0.10),   # B
}
BASE_LAYOUT = {  # base bddl -> expected layout_variant
    "pick_up_the_milk_and_place_it_in_the_basket.bddl": 0,
    "pick_up_the_bbq_sauce_and_place_it_in_the_basket.bddl": 1,
}


def category_of(key):
    return key.split("__", 1)[0]


def region_block(raw):
    return raw[raw.index("(:regions"): raw.index("(:fixtures")]


def objects_block(raw):
    m = re.search(r"\(:objects(.*?)\)\s*\n\s*\(:obj_of_interest", raw, re.S)
    return m.group(1) if m else ""


def target_region_center(raw):
    m = re.search(r"target_object_region.*?\(\s*([-0-9.]+)\s+([-0-9.]+)\s+([-0-9.]+)\s+([-0-9.]+)\)",
                  raw, re.S)
    x0, y0, x1, y1 = (float(g) for g in m.groups())
    return round((x0 + x1) / 2, 3), round((y0 + y1) / 2, 3)


def parse_base(raw):
    obj_pairs = []
    in_objects = False
    for ln in raw.split("\n"):
        s = ln.strip()
        if s.startswith("(:objects"):
            in_objects = True
            continue
        if in_objects:
            if s.startswith("(:") or s == ")":
                break
            m = re.match(r"(\S+) - (\S+)$", s)
            if m:
                obj_pairs.append((m.group(1), m.group(2)))
    stock_target = obj_pairs[0][1]
    distractor_cats = [t for inst, t in obj_pairs[1:] if t != "basket"]
    return stock_target, distractor_cats


def check_task(task, bddl_dir, stock_dir, base_cache, errors):
    name = task["name"]
    key = task["target_key"]
    cat = task["target_category"]
    base_fname = task["base_bddl"]
    path = os.path.join(bddl_dir, name + ".bddl")
    err = lambda msg: errors.append(f"{name}: {msg}")  # noqa: E731

    if not os.path.isfile(path):
        err("emitted BDDL missing")
        return
    emitted = open(path).read()

    # base validity (#4)
    if base_fname not in BASE_SLOT:
        err(f"base_bddl {base_fname!r} not in {{milk, bbq_sauce}}")
        return
    if task["layout_variant"] != BASE_LAYOUT[base_fname]:
        err(f"layout_variant {task['layout_variant']} != base layout {BASE_LAYOUT[base_fname]}")
    if base_fname not in base_cache:
        base_cache[base_fname] = open(os.path.join(stock_dir, base_fname)).read()
    base = base_cache[base_fname]
    stock_target, dcats = parse_base(base)

    # #1 minimal diff: same line count, only the 5 target lines differ, each an allowed transform
    b_lines, e_lines = base.split("\n"), emitted.split("\n")
    if len(b_lines) != len(e_lines):
        err(f"line count {len(e_lines)} != base {len(b_lines)}")
        return
    allowed = {
        f"{stock_target}_1 - {stock_target}": f"{key}_1 - {key}",
        f"{stock_target}_1": f"{key}_1",
        f"(On {stock_target}_1 floor_target_object_region)": f"(On {key}_1 floor_target_object_region)",
        f"(And (In {stock_target}_1 basket_1_contain_region))":
            f"(And (In {key}_1 basket_1_contain_region))",
    }
    changed = []
    for bl, el in zip(b_lines, e_lines):
        if bl == el:
            continue
        bs, es = bl.strip(), el.strip()
        if bs.startswith("(:language") and es == f"(:language pick up the {cat.replace('_', ' ')} and place it in the basket)":
            changed.append("language")
        elif bs in allowed and es == allowed[bs]:
            changed.append(bs)
        else:
            err(f"disallowed change:\n    base: {bl!r}\n    got:  {el!r}")
    if len(changed) != 5:
        err(f"expected exactly 5 changed lines, got {len(changed)}: {changed}")

    # #3 regions untouched (explicit, byte-for-byte)
    if region_block(emitted) != region_block(base):
        err("(:regions ...) block differs from base")

    # #2 distractors untouched (explicit): every non-target :objects/:init line equals the base's
    for bl, el in zip(b_lines, e_lines):
        s = bl.strip()
        if re.match(r"\S+_1 - \S+$", s) and not s.startswith(f"{stock_target}_1 "):
            if bl != el:
                err(f"distractor objects line changed: {bl!r} -> {el!r}")
        if s.startswith("(On ") and "other_object_region_" in s:
            if bl != el:
                err(f"distractor init line changed: {bl!r} -> {el!r}")
        if s.startswith("(On basket_1 ") and bl != el:
            err("basket init line changed")

    # #4 target slot coords match the base layout
    if target_region_center(emitted) != BASE_SLOT[base_fname]:
        err(f"target_object_region center {target_region_center(emitted)} != "
            f"{BASE_SLOT[base_fname]} for {base_fname}")

    # #5 no category collision
    if cat in dcats:
        err(f"target category {cat!r} collides with base distractors {dcats}")

    # #6 target consistency
    ooi = re.search(r"\(:obj_of_interest\s+(\S+)\s+(\S+)\s*\)", emitted)
    if not ooi or ooi.group(1) != f"{key}_1" or ooi.group(2) != "basket_1":
        err(f"obj_of_interest != ({key}_1 basket_1)")
    if f"{key}_1 - {key}" not in objects_block(emitted):
        err(f"objects block missing {key}_1 - {key}")
    if f"(In {key}_1 basket_1_contain_region)" not in emitted:
        err("goal does not reference the target")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--stock-dir", default=STOCK_DIR_DEFAULT)
    ap.add_argument("--targets-file", default=os.path.join(REPO, "full_pool_targets.txt"))
    args = ap.parse_args()

    manifest = json.load(open(args.manifest))
    bddl_dir = os.path.dirname(os.path.abspath(args.manifest))
    tasks = manifest["tasks"]
    errors = []
    base_cache = {}

    for task in tasks:
        check_task(task, bddl_dir, args.stock_dir, base_cache, errors)

    # #7 coverage / bijection
    targets = [ln.strip() for ln in open(args.targets_file) if ln.strip()]
    man_keys = [t["target_key"] for t in tasks]
    if set(man_keys) != set(targets):
        miss = set(targets) - set(man_keys)
        extra = set(man_keys) - set(targets)
        errors.append(f"coverage mismatch: {len(miss)} missing, {len(extra)} extra "
                      f"(e.g. missing {sorted(miss)[:3]}, extra {sorted(extra)[:3]})")
    if len(man_keys) != len(set(man_keys)):
        errors.append("duplicate target_key in manifest")
    on_disk = {f[:-5] for f in os.listdir(bddl_dir) if f.endswith(".bddl")}
    man_names = {t["name"] for t in tasks}
    if on_disk != man_names:
        errors.append(f"file/manifest name mismatch: "
                      f"{len(on_disk - man_names)} orphan .bddl, {len(man_names - on_disk)} missing")

    n = len(tasks)
    if errors:
        print(f"[FAIL] {len(errors)} violation(s) across {n} tasks:")
        for e in errors[:50]:
            print("  -", e)
        if len(errors) > 50:
            print(f"  ... and {len(errors) - 50} more")
        sys.exit(1)
    n_a = sum(t["layout_variant"] == 0 for t in tasks)
    print(f"[PASS] {n} tasks: all constraints hold "
          f"(A={n_a}, B={n - n_a}; each = 1 of 2 stock files, target-only swap).")


if __name__ == "__main__":
    main()
