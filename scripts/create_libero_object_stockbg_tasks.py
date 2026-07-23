"""Generate ``libero_object_unseen_stockbg``: stock LIBERO-Object scenes, target-only swap.

Each task is exactly one of **two** stock BDDL files with **only the target object's 5 lines** rewritten
to a RoboCasa object; the 5 distractors, the basket, every region, and the layout stay the byte-identical
stock bytes:

  * layout A -> ``pick_up_the_milk_and_place_it_in_the_basket.bddl``      (target at slot A)
  * layout B -> ``pick_up_the_bbq_sauce_and_place_it_in_the_basket.bddl`` (target at slot B)

There is **no scene randomizer** here: distractors/regions are copied verbatim, never selected by
``pick_distractors`` nor rebuilt by ``InitialSceneTemplates`` / ``generate_bddl_from_task_info``. This
module imports nothing from ``task_generation_utils`` / ``bddl_generation_utils`` / ``mu_utils`` (pure
text). ``scripts/verify_stockbg_suite.py`` re-checks every constraint over all emitted files.

Targets come from ``full_pool_targets.txt`` (all eligible RoboCasa instances, seen/overlap categories
included); per-key ``libero_overlap`` / ``prior_status`` tags from ``full_pool_meta.json``.

Layout is ``index % 2`` over the target list (mirrors stock LIBERO-Object's own two-layout design).
Collision fallback: the only RoboCasa target categories that also appear as a stock distractor are
``ketchup``/``milk``; the milk (A) base contains neither, the bbq_sauce (B) base contains ``ketchup`` --
so a target whose category is among its base's distractors is reassigned to the A base (currently only
the ketchup targets that land in B). Keeps every "pick up the <category>" unambiguous.

Example:
    python scripts/create_libero_object_stockbg_tasks.py \
        --targets-file full_pool_targets.txt --key-meta full_pool_meta.json \
        --out-dir libero/libero/bddl_files/libero_object_unseen_stockbg
"""

import argparse
import json
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STOCK_DIR_DEFAULT = os.path.join(REPO, "libero/libero/bddl_files/libero_object")
# (layout_variant, stock base filename) -- layout 0 = slot A, layout 1 = slot B.
BASE_FILES = {
    0: "pick_up_the_milk_and_place_it_in_the_basket.bddl",
    1: "pick_up_the_bbq_sauce_and_place_it_in_the_basket.bddl",
}


def category_of(key: str) -> str:
    """`apple__objaverse_15` -> `apple`; bare keys are their own category."""
    return key.split("__", 1)[0]


def parse_base(raw: str):
    """Return (stock_target_type, [distractor_categories]) from a stock libero_object BDDL."""
    obj_pairs = []  # (instance, type) in file order, from the (:objects ...) block
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
    ooi = re.search(r"\(:obj_of_interest\s+(\S+)", raw)
    assert ooi, "no (:obj_of_interest ...) in base"
    target_inst = ooi.group(1)
    assert obj_pairs and obj_pairs[0][0] == target_inst, (
        f"target {target_inst} is not the first (:objects) entry {obj_pairs[:1]}")
    stock_target = obj_pairs[0][1]
    distractor_cats = [t for inst, t in obj_pairs[1:] if t != "basket"]
    assert len(distractor_cats) == 5, f"expected 5 distractors, got {distractor_cats}"
    return stock_target, distractor_cats


def swap_target(raw: str, stock_target: str, key: str, category: str) -> str:
    """Byte-identical to `raw` except the 5 target lines rewritten for `key`.

    Uses `str.split("\\n")` (not splitlines) so a trailing newline round-trips exactly.
    """
    lines = raw.split("\n")
    hits = {"language": 0, "objects": 0, "ooi": 0, "init": 0, "goal": 0}
    out = []
    for ln in lines:
        s = ln.strip()
        indent = ln[: len(ln) - len(ln.lstrip())]
        if s.startswith("(:language"):
            out.append(f"{indent}(:language pick up the {category} and place it in the basket)")
            hits["language"] += 1
        elif s == f"{stock_target}_1 - {stock_target}":
            out.append(f"{indent}{key}_1 - {key}")
            hits["objects"] += 1
        elif s == f"{stock_target}_1":  # obj_of_interest target token
            out.append(f"{indent}{key}_1")
            hits["ooi"] += 1
        elif s == f"(On {stock_target}_1 floor_target_object_region)":
            out.append(f"{indent}(On {key}_1 floor_target_object_region)")
            hits["init"] += 1
        elif s == f"(And (In {stock_target}_1 basket_1_contain_region))":
            out.append(f"{indent}(And (In {key}_1 basket_1_contain_region))")
            hits["goal"] += 1
        else:
            out.append(ln)
    assert all(v == 1 for v in hits.values()), f"target lines matched {hits} for {key}"
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets-file", default=os.path.join(REPO, "full_pool_targets.txt"),
                    help="one RoboCasa registry key per line (all eligible instances)")
    ap.add_argument("--key-meta", default=os.path.join(REPO, "full_pool_meta.json"),
                    help="full_pool_meta.json: per-key libero_overlap / prior_status tags")
    ap.add_argument("--stock-dir", default=STOCK_DIR_DEFAULT,
                    help="folder holding the stock libero_object BDDLs")
    ap.add_argument("--suite-name", default="libero_object_unseen_stockbg")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    with open(args.targets_file) as f:
        targets = [ln.strip() for ln in f if ln.strip()]
    assert targets, "no targets"

    key_meta = {}
    if args.key_meta and os.path.isfile(args.key_meta):
        key_meta = json.load(open(args.key_meta)).get("keys", {})

    # Load + parse the two stock bases once.
    bases = {}
    for layout, fname in BASE_FILES.items():
        raw = open(os.path.join(args.stock_dir, fname)).read()
        stock_target, dcats = parse_base(raw)
        bases[layout] = {
            "template": stock_target, "distractor_cats": dcats,
            "raw": raw, "fname": fname,
        }

    os.makedirs(args.out_dir, exist_ok=True)
    tasks = []
    for i, key in enumerate(targets):
        cat = category_of(key)
        layout = i % 2
        if cat in bases[layout]["distractor_cats"]:  # collision -> A base (universal safe harbor)
            layout = 0
            assert cat not in bases[0]["distractor_cats"], (
                f"[error] target {key} (cat {cat}) collides with BOTH stock bases; "
                "a 2-file swap can't place it -- needs a new base file")
        base = bases[layout]
        cat_lang = cat.replace("_", " ")
        language = f"pick up the {cat_lang} and place it in the basket"
        bddl = swap_target(base["raw"], base["template"], key, cat_lang)

        name = f"pick_up_the_{key}_and_place_it_in_the_basket"
        with open(os.path.join(args.out_dir, name + ".bddl"), "w") as f:
            f.write(bddl)

        task = {
            "name": name,
            "language": language,
            "target_key": key,
            "target_category": cat,
            "layout_variant": layout,
            "base_template": base["template"],
            "base_bddl": base["fname"],
            "distractors": list(base["distractor_cats"]),
        }
        if key in key_meta:
            task["libero_overlap"] = key_meta[key].get("libero_overlap")
            task["prior_status"] = key_meta[key].get("prior_status")
        tasks.append(task)

    manifest = {
        "suite": args.suite_name,
        "source_targets_file": os.path.abspath(args.targets_file),
        "source_key_meta": os.path.abspath(args.key_meta) if key_meta else None,
        "base_bddls": {str(k): v["fname"] for k, v in bases.items()},
        "num_tasks": len(tasks),
        "tasks": tasks,
    }
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    n_a = sum(t["layout_variant"] == 0 for t in tasks)
    n_bump = sum(1 for i, t in enumerate(tasks)
                 if t["layout_variant"] == 0 and i % 2 == 1)
    print(f"[generated] {len(tasks)} tasks in {args.out_dir}  "
          f"(A={n_a}, B={len(tasks) - n_a}; {n_bump} collision-bumped to A)")


if __name__ == "__main__":
    main()
