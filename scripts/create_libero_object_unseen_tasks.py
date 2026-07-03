"""Regenerate the LIBERO-Object suite with unseen RoboCasa objects (same goal).

Mirrors the stock ``libero_object`` design exactly — floor workspace, basket at the bin
slot, target at ``target_object_region``, 5 distractors on the remaining fixed slots,
goal ``(In <target>_1 basket_1_contain_region)`` — but draws targets/distractors from the
converted RoboCasa object pool (LIBERO registry keys like ``apple__objaverse_15``).

Pool-agnostic: pass explicit ``--targets``/``--distractor-pool`` keys, or point
``--tier-split`` at a scale-pipeline ``tier_split.json`` and select tiers. Distractors are
chosen per target with a deterministic per-key RNG (stable when the pool later grows),
never share the target's category (keeps "pick up the apple" unambiguous in-scene), and
round-robin over categories for diversity.

Outputs ``<out-dir>/pick_up_the_<key>_and_place_it_in_the_basket.bddl`` per target plus
``manifest.json`` (the single source of truth the benchmark registration reads).

Example (pilot pool):
    python scripts/create_libero_object_unseen_tasks.py \
        --tier-split /tmp2/leocheng/ricl_scratch/scale_pilot/tier_split.json \
        --target-tiers seen tier_a tier_b --exclude-target-categories canned_food \
        --distractor-tiers seen tier_a tier_b \
        --seed 0 --out-dir libero/libero/bddl_files/libero_object_unseen
"""

import argparse
import json
import os
import random
import re

import init_path  # noqa: F401
from libero.libero.envs.objects import get_object_dict
from libero.libero.utils.bddl_generation_utils import (
    get_xy_region_kwargs_list_from_regions_info,
)
from libero.libero.utils.mu_utils import register_mu, InitialSceneTemplates
from libero.libero.utils import task_generation_utils
from libero.libero.utils.task_generation_utils import (
    register_task_info,
    generate_bddl_from_task_info,
)
import libero.libero.envs.bddl_utils as BDDLUtils

# Stock libero_object layout (verified against the shipped BDDLs): 6 object slots of
# half-len 0.025 plus the basket's bin slot of half-len 0.01, all on the floor workspace.
SLOTS = {
    "A": (-0.12, -0.24),
    "B": (0.05, -0.10),
    "C": (-0.15, 0.06),
    "D": (0.10, -0.20),
    "E": (0.15, 0.03),
    "F": (-0.20, -0.08),
}
SLOT_HALF_LEN = 0.025
BIN_XY, BIN_HALF_LEN = (0.0, 0.26), 0.01
# Stock suite alternates two layouts: target at A (others B..F) or at B (others A,C..F).
VARIANTS = [
    ("A", ["B", "C", "D", "E", "F"]),
    ("B", ["A", "C", "D", "E", "F"]),
]

RECEPTACLE = "basket"
TIERS = ("seen", "tier_a", "tier_b")


def _mu_key(class_name: str) -> str:
    """Reproduce mu_utils.register_mu's class-name -> registry-key mangling."""
    return "_".join(re.sub(r"([A-Z])", r" \1", class_name).split()).lower()


def _camel(key: str) -> str:
    return "".join(p.capitalize() for p in key.split("_") if p)


def category_of(key: str) -> str:
    """`apple__objaverse_15` -> `apple`; bare keys (`apple`) are their own category."""
    return key.split("__", 1)[0]


def make_scene(target: str, distractors: list, variant_idx: int) -> str:
    """Define + register one libero_object-style floor scene; return its registry key."""
    cls_name = "LiberoObjectUnseen" + _camel(target)
    target_slot, other_slots = VARIANTS[variant_idx]

    class _Scene(InitialSceneTemplates):
        def __init__(self):
            object_num_info = {target: 1}
            object_num_info.update({d: 1 for d in distractors})
            object_num_info[RECEPTACLE] = 1
            super().__init__(
                workspace_name="floor",
                fixture_num_info={"floor": 1},
                object_num_info=object_num_info,
            )

        def define_regions(self):
            self.regions.update(self.get_region_dict(
                region_centroid_xy=list(BIN_XY), region_name="bin_region",
                target_name=self.workspace_name, region_half_len=BIN_HALF_LEN))
            self.regions.update(self.get_region_dict(
                region_centroid_xy=list(SLOTS[target_slot]),
                region_name="target_object_region",
                target_name=self.workspace_name, region_half_len=SLOT_HALF_LEN))
            for i, slot in enumerate(other_slots):
                self.regions.update(self.get_region_dict(
                    region_centroid_xy=list(SLOTS[slot]),
                    region_name=f"other_object_region_{i}",
                    target_name=self.workspace_name, region_half_len=SLOT_HALF_LEN))
            self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(self.regions)

        @property
        def init_states(self):
            states = [("On", f"{target}_1", "floor_target_object_region")]
            states += [("On", f"{d}_1", f"floor_other_object_region_{i}")
                       for i, d in enumerate(distractors)]
            states.append(("On", f"{RECEPTACLE}_1", "floor_bin_region"))
            return states

    _Scene.__name__ = cls_name
    _Scene.__qualname__ = cls_name
    register_mu(scene_type="floor")(_Scene)
    return _mu_key(cls_name)


def pick_distractors(target: str, pool: dict, seed: int, k: int = 5) -> list:
    """Deterministic per-target choice of k distractors from pool {key: category}.

    Excludes the target's category entirely, then round-robins over the remaining
    categories (one instance each before any category repeats). Seeded by
    (seed, target key) so a task's distractors survive pool growth.
    """
    rng = random.Random(f"{seed}:{target}")
    by_cat = {}
    for key in sorted(pool):
        cat = pool[key]
        if cat != category_of(target):
            by_cat.setdefault(cat, []).append(key)
    cats = sorted(by_cat)
    rng.shuffle(cats)
    for c in cats:
        rng.shuffle(by_cat[c])

    picks = []
    while len(picks) < k:
        progressed = False
        for c in cats:
            if len(picks) >= k:
                break
            if by_cat[c]:
                picks.append(by_cat[c].pop(0))
                progressed = True
        if not progressed:
            raise SystemExit(
                f"[error] distractor pool too small for target {target}: "
                f"need {k}, got {len(picks)} (pool excludes category "
                f"'{category_of(target)}')")
    return picks


def load_tier_split(path: str):
    """-> ({key: tier}, {key: entry}) from a scale-pipeline tier_split.json."""
    with open(path) as f:
        split = json.load(f)
    tier_of, entries = {}, {}
    for tier in TIERS:
        for entry in split.get(tier, []):
            tier_of[entry["key"]] = tier
            entries[entry["key"]] = entry
    return tier_of, entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier-split", help="scale-pipeline tier_split.json (key metadata source)")
    ap.add_argument("--target-tiers", nargs="+", choices=TIERS,
                    help="tiers whose keys become targets (with --tier-split)")
    ap.add_argument("--distractor-tiers", nargs="+", choices=TIERS,
                    help="tiers whose keys form the distractor pool (with --tier-split)")
    ap.add_argument("--exclude-target-categories", nargs="*", default=[],
                    help="categories to drop from targets (still distractor-eligible)")
    ap.add_argument("--targets", nargs="+",
                    help="explicit target registry keys (overrides --target-tiers)")
    ap.add_argument("--distractor-pool", nargs="+",
                    help="explicit distractor-pool registry keys (overrides --distractor-tiers)")
    ap.add_argument("--max-xy-half", type=float, default=None,
                    help="drop keys whose bbox xy half-extent exceeds this (needs --tier-split)")
    ap.add_argument("--num-distractors", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--suite-name", default="libero_object_unseen")
    ap.add_argument("--out-dir", required=True,
                    help="folder for generated .bddl files + manifest.json")
    args = ap.parse_args()

    tier_of, entries = ({}, {})
    if args.tier_split:
        tier_of, entries = load_tier_split(args.tier_split)

    def gate(keys):
        if args.max_xy_half is None:
            return list(keys)
        kept = [k for k in keys
                if k not in entries
                or max(entries[k]["bbox_half"][:2]) <= args.max_xy_half]
        for k in sorted(set(keys) - set(kept)):
            print(f"[gate] dropping {k}: xy bbox half "
                  f"{max(entries[k]['bbox_half'][:2]):.3f} > {args.max_xy_half}")
        return kept

    if args.targets:
        targets = list(args.targets)
    elif args.target_tiers:
        targets = [k for k, t in tier_of.items() if t in args.target_tiers]
    else:
        raise SystemExit("[error] need --targets or (--tier-split + --target-tiers)")
    targets = [t for t in gate(targets)
               if category_of(t) not in args.exclude_target_categories]

    if args.distractor_pool:
        pool_keys = list(args.distractor_pool)
    elif args.distractor_tiers:
        pool_keys = [k for k, t in tier_of.items() if t in args.distractor_tiers]
    else:
        pool_keys = list(targets)
    pool_keys = gate(pool_keys)
    pool = {k: category_of(k) for k in pool_keys}

    targets = sorted(set(targets))
    if not targets:
        raise SystemExit("[error] no targets left after filtering")

    registry = get_object_dict()
    missing = [k for k in set(targets) | set(pool) | {RECEPTACLE} if k not in registry]
    if missing:
        raise SystemExit(f"[error] not registered in OBJECTS_DICT: {sorted(missing)}")

    # Global TASK_INFO accumulates across calls; make this invocation self-contained.
    task_generation_utils.TASK_INFO.clear()
    os.makedirs(args.out_dir, exist_ok=True)

    tasks = []
    for i, target in enumerate(targets):
        distractors = pick_distractors(target, pool, args.seed, args.num_distractors)
        variant_idx = i % len(VARIANTS)
        scene_key = make_scene(target, distractors, variant_idx)
        category = category_of(target)
        language = f"pick up the {category.replace('_', ' ')} and place it in the basket"
        register_task_info(
            language=language,
            scene_name=scene_key,
            objects_of_interest=[f"{target}_1", f"{RECEPTACLE}_1"],
            goal_states=[("In", f"{target}_1", f"{RECEPTACLE}_1_contain_region")],
        )
        tasks.append({
            "name": f"pick_up_the_{target}_and_place_it_in_the_basket",
            "language": language,
            "target_key": target,
            "target_category": category,
            "tier": tier_of.get(target, "unknown"),
            "distractors": distractors,
            "layout_variant": variant_idx,
            "scene_key": scene_key,
        })

    bddl_files, failures = generate_bddl_from_task_info(folder=args.out_dir)
    if failures:
        raise SystemExit(f"[error] {len(failures)} task(s) FAILED to generate: {failures}")
    if len(bddl_files) != len(tasks):
        raise SystemExit(f"[error] expected {len(tasks)} bddl files, got {len(bddl_files)}")

    # Rename emitted SCENEKEY_language.bddl files to canonical per-instance stems.
    for task in tasks:
        emitted = os.path.join(
            args.out_dir,
            task["scene_key"].upper() + "_"
            + "_".join(task["language"].lower().split(" ")) + ".bddl")
        final = os.path.join(args.out_dir, task["name"] + ".bddl")
        if not os.path.isfile(emitted):
            raise SystemExit(f"[error] expected emitted file missing: {emitted}")
        os.replace(emitted, final)

    # Syntax gate: the parser the env uses must accept what we emitted.
    probe = os.path.join(args.out_dir, tasks[0]["name"] + ".bddl")
    parsed = BDDLUtils.robosuite_parse_problem(probe)
    assert parsed["problem_name"] == "libero_floor_manipulation", parsed["problem_name"]

    manifest = {
        "suite": args.suite_name,
        "seed": args.seed,
        "num_distractors": args.num_distractors,
        "source_tier_split": os.path.abspath(args.tier_split) if args.tier_split else None,
        "tasks": tasks,
    }
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[generated] {len(tasks)} tasks in {args.out_dir}")
    for task in tasks:
        print(f"    {task['tier']:<7} {task['name']}")


if __name__ == "__main__":
    main()
