# `libero_object_unseen_full` — the extended (ungated) unseen-object suite

**Built:** 2026-07-04 · **1,557 generated tasks, 1,421 feasible, 110 target categories** · task design identical to stock LIBERO-Object (floor scene, basket, 6 slots, `pick up the <category> and place it in the basket`, goal `(In <target> basket_contain_region)`) — only object identities and language differ. **Verified for all 1,557 tasks** (see "Layout alignment" below).

## Why this suite exists

The original 92-task `libero_object_unseen` was shaped by two budget-era filters that were removed here:

1. **Scripted-demo gate (dropped).** A key was previously discarded if an unstable scripted policy failed to bank 50 demos in 100 attempts — a *policy* judgment that eliminated 176/384 keys and 19 whole categories (mostly long/thin and wide shapes). This suite replaces it with **policy-independent physics checks only**.
2. **3-instances-per-category budget (dropped).** The cap existed solely to bound scripted-collection cost (undocumented; the experiment plan actually asked for ≥5). With no demo collection, every eligible instance is included.

Purpose: expose **every unseen object** so train/eval splits can be chosen from the full picture, and so per-instance failure analysis isn't censored by survivor bias. Expected consequence, already visible: π0.5 scores ~29% here vs 46.6% on the gated 92-task suite — the old gate was hiding much of the failure surface.

## What's in the pool

| slice | count | notes |
|---|---|---|
| strict-pool instances (unseen categories) | 1,452 | every `graspable && !excluded` instance of the 135-strict categories; 96 categories have instances |
| LIBERO-overlap instances | 105 | on-disk keys of 14 categories semantically present in LIBERO (mug, ketchup, milk, bowl, pan, canned_food, wine, corn, cup, coffee_cup, condiment_bottle, cherry, boxed_food, can) — tagged `libero_overlap: true` so they can be filtered |
| **generated tasks** | **1,557** | one task per instance; distractor pool = the full set (5 distractors/task, deterministic per-target draw, never the target's own category) |
| physics-excluded | 136 | tagged in manifest, BDDLs kept (see below) |
| **feasible / evaluated** | **1,421** | all 96 importable strict categories + 14 overlap categories retain ≥1 task — **no category fully lost** |

- Sources: 853 aigen / 568 objaverse (feasible tasks).
- Prior-gate provenance of the 1,421 feasible tasks: 1,089 never attempted under the old pipeline, 199 old gate survivors, 131 previously gate-dropped (now included), 2 previously lost to a loader crash (now fixed).
- Instances per category range 1–32 (median ~12; largest: potato 32, cupcake 31, lemon 30, egg 28, bottled_water 27).
- 39 strict categories have **zero eligible instances** in `robocasa_object_meta.json` (nothing to import): lemon_wedge, zucchini, radish, artichoke, tofu, pork_loin, pork_chop, sausage, salami, chicken_drumstick, scone, hotdog_bun, hamburger, tacos, sugar_cube, cookie_dough_ball, marshmallow, cinnamon, paprika, turmeric, straw, oil_and_vinegar_bottle, vinegar, salsa, mustard, saucepan_with_lid, saucepan_lid, jar, measuring_cup, salt_and_pepper_shaker, blender_jug, aluminum_foil, ice_cube, skewers, kebab_skewer, wooden_spoon, reamer, peeler, dish_brush.
- Excluded from targets by construction: `rc_tray` (receptacle) and 6 bare pilot keys (`apple`, `bell_pepper`, `lemon`, `lime`, `orange`, `rc_tray` — unknown instance provenance, categories fully covered by properly-keyed imports).

## The physics gate (the only gate)

Every task ran the policy-independent smoke (`scripts/smoke_check_suite.py`): env builds, placement samples (5 seeded resets), goal false at reset, goal fires with target teleported to the basket contain-region center, target dropped from 3 cm settles and **stays** in the basket, init-state round-trip. Plus 50 baked init states per task (`create_suite_init_states.py`, seed 0).

**136 exclusions, fully accounted for:**
- 133 × *"target did not stay in basket after settling"* — objects that physically cannot rest inside the 0.061 m contain box: tall bottles/dispensers (spray 12, soap_dispenser 11, thermos 11, olive_oil_bottle 8, pitcher 5, beer 3) and wide/rigid shapes (bowl 7, fish 7, rolling_pin 7, ladle 6, baguette 5, pan 5, teapot 5, …)
- 2 × placement sampling never converged (`bowl__aigen_5`, `banana__objaverse_10`)
- 1 × init-state generation failure (`kettle_non_electric__objaverse_kettle_21`)

Excluded tasks keep their BDDLs and stay in the manifest with `excluded: true` + `exclusion_reason` (regenerating would flip layout variants of later tasks); eval clients run only `feasible_task_ids.txt`. Full detail: `exclusion_report.json`.

## Layout alignment with stock LIBERO-Object (verified 2026-07-04)

Every BDDL in the suite was parsed with the same parser the env uses (`bddl_utils.robosuite_parse_problem`) and its scene structure compared against the two stock `libero_object` layouts, with object identities masked out. Compared per task: floor workspace + fixtures, all six slot regions (centroids and half-extents), the bin region, the basket contain region, yaw rotations, the goal-predicate form `(In TARGET basket_1_contain_region)`, and the init-state structure (target on `target_object_region`, five distractors on `other_object_region_0..4`, basket on the bin region).

**Result: all 1,557 tasks match a stock layout exactly — 779 match "target at slot A", 778 match "target at slot B" (the same two-layout alternation stock uses), 0 mismatches.** Only the object identities (and hence the language string) differ from stock LIBERO-Object.

Two parser-level artifacts surfaced during verification; both are non-differences:

1. **Yaw serialization.** Stock BDDLs omit the yaw block, so the parser defaults to integer `(0, 0)`; our generator writes it explicitly and it parses to `(0.0, 0.0)`. Numerically identical — a naive string/JSON diff flags it, a value comparison does not.
2. **Object-internal marker sites.** Some converted RoboCasa models (e.g. `jug_wide_opening`) carry inert internal sites (`*_liquid`, `*_reg_int`) inside their own MJCF. The parser lists them alongside scene regions, but they are attached to the object model, are referenced by no init/goal predicate, and do not affect placement or goals — object mesh detail, not scene layout.

## Evaluation protocol (running)

- Models: released **π0.5** (`pi05_libero`) and official **π0-FAST** (`pi0_fast_libero`), openpi server/client harness, unmodified inference settings.
- 10 trials/task, seed 7, fixed baked init states (indices 0–9), max 280 steps + 10 settle steps, **all episode videos saved** (successes included).
- Control gate passed before each lane: 2/2 on stock `libero_object` task 0 for both models.
- **π0.5 FINAL (2026-07-07): 29.1% (4,129/14,210 episodes over all 1,421 feasible tasks)** — vs 46.6% on the old gated 92-task suite. Aggregated at `/tmp2/leocheng/eval_results/full/summary_pi05_final/`. Best categories: condiment_bottle/orange/shaker/can/mug (60–80%); worst: cereal (23 tasks), rolling_pin, thermos, bowl at 0%, cheese_grater/canola_oil/tongs/spray ≤2.5%.
- **π0-FAST FINAL (2026-07-08): 15.9% (2,256/14,210 episodes over all 1,421 feasible tasks)** — vs ~19% on the old gated suite. 20/110 categories at exactly 0% (vs 4/110 for π0.5); best category (saucepan 50%) is below π0.5's tenth-best. Combined 2-model aggregation: `/tmp2/leocheng/eval_results/full/summary_final/`.

## Artifacts

| path | what |
|---|---|
| `libero/libero/bddl_files/libero_object_unseen_full/` | 1,557 BDDLs + `manifest.json` (per-task tags: category, `libero_overlap`, `prior_status`, exclusions) + `exclusion_report.json` + `feasible_task_ids.txt` + smoke shard reports |
| `libero/libero/init_files/libero_object_unseen_full/` | 50-state `.pruned_init` per feasible task + init-failure shard logs |
| `full_pool_manifest.json` / `full_pool_meta.json` / `full_pool_targets.txt` | pool selection (all eligible strict instances) + per-key tags |
| `import_report.json` | conversion report (1,171 new keys, 0 quarantined on final pass) |
| `coverage_report.json` | audit: all 135 strict categories → evaluated (96) / no-eligible-instances (39); per-key states |
| `/tmp2/leocheng/eval_results/full/` | per-model results JSONs + `videos_pi05/`, `videos_pi0fast/` (every episode) |
| suite registration | `libero_object_unseen_full` in `libero/libero/benchmark/__init__.py` (n_tasks = 1,557; eval uses feasible ids) |

## Reproduction

```bash
# 1. pool manifest (all eligible strict instances + tags)
python scripts/make_full_pool_manifest.py
# 2. convert (MUST run under a python with robosuite installed — texture search path)
<venv>/python tools/convert_robocasa_xml.py --manifest full_pool_manifest.json \
    --max-xy-half 0.075 --report import_report.json
# 3. suite
<venv>/python scripts/create_libero_object_unseen_tasks.py \
    --targets-file full_pool_targets.txt --key-meta full_pool_meta.json \
    --suite-name libero_object_unseen_full --seed 0 \
    --out-dir libero/libero/bddl_files/libero_object_unseen_full
# 4. init states + smoke (12 shards) + exclusion tagging
bash /tmp2/leocheng/eval_results/run_full_suite_prep.sh
# 5. coverage audit
python scripts/full_pool_coverage.py
```

Caveats: 10 trials/task → ±10 pp resolution per task; category scores pool 1–32 instances, so small categories carry wide error bars; the physics gate guarantees *placeability*, not graspability — a 0% task may still be a legitimately hard-but-possible grasp (that's the point of removing the scripted gate).
