# `libero_object_unseen_stockbg` — stock distractors, target-only swap

**1,557 tasks · 1,417 physically feasible · 110 target categories.** Each task is a **stock
LIBERO-Object scene with only the target object swapped** to a RoboCasa instance — the five distractors
and the basket stay the familiar stock objects at their stock positions. This isolates *target novelty*
as the single variable, in contrast to `libero_object_unseen_full`, which replaces the distractors too
(and varies them per task).

## Design

- **Targets**: every eligible RoboCasa instance from `full_pool_targets.txt` (all 110 categories,
  seen/LIBERO-overlap categories included) — one task per instance.
- **Exactly two base BDDL files**, copied verbatim with only the target's 5 lines rewritten
  (`:language`, the target `:objects` line, `:obj_of_interest`, the target `:init` `On`, the `:goal`
  `In`); regions, distractors, basket, and layout are byte-identical to stock:
  - layout A → stock `pick_up_the_milk_and_place_it_in_the_basket.bddl`
  - layout B → stock `pick_up_the_bbq_sauce_and_place_it_in_the_basket.bddl`
- **No scene randomizer**: distractors/regions are fixed stock bytes — never drawn by `pick_distractors`
  nor rebuilt by `InitialSceneTemplates`. Layout alternates `index % 2` (mirrors stock's two layouts).
- **Collision fallback**: `ketchup`/`milk` are the only RoboCasa target categories that also appear as a
  stock distractor; the milk (A) base contains neither, so the 4 ketchup targets that land in B are
  reassigned to the A base — keeping every "pick up the *category*" unambiguous.

## Feasibility

Every task ran the policy-independent physics smoke check (`scripts/smoke_check_suite.py`): env builds,
placement samples, goal false at reset, target teleported to the basket fires and *stays* successful
after settling, init-state round-trip. **140 tasks excluded** (all "target did not stay in the basket
after settling" — target+basket-only, so essentially the same set the full suite excludes). Eval clients
run only `feasible_task_ids.txt` (1,417 ids). Excluded tasks keep their BDDLs and are tagged in
`manifest.json` (`excluded: true` + `exclusion_reason`); full detail in `exclusion_report.json`.

## Regenerate

```bash
python scripts/create_libero_object_stockbg_tasks.py \
    --targets-file full_pool_targets.txt --key-meta full_pool_meta.json \
    --out-dir libero/libero/bddl_files/libero_object_unseen_stockbg
python scripts/verify_stockbg_suite.py \
    --manifest libero/libero/bddl_files/libero_object_unseen_stockbg/manifest.json
# then (libero env): create_suite_init_states.py -> smoke_check_suite.py -> tag_suite_exclusions.py
```

`scripts/verify_stockbg_suite.py` re-proves, over all 1,557 files, that each is one of the two stock
files with only the target changed (minimal-diff, distractors/regions untouched, no collision, coverage).

Registered as `LIBERO_OBJECT_UNSEEN_STOCKBG` in `libero/libero/benchmark/__init__.py`.
