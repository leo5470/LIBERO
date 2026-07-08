# `libero_object_unseen_full` — ungated unseen-object suite for LIBERO

**1,557 tasks · 1,421 physically feasible · 110 target categories · task design identical to stock LIBERO-Object.** One task per object instance: *"pick up the `<category>` and place it in the basket"*, goal `(In <target> basket_1_contain_region)`. Only the object identities (and hence the language) differ from stock LIBERO-Object — every task was verified to match one of the two stock layout variants exactly (779 A / 778 B, 0 mismatches).

This is the ungated successor to the 92-task `libero_object_unseen`: it drops the scripted-demo gate (a *policy* judgment that censored 176/384 keys) and the 3-instances-per-category budget, so **every eligible unseen RoboCasa instance** is represented. The only filter is policy-independent physics (below). Full design rationale, verification detail, and reproduction commands: [`extended.md`](../../../../extended.md) at the repo root.

## Pool composition

| slice | count |
|---|---|
| strict-unseen instances (96 of 135 strict categories have eligible instances) | 1,452 |
| LIBERO-overlap instances (14 categories, tagged `libero_overlap: true`) | 105 |
| **generated tasks** | **1,557** |
| physics-excluded (tagged, BDDLs kept) | 136 |
| **feasible / evaluated** | **1,421** |

Sources: 853 aigen / 568 objaverse (feasible). Prior-pipeline provenance per task in the manifest (`prior_status`): 1,089 never attempted, 199 old gate survivors, 131 previously gate-dropped, 2 previously lost to a loader crash. Instances per category range 1–32. 39 strict categories have zero eligible instances in `robocasa_object_meta.json` (listed in `extended.md`).

## The physics gate (the only gate)

Every task passed `scripts/smoke_check_suite.py`: env builds, placement samples (5 seeded resets), goal false at reset, goal fires with the target teleported into the basket contain region, target dropped from 3 cm settles and **stays** in the basket, init-state round-trip. 136 exclusions (133 settle — tall/wide objects vs the 0.061 m contain box; 2 placement; 1 init-state), each recorded in `exclusion_report.json` and tagged `excluded: true` + `exclusion_reason` in `manifest.json`. Excluded BDDLs are kept — regenerating the suite would flip layout variants of later tasks and stale the baked init states.

## Files in this directory

| file | what |
|---|---|
| `*.bddl` (1,557) | task definitions, stock LIBERO-Object layout |
| `manifest.json` | single source of truth: per-task `target_key`, `category`, `source`, `libero_overlap`, `prior_status`, `excluded`/`exclusion_reason` |
| `feasible_task_ids.txt` | the 1,421 task ids to evaluate (comma-separated) |
| `exclusion_report.json` | per-exclusion detail |
| `smoke_report_shard*.json` | raw physics-check verdicts (12 shards) |

Baked init states (50 per feasible task, seed 0): `libero/libero/init_files/libero_object_unseen_full/*.pruned_init`.

## Usage

```python
from libero.libero.benchmark import get_benchmark_dict
suite = get_benchmark_dict()["libero_object_unseen_full"]()  # n_tasks = 1557
ids = open(".../feasible_task_ids.txt").read().split(",")     # evaluate these only
```

- **Assets required**: the converted RoboCasa objects (`libero/libero/assets/robocasa_objects/`, ~9 G) are gitignored. Regenerate with `tools/convert_robocasa_xml.py --manifest full_pool_manifest.json --max-xy-half 0.075` — it must run under a Python with robosuite installed (texture search paths).
- **Protocol used for baselines**: 10 trials/task, seed 7, init-state indices 0–9, max 280 steps + 10 settle steps.
- **Aggregation**: `scripts/aggregate_unseen_eval.py --results <results globs> --manifest manifest.json --out-dir <dir>` merges partial per-task results JSONs into per-category CSV, failure index, and per-instance summary.

## Baseline results (all 1,421 feasible tasks, 10 trials each)

| model | this suite | gated 92-task suite |
|---|---|---|
| π0.5 (`pi05_libero`) | **29.1%** | 46.6% |
| π0-FAST (`pi0_fast_libero`) | **15.9%** | ~19% |

The gap versus the gated suite is the point: the old scripted-policy gate was hiding roughly a third of the failure surface. π0-FAST zeroes out on 20/110 categories (π0.5: 4/110). Per-instance variance within a category is large (adjacent instances differing by >50 pp), so category scores with few instances carry wide error bars.

## Caveats

- 10 trials/task → ±10 pp resolution per task.
- The physics gate guarantees *placeability*, not graspability — a 0% task may still be a legitimately hard-but-possible grasp.
- `libero_overlap` categories are semantically present in LIBERO training data; filter on the manifest tag if you need strictly-unseen-only numbers.
