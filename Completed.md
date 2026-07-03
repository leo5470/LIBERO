# Completed — RoboCasa → LIBERO object-pool scale-up (Phase 1)

This document records the code and data changes that scale the RoboCasa→LIBERO object import
(validated on 6 objects in Phase 0) into a **curated, gated Seen / Tier-A / Tier-B object pool**
with per-object scripted demos rendered to robomimic HDF5. Scope is **LIBERO-only** (no RICL
training/serving/retrieval). It corresponds to the approved plan
`~/.claude/plans/think-about-whether-there-composed-feather.md`.

The headline outcome: downloading the RoboCasa **aigen** asset set expanded the on-disk pool from
**71 → 110 graspable categories**, which (a) removed the category-count squeeze that made the
`≥30 Seen + ≥15 Tier B` target barely feasible and (b) supplied a real **objaverse→aigen
source-shift** for the Tier-A (unseen-instance) holdout. Three real bugs surfaced and were fixed
while de-risking aigen. The whole pipeline is validated end-to-end and a 44-key pilot is running.

---

## 1. Change set at a glance

| File | Status | What changed |
|---|---|---|
| `robocasa_object_meta.json` | regenerated | Re-harvested after the aigen download: **875 → 2074 instances** (701 objaverse + 1001 aigen graspable), **71 → 110 graspable categories**. |
| `scale_pool_manifest.json` | **new (data)** | Output of `select_scale_pool.py`: the pre-registered Seen/Tier-A/Tier-B **candidate** assignment — 384 keys, 40 Seen + 25 Tier-B categories. |
| `tools/select_scale_pool.py` | **new** | Selects the pool from the meta (source-aware, group-stratified, seeded); emits the manifest. |
| `tools/convert_robocasa_xml.py` | **modified** | Added shared-texture fallback resolution, `--manifest` batch mode, auto-fit-scale, quarantine, conversion report. |
| `tools/make_tier_split.py` | **new** | Emits `tier_split.json` over gate survivors with disjointness asserts + covariates. |
| `scripts/run_scale_pipeline.py` | **new** | Orchestrator: convert → register-check → BDDL → parallel gated collect → serialized render → report. Resumable. |
| `scripts/collect_scripted_demonstrations.py` | **modified** | Added `--result-json` (machine-readable gate result). |
| `scripts/create_dataset.py` | **modified** | Added `--output-root` (redirect renders off the small `/home` disk) and `--compress` (gzip images). |
| `libero/libero/envs/objects/robocasa_objects.py` | **modified** | Register objects under their **exact dir name** (fixes per-instance key mangling). |

Unchanged but reused as-is: `tools/harvest_robocasa_objects.py`, `scripts/create_robocasa_pickplace_tasks.py`
(already parameterized `--manipulanda/--receptacle`), `libero_scripted_policy.py`,
`libero/libero/envs/objects/__init__.py` (line 9 `from .robocasa_objects import *` runs registration at import).

Nothing is committed — all changes are in the working tree.

---

## 2. The three bugs fixed (all were predicted risks in the plan)

### 2.1 Converter: aigen shared-texture absolute paths (`convert_robocasa_xml.py`)
aigen `model.xml` files reference a shared robosuite texture (e.g. `ceramic.png`) by an **absolute
path baked on the asset-generation machine** (`/home/abhiram03/.../textures/ceramic.png`), which
does not exist locally, so `_localize_assets` left a broken `file=` in the XML → MuJoCo compile
failure. Fix: when a referenced asset is missing, resolve it **by basename** against local asset
dirs (`_default_search_dirs()` auto-includes the installed robosuite `models/assets/textures`), copy
it in, and rewrite the reference. `_localize_assets` now returns the list of still-unresolved files
so `--manifest` mode can quarantine a key rather than ship a broken object.

### 2.2 Registration: per-instance key mangling (`robocasa_objects.py`)
`base_object.register_object` derives the registry key from the class name via
`re.sub(r"([A-Z0-9])", r" \1", ClassName)` — which **splits on every digit and collapses double
underscores**. Per-instance dir names do not survive the snake→camel→snake round-trip:
`apple__objaverse_10` → class `AppleObjaverse10` → registered key `apple_objaverse_1_0` (≠ dir name).
So BDDL lookups by the manifest key failed. Fix: `register_robocasa_objects()` now inserts
`OBJECTS_DICT[dir_name] = _make_class(dir_name)` directly, keeping **registry key == manifest key ==
BDDL object name**. (Phase-0 single-word keys like `apple`/`rc_tray` still register identically.)

### 2.3 Selector: aigen-only categories in Seen with zero priming (`select_scale_pool.py`)
The first cut put aigen-only categories (e.g. `burrito`, `ice_cream`, overlap `cherry`) into the Seen
slice with `prime=0` — a Seen category with no priming instances is meaningless, and an aigen-only
*novel* category belongs in Tier B. Fix: the Seen stratified slice is drawn only from categories that
**have objaverse instances** (priming needs a uniform objaverse distribution); aigen-only novel
categories flow to Tier B; overlap categories lacking objaverse (`cherry`) prime on aigen as a
documented fallback.

---

## 3. Per-instance key scheme

Phase 0 used one object per category (`apple`, `rc_tray`). Scaling to many instances/category, the
registry key is now **per instance**:

```
key = <category>__<source>_<suffix>        e.g.  bowl__objaverse_14,  ketchup__aigen_6
```

One converted object dir per key under `libero/libero/assets/robocasa_objects/<key>/`. The auto-scan
in `robocasa_objects.py` registers each. In a task the scene object is `<key>_1`, so the observation
key is **`<key>_1_pos`** (e.g. `bowl__objaverse_14_1_pos`) — not `<key>_pos`.

---

## 4. Tier design and manifest semantics

- **Seen (priming)** — 13/14 LIBERO-overlap categories (base-familiar semantics; kept as candidates
  even though most hollow/handled ones fail the grasp gate) **+** a group-stratified slice of
  objaverse-having RoboCasa-only categories. Primed on **objaverse** instances (uniform visual
  distribution).
- **Tier A (unseen instance)** — held-out **aigen** instances of the Seen categories (a sharp
  instance-novelty axis with a logged objaverse→aigen source shift). Falls back to held-out objaverse
  where a category has no aigen (logged as `objaverse_fallback`).
- **Tier B (unseen category)** — RoboCasa-only categories not in Seen, held out entirely; richest
  from the 39 **aigen-only** produce/protein/bakery categories that LIBERO's base has never seen.

Tiers are **pre-registered** in `select_scale_pool.py` (before any collection) and the candidate
slices are **over-provisioned** (40 Seen / 25 Tier-B categories vs the `≥30 / ≥15` post-gate targets),
so gate attrition never forces a post-hoc rebalance. `make_tier_split.py` assigns **only gate
survivors** to a final tier and asserts:
`Seen∩TierA == ∅` (instances) and `Seen-cats ∩ TierB-cats == ∅` (categories).

Per-key covariates carried through manifest → report: `source`, `is_overlap`, `group`, `bbox_half`,
`applied_scale`, `chosen_grasp_frac`, `gate_yield`, `n_demos`.

---

## 5. End-to-end pipeline

All simulation runs in the **`libero`** conda env; harvest/download run in the **`robocasa`** env
(robosuite 1.4 vs 1.5 conflict — see §8). Paths below are relative to the repo root.

### Step 0 — expand + harvest (robocasa env, asset-only)
```bash
# download the AI-generated object set (~5.8 GB zip -> ~4.9 GB extracted; prompts y/n)
python -m robocasa.scripts.download_kitchen_assets --type objs_aigen
# re-harvest so the meta picks up on-disk aigen instances
python tools/harvest_robocasa_objects.py --out robocasa_object_meta.json
```

### Step 1 — select the pool (libero env)
```bash
python tools/select_scale_pool.py --meta robocasa_object_meta.json \
    --out scale_pool_manifest.json --seed 0
# knobs: --seen-target 40 --tierb-target 25 --n-prime 5 --n-tiera 3 --n-tierb 3 --max-xy-half 0.075
```

### Steps 2–7 — the orchestrator (libero env)
`run_scale_pipeline.py` drives the rest. It is resumable (skips keys with an existing
`result.json` / rendered HDF5) and logs per key under `<out-root>/logs/`.

```bash
# PILOT: a representative subset to measure real gate-yield / time / disk first
python scripts/run_scale_pipeline.py --manifest scale_pool_manifest.json \
    --out-root /tmp2/leocheng/ricl_scratch/scale_pilot \
    --categories canned_food bowl egg bar apple banana avocado chicken_breast \
    --num-success 50 --max-attempts 100 --workers 8 --render

# FULL RUN: all 384 keys (multi-day; renders to /tmp2, compressed, resumable)
python scripts/run_scale_pipeline.py --manifest scale_pool_manifest.json \
    --out-root /tmp2/leocheng/ricl_scratch/scale \
    --num-success 50 --max-attempts 100 --workers 8 --render
```

Stage breakdown (all inside the orchestrator; `--stages convert,bddl,collect,render,report` to subset):
1. **convert** → `convert_robocasa_xml.py --manifest` writes `<out-root>/convert_report.json`;
   quarantines any key with a parse failure or unresolved asset; auto-fit-scale shrinks oversize
   meshes so they place on the table region.
2. **register-check** → a fresh subprocess imports `get_object_dict()` and keeps only registered keys.
3. **bddl** → `create_robocasa_pickplace_tasks.py --manipulanda <keys> --receptacle rc_tray`
   (one pick-place task/key vs `rc_tray`) into `libero/libero/bddl_files/robocasa_pickplace_scale/`.
4. **collect** (parallel, CPU) → `collect_scripted_demonstrations.py --num-success 50 --max-attempts
   100 --result-json`; the gate keeps a key iff `reached_target == True`.
5. **render** (serialized, GPU/egl) → `create_dataset.py --use-camera-obs --compress --output-root
   <out-root>/rendered` for survivors only.
6. **report** → `scale_pool_report.{json,md}` + `make_tier_split.py` → `tier_split.json`.

### The gate
`collect_scripted_demonstrations.py --result-json` writes:
```json
{ "chosen_grasp_frac": 0.5, "calib_yields": {"0.5":[2,2],"0.65":[2,2],"0.8":[0,2]},
  "collect_successes": ..., "collect_attempts": ..., "total_successes": ..., "total_attempts": ...,
  "reached_target": true, "gate_yield": 42.9, "collect_yield": 100.0, "n_demos": 50 }
```
**Gate on `reached_target`** (did the chosen grasp_frac hit `--num-success` within `--max-attempts`),
not on `gate_yield` — the latter is diluted by the deliberately-bad grasp_fracs tried during
calibration.

---

## 6. Output layout

```
/tmp2/leocheng/ricl_scratch/scale[_pilot]/
├── working_manifest.json          # the (subset) manifest this run processed
├── convert_report.json            # per-key converted / quarantined + reason
├── collect/<key>/demo.hdf5        # low-dim successful demos (robosuite format)
├── collect/<key>/result.json      # the gate result above
├── rendered/robocasa_pickplace_scale/<task>_demo.hdf5   # robomimic HDF5 w/ images
├── logs/<key>.{collect,render}.log
├── scale_pool_report.{json,md}    # survivors / dropped (ceilings) / tier counts / covariates
└── tier_split.json                # Seen / Tier-A / Tier-B over survivors (disjointness asserted)
```
Registered object assets (`libero/libero/assets/robocasa_objects/<key>/`) and BDDL files
(`libero/libero/bddl_files/robocasa_pickplace_scale/`) live **in the repo working tree**; demos and
rendered datasets live under `/tmp2` scratch.

Rendered image HDF5 keys (validated): `obs/{agentview_rgb, eye_in_hand_rgb (128×128×3 uint8),
ee_states (6), gripper_states (2), joint_states, ee_pos, ee_ori}`, `actions (7)`, plus
`states/robot_states/rewards/dones` and per-demo `model_file`/`init_state`.

---

## 7. Verification evidence (dry-run, 6 keys spanning tiers + both sources)

- **Conversion:** aigen objects convert cleanly after the texture fix; auto-fit-scale shrank oversize
  meshes (`bowl` 0.115 m and `pan__aigen_4` 0.19 m horizontal half-extent) so they place.
- **Registration + BDDL + reset:** all 6 load headless and reset; obs exposes `<key>_1_pos` and
  `robot0_eef_pos` (confirming the double-underscore key resolves).
- **Gate:** `onion__objaverse_7` **passes** (grasp_frac auto-picks 0.5, wrote demos,
  `reached_target=True`); `bowl__objaverse_14` **fails** (0/14, `reached_target=False`) — the wide/
  hollow overlap category is gated out exactly as expected, with its ceiling logged.
- **Render:** produces the expected obs keys/shapes above; `num_demos > 0`.
- **Orchestrator:** smoke test (2 categories) ran convert→bddl→parallel-collect→report green.

---

## 8. Environment & gotchas

- **Two conda envs, never mixed:** `libero` (robosuite 1.4, mujoco 3.2.3 — all sim/collect/render)
  and `robocasa` (robosuite 1.5.2 — asset harvest/download only).
- **`objs_aigen` download** goes to `.../robocasa/models/assets/objects/aigen_objs/`; the harvester
  only records instances whose `model.xml` is on disk, so re-harvest **after** the download.
- **Disk (binding constraint):** `/tmp2` ≈ 470 GB free; **`/home` only ≈ 203 GB free**.
  `create_dataset.py` defaults its output to `get_libero_path("datasets")` on `/home` — always pass
  `--output-root` (the orchestrator does) to keep renders on `/tmp2`, and `--compress` to gzip images.
  A full run is ~50–150 GB compressed.
- **Rendering** needs `MUJOCO_GL=egl` and a (shared) GPU; collection is headless and CPU-only.
- **Per-instance keys** contain double underscores and multi-digit indices — do not route them through
  any camel/snake name derivation (see §2.2).

---

## 9. Current status & remaining work

**Running:** a 44-key pilot (`canned_food bowl egg bar` Seen + `apple banana avocado chicken_breast`
Tier B), 50 demos/object, 8 parallel collect workers, rendering to `/tmp2`. On completion it yields
real gate-yield-per-category, per-key collect time, compressed disk/demo, and tier counts — the
inputs to extrapolate and green-light the **full 384-key run**.

**Not yet done (next phase, out of scope here):** the full-pool run itself; and everything RICL —
`libero_to_ricl.py`, retrieval, norm stats, training/serving/eval, and the optional lightwheel
category expansion.
