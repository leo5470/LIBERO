# Project — RoboCasa objects in LIBERO: what changed inside LIBERO

A walkthrough of the changes **to LIBERO itself** (the `libero/libero/…` package and LIBERO's
`scripts/`) that turn the RoboCasa→LIBERO import into a scalable, gated Seen / Tier-A / Tier-B object
pool. Scope is **LIBERO-only** (no RICL). External tooling that merely *produces* LIBERO artifacts
(`tools/…`, `scripts/run_scale_pipeline.py`) is summarized at the end and documented fully in
[`Completed.md`](Completed.md).

Integration happens at six layers along the data flow. Each layer below is tagged **[LIBERO
library]** (`libero/libero/…`), **[LIBERO scripts]** (`scripts/`), or notes where an external tool
generates the artifact LIBERO consumes.

---

## Layer 1 — Object registry `[LIBERO library]`

**`libero/libero/envs/base_object.py`** — *unchanged, but now deliberately bypassed.* LIBERO keeps a
global `OBJECTS_DICT`, keyed by a camel→snake transform of the class name (`register_object`,
base_object.py:7‑12). That transform (`re.sub(r"([A-Z0-9])", …)`) **splits on every digit and
collapses underscores**, so it cannot round-trip a per-instance key: `apple__objaverse_10` → class
`AppleObjaverse10` → `apple_objaverse_1_0` (≠ the dir name). Under this transform, BDDL lookups by the
manifest key failed.

**`libero/libero/envs/objects/robocasa_objects.py`** — *the core integration point; the one real
LIBERO-library change this session.*
- `RoboCasaObject(MujocoXMLObject)` (Phase 0) loads a converted `model.xml` and sets the LIBERO-required
  attributes (`category_name`, `object_properties`).
- `register_robocasa_objects()` auto-scans `assets/robocasa_objects/<key>/model.xml` and registers one
  class per dir, as an import side-effect (bottom of the file).
- **Fix (this session):** register **directly** under the exact dir name —
  `OBJECTS_DICT[dir_name] = _make_class(dir_name)` — instead of via `register_object`. This keeps
  **registry key == manifest key == BDDL object name** for double-underscore, multi-digit per-instance
  keys. Phase-0 single-word keys (`apple`, `rc_tray`) still register identically.

**`libero/libero/envs/objects/__init__.py`** — line 9 `from .robocasa_objects import *` makes
registration fire on import of LIBERO's env package. (Phase 0; unchanged this session.)

*Net effect:* `get_object_dict()` now returns hundreds of RoboCasa instance keys alongside LIBERO's
native objects, usable anywhere LIBERO looks up an object by name.

---

## Layer 2 — Assets `[LIBERO library data]`

New per-instance object dirs under **`libero/libero/assets/robocasa_objects/<key>/`** (`model.xml` +
`.obj` meshes + textures). Produced by `tools/convert_robocasa_xml.py` (external), but they *are* the
LIBERO asset change. The crux is a **format bridge**: RoboCasa expresses extents via a `reg_bbox` geom
and applies a category scale at load, whereas LIBERO's `MujocoXMLObject`/placement code expects
**sites**. So each converted XML gets synthesized `bottom_site` / `top_site` /
`horizontal_radius_site` (robosuite placement + collision) and, for the receptacle, a `contain_region`
box site so LIBERO's `(In …)` affordance works (cf. `basket.xml`).

Two robustness behaviors are baked into these assets by the converter:
- **shared-texture resolution** — aigen XMLs referenced textures by an absolute path from the
  generation machine; missing files are now resolved by basename against the local robosuite textures
  and copied in (otherwise MuJoCo compile fails).
- **auto-fit-scale** — oversize meshes are shrunk so their baked horizontal half-extent fits LIBERO's
  small table placement region.

---

## Layer 3 — Task / BDDL generation `[LIBERO scripts + library utils]`

**`scripts/create_robocasa_pickplace_tasks.py`** (Phase 0, parameterized; unchanged this session) turns
RoboCasa objects into LIBERO tasks using LIBERO's own machinery — `register_mu` +
`InitialSceneTemplates` (mu_utils), `register_task_info` + `generate_bddl_from_task_info`
(task_generation_utils), `get_object_dict`. It emits one `kitchen_table` pick-place task per key: table
+ fixed `rc_tray` + the one object, goal `(In <key>_1 rc_tray_1_contain_region)`
(create_robocasa_pickplace_tasks.py:43‑78). The placement regions are tiny (manipuland
`region_half_len=0.025`), which is *why* auto-fit-scale (Layer 2) is needed. Generated BDDL lives in
the LIBERO tree at **`libero/libero/bddl_files/robocasa_pickplace_scale/`**.

---

## Layer 4 — Env & observations `[LIBERO library]`

No code change, but a consequence to flag: the scene object is `<key>_1`, so LIBERO exposes the pose as
**`obs["<key>_1_pos"]`** (e.g. `bowl__objaverse_14_1_pos`), not `<key>_pos`. Tasks reset via
`TASK_MAPPING[problem_name]`; success is LIBERO's own `env._check_success()`. This is the contract the
collector and any downstream consumer rely on.

---

## Layer 5 — Scripted demo collection `[LIBERO scripts]`

**`scripts/collect_scripted_demonstrations.py`** builds a headless LIBERO env, wraps it in robosuite's
`DataCollectionWrapper`, runs the scripted top-down policy, keeps only `_check_success()` episodes, and
auto-calibrates `grasp_frac` per object. **Change this session:** added `--result-json`, a
machine-readable gate result (`reached_target`, per-frac `calib_yields`, collect counts, `n_demos`) so
the orchestrator gates without parsing stdout. Output `demo.hdf5` is robosuite low-dim format, feeding
Layer 6. **Gate on `reached_target`**, not the calibration-diluted `gate_yield`.

---

## Layer 6 — Dataset rendering `[LIBERO scripts]`

**`scripts/create_dataset.py`** is LIBERO's replay→robomimic-HDF5 renderer (adds camera images).
**Changes this session:**
- `--output-root` — redirect the rendered `<task>_demo.hdf5` off the default
  `get_libero_path("datasets")` location. This matters because of LIBERO's **path-split gotcha**:
  `get_libero_path` resolves to a *different* clone at `/home/leocheng/LIBERO` (per
  `~/.libero/config.yaml`), whose disk has only ~203 GB free, vs ~470 GB on `/tmp2`. The orchestrator
  passes `--output-root` to keep renders on `/tmp2`.
- `--compress` — gzip the RGB/depth image datasets (~3× smaller).

Rendered HDF5 schema (validated): `obs/{agentview_rgb, eye_in_hand_rgb (128²), ee_states,
gripper_states, joint_states, ee_pos, ee_ori}`, `actions (7)`, plus `states/robot_states/rewards/dones`
and per-demo `model_file`/`init_state`.

---

## Cross-cutting: the per-instance key scheme

Phase 0 used one object per category (`apple`, `rc_tray`). Scaling to many instances/category, the
LIBERO registry key is now per instance: **`<category>__<source>_<suffix>`** (e.g.
`bowl__objaverse_14`, `ketchup__aigen_6`), one converted dir per key. Do **not** route these through any
camel/snake name derivation (Layer 1).

---

## Not part of LIBERO (external tooling that produces LIBERO artifacts)

`tools/harvest_robocasa_objects.py` (robocasa env → `robocasa_object_meta.json`),
`tools/select_scale_pool.py` (→ `scale_pool_manifest.json`), `tools/convert_robocasa_xml.py` (→ Layer‑2
assets), `tools/make_tier_split.py`, and `scripts/run_scale_pipeline.py` (orchestrator). These live
outside the LIBERO package and don't change LIBERO's runtime; they generate the assets/BDDL/datasets
LIBERO consumes. Full detail in [`Completed.md`](Completed.md).

---

## Pilot results (44 keys — complete)

Ran `canned_food bowl egg bar` (Seen) + `apple banana avocado chicken_breast` (Tier B), 50 demos/object,
8 parallel collect workers, render pinned to a single GPU. **23/44 keys survived the gate (52%).**

| category | tier role | survived / collected | note |
|---|---|---|---|
| `canned_food` | Seen (overlap, graspable) | **8 / 8** | passes; incl. 3 aigen Tier-A |
| `egg` | Seen (produce) | **8 / 8** | passes; incl. 3 aigen Tier-A |
| `apple` | Tier B | **3 / 3** | |
| `avocado` | Tier B | **2 / 3** | |
| `chicken_breast` | Tier B (aigen novel protein) | **2 / 3** | |
| `banana` | Tier B | **0 / 3** | elongated/curved → top-down grasp slips |
| `bar` | Seen (packaged) | **0 / 8** | thin flat snack bar → nothing to grip |
| `bowl` | Seen (overlap, hollow) | **0 / 8** | gated out (expected; ceiling logged) |

**Tiers over survivors** (`tier_split.json`, disjointness asserted): Seen **2 cats / 10 inst**;
Tier A **2 cats / 6 inst** (all real **aigen** source-shift; 3 overlap + 3 non-overlap); Tier B
**3 cats / 7 inst**. Grasp auto-calibration picked `grasp_frac` 0.5 for most, 0.65 for some
`canned_food`/`avocado` — it is doing real per-object work.

**Throughput / footprint (the numbers that drive the full run):**
- **Collect:** 44 keys in **19.6 min** (8 workers) — dominated by the 16 doomed `bowl`/`bar` keys, which
  each burn the full 118-episode budget to confirm their ceiling.
- **Render:** **2.65 MB / demo** compressed (128², gzip); 23 survivors × 50 = 1150 demos = **3.05 GB**.
- **GPU pinning:** render is pinned to a single idle GPU via the new orchestrator flag `--gpu <id>`
  (sets `MUJOCO_EGL_DEVICE_ID` + `CUDA_VISIBLE_DEVICES`) after the shared GPU 0 hit 99% from another
  user — verified our EGL graphics context sat on the pinned GPU.

**Full-run extrapolation (384 keys):** ~**200 survivors** (at the pilot's 52%, likely higher — the pilot
was deliberately loaded with hard shapes), collect ≈ **~3 hr** @ 8 workers, render disk ≈ **~27 GB**
(trivial vs ~470 GB free on `/tmp2`). **Disk is not the constraint; single-GPU render wall-time is the
long pole.** Tier targets (`≥30 Seen / ≥15 Tier B` categories) — mechanism validated; the full run
confirms the counts.

---

## Next steps

**Ready to launch — the full 384-key run** (pilot validated the pipeline + numbers):
```bash
python scripts/run_scale_pipeline.py --manifest scale_pool_manifest.json \
    --out-root /tmp2/leocheng/ricl_scratch/scale \
    --num-success 50 --max-attempts 100 --workers 8 --render --gpu 1
```
- Resumable; ~3 hr collect + ~27 GB render. **Single-GPU render wall-time is the long pole.**
- Optional speedup: render currently runs serialized. Adding a small `--render-workers` (e.g. 2
  concurrent renders on the *same* pinned GPU — each ~300 MiB, so still one GPU / no collision) would
  roughly halve render wall-time; worth adding before the full run if turnaround matters.
- On completion `make_tier_split.py` asserts `Seen∩TierA==∅` (instances) and
  `Seen-cats∩TierB-cats==∅` (categories) and reports final tier counts against the `≥30 / ≥15` targets.

**LIBERO-side cleanups to do with the full run**
- Re-key or remove the 6 Phase-0 bare dirs (`apple`, `lemon`, … `rc_tray`), which now coexist as
  orphans beside per-instance keys (fold `rc_tray` explicitly into the manifest).
- Decide git-LFS vs. out-of-VCS for the ~2 GB of meshes the full run adds under
  `libero/libero/assets/robocasa_objects/`.
- Optional: tune `--max-xy-half` / init-region half-lens if produce gets over-shrunk; fix
  `get_libero_path`'s path-split at the source rather than routing around it.

**Later phase (out of scope now — RICL):** `libero_to_ricl.py` → `processed_demo.npz`, retrieval, norm
stats, RICL training/serving/eval, and the optional lightwheel category expansion.
