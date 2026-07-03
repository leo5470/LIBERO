# CLAUDE.md — Import RoboCasa objects into LIBERO (LIBERO-only phase)

**Scope for now: LIBERO only.** Bring RoboCasa365 objects into LIBERO, build pick-and-place tasks, and
auto-collect scripted demos. **Do NOT touch the RICL training/serving pipeline yet** — it is a later
phase. Companion refs (read before coding): `RICL_RoboCasa_unseen_object_experiment_plan.md` (full plan;
its RICL sections are deferred) and `robocasa_object_pool.md` (object selection + graspability lists).

## Goal (this phase only)

Keep the LIBERO embodiment (fixed-base Franka, robosuite 1.4, OSC 7-D action, cameras `agentview` +
`robot0_eye_in_hand`). Import RoboCasa object meshes, register them as LIBERO objects, generate
pick-and-place BDDL tasks, run the scripted top-down policy, and save successful demos as robomimic
HDF5 via LIBERO's own `create_dataset.py`. **Stop there.**

## RoboCasa is NOT provided yet — set it up as an external asset source

RoboCasa needs robosuite **master (v1.5) / MuJoCo 3.x**, which CONFLICTS with LIBERO's robosuite 1.4 /
MuJoCo 2.x. Use **two separate Python envs**. RoboCasa is used ONLY to (a) read the `OBJ_CATEGORIES`
registry and (b) harvest object `model.xml` + meshes. All simulation work happens in the LIBERO env.

robocasa env (separate; per RoboCasa README):
```bash
conda create -c conda-forge -n robocasa python=3.11 -y && conda activate robocasa
git clone https://github.com/ARISE-Initiative/robosuite && pip install -e robosuite   # master / v1.5
git clone https://github.com/robocasa/robocasa && pip install -e robocasa
python -m robocasa.scripts.setup_macros
python -m robocasa.scripts.download_kitchen_assets    # ~10GB of object + scene meshes
```
Then export the object metadata to JSON so the LIBERO env never needs robocasa importable:
```python
# run in the robocasa env
import json, os, glob, robocasa
from robocasa.models.objects.kitchen_objects import OBJ_CATEGORIES
root = os.path.join(robocasa.models.assets_root, "objects")
def types_of(v):
    t = v.get("types", ())
    return [t] if isinstance(t, str) else list(t)
cats = {k: {"graspable": v.get("graspable"), "types": types_of(v)} for k, v in OBJ_CATEGORIES.items()}
# map instances from the filesystem; instance dirs are named like "<category>_<n>" (objaverse) etc.
inst = sorted(os.path.dirname(p) for p in glob.glob(os.path.join(root, "**", "model.xml"), recursive=True))
json.dump({"assets_root": root, "categories": cats, "instance_model_xmls": inst},
          open("robocasa_object_meta.json", "w"), indent=2)
print(len(cats), "categories;", len(inst), "instances")
```
Hand `robocasa_object_meta.json` + the assets dir to the LIBERO side.

## Object selection (graspability filter)

From `robocasa_object_meta.json`: drop the 41 `graspable=False` categories as manipulanda; reuse the 7
`receptacle` categories (plate/tray/basket/cutting_board/oven_tray/baking_sheet/placemat) as the
place-target; pick manipulanda from the graspable set. Full lists + rationale in `robocasa_object_pool.md`.
**Derive lists from the JSON, not from hand-typed names.**

## START HERE — Phase 0 (LIBERO only; scope your first work to THIS)

Prove the import path on a tiny set before scaling. Do not import all objects yet.

1. Set up the robocasa env, download assets, export `robocasa_object_meta.json` (above).
2. Pick **5 easy graspable objects** (small, convex: e.g. apple, lemon, bell_pepper, a can-like, a mug-like).
3. Copy each `model.xml` + meshes into `LIBERO/libero/libero/assets/robocasa_objects/<inst>/`.
4. Write `tools/convert_robocasa_xml.py` — batch-rewrite RoboCasa `model.xml` -> robosuite-1.4 / MuJoCo-2.x
   loadable XML (fix `<compiler>` attrs, mesh/material/texture paths, drop MuJoCo-3-only fields). Iterate
   against real load errors.
5. Register: add `libero/libero/envs/objects/robocasa_objects.py` (`RoboCasaObject(MujocoXMLObject)` +
   `@register_object`, see plan section 2) and import it in `objects/__init__.py`.
   - Gate: `from libero.libero.envs.objects import get_object_dict; assert "<inst>" in get_object_dict()`.
6. Build one BDDL pick-place task per object (fixed table + one fixed receptacle target) via the
   `scripts/create_libero_task_example.py` pattern.
   - Gate: `OffScreenRenderEnv(bddl_file_name=...)` resets; `obs["<inst>_pos"]` and `obs["robot0_eef_pos"]`
     exist; the object is visible in `agentview`.
7. Run the scripted policy (`libero_scripted_policy.py`); collect rollouts.
   - Gate: some rollouts hit `env._check_success()`. Record per-object success yield.
8. (LIBERO finish line) render the successful demos to robomimic HDF5 via LIBERO `create_dataset.py`.
   - Gate: HDF5 has `data/demo_*/obs/{agentview_rgb, eye_in_hand_rgb, ee_states, gripper_states}` + `actions`.

When all gates pass on 5 objects, **stop and report** per-object yields + any XML quirks. Scaling to the
full pool, the seen/unseen split, and anything RICL come later.

## Key files

- LIBERO: `libero/libero/envs/base_object.py` (`register_object`, `OBJECTS_DICT`),
  `libero/libero/envs/objects/google_scanned_objects.py` (object-class template),
  `libero/libero/envs/objects/__init__.py`, `scripts/create_libero_task_example.py`,
  `scripts/create_dataset.py`, `libero/libero/envs/env_wrapper.py` (`OffScreenRenderEnv`, `set_init_state`).
- From the robocasa env: `robocasa_object_meta.json` + `<assets_root>/.../<inst>/model.xml`.
- Scripted demo generator: `libero_scripted_policy.py`.

## Gotchas

- **Two envs, never mixed**: do not pip-install robocasa and libero in the same env (robosuite 1.4 vs
  1.5 conflict). The robocasa env is asset-harvest only.
- robosuite/MuJoCo version mismatch in `model.xml` (step 4) — expect to iterate; validate by loading in a
  LIBERO env, not by reasoning about XML.
- Top-down scripted grasp fails on tall/handled/fragile objects even if `graspable=True` — filter empirically.

## Out of scope for now (later phases — ignore until LIBERO import + demos are solid)

RICL conversion (`libero_to_ricl.py`), retrieval, norm stats, RICL training/serving (`ricl_openpi`),
the seen/unseen tier split, and demo-diversity scaling. These live in the experiment plan; do not start
them in this phase.