# Experiment Plan — RICL with unseen objects (RoboCasa365 objects in LIBERO)

## 摘要

設定：**embodiment 維持 LIBERO**（固定底座 Franka、OSC 7 維、2 鏡頭、robosuite 1.4），只從 **RoboCasa365 匯入物件 asset** 當作多樣性 / unseen 來源。因此前面建好的整條管線——scripted policy 收 demo、`libero_to_ricl.py`、`action_dim=8`、RICL config、LIBERO sim runtime、以及你已訓練的 LIBERO π0-FAST 當 init——**全部沿用**。

核心科學問題：在多樣物件上做 RICL 重訓後，模型能不能靠「測試時給幾條該物件的 in-context demo（retrieval）」就操作**沒見過的物件**，而不需要再更新權重？用 **兩個 tier** 衡量：Tier A 已見類別的新個體、Tier B 全新類別（RoboCasa365 的 57 個新類別）。

唯一的新工程是把 RoboCasa 的 `MujocoXMLObject` 物件註冊進 LIBERO（`@register_object` + 一個 XML 路徑）。最大風險是 robosuite 1.4(LIBERO)/MuJoCo2 與 robosuite 1.5(RoboCasa)/MuJoCo3 的 asset XML 相容性，先用 Phase 0 把它 de-risk。

---

## 1. Setup and what gets reused

| Component | Choice | Reuse status |
|---|---|---|
| Embodiment / sim | LIBERO (robosuite 1.4, fixed-base Franka, OSC 7-D, agentview + eye_in_hand) | unchanged |
| Object source | **RoboCasa365 assets** (3,200+ objects, objaverse + AI-gen), imported into LIBERO | **new** |
| Demo collection | scripted top-down pick-place policy (turn 1) → robomimic HDF5 | unchanged |
| RICL conversion | `libero_to_ricl.py` → `processed_demo.npz` (state 8, action 8 w/ pad, top/right(zeros)/wrist) | unchanged |
| Retrieval / norm stats / training | `retrieve_within_collected_demo_groups.py`, `setup_norm_states_for_ricl.py`, `pi0_fast_libero_ricl` | unchanged |
| RICL init (`weight_loader`) | **your LIBERO π0-FAST** checkpoint | per your choice |
| Serving | LIBERO sim runtime (`examples/libero/main_ricl.py`) | unchanged |

Because we keep the LIBERO embodiment, **none of RoboCasa's mobile-base / 12-D action / 3-camera machinery is involved** — we only borrow the 3D object meshes. This is the lowest-risk way to get RoboCasa-scale object diversity into the RICL pipeline you already have.

> Why this is the right call: the unseen-object question is about *object* generalization. Keeping skill, scene, receptacle, camera, and action space fixed means any success-rate change is attributable to object novelty, not embodiment shift.

### 1.1 Alignment with the original RICL regime (verified against the paper)

This design **matches** RICL's intended use. In the RICL paper (Sridhar et al., CoRL 2025): RICL is *primed/re-trained* on a fixed set of **20 pick-and-place tasks (400 demos)**, then **deployed on new tasks**, and its headline evaluation category is literally **"Unseen Objects"** (pokeball, idli plate, squeegee) — objects **not** in the priming set. So "train (prime) on seen objects → test on unseen objects, no weight update" is exactly what RICL is for; we are not going off-design. Implications we fold in:

- **Two senses of "seen."** RICL's test objects are unseen both in the *base VLA* (DROID pretraining) **and** in the *RICL priming* set. So make our Tier A/B objects unseen in **both** your LIBERO π0-FAST base **and** the RoboCasa priming pool (RoboCasa objects are mostly absent from LIBERO, so this is naturally satisfied — but verify by asset id).
- **Priming was tiny (20 tasks/400 demos).** Our large priming-diversity sweep is a deliberate scale-up — the authors explicitly list "scaling RICL with more priming demonstrations" as future work, so this is a sensible, novel extension rather than a deviation.
- **Test-time demo count matters.** RICL's ablation found **≥10 in-context demos are needed** (5 reverts to baseline); use this to set k (below).
- Retrieval in the paper embeds the **top (third-person) image** by default; we keep wrist/both as ablations (§7).

## 2. Importing RoboCasa objects into LIBERO (the one new piece)

Both stacks use robosuite `MujocoXMLObject`, and LIBERO's object registry is just a decorator (`base_object.register_object` → `OBJECTS_DICT[snake_case_name]`). RoboCasa objects live at `robocasa/models/assets/objects/<category>/<instance>/model.xml` (+ meshes; ~10GB via `download_kitchen_assets`).

Recipe (script this — you'll import hundreds):
1. Copy each object's `model.xml` + meshes into `LIBERO/.../assets/robocasa_objects/<instance>/`.
2. **Fixup XML for robosuite 1.4 / MuJoCo 2.x** (the real work): adjust `<compiler>` attributes, mesh/material/texture paths, and any MuJoCo-3-only fields. Write one converter that batch-rewrites RoboCasa `model.xml` → LIBERO-loadable XML.
3. Register a class per object (or generate them programmatically from RoboCasa's category registry):

```python
# libero/libero/envs/objects/robocasa_objects.py
import os, re, pathlib, numpy as np
from robosuite.models.objects import MujocoXMLObject
from libero.libero.envs.base_object import register_object

_ASSETS = pathlib.Path(__file__).parent.parent.parent / "assets" / "robocasa_objects"

class RoboCasaObject(MujocoXMLObject):
    def __init__(self, name, obj_name, joints=[dict(type="free", damping="0.0005")]):
        super().__init__(str(_ASSETS / obj_name / "model.xml"),
                         name=name, joints=joints, obj_type="all",
                         duplicate_collision_geoms=False)
        self.category_name = re.sub(r"([A-Z])", r" \1", type(self).__name__).strip().replace(" ", "_").lower()
        self.rotation, self.rotation_axis = (0, 0), "x"
        self.object_properties = {"vis_site_names": {}}

def make_and_register(class_name, obj_name):           # programmatic registration
    cls = type(class_name, (RoboCasaObject,),
               {"__init__": lambda self, name=obj_name, obj_name=obj_name: RoboCasaObject.__init__(self, name, obj_name)})
    return register_object(cls)

# e.g. iterate RoboCasa's category->instance registry:
# for cat, instances in robocasa_object_registry.items():
#     for inst in instances: make_and_register(to_camel(inst), inst)
```
4. Import it in `objects/__init__.py`, then reference objects in BDDL tasks. Generate tasks programmatically with LIBERO's `scripts/create_libero_task_example.py` pattern (`register_mu` scene + `register_task_info` + `generate_bddl_from_task_info`) — one pick-and-place task per object, fixed table + fixed receptacle.

Browse/enumerate RoboCasa objects with `python -m robocasa.demos.demo_objects` (and `--obj_types aigen`); the category→instance registry is what defines your split in §3.

## 3. Object split — both tiers

Pick **one skill, pick-and-place**, and vary only the manipulated object. Define three disjoint object pools:

| Pool | Definition | Used for |
|---|---|---|
| **Seen** | A set of categories, a subset of instances each | RICL **priming** (re-training) |
| **Tier A — unseen instance** | *Held-out instances* of the **seen** categories | test only |
| **Tier B — unseen category** | **Entirely new categories** (use RoboCasa365's 57 new categories) | test only |

Guidelines: ≥ ~30–50 seen categories × several instances for priming diversity (the RoboCasa365 paper shows object/task diversity is the main driver of unseen generalization). Keep ≥5 held-out instances/category for Tier A and ≥15 unseen categories for Tier B. Match object scale/pose distributions across pools so difficulty is comparable; log per-object bbox/mass as covariates.

## 4. Data generation (scripted policy = the scalable demo source)

Teleoperating demos for hundreds of objects is infeasible — the **scripted top-down pick-place policy from turn 1 is the demo generator**. Because the motion is the same across objects and only the object changes, this *cleanly isolates object generalization* and gives RICL a consistent retrieval structure.

Per object-task: run the scripted policy over randomized object poses → keep **only successful** trajectories (filter on `env._check_success()`), e.g. 20–50 demos/object. Then the existing flow:
`create_dataset.py` (render → robomimic HDF5) → `libero_to_ricl.py` (→ `processed_demo.npz`, one task-group per object) → `retrieve_within_collected_demo_groups.py` → `setup_norm_states_for_ricl.py`.

- **Priming set** = seen-object task-groups → `collected_demos_training/`.
- **Retrieval/test set** = Tier A + Tier B object task-groups → `collected_demos/` (these are the test-time in-context demos; ensure ≥ k+1 demos/object so retrieval has neighbors).
- Caveat: the simple top-down grasp won't suit every geometry. Filter to objects the scripted policy can grasp, and **report the scripted-policy success rate per object as a ceiling** so RICL numbers are normalized against achievability.

## 5. RICL re-training

Init from **your LIBERO π0-FAST** (`weight_loader = CheckpointWeightLoader("checkpoints/pi0_fast_libero/<exp>/<step>/params")`), config `pi0_fast_libero_ricl` (action_dim 8, RICL transforms, 2 real cameras + zero right, frozen image encoder as in the DROID recipe). Train on the seen-object priming groups only; retrieval is within-group (same object, other demos).

Primary independent variable to sweep: **priming object diversity** (#categories × #instances). Hypothesis H2 says unseen success rises with priming diversity.

## 6. Evaluation

**Research question.** After RICL re-training on diverse objects, can a π0-FAST policy manipulate *unseen* objects from a few in-context demos at test time, *without* weight updates — beating zero-shot and approaching per-object finetuning?

**Hypotheses.** H1: RICL-retrieval > zero-shot on unseen objects. H2: unseen success ↑ with k (in-context demos) and with priming diversity. H3: Tier A (unseen instance) > Tier B (unseen category); the gap shrinks as priming diversity grows.

**Baselines (all on the same unseen-object tasks):**

| # | Method | Weight update at test? | Purpose |
|---|---|---|---|
| 1 | LIBERO π0-FAST, zero-shot (no context) | no | lower bound |
| 2 | Multi-task π0-FAST trained on all seen-object demos | no | does plain multi-task already generalize? |
| 3 | Per-object finetune on the k test demos | **yes** | reference upper bound + cost contrast |
| 4 | RICL with **random** in-context demos | no | isolates value of *retrieval* vs mere conditioning |
| 5 | **RICL with retrieval (proposed)** | no | main method |

**Metrics & protocol.** Success rate via `env._check_success()`, ≥30 rollouts/object with randomized object poses, stratified by **Seen / Tier A / Tier B**. Report: success vs **k ∈ {5,10,15,20}** in-context demos (RICL needs ≥10 — fewer reverts to baseline per the paper's ablation; ensure ≥ k+1 demos/test object); success vs priming diversity; success normalized by the scripted-policy ceiling (§4). Retrieval analysis: distribution of retrieved-neighbor embedding distances, and whether neighbors share object/grasp state.

## 7. Ablations

- **`num_retrieved_observations`** (RICL context size N) and **`lamda`** (exp(−λ·dist) weighting), `use_action_interpolation` on/off.
- **Retrieval embedding**: agentview vs wrist vs both. (Wrist view is highly informative for object identity at grasp — likely the strongest single view here.)
- **Priming diversity sweep**: #categories and #demos/object.
- **Prompt dependence**: category-name prompt ("pick up the {category}") vs neutral ("pick up the object"). For Tier B the category word is itself novel, so this disentangles language-novelty from visual-novelty.
- (Fixed by your choice, but worth one run) init from LIBERO π0-FAST vs `pi0_fast_base`.

## 8. Confounds & controls

- Fix skill, receptacle, table/scene style, camera, action space, language template → only object identity varies.
- Match object scale/pose across pools; report bbox/mass as covariates so a Tier-B drop isn't just "bigger/heavier objects."
- Filter to scripted-graspable objects and normalize by the per-object ceiling.
- Ensure Tier A/B objects are truly absent from priming (instance- and category-level disjointness; verify by asset id).
- ≥ k+1 demos per test object so retrieval has within-group neighbors.

## 9. Phases & milestones

0. **Integration sanity (de-risk):** import ~5 RoboCasa objects, fix XML for robosuite 1.4, generate scripted demos, run the *entire* pipeline end-to-end on this toy set, serve in LIBERO. Confirms the only novel risk before scaling.
1. **Object import at scale:** batch XML-fixup + programmatic registration; define Seen / Tier A / Tier B; auto-generate BDDL pick-place tasks.
2. **Data generation:** scripted demos for all pools; filter successes; convert + retrieval + norm stats.
3. **RICL training:** train from LIBERO π0-FAST; priming-diversity sweep.
4. **Evaluation:** baselines 1–5 across tiers; k-sweep; ablations.
5. **Analysis & writeup.**

## 10. Risks & mitigations

| Risk | Mitigation |
|---|---|
| **robosuite 1.4/MuJoCo2 (LIBERO) vs 1.5/MuJoCo3 (RoboCasa) asset incompatibility** (biggest) | Phase-0 toy import; write a batch `model.xml` converter; if intractable, upgrade LIBERO's robosuite or re-export objects via robosuite 1.4. |
| Scripted top-down grasp fails on odd geometries | Per-category grasp-offset tuning; filter objects; fall back to teleop/MimicGen for a few hard categories; report ceilings. |
| LIBERO π0-FAST init weak on novel RoboCasa textures | Acceptable (that's the test); optionally compare `pi0_fast_base` init in one ablation. |
| Too few demos per unseen object → no retrieval neighbors | Enforce ≥ k+1 demos/test object. |
| Tier-B category word is OOD for the tokenizer | Run the neutral-prompt variant (§7) to separate language vs visual novelty. |

## 11. Verification checklist

1. Phase-0 object loads in a LIBERO env; `obs["<obj>_pos"]` present; scripted policy grasps it; `_check_success()` fires.
2. `processed_demo.npz` for a RoboCasa-object task has correct shapes (state 8, actions 8, images 224³); retrieval + norm stats run.
3. Seen / Tier A / Tier B object id lists are disjoint (assert programmatically).
4. RICL trains from the LIBERO π0-FAST init (first-batch `query_actions (B,H,8)`, no NaN).
5. Eval harness reports success stratified by tier and by k, plus per-object scripted ceiling.
