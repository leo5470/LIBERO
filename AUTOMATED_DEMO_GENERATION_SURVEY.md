# Automated (Zero-Human) Demonstration Generation for `libero_object_unseen_stockbg`

**Scope.** Survey of every method/tool/repo that can produce expert state–action trajectories for the
`libero_object_unseen_stockbg` suite **with zero human demonstrations, zero teleoperation, and zero human
seed trajectories**. Target: Franka Panda, robosuite `OSC_POSE`, 7-D relative actions
`[dx, dy, dz, dax, day, daz, gripper]` @ 20 Hz, output in robomimic/LIBERO HDF5.

**Status of the suite (measured this session, not quoted):**

| Fact | Value | Source |
| :--- | :--- | :--- |
| Tasks in suite | 1,557 | `bddl_files/libero_object_unseen_stockbg/manifest.json` |
| Physics-feasible tasks | 1,417 | `feasible_task_ids.txt` / `exclusion_report.json` |
| Task template | `pick up the <X> and place it in the basket` — **one** motion template, 1 swapped target + 5 stock distractors, goal `(In <target> basket_1_contain_region)` | BDDL files |
| Feasible tasks with a *known-good* scripted demo path | **199** (`gate_survivor_*`) | `manifest.json:prior_status` |
| Feasible tasks that *failed* the old scripted gate | **129** (127 dropped + 2 crashed) | same |
| Feasible tasks **never attempted** | **1,089 (77%)** | same |
| Existing demos on disk for this suite | **none** | repo scan |

The suite is therefore **~14% covered** by the existing generator and the remaining 86% is unknown or failing.
Everything below is judged against closing that gap.

---

## 0. Headline empirical result (new, measured in this session)

The repo already ships a zero-human generator (`libero_scripted_policy.py` +
`scripts/collect_scripted_demonstrations.py`). It was run at full scale once, over 382 RoboCasa object keys
(`/tmp2/leocheng/ricl_scratch/scale/collect/*/result.json`). Aggregating those results:

| Outcome | Objects | Share |
| :--- | ---: | ---: |
| Passed gate (≥50 successes in ≤100 attempts) | 206 / 382 | **53.9%** |
| **100% collect yield** | 177 | 46.3% |
| 1–99% yield | 48 | 12.6% |
| **0 demos — total failure** | **157** | **41.1%** |

The distribution is **bimodal, not noisy** — an object either works every time or never works. Cause: LIBERO
spawns objects at a fixed yaw (`yaw_rotation (0 0)`) and the policy holds `a[3:6] = 0`, i.e. **it never rotates
the wrist**. Grasp geometry is therefore deterministic per object.

The zero-demo categories are almost entirely *elongated* (baguette, bar, hot_dog, rolling_pin, corn, croissant,
banana, carrot, celery, cucumber, asparagus) or *wide/hollow* (bowl, pan, ladle, jug, wine).

**Test.** I added a world-z yaw servo that drives the gripper's local **y-axis** (the Panda finger-closing
direction — `panda_gripper.xml` finger joints slide on `axis="0 1 0"`, 0.04 m each ⇒ **0.08 m max opening**)
onto the object's horizontal **minor** principal axis, computed by PCA over the object's MuJoCo mesh vertices at
reset. Everything else identical. 5–6 rollouts per cell, on tasks that scored **0 demos** in the scale run:

| Object (feasible stockbg task) | xy extents (cm) | baseline success | baseline lifted | **yaw success** | **yaw lifted** |
| :--- | :--- | ---: | ---: | ---: | ---: |
| `carrot__objaverse_0` | 14.5 × 4.2 | 0/6 | 0/6 | **6/6** | 6/6 |
| `hot_dog__aigen_0` | 12.8 × 5.0 | 0/6 | 0/6 | **6/6** | 6/6 |
| `corn__aigen_0` | 13.7 × 7.2 | 0/5 | 0/5 | **5/5** | 5/5 |
| `bar__aigen_0` | 12.1 × 3.2 | 0/5 | 2/5 | **5/5** | 5/5 |
| `croissant__aigen_1` | 9.1 × 5.7 | 0/5 | 0/5 | **5/5** | 5/5 |
| `eggplant__objaverse_0` | 10.5 × 5.8 | 0/5 | 0/5 | **5/5** | 5/5 |
| `sponge__aigen_0` | 8.7 × 6.3 | 0/5 | 5/5 | **5/5** | 5/5 |
| `asparagus__aigen_11` | 15.0 × 4.5 | 0/5 | 0/5 | 2/5 | 5/5 |
| `fish__objaverse_1` | 17.1 × 4.8 | 0/5 | 0/5 | 2/5 | 5/5 |
| `rolling_pin__aigen_2` | 18.0 × 3.8 | 0/5 | 0/5 | 0/5 | **5/5** |
| `baguette__objaverse_0` | 16.4 × 6.7 | 0/5 | 0/5 | 0/5 | **5/5** |
| `celery__aigen_8` | 15.2 × 7.1 | 0/5 | 1/5 | 0/5 | **5/5** |
| `wine__aigen_4` | 7.4 × 7.6 | 0/5 | 3/5 | 0/5 | **5/5** |
| `ladle__objaverse_0` | 18.0 × 5.5 | 0/5 | 0/5 | 0/5 | 0/5 |
| `steak__aigen_7` | 8.5 × 6.9 | 0/5 | 0/5 | 0/5 | 0/5 |
| `donut__aigen_1` | 9.4 × 9.1 | 0/5 | 5/5 | 0/5 | 3/5 |
| `banana__objaverse_14` | 16.4 × **9.0** | 0/5 | 0/5 | 0/5 | 0/5 |
| `jug__objaverse_0` | 13.0 × **10.8** | 0/5 | 4/5 | 0/5 | 0/5 |
| `pan__aigen_4` | 18.0 × **13.2** | 0/5 | 0/5 | 0/5 | 0/5 |
| `bowl__objaverse_14` | 18.0 × **18.0** | 0/5 | 0/5 | 0/5 | 0/5 |
| *control* `apple__objaverse_15` (gate survivor) | 5.4 × 5.5 | 5/6 | 5/6 | **6/6** | 6/6 |

**Yaw error at grasp was 81–90° for the baseline on every failing object, and 0–1° with the servo.** The fingers
were closing along the object's *long* axis. This is the single dominant failure mechanism, and it is a
~15-line fix.

Reading the residuals, the 41% failure mass decomposes into four *distinct* mechanisms:

1. **Wrist yaw (fixed by the servo).** ~44% of the sampled failure set goes 0 → full or partial success.
   Note the sample was deliberately drawn from the *worst* categories, so recovery over the whole zero-demo set
   should be better than 44%.
2. **Minor extent > 0.08 m jaw limit** (banana 9.0, donut 9.1, jug 10.8, pan 13.2, bowl 18.0). *No* top-down
   parallel-jaw grasp exists at any yaw. Needs rim/handle grasp synthesis or exclusion.
3. **Grasp fixed, placement still fails** (rolling_pin, baguette, celery, wine — `lifted 5/5, success 0/5`).
   Long/tall objects don't settle inside the basket's `contain_region`. Needs place-orientation control
   (align long axis to the basket) or honest exclusion.
4. **Centroid grasp is wrong locally** (ladle: 18.0 × 5.5, never lifts — PCA centroid sits between the bowl and
   the handle). Needs a *local* grasp sampler, not a global principal axis.

**Throughput measured:** 41.3 s for 10 headless low-dim rollouts incl. env build ⇒ **≈3.5–4 s per rollout on one
core**. At ~85% yield and 40 cores that is **≈40k demos/hour** pre-rendering. Rendering (`create_dataset.py`) is
the real bottleneck and is GPU-serialized.

Experiment script: `/tmp/claude-1002/-tmp2-leocheng-forks-LIBERO/d9d4f378-c8b5-47f0-ad78-f0950d387def/scratchpad/yaw_experiment.py`

---

# Part 1: Structured Method Entries

## Category 1 — Fully Automated Motion Planning & Waypoint Controllers

### 1.1 `ScriptedPickPlacePolicy` (in-repo baseline)

| Field | Details |
| :--- | :--- |
| **Name** | `libero_scripted_policy.ScriptedPickPlacePolicy` |
| **Type** | Procedural Script (ground-truth waypoint state machine) |
| **Suite coverage** | **Fully supports suite** — written for exactly this suite; `infer_obj_and_target` reads `env.obj_of_interest` from the BDDL |
| **Zero human seed?** | **Strictly Yes (0 human seeds)** |
| **Action space** | **Direct 7-D `OSC_POSE`** — `a[:3] = clip(pos_err·20, ±1)`, `a[3:6]=0`, `a[6]=±1` |
| **Output format** | HDF5 robomimic/LIBERO (`states`/`actions`/`model_file` + `data` attrs) via `collect_scripted_demonstrations.py` |
| **Code & status** | This repo, `libero_scripted_policy.py` (200 lines). Maintained, working. |
| **Paper** | None (repo-original) |
| **Success & quality** | **Measured: 53.9% of 382 objects pass; 46.3% at 100% yield; 41.1% at 0%.** Median collect yield 84.8%. Trajectories are smooth P-controlled straight lines; 8-phase machine `APPROACH→DESCEND→GRASP→LIFT→MOVE→LOWER→DROP→SETTLE`. |
| **Scalability** | **≈3.5–4 s/rollout/core measured**, headless, no GPU |
| **Integration** | **Plug-and-play** |
| **Limitations** | **Never rotates the wrist** (`a[3:6]≡0`) ⇒ deterministic failure on all elongated geometry (measured 81–90° yaw error at grasp). No collision awareness — relies on a high `carry_z` to clear distractors. Grasp height is a hand-tuned `grasp_frac` swept over a 3-value grid. No re-grasp/retry. |

### 1.2 Yaw-aligned PCA grasp extension (measured this session)

| Field | Details |
| :--- | :--- |
| **Name** | Minor-axis yaw servo on top of 1.1 (`YawAlignedPolicy`) |
| **Type** | Procedural Script + analytic geometry |
| **Suite coverage** | **Fully supports suite** |
| **Zero human seed?** | **Strictly Yes** |
| **Action space** | **Direct 7-D** — adds only `a[5] = clip(yaw_err·gain/0.5, ±1)`; verified that robosuite composes the delta as `R_delta @ R_current` (`control_utils.set_goal_orientation`), i.e. **world-frame** z rotation |
| **Output format** | HDF5 (unchanged collector) |
| **Code & status** | Prototype written + validated this session; ~15 lines to land in the repo |
| **Paper** | Standard antipodal/principal-axis heuristic (Dex-Net 1.0 lineage) |
| **Success & quality** | **Measured:** 7/21 objects 0 → full success, 2 more partial, 4 more now lift reliably; control object 5/6 → 6/6 (no regression on round objects) |
| **Scalability** | Same as 1.1 (PCA over mesh verts is ~ms, once per reset) |
| **Integration** | **Plug-and-play** (subclass + one obs hook) |
| **Limitations** | Global PCA fails when the grasp region is *local* (ladle handle). Does not fix jaw-width or placement failures. |

### 1.3 Analytic antipodal grasp sampling on MuJoCo meshes

| Field | Details |
| :--- | :--- |
| **Name** | Antipodal / force-closure rejection sampling (Dex-Net 1.0 methodology) |
| **Type** | Motion Planning / analytic grasp synthesis |
| **Suite coverage** | **Directly adaptable** — mesh vertices + normals are already reachable via `sim.model.mesh_vert` / `mesh_normal`; no new dependency (`trimesh`/`open3d` are **not** installed in the `libero` env, but MuJoCo alone suffices) |
| **Zero human seed?** | **Strictly Yes** |
| **Action space** | Requires conversion — yields a 6-DoF grasp pose; convert to `OSC_POSE` deltas with the servo of Part 4 |
| **Output format** | Grasp poses only; HDF5 comes from the existing collector |
| **Code & status** | [Dex-Net](https://github.com/BerkeleyAutomation/dex-net) (archived, Python 2 era); [ACRONYM](https://github.com/NVlabs/acronym) dataset + sampler (maintained). Re-implementing the sampler in ~80 lines of NumPy is the pragmatic path. |
| **Paper** | Mahler et al., *Dex-Net 1.0*, ICRA 2016 ([pdf](https://goldberg.berkeley.edu/pubs/icra16-submitted-Dex-Net.pdf)); *Dex-Net 2.0*, RSS 2017 ([pdf](https://goldberg.berkeley.edu/pubs/dex-net-2.0-Camera-Ready-RSS-2017.pdf)); Eppner et al., *ACRONYM*, ICRA 2021 |
| **Success & quality** | Analytic force-closure metrics correlate well with physical success for parallel jaws; directly addresses failure mechanisms 2 and 4 above (rim grasps on bowls/pans, handle grasps on ladles/jugs) |
| **Scalability** | Offline, once per object mesh; ~0.1–1 s/object, cacheable across all 1,557 tasks |
| **Integration** | **Engineering wrapper required** (sampler + reachability filter + width filter ≤ 0.08 m) |
| **Limitations** | Needs surface normals and a friction assumption; grasps must additionally be filtered for Panda reachability and for the fixed top-down approach that the rest of the pipeline assumes. |

### 1.4 mplib

| Field | Details |
| :--- | :--- |
| **Name** | [MPlib](https://github.com/haosulab/MPlib) (Hao Su Lab) |
| **Type** | Motion Planning (OMPL-backed, SAPIEN-native) |
| **Suite coverage** | **Conceptual only → adaptable** — plans against its own URDF + collision world, *not* MuJoCo; scene geometry must be exported from `env.sim` |
| **Zero human seed?** | **Strictly Yes** |
| **Action space** | **Requires conversion** — outputs joint-space waypoints; needs joint → EEF → OSC delta conversion |
| **Output format** | Other (trajectory arrays) |
| **Code & status** | Active, pip-installable. **Not installed** in the `libero` env. |
| **Paper** | No standalone paper; ships with ManiSkill |
| **Success & quality** | High reliability for free-space transit; used to auto-generate all ManiSkill motion-planned demos |
| **Scalability** | ~0.05–0.5 s/plan |
| **Integration** | **Engineering wrapper required** — duplicate collision geometry from MuJoCo, plus the joint→OSC conversion |
| **Limitations** | Does not solve *grasp selection*, which is the actual bottleneck here. Adds a second kinematic model that can desync from MuJoCo. |

### 1.5 cuRobo

| Field | Details |
| :--- | :--- |
| **Name** | [cuRobo](https://github.com/NVlabs/curobo) — CUDA Accelerated Robot Library |
| **Type** | Motion Planning (GPU trajectory optimization) |
| **Suite coverage** | **Directly adaptable** — Franka Panda is a first-class supported robot; published MuJoCo integrations build a cuboid-approximated collision world from the MuJoCo scene |
| **Zero human seed?** | **Strictly Yes** |
| **Action space** | **Requires conversion** (joint trajectories → EEF → OSC deltas) |
| **Output format** | Other |
| **Code & status** | NVlabs, actively maintained. **Not installed**; needs CUDA toolchain. |
| **Paper** | Sundaralingam et al., *cuRobo: Parallelized Collision-Free Minimum-Jerk Robot Motion Generation*, ICRA 2023 ([arXiv:2310.17274](https://arxiv.org/abs/2310.17274), [report](https://curobo.org/reports/curobo_report.pdf)) |
| **Success & quality** | >7,000 collision-free IK queries/s; sub-5 ms MuJoCo round-trip reported; minimum-jerk (very smooth) trajectories |
| **Scalability** | Excellent — batched planning over many goal poses in parallel |
| **Integration** | **Engineering wrapper required** (URDF + sphere model + collision-world export) |
| **Limitations** | Heavy dependency (CUDA/Warp) for a task whose bottleneck is grasp geometry, not path search. The stockbg scene is a bare floor with 6 objects — free-space planning is nearly trivial here, so cuRobo's strength is largely wasted. |

### 1.6 OMPL

| Field | Details |
| :--- | :--- |
| **Name** | [OMPL](https://ompl.kavrakilab.org/) (Open Motion Planning Library) |
| **Type** | Motion Planning (sampling-based: RRT-Connect, BIT*, …) |
| **Suite coverage** | **Conceptual only** — no MuJoCo/robosuite binding; requires a custom state-validity checker calling `mj_forward` + contact checks |
| **Zero human seed?** | **Strictly Yes** |
| **Action space** | **Requires conversion** |
| **Output format** | Other |
| **Code & status** | Mature, maintained; awkward Python install. **Not installed.** |
| **Paper** | Şucan, Moll, Kavraki, *The Open Motion Planning Library*, IEEE RAM 2012 |
| **Success & quality** | Probabilistically complete; paths need post-hoc smoothing/shortcutting to be demo-quality |
| **Scalability** | ~0.1–1 s/plan |
| **Integration** | **Engineering wrapper required** (largest of the three planners) |
| **Limitations** | Raw OMPL paths are jerky and non-monotonic — poor imitation-learning targets without smoothing. Same misdiagnosis as 1.4/1.5: path search is not the failure mode. |

### 1.7 Differential IK / joint→OSC delta conversion utilities

| Field | Details |
| :--- | :--- |
| **Name** | robosuite `IK_POSE` controller; `robosuite.utils.transform_utils`; MuJoCo Jacobian (`mj_jacBody`) |
| **Type** | Motion Planning (kinematic utility) |
| **Suite coverage** | **Fully supports suite** — `transform_utils` is already imported by `collect_scripted_demonstrations.py` |
| **Zero human seed?** | **Strictly Yes** |
| **Action space** | **This *is* the conversion layer.** Verified: `OSC_POSE` `output_max = [0.05,0.05,0.05,0.5,0.5,0.5]`, `control_delta=True`, and `set_goal_orientation` computes `R_goal = R_delta @ R_current` ⇒ **rotation deltas are world-frame axis-angle**, position deltas are world-frame metres/0.05. |
| **Output format** | n/a |
| **Code & status** | Installed (robosuite 1.4.0) |
| **Paper** | Zhu et al., *robosuite* ([arXiv:2009.12293](https://arxiv.org/abs/2009.12293)) |
| **Success & quality** | Exact; this is the ground truth for Part 4's snippet |
| **Scalability** | Free |
| **Integration** | **Plug-and-play** |
| **Limitations** | `IK_POSE` needs PyBullet (**not installed**). Prefer OSC + a P-servo, which is what the repo already does and what I validated. |

---

## Category 2 — Autonomous RL Policy Training & Rollout

### 2.1 RL-from-scratch experts (SAC / PPO / TD3) on the suite

| Field | Details |
| :--- | :--- |
| **Name** | Zero-human RL expert training on LIBERO envs |
| **Type** | RL Expert |
| **Suite coverage** | **Directly adaptable** (envs are gym-like) but see limitations |
| **Zero human seed?** | **Strictly Yes** |
| **Action space** | **Direct 7-D `OSC_POSE`** (native env action space) |
| **Output format** | Requires a rollout-logging wrapper → HDF5 |
| **Code & status** | stable-baselines3 / rl_games / robomimic; none wired to LIBERO in this repo |
| **Paper** | Haarnoja et al. SAC (ICML 2018); Schulman et al. PPO (2017) |
| **Success & quality** | **The blocker: LIBERO exposes only a sparse binary BDDL predicate as reward.** Sparse-reward SAC/PPO on 7-DoF contact-rich pick-place is a well-known near-impossible exploration problem without shaping, HER, or demos. |
| **Scalability** | **Terrible for this use case: ~1,417 tasks × (hours–days of training each).** Even at 1 GPU-hour/task that is ≥1,400 GPU-hours before a single demo is written. |
| **Integration** | **Engineering wrapper required** — plus per-task dense reward authoring, which is itself the hard part |
| **Limitations** | Would need hand-written dense rewards per object (defeating "algorithmic"), or LLM-authored rewards (Eureka-style), adding another failure surface. RL trajectories are also typically jittery — poor BC targets. **Not recommended**, given a 4-second scripted rollout already solves ~54–75% of objects. |

### 2.2 Automated policy-rollout → HDF5 pipelines

| Field | Details |
| :--- | :--- |
| **Name** | robosuite `DataCollectionWrapper` + `scripts/collect_scripted_demonstrations.py`; robomimic rollout utils |
| **Type** | RL Expert / infrastructure |
| **Suite coverage** | **Fully supports suite** |
| **Zero human seed?** | **Strictly Yes** — policy-agnostic |
| **Action space** | **Direct 7-D** |
| **Output format** | **HDF5 (robomimic/LIBERO)** — writes `data/demo_N/{states,actions}` + `model_file` and the `env`/`env_info`/`problem_info`/`bddl_file_name`/`bddl_file_content` attrs, then `scripts/create_dataset.py --use-camera-obs` renders images |
| **Code & status** | In-repo, working; robomimic 0.2.0 installed |
| **Paper** | Mandlekar et al., *robomimic* ([arXiv:2108.03298](https://arxiv.org/abs/2108.03298)) |
| **Success & quality** | Filters on `env._check_success()` so only successful episodes are written; drops the trailing state to keep `len(states)==len(actions)` |
| **Scalability** | Parallel across CPU workers via `scripts/run_scale_pipeline.py`; render stage GPU-serialized |
| **Integration** | **Plug-and-play** |
| **Limitations** | Rendering dominates wall-clock and disk. Env constructor itself samples a placement, so `RandomizationError` must be retried at *build* time, not just reset time. |

---

## Category 3 — Procedural Solvers & Symbolic Goal Parsers (BDDL / TAMP)

### 3.1 In-repo BDDL parser + predicate checker

| Field | Details |
| :--- | :--- |
| **Name** | `libero/libero/envs/bddl_utils.py` (`get_problem_info`), `env.obj_of_interest`, `env._check_success()` |
| **Type** | TAMP (symbolic goal parsing) |
| **Suite coverage** | **Fully supports suite** |
| **Zero human seed?** | **Strictly Yes** |
| **Action space** | n/a (specification layer) |
| **Output format** | n/a |
| **Code & status** | In-repo, working |
| **Paper** | Liu et al., *LIBERO*, NeurIPS 2023 ([pdf](https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/liu_zhu_NeurIPS2023.pdf)) |
| **Success & quality** | The `(:obj_of_interest ...)` list gives `[manipuland, receptacle]` for free — the entire "symbolic → subtask" step this suite needs is a **two-element list lookup**. The goal is a single `(In X basket_1_contain_region)` predicate. |
| **Scalability** | Free |
| **Integration** | **Plug-and-play** |
| **Limitations** | Trivially sufficient *here*; would not scale to multi-step or chained goals. |

### 3.2 PDDLStream / pybullet-planning

| Field | Details |
| :--- | :--- |
| **Name** | [PDDLStream](https://github.com/caelan/pddlstream), [pybullet-planning](https://github.com/caelan/pybullet-planning) |
| **Type** | TAMP |
| **Suite coverage** | **Conceptual only** — PyBullet-based; would need the whole LIBERO scene re-expressed in PyBullet |
| **Zero human seed?** | **Strictly Yes** |
| **Action space** | **Requires conversion** |
| **Output format** | Other |
| **Code & status** | Maintained by Caelan Garrett; a documented fork is on PyPI as `pybullet_planning`. **PyBullet not installed.** |
| **Paper** | Garrett, Lozano-Pérez, Kaelbling, *PDDLStream*, ICAPS 2020 |
| **Success & quality** | Sound and complete for the standard pick-and-place domain |
| **Scalability** | Seconds per plan; heavier than a state machine |
| **Integration** | **Engineering wrapper required (large)** — a second simulator |
| **Limitations** | **Massive overkill.** This suite has a 1-action symbolic plan (`pick` → `place`) that is already given by `obj_of_interest`. TAMP solves task *sequencing*, which is not a problem here. |

### 3.3 OPTIMUS

| Field | Details |
| :--- | :--- |
| **Name** | [OPTIMUS](https://mihdalal.github.io/optimus/) — Imitating Task and Motion Planning with Visuomotor Transformers |
| **Type** | TAMP → imitation data generation |
| **Suite coverage** | **Conceptual only** (design pattern, not a drop-in) |
| **Zero human seed?** | **Strictly Yes** — the headline claim is explicitly replacing human supervision with a TAMP supervisor |
| **Action space** | Requires conversion |
| **Output format** | Other (its own pipeline), HDF5-adjacent |
| **Code & status** | [NVlabs/Optimus](https://github.com/NVlabs/Optimus) — **training/eval code only**, the TAMP data generator is not the released part |
| **Paper** | Dalal, Mandlekar et al., *Imitating Task and Motion Planning with Visuomotor Transformers*, CoRL 2023 ([arXiv:2305.16309](https://arxiv.org/abs/2305.16309)) |
| **Success & quality** | 70–80% success across 300+ tasks and up to 72 objects, trained purely on TAMP-generated data — **the strongest published evidence that zero-human demo generation at object scale works** |
| **Scalability** | Large-scale by design |
| **Integration** | **Engineering wrapper required** — best used as validation of the approach, not as code |
| **Limitations** | Released repo doesn't give you the generator. Its TAMP stack assumes known object models — true here, which is why the simpler in-repo route suffices. |

### 3.4 RLBench (design pattern)

| Field | Details |
| :--- | :--- |
| **Name** | [RLBench](https://github.com/stepjam/RLBench) |
| **Type** | Procedural Script + Motion Planning |
| **Suite coverage** | **Conceptual only** — CoppeliaSim/PyRep, not MuJoCo |
| **Zero human seed?** | **Strictly Yes** — *"an infinite supply of demonstrations through the use of motion planners operating on a series of waypoints given during task creation time"* |
| **Action space** | Incompatible (its own EEF-pose action modes) |
| **Output format** | Other |
| **Code & status** | Maintained |
| **Paper** | James et al., *RLBench*, RA-L 2020 ([arXiv:1909.12271](https://arxiv.org/abs/1909.12271)) |
| **Success & quality** | 100 tasks, all demos machine-generated, guaranteed collision-free |
| **Scalability** | Unbounded |
| **Integration** | Not integrable — **borrow the architecture** (hand-authored waypoints per task + planner between them), which is exactly what the in-repo phase machine already is |
| **Limitations** | Waypoints are hand-placed *per task* by the task author. For 1,557 auto-generated tasks you need waypoints derived from geometry — i.e. the grasp synthesis problem again. |

### 3.5 Meta-World scripted policies

| Field | Details |
| :--- | :--- |
| **Name** | [Meta-World](https://github.com/Farama-Foundation/Metaworld) `policies/` |
| **Type** | Procedural Script |
| **Suite coverage** | **Conceptual only** |
| **Zero human seed?** | **Strictly Yes** |
| **Action space** | Incompatible (Sawyer, 4-D `[dx,dy,dz,grip]` position control) |
| **Output format** | Other |
| **Code & status** | Maintained (Farama) |
| **Paper** | Yu et al., *Meta-World*, CoRL 2019 ([arXiv:1910.10897](https://arxiv.org/abs/1910.10897)) |
| **Success & quality** | Scripted oracle policies for all 50 tasks; commonly used to emit thousands of episodes (e.g. 4,000 episodes / 350K steps) |
| **Scalability** | Very fast |
| **Integration** | **Borrow the pattern only** — `p_gain * (target - current)` phase machines, identical in spirit to `ScriptedPickPlacePolicy` |
| **Limitations** | Every policy is hand-written per task against known object geometry, with a fixed gripper yaw — it inherits the *same* elongated-object blind spot found here. |

### 3.6 ManiSkill motion-planning solutions

| Field | Details |
| :--- | :--- |
| **Name** | [`mani_skill/examples/motionplanning/panda`](https://github.com/haosulab/ManiSkill/tree/main/mani_skill/examples/motionplanning/panda) |
| **Type** | Motion Planning + Procedural Script |
| **Suite coverage** | **Conceptual only → directly adaptable logic** — SAPIEN, not MuJoCo, but **the robot is the same Panda** |
| **Zero human seed?** | **Strictly Yes** — the canonical "generate demos without teleoperation" reference implementation |
| **Action space** | **Requires conversion** — emits joint/EEF-pose targets; ManiSkill's `pd_ee_delta_pose` is conceptually close to `OSC_POSE` |
| **Output format** | `.h5` trajectories (ManiSkill schema, not robomimic) |
| **Code & status** | Actively maintained; `python -m mani_skill.examples.motionplanning.panda.run -e PickCube-v1` |
| **Paper** | Tao et al., *ManiSkill3*, RSS 2025 ([arXiv:2410.00425](https://arxiv.org/abs/2410.00425)) |
| **Success & quality** | High on primitive-shape tasks (`PickCube`, `StackCube`); grasp poses are computed from **known primitive geometry**, so it does not transfer to 1,557 arbitrary Objaverse/AIGen meshes without a grasp sampler |
| **Scalability** | Seconds per demo; docs note the approach suits "simple manipulation tasks" |
| **Integration** | **Engineering wrapper required** (different sim, different action space, different HDF5 schema) |
| **Limitations** | Same core gap: its grasp poses are analytic for cubes, not learned/sampled for meshes. |

---

## Category 4 — LLM & Foundation-Model-Guided Scripting

### 4.1 Code-as-Policies

| Field | Details |
| :--- | :--- |
| **Name** | Code as Policies (CaP) |
| **Type** | LLM |
| **Suite coverage** | **Conceptual only** |
| **Zero human seed?** | **Strictly Yes** (needs a hand-written primitive API, not demos) |
| **Action space** | Requires conversion (emits calls to your primitives) |
| **Output format** | Other |
| **Code & status** | [google-research/code-as-policies](https://github.com/google-research/google-research/tree/master/code_as_policies); reference-quality, colab-oriented |
| **Paper** | Liang et al., *Code as Policies*, ICRA 2023 ([arXiv:2209.07753](https://arxiv.org/abs/2209.07753)) |
| **Success & quality** | Strong at *composing* primitives; the quality ceiling is set entirely by the primitives you supply |
| **Scalability** | An LLM call per task — 1,557 calls, and the generated code needs validation |
| **Integration** | Engineering wrapper required |
| **Limitations** | **Solves the wrong layer.** All 1,557 tasks share *one* instruction template ("pick up X, place in basket"). There is no compositional variety for an LLM to resolve — the hard part is per-mesh grasp geometry, which an LLM cannot see. |

### 4.2 VoxPoser

| Field | Details |
| :--- | :--- |
| **Name** | [VoxPoser](https://github.com/huangwl18/VoxPoser) |
| **Type** | LLM + model-based planning |
| **Suite coverage** | **Conceptual only** |
| **Zero human seed?** | **Strictly Yes** — zero-shot, no robot data, no LLM training |
| **Action space** | Requires conversion (synthesizes EEF trajectories) |
| **Output format** | Other |
| **Code & status** | Released; research-grade, RLBench-oriented |
| **Paper** | Huang et al., *VoxPoser*, CoRL 2023 ([arXiv:2307.05973](https://arxiv.org/abs/2307.05973)) |
| **Success & quality** | Composes 3-D affordance/value maps from an LLM+VLM, then optimizes trajectories — genuinely zero-shot, but reported *"only for relatively simple manipulations"*, dependent on careful prompt engineering, heavy test-time compute |
| **Scalability** | Poor — VLM + voxel-map optimization per episode |
| **Integration** | Engineering wrapper required (large) |
| **Limitations** | Value-map grasping is far less reliable than analytic antipodal sampling *when you already have the exact mesh*, which you do. Wrong tool for a full-state simulator. |

### 4.3 RoboGen

| Field | Details |
| :--- | :--- |
| **Name** | [RoboGen](https://github.com/Genesis-Embodied-AI/RoboGen) |
| **Type** | LLM + RL + Motion Planning (generative simulation) |
| **Suite coverage** | **Conceptual only** — generates *its own* tasks and scenes; here the tasks already exist |
| **Zero human seed?** | **Strictly Yes** — *"minimal human involvement beyond several prompt designs and in-context examples"* |
| **Action space** | Requires conversion |
| **Output format** | Other |
| **Code & status** | Released |
| **Paper** | Wang et al., *RoboGen*, ICML 2024 ([arXiv:2311.01455](https://arxiv.org/abs/2311.01455)) |
| **Success & quality** | Delivers "a continuous stream of diversified skill demonstrations"; the LLM authors *reward functions* which RL then optimizes |
| **Scalability** | Expensive (RL per skill) |
| **Integration** | Engineering wrapper required (large) |
| **Limitations** | Its value is *task proposal*, which this suite does not need — the 1,557 BDDLs are already generated. Using RoboGen here means paying for RL you don't need. |

### 4.4 GenSim / GenSim2

| Field | Details |
| :--- | :--- |
| **Name** | [GenSim](https://github.com/liruiw/GenSim), [GenSim2](https://github.com/GenSim2/GenSim2) |
| **Type** | LLM + Procedural Script / RL solvers |
| **Suite coverage** | **Conceptual only** (Ravens/CLIPort-based) |
| **Zero human seed?** | **Strictly Yes** — LLM writes the task code *and* the oracle policy |
| **Action space** | Incompatible (Ravens uses 2-pose pick-and-place primitives, not 20 Hz 7-D deltas) |
| **Output format** | Other |
| **Code & status** | Both released and maintained |
| **Paper** | Wang et al., *GenSim*, ICLR 2024 ([arXiv:2310.01361](https://arxiv.org/abs/2310.01361)); *GenSim2*, CoRL 2024 ([arXiv:2410.03645](https://arxiv.org/abs/2410.03645)) |
| **Success & quality** | GenSim2 scales to ~100 articulated tasks / 200 objects with planning + RL solvers that "generalize within object categories" |
| **Scalability** | Good |
| **Integration** | Engineering wrapper required (large) |
| **Limitations** | The Ravens action abstraction hides exactly the 20 Hz continuous-control detail that LIBERO demos must contain. The *category-generalizing solver* idea is the transferable insight. |

### 4.5 BLAZER

| Field | Details |
| :--- | :--- |
| **Name** | BLAZER — Bootstrapping LLM-based Manipulation Agents with Zero-Shot Data Generation |
| **Type** | LLM / Synthetic |
| **Suite coverage** | **Conceptual only** |
| **Zero human seed?** | **Strictly Yes** |
| **Action space** | Requires conversion |
| **Output format** | Other |
| **Code & status** | Recent; see [arXiv:2510.08572](https://arxiv.org/html/2510.08572) |
| **Paper** | *BLAZER*, 2025 |
| **Success & quality** | Self-bootstraps: LLM writes manipulation code, successful zero-shot rollouts become the training set |
| **Scalability** | LLM-call bound |
| **Integration** | Engineering wrapper required |
| **Limitations** | Same layer mismatch as 4.1. Worth noting as the "LLM proposes, simulator verifies" pattern — but the repo's `grasp_frac` grid search already *is* a propose-and-verify loop, at zero LLM cost. |

---

## Category 5 — Algorithmic Data Synthesis

### 5.1 MimicGen

| Field | Details |
| :--- | :--- |
| **Name** | [MimicGen](https://github.com/NVlabs/mimicgen) |
| **Type** | Synthetic (SE(3) trajectory retargeting) |
| **Suite coverage** | **Directly adaptable** — robosuite/MuJoCo native, robomimic HDF5 native, same simulator family |
| **Zero human seed?** | **No as published (requires human seeds)** — *"adapts some human-collected source demonstrations to novel object configurations"*. **But**: the algorithm only consumes *object-centric subtask segments*; nothing in the transform is human-specific. **Seeding it with scripted trajectories from §1.1/1.2 makes the combined pipeline strictly zero-human.** |
| **Action space** | **Direct 7-D `OSC_POSE`** — MimicGen's canonical robosuite setting |
| **Output format** | **HDF5 (robomimic)** — native |
| **Code & status** | NVlabs, maintained. **Not installed** in the `libero` env. |
| **Paper** | Mandlekar et al., *MimicGen*, CoRL 2023 ([arXiv:2310.17596](https://arxiv.org/abs/2310.17596)) |
| **Success & quality** | 50K+ demos across 18 tasks from ~200 source demos; RoboCasa (source of these very objects) is built on it |
| **Scalability** | Very high — replay + transform is cheaper than re-planning |
| **Integration** | **Engineering wrapper required** (task-spec + subtask segmentation per task) |
| **Limitations** | **Critical caveat for this suite:** MimicGen transforms a grasp pose *rigidly* into the new object's frame. That is valid across *poses of the same object*, not across *different meshes*. Since stockbg's only variation is the object **identity** (the layout is fixed and the target region already varies via 50 init states), MimicGen's spatial-transform axis is the one axis this suite does **not** vary. Low marginal value here. |

### 5.2 SkillMimicGen / DexMimicGen

| Field | Details |
| :--- | :--- |
| **Name** | SkillMimicGen (SkillGen), DexMimicGen |
| **Type** | Synthetic |
| **Suite coverage** | Conceptual only (bimanual/dexterous focus for DexMimicGen) |
| **Zero human seed?** | **No (requires human seeds)** — DexMimicGen: 20,000+ demos from 60 human source demos |
| **Action space** | Requires conversion (DexMimicGen: bimanual/dexterous hands) |
| **Output format** | HDF5 (robomimic-family) |
| **Code & status** | [DexMimicGen](https://dexmimicgen.github.io/) released |
| **Paper** | *SkillMimicGen* ([arXiv:2410.18907](https://arxiv.org/abs/2410.18907)); *DexMimicGen* ([arXiv:2410.24185](https://arxiv.org/abs/2410.24185)) |
| **Success & quality** | Strong amplification factors; SkillGen adds motion-planned transit between skill segments — that planner-between-skills idea is directly reusable |
| **Scalability** | High |
| **Integration** | Engineering wrapper required |
| **Limitations** | Single-arm Panda parallel jaw here makes DexMimicGen irrelevant; both still need seeds. |

### 5.3 DemoGen

| Field | Details |
| :--- | :--- |
| **Name** | [DemoGen](https://github.com/TEA-Lab/DemoGen) |
| **Type** | Synthetic (TAMP-style action adaptation + 3-D point-cloud editing) |
| **Suite coverage** | **Conceptual only** — point-cloud observation modality, not LIBERO's RGB+low-dim |
| **Zero human seed?** | **No (requires 1 human seed per task)** — headline is "one human demonstration → hundreds of synthetic demos" |
| **Action space** | Requires conversion |
| **Output format** | Other (point-cloud datasets) |
| **Code & status** | Released, RSS 2025 |
| **Paper** | Xue et al., *DemoGen*, RSS 2025 ([arXiv:2502.16932](https://arxiv.org/abs/2502.16932)) |
| **Success & quality** | Hundreds of spatially-augmented demos in seconds; ~20× reduction in human effort — reduction, **not elimination** |
| **Scalability** | Seconds per batch |
| **Integration** | Engineering wrapper required (large) |
| **Limitations** | Same objection as MimicGen — augments *spatial* configuration, which is not this suite's axis of variation. And the "1 human demo" would have to be replaced by a scripted seed anyway. |

### 5.4 LIBERO-Gen (Behavior Prompting Policy)

| Field | Details |
| :--- | :--- |
| **Name** | LIBERO-Gen |
| **Type** | Procedural Script + replay |
| **Suite coverage** | **Directly adaptable** (LIBERO-native, BDDL-native) — the closest published work to this task |
| **Zero human seed?** | **No (requires human seeds).** Verified from the paper: *"From the existing teleoperation data, we extract object-relative grasp poses for each of the target objects"* and *"many tasks also involve replaying a portion of teleoperation data and then switching to a scripted policy."* |
| **Action space** | **Direct 7-D `OSC_POSE`** |
| **Output format** | **HDF5 (LIBERO)** |
| **Code & status** | **No public release stated in the paper** |
| **Paper** | *Behavior Prompting Policy: Demonstrations as Prompts for Manipulation* ([arXiv:2606.30457](https://arxiv.org/html/2606.30457v1), [site](https://behavior-prompting.github.io/)) |
| **Success & quality** | 174-task and 321-task generated suites; validates every demo with LIBERO's predicate checker — same gate this repo uses |
| **Scalability** | Good |
| **Integration** | Would be plug-and-play **if released** and **if you had human demos** — you have neither |
| **Limitations** | **Disqualified by the core constraint.** Its grasp poses come from human teleop. Instructive as confirmation that *scripted policy + predicate validation* is the right architecture; its human-seeded grasp-pose extraction is exactly the component that §1.3 (analytic antipodal sampling) replaces algorithmically. |

### 5.5 Init-state resampling / trajectory perturbation (in-repo)

| Field | Details |
| :--- | :--- |
| **Name** | `scripts/create_suite_init_states.py` (50 init states/task) + policy-knob randomization |
| **Type** | Synthetic (zero-seed) |
| **Suite coverage** | **Fully supports suite** — `.pruned_init` files already exist for all 1,557 tasks |
| **Zero human seed?** | **Strictly Yes** |
| **Action space** | **Direct 7-D** |
| **Output format** | **HDF5** |
| **Code & status** | In-repo, working |
| **Paper** | n/a |
| **Success & quality** | Every rollout starts from a different sampled placement, so demos are diverse by construction — this is *already* the diversity mechanism, and it is why MimicGen adds little |
| **Scalability** | Free |
| **Integration** | **Plug-and-play** |
| **Limitations** | Diversity is bounded by the BDDL region ranges; no grasp-strategy diversity unless you also randomize `grasp_frac`/yaw/approach. |

---

## Category 6 — Codebase Inspection & Provenance (this repo)

| Utility | Purpose | Zero-human? | Status |
| :--- | :--- | :--- | :--- |
| `libero_scripted_policy.py` | 8-phase top-down pick-place state machine; reads only `<obj>_pos`, `<tgt>_pos`, `robot0_eef_pos` (+ injected `<obj>_bottom_z`/`_top_z`) | **Yes** | Working; **no wrist rotation** — the key defect |
| `scripts/collect_scripted_demonstrations.py` | Headless collector; `DataCollectionWrapper`; `grasp_frac` auto-calibration grid; success-filtered HDF5; `--result-json` gate | **Yes** | Working |
| `scripts/run_scale_pipeline.py` | Orchestrator: convert → BDDL → **parallel gated collect** → serialized GPU render → report; resumable | **Yes** | Working; already executed over 382 keys |
| `scripts/create_dataset.py` | Re-renders low-dim demos into image HDF5 (`--use-camera-obs --compress`) | **Yes** | Working |
| `scripts/create_suite_init_states.py` | 50 init states/task → `.pruned_init` | **Yes** | Done for all 1,557 |
| `scripts/smoke_check_suite.py` | Physics feasibility gate (placement / teleport-goal-fires / settle) | **Yes** | Produced the 1,417-task feasible set |
| `scripts/tag_suite_exclusions.py` | Writes `excluded:true` into the manifest; emits `exclusion_report.json`, `feasible_task_ids.txt` | **Yes** | Done |
| `scripts/verify_stockbg_suite.py` | Suite integrity check | **Yes** | Working |
| `scripts/render_demo_videos.py` | Renders collected demos to video for eyeballing | **Yes** | Working |
| `scripts/check_dataset_integrity.py` / `get_dataset_info.py` | HDF5 validation | **Yes** | Working |
| `libero/libero/envs/bddl_utils.py` | BDDL parser → `problem_name`, `language_instruction`, `obj_of_interest` | **Yes** | Working |
| `scripts/collect_demonstration.py` | **Human** SpaceMouse teleop (upstream LIBERO) | **No** | Present but **must not be used** |
| `tools/convert_robocasa_xml.py` | RoboCasa/Objaverse mesh → LIBERO object XML | **Yes** | Working (must run under a robosuite-enabled Python) |

**Provenance conclusion:** the repo already contains a complete, working, zero-human generation pipeline. Nothing
needs to be built from scratch — one component (grasp pose synthesis) is underpowered, and that is the whole gap.
Upstream LIBERO's own demos, by contrast, are **human SpaceMouse teleop at 20 Hz** — there is no zero-human
generator to inherit from upstream.

---

# Part 2: Comparative Master Table

| Name | Category | Zero-Human-Seed? | Native 7D `OSC_POSE`? | Direct LIBERO Fit? | Key Limitation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`ScriptedPickPlacePolicy` (in-repo)** | Procedural Script | **Yes** | **Yes** | **Native** | Fixed wrist ⇒ 41% of objects yield 0 demos |
| **Yaw-aligned PCA extension** | Procedural Script | **Yes** | **Yes** | **Native** | Global PCA fails on locally-graspable shapes (ladle) |
| **Antipodal grasp sampling (Dex-Net/ACRONYM)** | Motion Planning | **Yes** | Convert | Adaptable (MuJoCo meshes in hand) | Needs reachability + ≤0.08 m width filtering |
| mplib | Motion Planning | **Yes** | Convert | Wrapper (SAPIEN→MuJoCo) | Doesn't solve grasp selection |
| cuRobo | Motion Planning | **Yes** | Convert | Wrapper (CUDA) | Heavy dep; path search isn't the bottleneck |
| OMPL | Motion Planning | **Yes** | Convert | Wrapper (large) | Jerky paths need smoothing |
| robosuite IK / `transform_utils` | Kinematic utility | **Yes** | **Is the conversion layer** | **Native** | `IK_POSE` needs PyBullet (absent) |
| RL from scratch (SAC/PPO/TD3) | RL Expert | **Yes** | **Yes** | Adaptable | Sparse BDDL reward ⇒ ≥1,400 GPU-hours, jittery demos |
| `DataCollectionWrapper` + collector | Infrastructure | **Yes** | **Yes** | **Native** | Rendering dominates cost |
| BDDL parser + `_check_success()` | TAMP | **Yes** | n/a | **Native** | Trivial here (1-step goal) |
| PDDLStream / pybullet-planning | TAMP | **Yes** | Convert | Wrapper (2nd simulator) | Massive overkill for a 1-action plan |
| OPTIMUS | TAMP | **Yes** | Convert | Conceptual | Data generator not released |
| RLBench | Procedural + Planning | **Yes** | Incompatible | Conceptual | Waypoints hand-authored per task |
| Meta-World scripted policies | Procedural Script | **Yes** | Incompatible (4-D, Sawyer) | Conceptual | Same fixed-yaw blind spot |
| ManiSkill motionplanning/panda | Motion Planning | **Yes** | Convert | Conceptual | Grasps analytic for primitives only |
| Code-as-Policies | LLM | **Yes** | Convert | Conceptual | Wrong layer — no task variety to resolve |
| VoxPoser | LLM | **Yes** | Convert | Conceptual | Slow, prompt-sensitive, "simple manipulations" |
| RoboGen | LLM + RL | **Yes** | Convert | Conceptual | Pays for task proposal you don't need |
| GenSim / GenSim2 | LLM + Script | **Yes** | Incompatible (Ravens) | Conceptual | 2-pose abstraction hides 20 Hz control |
| BLAZER | LLM / Synthetic | **Yes** | Convert | Conceptual | Propose-and-verify already exists in-repo |
| **MimicGen** | Synthetic | **No** (but scripted seeds work) | **Yes** | **Adaptable (robosuite-native)** | Transforms *pose*, not *mesh identity* — wrong axis here |
| SkillMimicGen / DexMimicGen | Synthetic | **No** | Convert | Conceptual | Needs human seeds; dexterous focus |
| DemoGen | Synthetic | **No** (1 human demo) | Convert | Conceptual | Point-cloud modality; spatial axis only |
| **LIBERO-Gen** | Procedural + replay | **No** — grasp poses from teleop | **Yes** | **Native (if released)** | Unreleased **and** human-seeded |
| Init-state resampling (in-repo) | Synthetic | **Yes** | **Yes** | **Native** | Diversity bounded by BDDL ranges |

---

# Part 3: Practical Recommendation

**Recommendation: do not adopt an external framework. Fix the grasp-synthesis component of the pipeline this
repo already has.**

The evidence is unusually clear-cut:

* The repo's generator is already zero-human, already emits robomimic/LIBERO HDF5, already runs at
  ~3.5 s/rollout headless, and already has a parallel orchestrator and a success gate.
* Its failure is **not** stochastic, **not** a planning failure, and **not** a controller failure. It is one
  missing degree of freedom: **the wrist never rotates**, so on every elongated object the fingers close along
  the long axis (measured: 81–90° yaw error at grasp, 0 lifts).
* A ~15-line yaw servo took 7 of 21 measured zero-demo objects to full success and 4 more to reliable lifting,
  with no regression on the round-object control.
* Every external candidate either (a) requires human seeds (MimicGen, DemoGen, LIBERO-Gen, SkillMimicGen),
  (b) solves task *sequencing* or *proposal*, which this single-template suite does not need (PDDLStream,
  RoboGen, GenSim, Code-as-Policies), or (c) solves *path search*, which on a bare floor with 6 objects is
  already trivial (OMPL, cuRobo, mplib).

**The recommended pipeline — "Geometry-Aware Scripted Generation":**

```
BDDL (:obj_of_interest) ──► manipuland + receptacle names
        │
        ▼
MuJoCo mesh of manipuland ──► analytic grasp candidates
        │                     (Stage 1: minor-axis PCA yaw   — fixes ~44%+ of failures)
        │                     (Stage 2: antipodal sampling   — fixes ladle/bowl-rim class)
        │                     filter: jaw width ≤ 0.08 m, top-down reachable
        ▼
8-phase state machine + P-servo ──► 7-D OSC_POSE deltas @ 20 Hz
        │
        ▼
env._check_success()  ──► keep only successes
        │
        ▼
DataCollectionWrapper ──► demo.hdf5 (robomimic/LIBERO)
        │
        ▼
create_dataset.py --use-camera-obs --compress ──► image HDF5
```

Staged, with honest expected coverage of the 1,417 feasible tasks:

| Stage | Change | Effort | Expected object coverage |
| :--- | :--- | :--- | :--- |
| 0 | Re-run existing collector over all 1,417 (1,089 never attempted) | Zero new code | ~54% (measured baseline) |
| **1** | **Minor-axis yaw servo** | **~15 lines** | **~70–75%** |
| 2 | Width pre-filter (reject minor extent > 0.075 m) → auto-exclude instead of burning compute | ~10 lines | Same coverage, ~25% less wasted compute, honest exclusion list |
| 3 | Antipodal grasp sampling + retry over top-k candidates | ~80–150 lines | ~80–85% (recovers ladle/handle/rim class) |
| 4 | Place-orientation control (align long axis to basket) for long objects | ~20 lines | Recovers the baguette/celery/rolling_pin/wine class |

Stage 1 alone is the highest return-on-effort action available and should be done first. Stages 2–4 are
independent and can be sequenced by measured need.

**Explicitly not recommended:** RL-from-scratch (≥1,400 GPU-hours against a sparse predicate, for demos that
would be jerkier than a P-servo's), and any LLM-in-the-loop generator (1,557 tasks share one instruction
template — there is nothing for an LLM to disambiguate, and it cannot see mesh geometry).

---

# Part 4: Technical Roadmap & Code Snippet

## 4.1 Verified control-layer facts

These are read from the installed robosuite 1.4.0, not assumed:

| Fact | Value | Source |
| :--- | :--- | :--- |
| Position scale | `output_max[:3] = 0.05` m at full deflection | `controllers/config/osc_pose.json` |
| Rotation scale | `output_max[3:6] = 0.5` rad at full deflection | same |
| Delta mode | `control_delta: true` | same |
| Rotation composition | `R_goal = R_delta @ R_current` ⇒ **world-frame** axis-angle | `control_utils.set_goal_orientation` |
| Finger axis | Panda fingers slide on gripper-local **y**, 0.04 m each | `panda_gripper.xml` |
| **Max jaw opening** | **0.08 m** | same |
| Gripper convention | `+1 = close`, `-1 = open` | `libero_scripted_policy.py` |
| Control rate | 20 Hz | `collect_scripted_demonstrations.py` |

## 4.2 Ground-truth extraction → 7-D relative action deltas

```python
import numpy as np
from robosuite.utils import transform_utils as T

POS_SCALE, ROT_SCALE = 0.05, 0.5   # OSC_POSE output_max (metres, radians)
MAX_JAW = 0.08                     # Panda parallel-jaw opening
OPEN, CLOSE = -1.0, +1.0


# ---------- 1. ground-truth geometry straight out of env.sim ------------------
def object_xy_axes(sim, obj_name):
    """World-frame horizontal principal axes of the object, from its mesh vertices.

    Returns (major, minor, ext_major, ext_minor). `minor` is the direction the
    gripper fingers must close along; `ext_minor` is the width they must span.
    """
    m, d = sim.model, sim.data
    bids = set(_body_subtree(m, f"{obj_name}_main"))
    pts = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] not in bids:
            continue
        gpos, gmat = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3)
        if m.geom_type[g] == 7:                       # mjGEOM_MESH
            mid = m.geom_dataid[g]
            v0, nv = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            V = m.mesh_vert[v0:v0 + nv].reshape(-1, 3)
        else:                                          # primitive -> AABB corners
            s = m.geom_size[g]
            V = np.array([[sx*s[0], sy*s[1], sz*s[2]]
                          for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
        pts.append(gpos + V @ gmat.T)                  # local -> world

    P = np.vstack(pts)[:, :2]
    P -= P.mean(0)
    _, _, Vt = np.linalg.svd(P, full_matrices=False)   # PCA on the xy point cloud
    major, minor = Vt[0], Vt[1]
    span = lambda ax: float((P @ ax).max() - (P @ ax).min())
    return major, minor, span(major), span(minor)


def graspable(ext_minor, margin=0.005):
    """Reject before burning a rollout: no top-down parallel-jaw grasp can exist."""
    return ext_minor <= MAX_JAW - margin


# ---------- 2. the 7-D OSC_POSE conversion -----------------------------------
def wrap_half_pi(a):
    """A parallel jaw is symmetric under a 180 deg flip -> wrap to (-pi/2, pi/2]."""
    return (a + np.pi / 2) % np.pi - np.pi / 2


def finger_axis_yaw(obs):
    """World heading of the gripper's local y-axis (the finger-closing direction)."""
    R = T.quat2mat(np.asarray(obs["robot0_eef_quat"], float))
    return float(np.arctan2(R[1, 1], R[0, 1]))


def osc_action(obs, target_xyz, target_yaw, grip,
               pos_gain=20.0, yaw_gain=2.0):
    """Ground-truth pose -> 7-D relative action [dx,dy,dz,dax,day,daz,gripper].

    Position: world-frame error, P-controlled, normalised by the 0.05 m output scale.
    Rotation: robosuite composes as R_goal = R_delta @ R_current, so a[5] is a
              *world-frame* z rotation -- exactly what a top-down yaw needs.
    """
    a = np.zeros(7, dtype=float)

    pos_err = np.asarray(target_xyz, float) - np.asarray(obs["robot0_eef_pos"], float)
    a[:3] = np.clip(pos_err * pos_gain, -1.0, 1.0)     # (implicitly / POS_SCALE via gain)

    yaw_err = wrap_half_pi(target_yaw - finger_axis_yaw(obs))
    a[5] = np.clip(yaw_err * yaw_gain / ROT_SCALE, -1.0, 1.0)

    a[6] = grip                                        # +1 close / -1 open
    return a


# ---------- 3. phase machine (drop-in over the existing 8 phases) ------------
def plan_grasp(sim, obs, obj_name, grasp_frac=0.5):
    """Everything the state machine needs, derived purely from simulator state."""
    major, minor, ext_maj, ext_min = object_xy_axes(sim, obj_name)
    if not graspable(ext_min):
        return None                                    # auto-exclude, don't waste rollouts

    pos = np.asarray(obs[f"{obj_name}_pos"], float)
    bz, tz = obs[f"{obj_name}_bottom_z"], obs[f"{obj_name}_top_z"]   # injected live extents
    return dict(
        grasp_xyz=np.array([pos[0], pos[1], bz + grasp_frac * (tz - bz)]),
        grasp_yaw=wrap_half_pi(np.arctan2(minor[1], minor[0])),      # fingers along minor axis
        place_yaw=wrap_half_pi(np.arctan2(major[1], major[0])),      # long axis into the basket
    )


# ---------- 4. rollout -> HDF5 (unchanged: the repo already does this) -------
# env = DataCollectionWrapper(env, tmp_dir)
#   ... step the phase machine with osc_action(...) ...
#   keep the episode iff env._check_success()
#   gather_successful_demos_as_hdf5(...)  ->  data/demo_N/{states,actions} + model_file
#   then: scripts/create_dataset.py --use-camera-obs --compress
```

`_body_subtree(model, root_name)` is a 6-line BFS over `model.body_parentid`; the full working version
(including the diagnostic harness that produced the Part 0 table) is at
`…/scratchpad/yaw_experiment.py`.

## 4.3 Integration points in this repo

| Step | File | Change |
| :--- | :--- | :--- |
| 1 | `libero_scripted_policy.py` | Add `yaw_target` + set `a[5]` in every phase (hold through carry, switch to `place_yaw` at `LOWER`) |
| 2 | `scripts/collect_scripted_demonstrations.py` | Call `plan_grasp()` after each `safe_reset`, next to the existing `inject_obj_extents()`; skip the task early when `graspable()` is False and record it in `--result-json` |
| 3 | `scripts/run_scale_pipeline.py` | Unchanged — re-run the `collect` stage over the 1,417 feasible task IDs |
| 4 | `scripts/tag_suite_exclusions.py` | Add a `jaw_width` exclusion reason so width-infeasible tasks leave an auditable trail |
| 5 | — | Re-render with `scripts/create_dataset.py`; validate with `scripts/check_dataset_integrity.py` |

**Two gotchas that will bite (both already known to this repo, both hit during this session's experiments):**

1. The robosuite env **constructor** runs one placement sample internally, so `RandomizationError` must be
   retried at **build** time, not only around `env.reset()`.
2. `tools/convert_robocasa_xml.py` must run under a Python with robosuite installed — texture paths resolve
   against robosuite's package directory.

---

## Sources

- [LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning](https://www.cs.utexas.edu/~pstone/Papers/bib2html-links/liu_zhu_NeurIPS2023.pdf)
- [MimicGen: A Data Generation System for Scalable Robot Learning using Human Demonstrations](https://arxiv.org/pdf/2310.17596)
- [SkillMimicGen](https://arxiv.org/pdf/2410.18907) · [DexMimicGen](https://dexmimicgen.github.io/)
- [DemoGen: Synthetic Demonstration Generation for Data-Efficient Visuomotor Policy Learning](https://arxiv.org/html/2502.16932v1) · [code](https://github.com/TEA-Lab/DemoGen)
- [Behavior Prompting Policy (LIBERO-Gen)](https://arxiv.org/html/2606.30457v1) · [site](https://behavior-prompting.github.io/)
- [Imitating Task and Motion Planning with Visuomotor Transformers (OPTIMUS)](https://arxiv.org/abs/2305.16309) · [code](https://github.com/NVlabs/Optimus) · [site](https://mihdalal.github.io/optimus/)
- [RLBench: The Robot Learning Benchmark & Learning Environment](https://arxiv.org/pdf/1909.12271)
- [ManiSkill3](https://arxiv.org/pdf/2410.00425) · [Motion Planning docs](https://maniskill.readthedocs.io/en/latest/user_guide/data_collection/motionplanning.html) · [MPlib](https://github.com/haosulab/MPlib)
- [cuRobo: Parallelized Collision-Free Minimum-Jerk Robot Motion Generation](https://ar5iv.labs.arxiv.org/html/2310.17274) · [report](https://curobo.org/reports/curobo_report.pdf) · [site](https://nvlabs.github.io/curobo/)
- [Dex-Net 1.0](https://goldberg.berkeley.edu/pubs/icra16-submitted-Dex-Net.pdf) · [Dex-Net 2.0](https://goldberg.berkeley.edu/pubs/dex-net-2.0-Camera-Ready-RSS-2017.pdf)
- [VoxPoser](https://arxiv.org/abs/2307.05973) · [code](https://github.com/huangwl18/VoxPoser)
- [RoboGen](https://arxiv.org/abs/2311.01455) · [site](https://robogen-ai.github.io/)
- [GenSim](https://arxiv.org/pdf/2310.01361) · [GenSim2](https://arxiv.org/abs/2410.03645)
- [BLAZER](https://arxiv.org/html/2510.08572)
- [pybullet-planning / PDDLStream](https://github.com/caelan/pybullet-planning)
- [Meta-World](https://github.com/rlworkgroup/metaworld)
