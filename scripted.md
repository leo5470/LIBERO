# The scripted demonstration generator

A zero-human-data pipeline that produces expert pick-and-place demonstrations for
`libero_object_unseen_stockbg` (1,417 physics-feasible tasks over 110 RoboCasa object
categories). No teleoperation, no seed trajectories, no human video: grasps are synthesised
from simulator geometry and executed by a hand-written phase machine, and only episodes that
pass the benchmark's own `env._check_success()` are kept.

**Current state (canonical): 67,615 demonstrations across 1,379 tasks and 109 of 110
categories** (`/tmp2/leocheng/ricl_scratch/stockbg_v2`, ~13 GB, 0 corrupt files, contact
0/10). Best on every aggregate metric AND fixes the per-task yield regressions of earlier
runs (diverse ladder + simulator selection; §5). `stockbg_rim`/`stockbg_fix`/`stockbg_final`
are superseded scaffolding (see §7.1).

Every number in this document was measured on this machine. Where a claim was later found
wrong it is marked **[corrected]** with what replaced it — the mistakes are as informative as
the results.

---

## 1. Components

| File | Lines | Role |
| :--- | ---: | :--- |
| `libero_scripted_policy.py` | 334 | phase machine; emits 7-D OSC actions. **Obs-only** — no `sim` access |
| `libero_grasp_synthesis.py` | 519 | antipodal grasp sampler + ranking, reads `env.sim` geometry |
| `scripts/collect_scripted_demonstrations.py` | 531 | one task: build/reset retries, candidate ladder, calibration, HDF5 output |
| `scripts/collect_suite_demonstrations.py` | 314 | suite orchestration: parallel, resumable, per-category report, escalation |
| `scripts/create_stockbg_train_variants.py` | 168 | per-object scene variants (slot permutation + distractor redraw) |
| `scripts/render_suite_videos.py` | 134 | batch mp4 spot-checks over a collection root |
| `scripts/render_demo_videos.py` | — | single-file renderer (`build_env` reused by the batch one) |

The policy is deliberately **obs-only**: all geometry reaches it through injected observation
keys, each optional, each falling back to prior behaviour when absent. That keeps it usable
as a plain robomimic-style policy and makes every feature independently switchable.

Injected keys: `<obj>_bottom_z`, `<obj>_top_z`, `<obj>_grasp_xyz`, `<obj>_grasp_yaw`,
`<obj>_grasp_width`, `<target>_rim_z`.

---

## 2. Embodiment facts (verified, not assumed)

- **Action**: 7-D `[dx, dy, dz, dax, day, daz, gripper]`, `OSC_POSE`, 20 Hz.
  `output_max = [0.05]*3 + [0.5]*3`.
- **Rotation is world-frame**: `control_utils.set_goal_orientation` composes
  `R_goal = R_delta @ R_current`, so `a[5]` is a world-z yaw — exactly what a top-down wrist
  needs. Confirmed by reading the controller, not inferred.
- **Fingers close along the gripper's local y-axis** (`panda_gripper.xml`, both finger joints
  on `axis="0 1 0"`, 0.04 m each) ⇒ **0.08 m maximum opening**; measured 0.078 m live.
- **The gripper action is a sign-based accumulator**, not a position command
  (`PandaGripper.format_action`, speed 0.01/step). `sign(0) == 0` freezes it, so an
  intermediate opening can be *servo'd* from `obs["robot0_gripper_qpos"]` but never commanded
  directly.
- **`In` is a point test on the object's body origin.** `contain_region` is a *site*; its
  `check_contact` returns `True` unconditionally (`base_object_states.py:172`) and
  `SiteObject.in_box` (`site_object.py:37`) explicitly "treat[s] the object as a point".
  Measured half-extents `[0.061, 0.061, 0.0695]`.
- **The basket drops ~4.2 cm into the floor at step 2 of every episode**, before the robot
  moves — universal, not robot-caused. So the containment box top is **0.137**, not the
  0.176 it measures at spawn. **[corrected]** I first attributed this sink to the wine
  bottle's weight; it happens with an idle arm too.
- Basket: 16.2 × 17.3 cm, rim top 0.183 at spawn (0.143 after settling), interior floor
  ≈0.036, **14.6 cm deep**.
- **Demos are collected in HEAVIER physics than eval runs. This is upstream LIBERO, not ours.**
  `DataCollectionWrapper._start_new_episode()` rebuilds the sim on *every* reset —
  `sim.model.get_xml()` → `reset_from_xml_string()` → `sim.reset()` → restore state, robosuite's
  "trick for ensuring that we can play MuJoCo demonstrations back deterministically".
  `model.get_xml()` is MuJoCo's `mj_saveLastXML`, and it is **lossy**: robosuite assembles a
  79,301-char task XML, MuJoCo re-serializes it to 72,541 (`scale=` 40→39, `solref=` 98→96,
  `friction=` 98→96, `condim=` 3→2, `group=` 180→73). Recompiling that shorter string gives a
  **different model**. Verified directly — compiling robosuite's task XML reproduces the live
  mass, compiling `get_xml()` output does not:

  | | manipuland (`boxed_drink__aigen_8`) | basket | stock distractors |
  | :--- | ---: | ---: | ---: |
  | plain bddl build (**what eval runs**) | 1.7 g | 136 g | — |
  | after the round-trip (**what demos are collected in**) | 20.3 g | 553 g | ~2× heavier |
  | ratio | **11.9×** | **4.06×** | 2.0–2.2× |

  Meshes, geom sizes and geom positions are untouched — objects are the same shape, just
  heavier, with inertia up to 6.8× and shifted `body_ipos`/`body_iquat`. The round-trip is
  idempotent after the first application. **Stock LIBERO-Object behaves identically** (milk
  31.7 g → 68.5 g, 2.16×; same basket 4.06×), so the original human demos were recorded in
  the heavy physics too and every LIBERO policy is trained on one physics and evaluated in
  another. We inherit that; we did not introduce it.

  How much it matters: on `boxed_drink__aigen_8` the same policy, grasp and seed give
  **10/10 wrapped and 0/10 unwrapped** — light objects skitter, so the jaw pre-shape hunts,
  closes on empty air and the arm carries nothing. Actions match bit-exactly at t=0 and
  diverge in the 4th decimal by t=1.

  Consequences: (1) **anything that re-steps recorded actions must build from the demo's
  recorded `model_file`** — `render_demo_videos.py` and `create_dataset.py` do
  (`reset_from_xml_string`), `render_suite_videos.py` inherits it; (2) pure state-playback
  analysis (`set_state_from_flattened` + `forward`, no stepping) is unaffected; (3) an A/B run
  unwrapped has valid *contrasts* but eval-physics absolute yields — 89.0% suite-weighted
  where the collector scores 93.4% against a recorded population mean of 92.3%. Pick the
  physics to match the question: `--roundtrip` to predict collection yield, plain to predict
  eval behaviour.

---

## 3. The policy

Phase machine, unchanged in structure since the first version:

```
APPROACH → DESCEND → GRASP → LIFT → MOVE → LOWER → DROP → SETTLE → done
```

Position control is P-only: `a[:3] = clip((target - eef) * 20.0, ±1)`. The object is carried
at a fixed high `carry_z` so it clears the rim and distractors, and only descends once
aligned over the target.

### 3.1 Yaw servo (the original defect and its fix)

The first version held `a[3:6] = 0` — **it never rotated the wrist**. Because LIBERO spawns
objects at a fixed yaw, grasp geometry is deterministic per object, which made yields
*bimodal rather than noisy*: over 382 objects, **46.3% hit 100% and 41.1% produced zero
demos**. Measured yaw error at grasp on every failing object: **81–90°** — the fingers closed
across the object's long axis.

Fix: servo `a[5]` so the gripper's local y-axis tracks the grasp's closing direction, in
**every** phase so alignment survives the carry.
`a[5] = clip(wrap_half_pi(grasp_yaw − finger_axis_yaw) · 2.0 / 0.5, ±1)`, wrapping to
(−π/2, π/2] because a parallel jaw is symmetric under a 180° flip. Took carrot, hot_dog,
corn, bar, croissant, eggplant and sponge from 0/5–0/6 to full success with no regression on
a round control.

### 3.2 Origin-centred placement

`In` tests the body origin, so centring the *end-effector* over the basket fails whenever the
grasp is off-centre — which is exactly what rim and handle grasps are. At `GRASP` the policy
latches `grasp_offset = obj_xy − eef_xy` and steers `target_xy − grasp_offset` through
MOVE/LOWER/DROP, so the **object origin** tracks the basket centre. This is what recovered
rolling_pin, celery and baguette, which previously lifted reliably and never placed.

### 3.3 Release above the rim

**Measured against the stock human demos** (`libero_object`, same basket, same two scenes):

| | EEF vs rim at release | Basket moved laterally | Arm–basket contact |
| :--- | :--- | ---: | :--- |
| stock human (teleop) | **+0.4 … +4.6 cm above** (median +2.1) | 0.0 cm | **0 steps, 10/10 demos** |
| ours, first run | **−3.3 cm below** | 0.2 – 39 cm | 82 steps median, **24/24 demos** |
| ours, after fix | +3.2 … +5.5 cm above | 0.0 – 1.2 cm | 0 steps, **0/20** |

Humans stop with the fingers just above the rim and let the object hang below the hand and
drop; tall objects still end deep inside (milk's base 3.8–9.0 cm below the rim) because it
dangles from the grasp. Targeting the *object's* floor clearance instead put our **hand**
below the rim and raked the basket on every episode — worst case the steak shoved it 39 cm.

Fix: clamp the LOWER target to `rim_z + release_above_rim` (default 0.02 m).
Cost: elongated objects lose yield (rolling_pin 65.7→24%, tongs 76.7→42%) because a gently
deposited long object beats a dropped one. Counter-intuitively a *smaller* margin is worse
(rolling_pin 3% @0.005, 9% @0.01, 16% @0.02, 24% @0.05); bigger margins help but leave the
stock range, so 0.02 is a deliberate trade-off.

**Gotcha**: the rim must be re-measured during the episode (`RIM_REFRESH_EVERY = 20` steps).
Caching it at reset is 4.2 cm too high (§2) and releases from +6 … +7.5 cm.

### 3.4 Jaw pre-shape

The jaw descends pre-shaped to `grasp_width + 2 × 0.012 m` rather than fully open, servo'd
closed-loop on `robot0_gripper_qpos`. This keeps the sampler's clearance model honest — the
two must stay in sync via `GS.pre_grasp_half_for`.

### 3.5 Transit height

**Measured against the stock human demos** (`libero_object`, same arena, floor plane at
z = 0), eef height above the table by *horizontal progress* toward the grasp — both start
from the same home pose and travel the same ~25 cm:

| | start | 10% | 50% | 90% | at grasp |
| :--- | ---: | ---: | ---: | ---: | ---: |
| stock human (teleop) | 25.8 | 25.5 | **25.4** | 24.2 | 4.9 |
| ours, stockbg_v2 | 26.1 | 24.2 | **16.4** | 14.6 | 4.8 |

The human holds its start altitude across the table and drops the last 20 cm vertically;
ours bled altitude the whole way and arrived low. Cause: `APPROACH` hovered at
`obj_origin_z + hover_height`, so the **transit height tracked the object's own height** —
13.8–19.4 cm across 23 categories, versus a flat 25.0 [22.7, 26.5] for the humans. That band
sits at the basket rim (14.0 cm) and below the tallest table object (orange juice, 19.1 cm),
and for anything taller than ~24 cm the rule inverts: `wine` is 27.0 cm tall with its origin
13.4 cm up, so the old hover put the hand **1.6 cm below the bottle's top** and swept through it.

Fix: `APPROACH` crosses at `table_z + transit_height`, where `table_z` is latched from the
manipuland's resting `bottom_z`, floored at `top_z + transit_clearance` so tall manipulands
are cleared. `DESCEND` then drops vertically onto the grasp, unchanged.

**Choosing the height — 60 tasks × 20 attempts, paired on identical initial states, grasp
pinned to each task's stored `chosen_entry`, so trajectory is the only variable.** Tasks are
stratified by their v2 collect-yield and re-weighted to the real suite composition (81% of
tasks sit in the ≥95% band, which is what makes the raw sample mean misleading).

⚠️ **The sweep below was measured in an *unwrapped* env, i.e. not the physics the collector
runs** (§2). Its contrasts are internally valid — both arms shared the physics — but its
absolute levels are pessimistic, and the ≥95 band it reports (94.0%) is 5.8 points below what
that band actually scores under the collector (99.8%). The corrected measurement follows.

| `transit_height` | achieved z@50% | ≥95 band | 50–95 | 20–50 | <20 | **suite-weighted** |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| *off (origin-relative)* | 16.9 cm | 94.0% | 73.9% | 37.5% | 0.4% | 86.8% |
| **0.24** | 21.3 | **95.4%** | 78.6% | 63.5% | 0.0% | **89.1%** (+2.3) |
| 0.26 | 23.2 | 91.0% | 80.0% | 77.5% | 2.7% | 86.1% (−0.7) |
| 0.28 | 24.9 | 90.6% | 81.4% | 64.0% | 0.0% | 85.6% (−1.2) |
| 0.30 | 27.0 | 89.8% | 83.9% | 78.0% | 5.5% | 85.7% (−1.0) |

**0.24 is the only value that does not cost the dominant band.** Going higher keeps buying
yield in the hard tail and keeps losing it in the ≥95 band, and because that band is 81% of
the suite the net turns negative — the raw sample mean says the opposite, which is the trap.
Decelerating the last 4 cm of the taller descent (tested at 0.30) does not recover it either:
per-task outcomes flip chaotically between nearby settings, the same "reliability is dynamic"
lesson as §5.2.

**Confirmed on 60 held-out tasks** (disjoint from the sweep above, different seed), because
0.24 was picked partly on how it landed in the ≥95 band of the tuning sample:

| sample | ≥95 | 50–95 | 20–50 | <20 | suite-weighted |
| :--- | ---: | ---: | ---: | ---: | ---: |
| tuning (60 tasks) | +1.5 | +4.6 | +26.0 | +0.4 | +2.3 [−0.9, +6.1] |
| held-out (60 tasks) | −1.0 | +6.4 | +18.0 | +2.5 | +0.5 [−3.9, +4.1] |
| **pooled (120 tasks)** | **+0.2** | **+5.5** | **+22.0** | **+1.5** | **+1.4 [−1.4, +4.0]** |

The tuning estimate was optimistic, as expected — the held-out ≥95 delta regresses to
roughly zero.

**Re-measured in the collector's physics (all 120 tasks, `ab_transit.py --roundtrip`), the
yield case all but disappears:**

| physics | baseline | with fix | ≥95 | 50–95 | 20–50 | <20 | weighted delta |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| unwrapped (wrong) | 89.0% | 90.3% | +0.2 | +5.5 | +22.0 | +1.5 | +1.4 [−1.5, +4.3] |
| **collector (correct)** | **93.4%** | **93.5%** | **−0.5** | **+0.5** | **+15.0** | **+5.6** | **+0.1 [−1.2, +1.4]** |

*(the correct baseline lands on the recorded population mean of 92.3%; the unwrapped one is
4.4 points low, which is what inflated every earlier delta)*

**The honest read: at the suite level this change is yield-NEUTRAL (+0.1). The dominant band
is saturated at 99.8% — there was never room to gain there — and the real gains are confined
to the hard tail (20–50 band +15.0, <20 band +5.6).** Paired McNemar over 2,400 rollouts still
favours it (gained 216, lost 131, p = 5.9e-6), but that count is dominated by tail tasks. The
justification for the change is therefore trajectory realism plus tail coverage, *not* suite
yield. One thing the measurement cannot capture in the fix's favour: the collector
re-calibrates the grasp under the new policy, whereas every number here pins v2's
`chosen_entry`.

⚠️ **Open: the 0.24 default was chosen on the unwrapped sweep, where the ≥95 band appeared to
fall off a cliff past 0.24 (94.0 → 91.0). Under the collector's physics that band sits at
99.8% and barely moves, so the cliff may be an artifact and taller transits — including 0.28,
which lands at the human 24.9 cm — may now be affordable.** Re-sweep 0.26/0.28/0.30 with
`--roundtrip` before treating 0.24 as final.

**Matching the humans exactly is not the optimum.** 0.28 lands at 24.9 cm, within a
millimetre of their 25.0 median — and costs 1.2 points. So 0.24 deliberately stops 3.7 cm
short of the reference distribution: what actually mattered was removing the *coupling* to
object height (13.8–19.4 cm, varying with the object → a flat 21.0–21.7 for everything),
not reproducing the absolute height.

Cost: +1.6 steps per episode (141.1 → 142.7 mean). Distractor disturbance is **unchanged**
(p90 = 0.09 cm in every condition) — this is not a collision fix.

### 3.6 Defaults

`hover_height 0.12 · transit_height 0.24 · transit_clearance 0.05 · grasp_frac 0.55 ·
lift_height 0.20 · place_drop 0.06 · basket_clear 0.02 · release_above_rim 0.02 ·
pre_grasp_clearance 0.012 · pos_gain 20 · xy_tol 0.012 · z_tol 0.015 · yaw_gain 2.0 ·
max_phase_steps 90`

`hover_height` is now only the fallback for when the live extents aren't injected.

Opt-outs restoring earlier behaviour exactly: `--no-align-yaw`, `--no-center-placement`,
`--no-rim-release`, `--no-preshape`, `--no-grasp-sampler`, `--no-high-transit`.

---

## 4. The grasp sampler

`sample_grasps` proposes antipodal top-down grasps from live `env.sim` geometry.

0. **Sample density** (`max_pairs_samples`, the single most important recall knob).
   The O(n²) pair search runs over `n` area-weighted surface samples. This was **300 and
   far too low**: a thin handle or a narrow ring is a tiny fraction of surface, so drawing
   two samples on *opposite* faces of it is negligibly likely and the candidate never
   exists to be tested. **[corrected]** I first attributed kettle/jug/teapot's zero
   candidates to the antipodal test and even rewrote the clearance model for nothing — the
   real cause was this constant. Raising it 300 → **3000** recovered them, mostly as
   *central* grasps: kettle_6 0→5, kettle_1 0→28 (1%→64% yield), teapot 0→37 (14%→70%),
   jug 0→11, steak_2 0→5. Cost 2–16 s once per task.
1. **Surface samples** — triangle centroids + normals, with mesh **subdivision to ≤2 cm
   edges**. Without it a 12-triangle cereal box yields a handful of samples and starves pair
   sampling; that was the original "no candidate" false negative on all 7 boxes.
2. **Width** in (0.004, 0.075) m, leaving 5 mm inside the jaw.
3. **Near-horizontal** closing direction (`|u_z| < 0.35`) — the wrist yaws, never tilts.
4. **Winding-agnostic antipodal test** (the single most expensive bug of the project):
   `|N₁·û| > cos(atan μ)` **and** `|N₂·û| > cos(atan μ)` **and** `(N₁·û)(N₂·û) < 0`, μ=0.5.
   These meshes have ~97% outward normals while `û` points *into* the object, so the textbook
   `N·û > cos θ` form silently returns **zero candidates on every object**. It produced a
   plausible-looking but meaningless first result set.
5. **Contacts at compatible heights** (<2 cm apart).
6. **Floor clearance**, `min(0.010, 0.35 · height)`. **[corrected]** A fixed 1 cm rejected
   *every* candidate on flat objects — measured **444 → 0 on pizza_cutter (0.9 cm tall)** and
   **393 → 0 on tongs (1.1 cm)**. Scaling with height restores 55 and 52.
7. **Finger-path clearance** — the descent column at the pre-grasp opening plus the closing
   sweep band, and a palm check. Models a real overhang, not curvature: the strict
   outward-projection form rejects an apple, a known 100%-yield object.
8. **Hold quality** — `object_com` (mass-weighted, from `body_mass`/`body_ipos`),
   `straddles_com` (is the COM *between* the fingers along û?) and `lever_m` (absolute xy
   distance midpoint→COM). Scored `+0.5` for straddling and `−4.0 · lever_m`; absolute, not
   normalised, because torque scales with distance.

`rank_grasps` interleaves central and side candidates 2:1 and **always appends a minor-axis
centroid pinch** so the retry ladder is never empty even when the sampler finds nothing.

---

## 5. How the policy decides what to do with each object

Nothing about the interaction is hard-coded per object or per category — there is no rule
table, no category lookup, no per-object tuning. The decision is made in two stages:
**geometry proposes, the simulator decides.**

### 5.1 Stage 1: geometry proposes a ranked ladder

At task start the sampler reads the object's actual mesh (§4) and enumerates candidate
finger-contact *pairs*. Each survivor is described by four numbers — where the fingers go,
which way they close, how wide the grip is, how high up the object it sits — and scored:

```
score = contact_alignment                       # how squarely opposed the two normals are
      + 0.5   if the COM lies between the fingers
      − 4.0 × lever_arm_to_COM (metres)
      − 0.3 × |height_fraction − 0.4|
      − 5.0 × height_mismatch_between_contacts
```

So the policy prefers grasps that are squarely opposed, that **command the object's mass**
rather than a distant appendage, that sit near mid-height, and that are level. The top
candidates are taken in a 2:1 central-to-side cadence — deliberately mixing strategies rather
than committing to one — and the guaranteed centroid pinch is appended at three heights
(0.5 / 0.65 / 0.8). Typical ladder: 4 sampled candidates + 3 fallbacks.

### 5.2 Stage 2: the simulator picks the winner

The ladder is a *hypothesis list*, not a decision. The collector runs a short warmup on each
entry in order: the first entry with a perfect warmup wins immediately, otherwise the
best-yielding one does, and the batch is then collected at that setting. This is what makes
the pipeline robust to the sampler being wrong — and it is wrong in both directions (§7, §10).
Only a grasp that actually works in physics is ever used.

The chosen grasp is then stored **relative to the object's pose** (a local offset plus a
relative yaw, `grasp_to_local`) and recomposed with the object's live pose on every step. That
is why it survives the ±5 cm per-reset placement jitter without re-sampling.

### 5.3 What execution does with the choice

The policy servos the end-effector to the candidate midpoint, rotates the wrist so the finger
axis matches the candidate's closing direction (§3.1), pre-shapes the jaw to that candidate's
width plus 1.2 cm (§3.4), and at `GRASP` latches the object-origin-to-hand offset so placement
steers the *object* rather than the hand (§3.2). Everything else is identical for every object
in the suite by design — same approach, carry height and release — so the data isolates
*object* generalization rather than motion variety.

### 5.4 The same machinery, different behaviours

| Object | What it settles on | Why |
| :--- | :--- | :--- |
| apple | central equatorial pinch, 5.4 cm | 276 central candidates, 0.1 cm lever |
| cereal box | short-side pinch, 5.0 cm at h≈0.83 | long side exceeds the jaw; short side does not |
| tongs | central pinch, 3.1 cm | only reachable once floor clearance scales with height (§4.6) |
| saucepan | 2.2 cm side grasp on the handle | all 7 candidates are side grasps; nothing central exists |
| bowl | rim pinch, 0.4 cm | 18 cm body admits nothing else; flagged 8.6 cm lever, no straddle |
| donut | fallback pinch at 8.0 cm | hole too narrow for a fingertip, body wider than the jaw |

### 5.5 What it deliberately does not consider

Three gaps, two of which cause measured failures:

- **The rest of the scene.** Scoring looks only at the target's own geometry, so a handle grasp
  can rank first while pointing straight at a neighbour — the saucepan failure, where the
  gripper strikes `alphabet_soup_1` in 7 of 7 failures (§8).
- **Dynamics.** `straddles_com` is a *static* proxy for hold quality; nothing simulates the
  forces of the carry, which is why handle grasps with 4–5 cm levers pass ranking and then drop
  the object mid-traverse (§6.3).
- **Memory across tasks.** Each object is solved from scratch. No learning, no transfer between
  instances of the same category, no reuse of a sibling's working grasp — even though categories
  like `kettle` have 25 instances that differ only slightly.

---

## 6. Collection

**Per task** (`collect_scripted_demonstrations.py`):
- **Build-time retries with a fresh seed per attempt** (`np.random.seed(seed + 7919·attempt)`).
  The constructor runs one placement sample itself, and reseeding matters — an unlucky RNG
  stream stays unlucky. This alone unblocked `pot` and `saucepan`, two of three singleton
  categories, which had been impossible to instantiate.
- Optional `--init-states` seeding from a stored pool for tasks whose sampler is too flaky.
- **Candidate ladder + calibration**: short warmup per ladder entry, first perfect warmup wins
  immediately, otherwise best-yielding; then collect at that setting.
- `_tmp_states` cleanup by default — measured 8–13 MB per task, i.e. 100–200 GB at suite
  scale against 158 GB free.

**Per suite** (`collect_suite_demonstrations.py`): `ThreadPoolExecutor`, per-task logs,
`result.json` resumability, `--per-category` pilots, `--select` filters, and an `--escalate`
pass over zero-demo *categories* (2× attempts, top-8 candidates, μ=0.8, fresh init pools).
Reports `categories_with_zero_demos` as the primary metric.

**Gotcha**: `feasible_task_ids.txt` holds comma-separated **manifest indices**, not names.

---

## 7. Results

### 7.1 Runs

| | `stockbg_rim` | `stockbg_fix` | `stockbg_final` | **`stockbg_v2`** |
| :--- | ---: | ---: | ---: | ---: |
| change | rim release | + density/floor/hold-quality | + top-k 8, 3-reset, timeout guard | + diverse ladder, sim-selection, incremental flush |
| demos | 65,007 | 66,329 | 67,151 | **67,615** |
| tasks with demos | 1,341 | 1,363 | 1,374 | **1,379** / 1,417 |
| categories | 108 | 109 | 109 | **109** / 110 |
| tasks at the full 50 | 1,260 | 1,287 | 1,318 | **1,336** |
| mean per-cat success | 81.5% | — | 86.0% | **86.9%** |
| arm–basket contact | 0/20 | 0/20 | 0/12 | **0/10** |
| corrupt files | 0 | 0 | 0 | **0** |

`stockbg_v2` is canonical. The `_final` -> `_v2` step was the hardest-won: `_final` fixed
coverage but its ranking baked a *reliability* prior into the geometric score
(`-4*lever + 0.5*straddle`), which is internally contradictory -- it must both favour and
disfavour a handle grasp depending on the object (a mug wants its handle, a pitcher wants
its body; only the simulator knows which slips). That regressed ~270 tasks' yield (mug 94->16%).
`_v2` demotes the score to **feasibility only**, builds a **diverse ladder** spanning
central/side x height, and lets **calibration pick by measured yield** -- so the simulator,
not a geometry guess, decides. mug recovered 16->98% choosing its handle; pitcher keeps its
body grasp. Collected in two passes (initial + a resumed re-collect of 139 tasks the first
1200s timeout had killed): the timeouts were slow low-yield tasks, not hangs, and the fix was
**incremental flush** (write demos as collected, atomic, + partial result.json) so a killed
task keeps its progress. 8 tasks remain unfinished, all in covered categories.

**`stockbg_final` is canonical** — best on every aggregate metric. Progression: the rim
release (§3.3) fixed the basket-raking and shortened episodes ~30%; the density fix (§4.0)
recovered the sampling-starved categories (donut +538 demos, jug/lobster/dumpling/pitcher
+108–121, kettle +77); top-k 8 + multi-reset (§5.2) removed the `stockbg_fix` regressions
(cup 11→100%, mug 6→86%, cucumber 0→26%) that came from a shallow ladder and single-reset
sampling. Per-category three-way in `scripted_comparison.csv`.

Residual per-task churn vs `stockbg_rim`: 16 tasks down ≥10, ~40 up ≥10; three dropped to 0
that a predecessor had ≥40 — `pizza_cutter_7` (thin disc, §8), `teapot_15` (the timeout),
`donut_14` (config picked a failing side grasp; 8+ other donut instances cover the
category). All are single instances in otherwise well-covered categories.

### 7.2 Success-rate distribution (run 2)

Per task — mean 86.6%, median 100%:

| Success rate | Tasks | Share |
| :--- | ---: | ---: |
| exactly 100% | 945 | 66.7% |
| 90–99% | 150 | 10.6% |
| 80–89% | 73 | 5.2% |
| 50–79% | 84 | 5.9% |
| 1–49% | 89 | 6.3% |
| 0% | 76 | 5.4% |

Per category — mean 81.6%, median 91.3%: 13 at exactly 100%, 43 at 90–99%, 26 at 75–89%,
14 at 60–74%, and **14 below 60%** which is where all the remaining work lives.
**72.7% of all demos come from perfect-yield tasks**; everything below 50% success
contributes under 4% of the dataset.

Note the 50-demo target conflates capability with budget: `bread__objaverse_21` banked 50 at
45.5% yield while `jug_wide_opening__objaverse_jug_3` stalled at 49 with 47% — the difference
is attempt luck, not skill. All 81 partials hit the attempt cap.

### 7.3 Failure taxonomy

From 108 instrumented failures across 11 weak objects:

| Mechanism | Share | Objects | Signature |
| :--- | ---: | :--- | :--- |
| carried 19–24 cm then lost | 60% | pitcher, jug_wide_opening, pan, kettle | ends 17–23 cm away, on the table |
| slips at 3–10 cm | 38% | pizza_cutter, lobster, donut, tongs | grasp never holds |
| never grasped | 2% | — | — |

**[corrected]** I first reported these as grasp failures. The vessels grasp and carry fine —
they fail *after* lifting. Only the flat/hollow group fails at the grasp.

Within the "carried then lost" group there are two sub-mechanisms: **mid-carry slip** (pitcher
23.5 cm away, jug 22.7, pan 17.0 — handle grasps with 4.1–4.8 cm levers to the COM that hold
at rest and let go under acceleration) and **doesn't fit** (pot rests on the rim with its base
0.3 cm below it, failing on z in 5/5 — the bowl mechanism).

---

## 8. Genuinely unachievable (verified, not merely unsolved)

Each was confirmed by placing the object into the basket **by hand**, settling the physics,
and evaluating the predicate with the box recomputed at the same instant — with an apple as a
passing control.

- **`bowl`** (1 instance, 17.9 cm wide): cannot enter a 16 cm opening; rests on the rim with
  its origin **1.9 cm above** the box top.
- **`wine`** (3 instances, 27 cm tall, origin 13.5 cm above its own base): an upright bottle
  lands centred with origin 0.152 against a box top of 0.137 — it misses by 1.6 cm against a
  target that sank underneath it. A **leaning** pose does satisfy `In` (origin 0.122), but
  four release mechanisms across ~150 rollouts never produced one: wrist tilt (rates
  0.08–0.6), pivot-about-base lean (35–100° commanded, 11–27° achieved), topple-nudge
  (2–17°), and higher release (lands upright every time). Root cause: the grasp is on the neck
  ~24 cm above the base, so the wrist has almost no angular authority — the bottle slips
  instead of rotating. Closed at the user's direction.
- **`donut`** (23 instances) **[corrected]**: I claimed my clearance test broke it, citing 683
  candidates in an earlier sweep. False. 275 candidates survive every other filter and are
  *correctly* rejected: the hole is **2.2 cm across** (min radius 1.1 cm), too narrow for a
  fingertip, and the body is 9.1 cm, beyond the jaw. The 683 were false positives of a laxer
  test. Relaxing the clearance model recovered **nothing**.
- **`pizza_cutter`** (7 instances, **0.9 cm thick**): a flat disc. The density fix gives it
  55–295 candidates, but the parallel jaw closes on a ~9 mm-tall edge with nothing to hold and
  the disc rotates out — every rollout slips (lift stalls at 4.7 cm). This is a "too thin to
  hold" limit, the vertical analogue of the bowl's "too wide". **[corrected]** I first
  diagnosed a floor-*reachability* block ("the eef bottoms out 3 cm above the grasp"); that
  3.6 cm is just the fixed **finger-length offset** — present on every grasp including apple
  and cereal, which lift fine. tongs (1.1 cm) succeeds at the *same* grasp height, so a
  height-based reachability filter would be both wrong and harmful. Left as-is; genuinely hard.

`In` systematically penalises **tall** objects, whose origin rides high even when genuinely
placed. This is a defect in the benchmark's success criterion, not the robot's behaviour — but
the same predicate scores evaluation, so demos of "genuinely placed but scored false" would
teach a behaviour that cannot score. That is why the collector's gate is kept rather than
overridden.

---

## 9. Open blockers

This table reflects understanding **before** the sample-density fix (§4.0). Kettle and
teapot are now largely **solved** by it (kettle_1 1%→64%, teapot 14%→70%); the numbers below
are pre-fix and will be re-measured by the pending full re-run.

| Family | Blocking filter | Instances | Status |
| :--- | :--- | ---: | :--- |
| kettle | ~~antipodal wipes 9,946 → 0~~ **sampling starvation** | 25 | **fixed** — density 300→3000 |
| teapot, jug | ~~antipodal/finger-path~~ **sampling starvation** | 41 | teapot fixed; jug improved, still weak |
| pitcher, pan, jug_wide_opening | only 0–1 candidates, all high-lever handle grasps | 13 | ranking cannot help without alternatives; mid-carry slip |
| saucepan | **gripper strikes a distractor** in 7/7 failures | 1 | sampler is blind to non-target objects |
| pizza_cutter | grasp slips off a 0.9 cm disc | 7 | moved to §8 — geometric, not a filter |

The saucepan case is worth isolating: its failures are perfectly separated by
grasp-to-distractor distance (**≤10.6 cm fails, ≥12.0 cm succeeds**) because the pan's spawn
position varies ±5 cm and its handle grasp reaches toward a neighbouring can. Only 6 of 108
failures elsewhere show this, so it is not the general mechanism — but the sampler scoring
*only* target geometry is a real design gap.

Also unresolved: saucepan measured 11/18 in the harness versus 16% recorded, while the harness
reproduced all 11 other objects faithfully. Treat both numbers as suspect until re-run.

---

## 10. Reproducing

```bash
# one task
python scripts/collect_scripted_demonstrations.py \
    --bddl-file libero/libero/bddl_files/libero_object_unseen_stockbg/<task>.bddl \
    --num-success 50 --max-attempts 100 --build-tries 40 --out-dir /tmp/demos/<task>

# 110-category pilot, then the full suite + escalation
python scripts/collect_suite_demonstrations.py --out-root <root> --per-category 1 --workers 16
python scripts/collect_suite_demonstrations.py --out-root <root> --workers 16 --escalate

# inspect one object's grasp ladder
python libero_grasp_synthesis.py --bddl-file <task>.bddl --top-k 6

# spot-check videos
python scripts/render_suite_videos.py --collect-root <root>/collect --select bowl,donut --per-task 1
```

Budget: **≈3.5–4 s per rollout per core**; the full suite is **9–11 h at 16 workers, ~12 GB**.
Collection is low-dim only and needs no GPU; rendering is a separate replay stage.

**A GitHub clone cannot reproduce any of this** — the ~13 GB converted RoboCasa/Objaverse
asset pool under `libero/libero/assets` is gitignored, so no simulation runs from the tracked
repo alone.

---

## 11. Method notes worth keeping

Four mistakes here produced confident, wrong conclusions. Each was caught by measuring rather
than reasoning:

1. **Textbook antipodal test, wrong winding assumption** → zero candidates everywhere, and a
   first result set that looked plausible.
2. **A box cached before the drop** → objects appeared "inside" the containment region when
   they were not. Recompute the box at the same instant as the object pose; the basket moves.
3. **Diagnosing by dropping objects from 15–20 cm** → they miss the basket or knock it, so
   feasibility looks impossible. Place at the intended resting pose with zero velocity.
4. **Attributing a universal artefact to a specific cause** → the 4.2 cm basket sink was
   blamed on the wine bottle's weight; it happens in every episode with an idle arm.
5. **A fixed geometric offset misread as a limit** → "the eef bottoms out 3 cm above the
   grasp" looked like a floor-reachability block; that 3.6 cm is the finger length, present on
   *every* grasp including the successes. Nearly built a "reachability filter" for a
   non-problem that would have retired the working `tongs`. Always check the suspected-broken
   signal against a known-good control before acting on it.

The recurring lesson: an existence proof from geometry (a valid grasp exists, a pose satisfies
the predicate) says nothing about reachability by the policy. Candidates exist on 98.9% of
objects that produced **zero** demos. The sampler maps *possibility*; only the simulator's
verdict maps probability.
