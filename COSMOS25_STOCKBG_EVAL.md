# Running Cosmos-Predict2.5-2B on `libero_object_unseen_stockbg` (another machine)

End-to-end guide to evaluate **`nvidia/Cosmos-Predict2.5-2B`** (`robot/policy/libero`) on the
**`libero_object_unseen_stockbg`** suite (stock LIBERO-Object scenes, target-only swapped to RoboCasa
objects; **1,417 feasible / 1,557 tasks**) on a fresh machine, under the same protocol as the π0.5 /
π0-FAST runs (10 trials/task, seed 7, deterministic).

The one-command driver is **`scripts/run_cosmos25_stockbg.sh`** — everything below is what it needs in
place first. Steps 6–7 (the Reason1 cache and the suite data) are the two that most often bite.

---

## 0. What you get

- Results JSON at `$OUT_DIR/<subdir>/cosmos-predict2.5-2b_libero_object_unseen_stockbg_trials10_seed7.json`
  — identical schema to the π0.5/π0-FAST results, consumable by `scripts/aggregate_unseen_eval.py`.
- Runtime ≈ **3–3.5 min/task** on a 24 GB GPU → ~70 GPU-hours for all 1,417 tasks; split across GPUs
  (see the two-lane example in the script header).

## 1. Machine prerequisites

- **1 NVIDIA GPU ≥ 24 GB** for eval (bf16 policy fits 24 GB). The Reason1 warm step (step 6, option B)
  transiently needs **~17 GB** for the Reason1-7B encoder — do it on a free GPU or just copy the pkl.
- CUDA driver compatible with **torch 2.7 + cu128**.
- `uv` (astral), `git`, and a conda install (only to source a prebuilt `egl_probe` — see step 4).
- **Hugging Face access** to the gated repos (accept the licenses in a browser first):
  `nvidia/Cosmos-Predict2.5-2B`, `nvidia/Cosmos-Policy-LIBERO-Predict2-2B`,
  `nvidia/LIBERO-Cosmos-Policy`, plus the Reason1 encoder/Qwen/VAE repos its checkpoint_db resolves.

## 2. Repos

```bash
export LIBERO_REPO=/path/to/LIBERO           # THIS fork, branch robocasa-unseen-objects
export COSMOS_REPO=/path/to/cosmos-predict2.5
git clone <your LIBERO fork> "$LIBERO_REPO"    # must contain the stockbg suite + registration
git clone <cosmos-predict2.5>  "$COSMOS_REPO"
```

The LIBERO fork must include:
- `libero/libero/benchmark/__init__.py` with the `LIBERO_OBJECT_UNSEEN_STOCKBG` registration
  (`_load_manifest_suite("libero_object_unseen_stockbg")` + the `_ManifestBenchmark` subclass);
- the generated suite data (step 5).

## 3. Build the Cosmos venv

```bash
cd "$COSMOS_REPO"
uv sync --extra cu128 --group libero --python 3.10 \
  --no-install-package libero --no-install-package hf-egl-probe --no-install-package egl-probe
```
(PyPI `libero` is HF's re-upload — we use the fork instead; the `egl-probe` variants fail to build
without EGL headers.)

## 4. Three post-sync fixups (⚠ re-doing `uv sync` wipes all three — redo them each time)

Let `SP=$COSMOS_REPO/.venv/lib/python3.10/site-packages`.

1. **Vendored `egl_probe`** — copy a *prebuilt* `egl_probe` package into the venv (it won't compile
   here). Source it from a conda env where it already built:
   ```bash
   cp -r /path/to/conda/envs/libero/lib/python3.8/site-packages/egl_probe "$SP"/
   ```
2. **Wire the LIBERO fork** — the editable install alone does **not** get picked up; add an explicit
   `.pth` too:
   ```bash
   "$COSMOS_REPO"/.venv/bin/uv pip install -e "$LIBERO_REPO" --no-deps
   echo "$LIBERO_REPO" > "$SP"/libero_fork.pth
   ```
3. **robosuite file-logging** — this shared-host crash-guard:
   ```bash
   echo "FILE_LOGGING_LEVEL = None" >> "$SP"/robosuite/macros_private.py
   ```

## 5. LIBERO suite data + path config

The eval loads the suite via `benchmark.get_benchmark_dict()["libero_object_unseen_stockbg"]()`, which
reads **BDDLs from `get_libero_path("bddl_files")`** and **init states from `get_libero_path("init_states")`**.
Those paths come from `~/.libero/config.yaml` — point them at the fork:

```yaml
# ~/.libero/config.yaml
benchmark_root: /path/to/LIBERO/libero/libero
bddl_files:     /path/to/LIBERO/libero/libero/bddl_files
init_states:    /path/to/LIBERO/libero/libero/init_files
assets:         /path/to/LIBERO/libero/libero/assets
datasets:       /path/to/LIBERO/libero/libero/../datasets   # unused by eval
```

Then get the suite data (BDDLs 8.8 MB, init states 80 MB). **Copy is far cheaper than regenerating**
(init-state baking is ~1.5 GPU-h):

```bash
# from the machine that built the suite:
rsync -a $SRC/libero/libero/bddl_files/libero_object_unseen_stockbg  "$LIBERO_REPO"/libero/libero/bddl_files/
rsync -a $SRC/libero/libero/init_files/libero_object_unseen_stockbg  "$LIBERO_REPO"/libero/libero/init_files/
```

*Fallback — regenerate from scratch* (needs the `libero` conda env, several GPU-hours):
```bash
python scripts/create_libero_object_stockbg_tasks.py --out-dir libero/libero/bddl_files/libero_object_unseen_stockbg
python scripts/verify_stockbg_suite.py --manifest libero/libero/bddl_files/libero_object_unseen_stockbg/manifest.json
# then create_suite_init_states.py (sharded) -> smoke_check_suite.py -> tag_suite_exclusions.py
```

Sanity check (in the cosmos venv):
```bash
"$COSMOS_REPO"/.venv/bin/python -c "
from libero.libero.benchmark import get_benchmark
b=get_benchmark('libero_object_unseen_stockbg')(0); print(b.get_num_tasks(),'tasks'); print(b.get_task_init_states(0).shape)"
# -> 1557 tasks ; (50, 110)
```

## 6. Reason1 text-embedding cache (OOM-critical)

The eval computes Reason1 embeddings on the fly for any instruction **missing** from the cache pkl,
which loads the **Reason1-7B encoder (~16 GB) next to the policy → OOM on 24 GB**. Pre-populate it.

`libero_object_unseen_stockbg` uses category-level language identical to the full suite — **the same 110
instructions** — so the existing `reason1_unseen_embeddings.pkl` already covers it (verified: stockbg
language set == full suite language set).

- **Option A (copy, ~11 GB):**
  ```bash
  rsync -a $SRC/cosmos-predict2.5/t5_cache/reason1_unseen_embeddings.pkl "$COSMOS_REPO"/t5_cache/
  ```
- **Option B (warm on a free ≥17 GB GPU):**
  ```bash
  cd "$COSMOS_REPO"
  HF_HOME=$HF_HOME HF_TOKEN=$(cat ~/.cache/huggingface/token) CUDA_VISIBLE_DEVICES=<free-gpu> \
    .venv/bin/python warm_reason1_cache.py \
      --manifests "$LIBERO_REPO"/libero/libero/bddl_files/libero_object_unseen_stockbg/manifest.json \
      --out "$COSMOS_REPO"/t5_cache/reason1_unseen_embeddings.pkl
  ```

## 7. Eval patch — register the suite (2 lines; already in this fork's companion patch)

In `cosmos_predict2/_src/predict2/cosmos_policy/experiments/robot/libero/run_libero_eval.py`:
```python
class TaskSuite(str, Enum):
    ...
    LIBERO_OBJECT_UNSEEN_STOCKBG = "libero_object_unseen_stockbg"   # add

TASK_MAX_STEPS = {
    ...
    TaskSuite.LIBERO_OBJECT_UNSEEN_STOCKBG: 280,                    # add (same design as libero_object)
}
```
No unnorm change is needed: `check_unnorm_key` is dead code; unnormalization uses the flat
`--dataset_stats_path` file (suite-independent).

**Also required** — the vendored eval already carries the standing local patches from the earlier unseen
runs (the unseen enum entries, `--task_ids`, `--save_videos`, `--model_name`, incremental
`--results_out_path`, the serial `text_embeddings_kind` bugfix, the CPU Reason1 memo, and the
`RandomizationError` retry/skip in `run_libero_eval.py` + `libero_utils.py`). On a **fresh** cosmos clone
these are missing. Capture them from the source machine and apply:
```bash
# on source (if cosmos repo is git):
git -C $SRC/cosmos-predict2.5 diff > cosmos25_libero_eval.patch
# on target:
git -C "$COSMOS_REPO" apply cosmos25_libero_eval.patch
# else rsync the patched files directly:
#   .../robot/libero/run_libero_eval.py, .../robot/libero/libero_utils.py, .../cosmos_policy/**/cosmos_utils.py
```

## 8. Run

```bash
export HF_HOME=/path/to/hf_cache
export HF_TOKEN=$(cat ~/.cache/huggingface/token)   # Cosmos-Predict2.5-2B is gated; pass explicitly
export OUT_DIR=/path/to/eval_out

# single GPU, all 1,417 feasible tasks (runs a seen-suite control smoke first, aborts if it fails):
COSMOS_REPO=$COSMOS_REPO LIBERO_REPO=$LIBERO_REPO GPU=0 \
  bash "$LIBERO_REPO"/scripts/run_cosmos25_stockbg.sh

# or split across two GPUs by feasible-id parity:
GPU=0 RESULTS_SUBDIR=c0 LANE_PARITY=even bash "$LIBERO_REPO"/scripts/run_cosmos25_stockbg.sh &
GPU=1 RESULTS_SUBDIR=c1 LANE_PARITY=odd  bash "$LIBERO_REPO"/scripts/run_cosmos25_stockbg.sh &
```
Run it inside `tmux` if the session may disconnect — the job dies with its parent otherwise.

## 9. Aggregate

```bash
python scripts/aggregate_unseen_eval.py \
  --results "$OUT_DIR"/**/cosmos-predict2.5-2b_libero_object_unseen_stockbg_*.json \
  --manifest "$LIBERO_REPO"/libero/libero/bddl_files/libero_object_unseen_stockbg/manifest.json \
  --out-dir "$OUT_DIR"/summary
```

## 10. Gotchas (learned the hard way)

- **`--config_file` must be repo-RELATIVE** (`cosmos_predict2/_src/.../config/config.py`) — an absolute
  path breaks the path→module conversion in `get_config_helper`.
- **Reason1 cache miss = OOM.** Never start the eval without step 6 done; a single missing instruction
  pulls in the 7B encoder.
- **HF token isn't auto-picked-up** — `HF_HOME` differs from `~/.cache/huggingface`, so pass
  `HF_TOKEN=$(cat ~/.cache/huggingface/token)` explicitly (the script does this).
- **GPU selection**: `--available_gpus "0"` is fixed in the args; the *physical* GPU is chosen by
  `CUDA_VISIBLE_DEVICES` (the script sets it from `GPU=`). Don't pass a physical index to
  `--available_gpus`.
- **Deterministic-protocol clustering**: cosmos success clusters all-or-nothing within
  (category × layout_variant) cells and aliases onto any even/odd id split. Whole-suite totals are fine,
  but **stratify by (category, layout_variant) for any subset/split analysis** — an even/odd lane split
  will look "broken" when it isn't.
- **Resuming**: the results JSON is per-task incremental; re-run with the uncovered ids via `TASK_IDS=…`
  and a fresh `RESULTS_SUBDIR` (same filename overwrites; the aggregator unions `*.json`).
