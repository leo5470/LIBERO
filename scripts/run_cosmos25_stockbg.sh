#!/bin/bash
# Run Cosmos-Predict2.5-2B (robot/policy/libero) on the libero_object_unseen_stockbg suite.
# Portable lane script: everything machine-specific is an env var with a default. Mirrors the
# validated pi05/pi0_fast protocol (10 trials/task, seed 7, deterministic) and writes the same
# results-JSON schema consumed by scripts/aggregate_unseen_eval.py.
#
# See COSMOS25_STOCKBG_EVAL.md for full setup (venv, HF assets, Reason1 cache, the eval patch).
#
# Quick start (after setup):
#   COSMOS_REPO=/path/to/cosmos-predict2.5 LIBERO_REPO=/path/to/LIBERO \
#   HF_HOME=/path/to/hf_cache OUT_DIR=/path/to/eval_out GPU=0 \
#   bash scripts/run_cosmos25_stockbg.sh
#
# Split across two GPUs by feasible-id parity (each lane a disjoint half, distinct RESULTS_SUBDIR):
#   GPU=0 RESULTS_SUBDIR=c0 LANE_PARITY=even bash scripts/run_cosmos25_stockbg.sh &
#   GPU=1 RESULTS_SUBDIR=c1 LANE_PARITY=odd  bash scripts/run_cosmos25_stockbg.sh &
set -u

# ---- config (override via env) ---------------------------------------------------------------
COSMOS_REPO=${COSMOS_REPO:-/tmp2/leocheng/forks/cosmos-predict2.5}
LIBERO_REPO=${LIBERO_REPO:-/tmp2/leocheng/forks/LIBERO}
HF_HOME=${HF_HOME:-/tmp2/leocheng/hf_cache}
HF_TOKEN=${HF_TOKEN:-$(cat ~/.cache/huggingface/token 2>/dev/null)}
OUT_DIR=${OUT_DIR:-/tmp2/leocheng/eval_results/stockbg}
GPU=${GPU:-0}
SUITE=${SUITE:-libero_object_unseen_stockbg}
TRIALS=${TRIALS:-10}
SEED=${SEED:-7}
MODEL_NAME=${MODEL_NAME:-cosmos-predict2.5-2b}
# Reason1 embedding cache (must contain the suite's 110 category instructions; see the guide).
REASON1_PKL=${REASON1_PKL:-$COSMOS_REPO/t5_cache/reason1_unseen_embeddings.pkl}
# Stock LIBERO reason1 cache (for the seen-suite control smoke).
STOCK_PKL=${STOCK_PKL:-datasets/nvidia/LIBERO-Cosmos-Policy/success_only/reason1_embeddings.pkl}
# Task ids: default = the suite's feasible set; override with TASK_IDS or LANE_PARITY=even|odd.
FEASIBLE_FILE=$LIBERO_REPO/libero/libero/bddl_files/$SUITE/feasible_task_ids.txt
SKIP_SMOKE=${SKIP_SMOKE:-0}
RESULTS_SUBDIR=${RESULTS_SUBDIR:-}
# ---------------------------------------------------------------------------------------------

[ -n "$HF_TOKEN" ] || { echo "[err] HF_TOKEN empty (need HF access to gated nvidia/Cosmos-* repos)"; exit 1; }
[ -f "$REASON1_PKL" ] || { echo "[err] Reason1 cache missing: $REASON1_PKL  (warm or copy it — a cache miss OOMs the GPU)"; exit 1; }
[ -f "$FEASIBLE_FILE" ] || { echo "[err] feasible id file missing: $FEASIBLE_FILE"; exit 1; }

OUT=$OUT_DIR${RESULTS_SUBDIR:+/$RESULTS_SUBDIR}
VID=$OUT_DIR/videos$([ -n "$RESULTS_SUBDIR" ] && echo "_$RESULTS_SUBDIR")
mkdir -p "$OUT" "$OUT_DIR/logs" "$VID"
cd "$VID"   # rollout videos land in ./rollouts/<date> under cwd

export HF_HOME HF_TOKEN
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$GPU CUDA_VISIBLE_DEVICES=$GPU
PY=$COSMOS_REPO/.venv/bin/python
RUN="$PY -m cosmos_predict2._src.predict2.cosmos_policy.experiments.robot.libero.run_libero_eval"

# feature flags shared by every invocation (identical to the validated full-suite lane)
COMMON_ARGS=(
  --config cosmos_predict2p5_2b_480p_libero__inference_only_no_s3
  --ckpt_path nvidia/Cosmos-Predict2.5-2B/robot/policy/libero
  --config_file cosmos_predict2/_src/predict2/cosmos_policy/config/config.py   # MUST be repo-relative
  --use_wrist_image True --use_proprio True --normalize_proprio True --unnormalize_actions True
  --dataset_stats_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_dataset_statistics.json
  --text_embeddings_kind reason1
  --trained_with_image_aug True --chunk_size 16 --num_open_loop_steps 16
  --num_denoising_steps_action 5
  --seed "$SEED" --deterministic True --randomize_seed False
  --available_gpus "0"                # physical GPU is picked by CUDA_VISIBLE_DEVICES above
  --model_name "$MODEL_NAME"
  --local_log_dir "$OUT_DIR/logs/cosmos25_stockbg"
)

# ---- 1) control smoke: paper policy is near-ceiling on the seen suite; <1/2 = harness bug ----
if [ "$SKIP_SMOKE" != "1" ]; then
  $RUN "${COMMON_ARGS[@]}" \
    --t5_text_embeddings_path "$STOCK_PKL" \
    --task_suite_name libero_object --task_ids 0 --num_trials_per_task 2 \
    --save_videos all --results_out_path "$OUT_DIR/smoketest" \
    > "$OUT_DIR/logs/control_smoke.log" 2>&1
  OK=$($PY -c "import json;print(1 if json.load(open('$OUT_DIR/smoketest/${MODEL_NAME}_libero_object_trials2_seed${SEED}.json'))['total_successes']>=1 else 0)" 2>/dev/null || echo 0)
  [ "$OK" = "1" ] || { echo "[err] control smoke FAILED — aborting (see logs/control_smoke.log)"; exit 1; }
  echo "[stockbg] control smoke passed $(date)"
fi

# ---- 2) select task ids -----------------------------------------------------------------------
if [ -n "${TASK_IDS:-}" ]; then
  IDS=$TASK_IDS
else
  IDS=$(tr ',' '\n' < "$FEASIBLE_FILE" | tr -s ' \n' '\n' | grep -E '^[0-9]+$')
  case "${LANE_PARITY:-}" in
    even) IDS=$(echo "$IDS" | awk '$1%2==0');;
    odd)  IDS=$(echo "$IDS" | awk '$1%2==1');;
  esac
  IDS=$(echo "$IDS" | paste -sd, -)
fi
echo "[stockbg] $(echo "$IDS" | tr ',' '\n' | wc -l) task ids | GPU $GPU | out $OUT $(date)"

# ---- 3) run the suite -------------------------------------------------------------------------
$RUN "${COMMON_ARGS[@]}" \
  --t5_text_embeddings_path "$REASON1_PKL" \
  --task_suite_name "$SUITE" \
  --task_ids "$IDS" --num_trials_per_task "$TRIALS" \
  --save_videos failures \
  --results_out_path "$OUT" \
  > "$OUT_DIR/logs/cosmos25_stockbg$([ -n "$RESULTS_SUBDIR" ] && echo "_$RESULTS_SUBDIR").log" 2>&1
echo "[stockbg] done $(date) -> $OUT/${MODEL_NAME}_${SUITE}_trials${TRIALS}_seed${SEED}.json"
