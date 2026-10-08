#!/usr/bin/env bash
# Inference runner for a trained checkpoint. Edit the constants below to
# change runs.

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$ROOT/scripts/_jsonl.sh"
RUN_NAME="${RUN_NAME:-timeomni_v-cls_all-lr1e-5-ep10}"
VARIANT=timeomni_v             # timeomni_v | baseline
FUSION_MODE=time_interleave # block_adjacent | time_interleave (timeomni_v only)
BATCH_SIZE=16               # fallback per-rank batch size; ignored when DYN_BS_MAX_TOKENS>0
# Token-budget batching mirrors training. Padded-token measure (max_len * bs);
# set to 0 to fall back to a fixed BATCH_SIZE.
DYN_BS_MAX_TOKENS=${DYN_BS_MAX_TOKENS:-45000}
DYN_BS_MAX_BS=${DYN_BS_MAX_BS:-50}
# Generation cap. infer.py's default (4) only covers single-letter classification;
# image `prediction` answers reach ~1286 Qwen2.5-Omni tokens (terra). 1408 covers
# the longest sample with ~10% margin. Override per-dataset if needed.
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-8}

# Stage toggles (true | false). Set RUN_INFER=false to re-score an existing
# predictions.jsonl without re-running generation; set RUN_EVAL=false to skip
# scoring (e.g. predictions only). RUN_REPARSE=true tells eval.py to re-apply
# the answer parser over the existing `raw` outputs and rewrite predictions.jsonl
# in place — useful when only the letter-extraction regex changed.
RUN_INFER=${RUN_INFER:-true}
RUN_EVAL=${RUN_EVAL:-true}
RUN_REPARSE=${RUN_REPARSE:-true}
RUN_SUMMARY=${RUN_SUMMARY:-true}  # write summary.csv after the sweep finishes

# Zero-shot eval of base Qwen2.5-Omni (no LoRA adapter, no Chronos TS tower).
# When true: forces VARIANT=baseline, drops --adapter_path, writes predictions
# under runs/zero_shot_qwen2_5_omni/ so the artifacts don't collide with a
# trained run. RUN_NAME / VARIANT above are ignored in this mode.
ZERO_SHOT=${ZERO_SHOT:-false}

if [ "$ZERO_SHOT" = "true" ]; then
    VARIANT=baseline
    ADAPTER="$ROOT/runs/zero_shot_qwen2_5_omni"
    mkdir -p "$ADAPTER"
else
    ADAPTER="$ROOT/runs/$RUN_NAME"
fi

# Test sets to evaluate. One run per file; predictions + metrics land under
# $ADAPTER/<ablation>/<dataset_stem>/.
#
# DATASET_GROUP picks the default test list:
#   video           — video sets under data/video_ts/jsonl
#   image           — image sets under data/image_ts/jsonl
#   both (default)  — video + image sets
#   classification  — every per-source classification test set (each source
#                     dataset scored independently so per-set metrics stay
#                     separable).
# Override mechanisms (in priority order):
#   * inline (same bash process):  TEST_JSONLS=(/a /b) bash scripts/eval.sh
#   * env-portable (cross-process): TEST_JSONLS_STR="/a /b" bash …
# We can't export bash arrays across `exec`, so the *_STR form splits on
# whitespace.
DATASET_GROUP="${DATASET_GROUP:-classification}"
VIDEO_JSONL_DIR="$ROOT/data/video_ts/jsonl"
IMAGE_JSONL_DIR="$ROOT/data/image_ts/jsonl"
VIDEO_TEST_JSONLS=(
    "$(resolve_jsonl "$VIDEO_JSONL_DIR" "covla_test")"
    "$(resolve_jsonl "$VIDEO_JSONL_DIR" "agibot_test")"
    "$(resolve_jsonl "$VIDEO_JSONL_DIR" "cuhk_x_har_test")"
    "$(resolve_jsonl "$VIDEO_JSONL_DIR" "holo_assist_test")"
    "$(resolve_jsonl "$VIDEO_JSONL_DIR" "future_factories_test")"
)
IMAGE_TEST_JSONLS=(
    "$(resolve_jsonl "$IMAGE_JSONL_DIR" "mimic_death_test")"
    "$(resolve_jsonl "$IMAGE_JSONL_DIR" "mimic_disch_test")"
    "$(resolve_jsonl "$IMAGE_JSONL_DIR" "pixelrec_test")"
    "$(resolve_jsonl "$IMAGE_JSONL_DIR" "sp500_test")"
    "$(resolve_jsonl "$IMAGE_JSONL_DIR" "terra_test")"
)
# Source-level classification test sets. Mirrors the union of VIDEO_ and
# IMAGE_ above, minus the prediction-task ones (pixelrec / sp500 / terra).
CLASSIFICATION_TEST_JSONLS=(
    "$(resolve_jsonl "$VIDEO_JSONL_DIR" "covla_test")"
    "$(resolve_jsonl "$VIDEO_JSONL_DIR" "agibot_test")"
    "$(resolve_jsonl "$VIDEO_JSONL_DIR" "cuhk_x_har_test")"
    "$(resolve_jsonl "$VIDEO_JSONL_DIR" "holo_assist_test")"
    "$(resolve_jsonl "$VIDEO_JSONL_DIR" "future_factories_test")"
    "$(resolve_jsonl "$IMAGE_JSONL_DIR" "mimic_death_test")"
    "$(resolve_jsonl "$IMAGE_JSONL_DIR" "mimic_disch_test")"
)
if [ -z "${TEST_JSONLS+x}" ] && [ -n "${TEST_JSONLS_STR-}" ]; then
    # shellcheck disable=SC2206  # intentional word-split on $TEST_JSONLS_STR
    TEST_JSONLS=( $TEST_JSONLS_STR )
fi
if [ -z "${TEST_JSONLS+x}" ]; then
    case "$DATASET_GROUP" in
        video)          TEST_JSONLS=("${VIDEO_TEST_JSONLS[@]}") ;;
        image)          TEST_JSONLS=("${IMAGE_TEST_JSONLS[@]}") ;;
        both)           TEST_JSONLS=("${VIDEO_TEST_JSONLS[@]}" "${IMAGE_TEST_JSONLS[@]}") ;;
        classification) TEST_JSONLS=("${CLASSIFICATION_TEST_JSONLS[@]}") ;;
        *)
            echo "[eval.sh] unknown DATASET_GROUP=$DATASET_GROUP (expected: video | image | both | classification)" >&2
            exit 1
            ;;
    esac
fi

# Ablations to sweep (timeomni_v backend honors --no_vision and
# --no_timeseries):
#   full          → vision (video / images) + structured TS element
#   no_vision     → drops the visual element entirely (text + TS only).
#                   Works for both video and image datasets.
#   no_timeseries → strips the inline TS block AND drops the structured TS
#                   element so the processor emits no <|ts_placeholder|>
# Override with ABLATIONS=("full") to run a single configuration.
if [ -z "${ABLATIONS+x}" ]; then
    ABLATIONS=("full" "no_vision" "no_timeseries")
fi

if [ "$VARIANT" = "timeomni_v" ]; then
    EXTRA="--backend timeomni_v --chronos_path $ROOT/ckpts/chronos-2 --fusion_mode $FUSION_MODE"
else
    EXTRA="--backend qwen2_5_omni"
fi
# Trained checkpoints pass --adapter_path; zero-shot omits it so infer.py
# skips the PeftModel wrap and runs the bare base model.
if [ "$ZERO_SHOT" = "true" ]; then
    ADAPTER_ARG=""
else
    ADAPTER_ARG="--adapter_path $ADAPTER"
fi

cd "$ROOT"

if [ -n "${CONDA_ENV:-}" ] && command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi

export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache_${USER:-$(id -un)}}"
mkdir -p "$TRITON_CACHE_DIR"

# Multi-node launchers should inject NODE_COUNT, NODE_RANK, PROC_PER_NODE,
# MASTER_ADDR (otherwise we fall back to single-node defaults).
# Pick a random high port when MASTER_PORT isn't injected — re-using 29500
# after a failed run gives EADDRINUSE because the prior TCPStore hasn't
# released the socket yet (mirrors scripts/train.sh).
: "${MASTER_PORT:=$((20000 + RANDOM % 20000))}"

# Single-GPU-per-node degenerate case: run plain python (device_map="auto") so
# a model that doesn't fit on one GPU can still tensor-parallel across whatever
# is visible. Otherwise data-parallel via torchrun (one full model copy per rank).
if [ "$NODE_COUNT" -eq 1 ] && [ "$PROC_PER_NODE" -eq 1 ]; then
    LAUNCHER=(python -m timeomni_v.inference.infer)
else
    LAUNCHER=(
        torchrun
        --nnodes="$NODE_COUNT"
        --node_rank="$NODE_RANK"
        --nproc_per_node="$PROC_PER_NODE"
        --master_addr="$MASTER_ADDR"
        --master_port="$MASTER_PORT"
        -m timeomni_v.inference.infer
    )
fi

for ABLATION in "${ABLATIONS[@]}"; do
    case "$ABLATION" in
        full)          ABLATION_ARGS=() ;;
        no_vision)     ABLATION_ARGS=(--no_vision) ;;
        no_timeseries) ABLATION_ARGS=(--no_timeseries) ;;
        *)
            echo "[eval.sh] unknown ablation: $ABLATION — skipping" >&2
            continue
            ;;
    esac

    for TEST_JSONL in "${TEST_JSONLS[@]}"; do
        if [ ! -f "$TEST_JSONL" ]; then
            echo "[eval.sh] missing test jsonl: $TEST_JSONL — skipping" >&2
            continue
        fi
        DATASET_STEM=$(jsonl_stem "$TEST_JSONL")
        OUT_DIR="$ADAPTER/$ABLATION/$DATASET_STEM"
        mkdir -p "$OUT_DIR"
        OUT_JSONL="$OUT_DIR/predictions.jsonl"
        LOG_FILE="$OUT_DIR/eval.log"

        echo "==================================================================" \
            | tee -a "$LOG_FILE"
        echo "[eval.sh] ablation=$ABLATION dataset=$DATASET_STEM  test_jsonl=$TEST_JSONL" \
            | tee -a "$LOG_FILE"
        echo "[eval.sh] out_dir=$OUT_DIR" | tee -a "$LOG_FILE"

        if [ "$RUN_INFER" = "true" ]; then
            PYTHONPATH=. "${LAUNCHER[@]}" \
                --model_path "$ROOT/ckpts/Qwen2.5-Omni-7B" \
                $EXTRA \
                $ADAPTER_ARG \
                --test_jsonl "$TEST_JSONL" \
                --out_jsonl "$OUT_JSONL" \
                --batch_size "$BATCH_SIZE" \
                --dynamic_bs_max_tokens "$DYN_BS_MAX_TOKENS" \
                --dynamic_bs_max_bs "$DYN_BS_MAX_BS" \
                --max_new_tokens "$MAX_NEW_TOKENS" \
                ${ABLATION_ARGS[@]+"${ABLATION_ARGS[@]}"} 2>&1 | tee -a "$LOG_FILE"
        else
            echo "[eval.sh] RUN_INFER=false — skipping inference, re-using $OUT_JSONL" \
                | tee -a "$LOG_FILE"
        fi

        # Score the predictions: accuracy / macro-F1 / weighted-F1 / UAR /
        # confusion + merge in success/failure stats from <out_jsonl>.metrics.json.
        # torchrun runs this whole script on every node; gate on NODE_RANK so
        # scoring + log append happen exactly once.
        if [ "$RUN_EVAL" = "true" ] && [ "${NODE_RANK:-0}" -eq 0 ]; then
            # Pass --test_jsonl so eval.py can read the `task` field from the
            # source dataset. predictions.jsonl rows don't carry `task`, so
            # without this eval falls back to "classification" and scores
            # forecasting datasets (e.g. terra) as 100% unparseable.
            EVAL_ARGS=("$OUT_JSONL" --test_jsonl "$TEST_JSONL")
            if [ "$RUN_REPARSE" = "true" ]; then
                EVAL_ARGS+=(--reparse)
            fi
            PYTHONPATH=. python -m timeomni_v.inference.eval "${EVAL_ARGS[@]}" \
                2>&1 | tee -a "$LOG_FILE"
        fi
    done
done

# Roll all per-(ablation, dataset) metrics.json into one summary.csv at the
# adapter root. Skipped when no scoring happened on this node (NODE_RANK!=0).
if [ "$RUN_SUMMARY" = "true" ] && [ "${NODE_RANK:-0}" -eq 0 ]; then
    python "$ROOT/scripts/summarize_eval.py" "$ADAPTER" \
        --out "$ADAPTER/summary.csv"
fi
