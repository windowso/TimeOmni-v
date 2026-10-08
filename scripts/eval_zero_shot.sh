#!/usr/bin/env bash
# Zero-shot evaluation of base Qwen2.5-Omni (no LoRA adapter, no Chronos TS
# tower). Mirrors scripts/eval.sh but hard-wires VARIANT=baseline, drops the
# adapter argument, and writes everything under runs/zero_shot_qwen2_5_omni/
# so artifacts don't collide with a trained run.

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$ROOT/scripts/_jsonl.sh"
RUN_DIR_NAME=${RUN_DIR_NAME:-zero_shot}
BATCH_SIZE=${BATCH_SIZE:-16} # fallback per-rank batch size; ignored when DYN_BS_MAX_TOKENS>0
# Token-budget batching mirrors training. Padded-token measure (max_len * bs);
# set to 0 to fall back to a fixed BATCH_SIZE. Only the qwen2_5_omni / qwen3_omni
# / timeomni_v backends honor it; backends without a Qwen2.5-style video_processor
# fall back to fixed BATCH_SIZE automatically.
DYN_BS_MAX_TOKENS=${DYN_BS_MAX_TOKENS:-45000}
DYN_BS_MAX_BS=${DYN_BS_MAX_BS:-50}
# Generation cap. infer.py's default (4) only covers single-letter classification;
# image `prediction` answers reach ~1286 Qwen2.5-Omni tokens (terra). 1408 covers
# the longest sample with ~10% margin. Override per-dataset if needed.
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-16}

# Backend(s) to evaluate. Defaults to a single backend so existing usage stays
# unchanged; pass BACKENDS=(qwen2_5_omni qwen3_omni gpt) to sweep. Each backend
# also takes its own MODEL_PATH from BACKEND_MODEL_PATH_<UPPER_NAME> if set,
# falling back to MODEL_PATH.
BACKEND=${BACKEND:-qwen2_5_omni}
# `BACKENDS` is an in-shell array; bash will not export arrays through env, so
# the wrappers must set the *scalar* `BACKEND` instead — we rebuild the array
# from it here. Direct invocations (no wrapper) can still pass BACKENDS=(...)
# inline since they share the same bash process.
if [ -z "${BACKENDS+x}" ]; then BACKENDS=("$BACKEND"); fi

# Default model paths. All HF checkpoints live under $ROOT/ckpts/. Override
# any backend with BACKEND_MODEL_PATH_<UPPER_NAME>=... at the env. Both
# size variants of InternVL3.5 and Qwen3-VL are wired in as separate
# backends so they don't overwrite each other's predictions tree.
MODEL_PATH=${MODEL_PATH:-$ROOT/ckpts/Qwen2.5-Omni-7B}
BACKEND_MODEL_PATH_QWEN2_5_OMNI=${BACKEND_MODEL_PATH_QWEN2_5_OMNI:-$ROOT/ckpts/Qwen2.5-Omni-7B}
BACKEND_MODEL_PATH_QWEN3_OMNI=${BACKEND_MODEL_PATH_QWEN3_OMNI:-$ROOT/ckpts/Qwen3-Omni-30B-A3B-Instruct}
BACKEND_MODEL_PATH_QWEN3_VL_8B=${BACKEND_MODEL_PATH_QWEN3_VL_8B:-$ROOT/ckpts/Qwen3-VL-8B-Instruct}
BACKEND_MODEL_PATH_QWEN3_VL_30B=${BACKEND_MODEL_PATH_QWEN3_VL_30B:-$ROOT/ckpts/Qwen3-VL-30B-A3B-Instruct}
BACKEND_MODEL_PATH_INTERNVL_4B=${BACKEND_MODEL_PATH_INTERNVL_4B:-$ROOT/ckpts/InternVL3_5-4B-HF}
BACKEND_MODEL_PATH_INTERNVL_8B=${BACKEND_MODEL_PATH_INTERNVL_8B:-$ROOT/ckpts/InternVL3_5-8B-HF}
BACKEND_MODEL_PATH_GPT=${BACKEND_MODEL_PATH_GPT:-gpt-5-mini}
BACKEND_MODEL_PATH_GEMINI=${BACKEND_MODEL_PATH_GEMINI:-gemini-2.5-flash}

# API knobs (ignored for non-API backends).
API_PROVIDER=${API_PROVIDER:-proxy}
API_BASE_URL=${API_BASE_URL:-}
API_KEY_ENV=${API_KEY_ENV:-OPENAI_API_KEY}
API_CONCURRENCY=${API_CONCURRENCY:-8}
# Crash safety: append completed rows to predictions.jsonl every N completions
# so a killed run resumes from the last checkpoint. 0 disables.
API_CHECKPOINT_EVERY=${API_CHECKPOINT_EVERY:-20}

# (No InternVL-specific knob: the backend derives per-row num_frames from
# --fps / --min_frames / --max_frames and forces --batch_size=1.)

# Stage toggles (true | false). Set RUN_INFER=false to re-score an existing
# predictions.jsonl without re-running generation; set RUN_EVAL=false to skip
# scoring (e.g. predictions only). RUN_REPARSE=true tells eval.py to re-apply
# the answer parser over the existing `raw` outputs and rewrite predictions.jsonl
# in place — useful when only the letter-extraction regex changed.
RUN_INFER=${RUN_INFER:-true}
RUN_EVAL=${RUN_EVAL:-true}
RUN_REPARSE=${RUN_REPARSE:-true}
RUN_SUMMARY=${RUN_SUMMARY:-true}  # write summary.csv per backend after the sweep

OUT_ROOT="$ROOT/runs/$RUN_DIR_NAME"
mkdir -p "$OUT_ROOT"

# Test sets to evaluate. One run per file; predictions + metrics land under
# $OUT_ROOT/<backend>/<ablation>/<dataset_stem>/. Override mechanisms:
#   * inline (same bash process):  TEST_JSONLS=(/a /b) bash eval_zero_shot.sh
#   * env-portable (cross-process): TEST_JSONLS_STR="/a /b" bash …
# We can't export bash arrays across `exec`, so cluster wrappers / all.sh use
# the scalar TEST_JSONLS_STR and we split on whitespace here.
#
# DATASET_GROUP picks the default test list:
#   video (default) — video sets under data/video_ts/jsonl
#   image           — image sets under data/image_ts/jsonl
# TEST_JSONLS / TEST_JSONLS_STR override entirely (mixed lists ok).
DATASET_GROUP="${DATASET_GROUP:-video}"
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
if [ -z "${TEST_JSONLS+x}" ] && [ -n "${TEST_JSONLS_STR-}" ]; then
    # shellcheck disable=SC2206  # intentional word-split on $TEST_JSONLS_STR
    TEST_JSONLS=( $TEST_JSONLS_STR )
fi
if [ -z "${TEST_JSONLS+x}" ]; then
    case "$DATASET_GROUP" in
        video) TEST_JSONLS=("${VIDEO_TEST_JSONLS[@]}") ;;
        image) TEST_JSONLS=("${IMAGE_TEST_JSONLS[@]}") ;;
        both)  TEST_JSONLS=("${VIDEO_TEST_JSONLS[@]}" "${IMAGE_TEST_JSONLS[@]}") ;;
        *)
            echo "[eval_zero_shot.sh] unknown DATASET_GROUP=$DATASET_GROUP (expected: video | image | both)" >&2
            exit 1
            ;;
    esac
fi

# Ablations to sweep. Each entry maps to a subdir under $OUT_ROOT and a set
# of extra flags passed to infer.py:
#   full          → vision (video / images) + inline TS
#   no_vision     → drops the visual element entirely (text + inline TS
#                   only). Works for both video and image datasets.
#   no_timeseries → strips the inline TS block (vision + text only)
# Override with ABLATIONS=("full") to run a single configuration.
if [ -z "${ABLATIONS+x}" ]; then
    ABLATIONS=("full" "no_vision" "no_timeseries")
fi

# Zero-shot eval has no Chronos tower and no LoRA adapter. The full inline TS
# block flows through to the model — over-long samples surface as failures in
# <out>.metrics.json instead of being silently truncated.
ADAPTER_ARG=""

cd "$ROOT"

if [ -n "${CONDA_ENV:-}" ] && command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi

# qwen3_omni's modeling code lives in transformers>=5.2 (not in the 4.57.x
# branch the other backends pin). If it's about to run and the active env is
# too old, the script can either abort with instructions or auto-install. Set
# ALLOW_PIP=true to opt into auto-install; otherwise we just abort with a
# clear message so the caller can switch envs themselves.
require_transformers_v5_for_qwen3_omni() {
    if python -c 'import sys, transformers as t; from packaging.version import Version; sys.exit(0 if Version(t.__version__) >= Version("5.2.0") else 1)' 2>/dev/null; then
        return 0
    fi
    if [ "${ALLOW_PIP:-false}" = "true" ]; then
        echo "[eval_zero_shot.sh] qwen3_omni requires transformers>=5.2.0 — installing (ALLOW_PIP=true)"
        pip install --quiet "transformers>=5.2.0"
    else
        echo "[eval_zero_shot.sh] qwen3_omni requires transformers>=5.2.0; current is $(python -c 'import transformers;print(transformers.__version__)')." >&2
        echo "[eval_zero_shot.sh] Either upgrade manually or rerun with ALLOW_PIP=true." >&2
        exit 1
    fi
}

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

for BACKEND in "${BACKENDS[@]}"; do
    # Resolve per-backend overrides. Look up BACKEND_MODEL_PATH_<UPPER>; if
    # unset/empty, fall back to MODEL_PATH.
    BACKEND_UP=$(echo "$BACKEND" | tr '[:lower:]' '[:upper:]')
    VAR_NAME="BACKEND_MODEL_PATH_${BACKEND_UP}"
    BACKEND_MODEL_PATH="${!VAR_NAME:-$MODEL_PATH}"

    # qwen3_omni needs transformers>=5.2 — verify (or install if opted in).
    if [ "$BACKEND" = "qwen3_omni" ]; then
        require_transformers_v5_for_qwen3_omni
    fi

    # API backends bypass DDP — they don't load a torch model. If the user
    # invoked us under torchrun, that path's already wired in the python
    # entrypoint to assert; avoid even spawning torchrun for those.
    case "$BACKEND" in
        gpt|gemini)
            EXTRA_BACKEND_ARGS=(
                --api_provider "$API_PROVIDER"
                --api_key_env "$API_KEY_ENV"
                --api_concurrency "$API_CONCURRENCY"
                --api_checkpoint_every "$API_CHECKPOINT_EVERY"
            )
            if [ -n "$API_BASE_URL" ]; then
                EXTRA_BACKEND_ARGS+=(--api_base_url "$API_BASE_URL")
            fi
            BACKEND_LAUNCHER=(python -m timeomni_v.inference.infer)
            ;;
        internvl_4b|internvl_8b)
            # The backend hard-enforces --batch_size=1; per-row num_frames
            # is derived from --fps / --min_frames / --max_frames inside
            # the collator. No backend-specific CLI flags needed.
            EXTRA_BACKEND_ARGS=()
            BACKEND_LAUNCHER=("${LAUNCHER[@]}")
            ;;
        *)
            EXTRA_BACKEND_ARGS=()
            BACKEND_LAUNCHER=("${LAUNCHER[@]}")
            ;;
    esac

    # API backends share one dispatch key (`gpt`, `gemini`) but the actual
    # model is whatever string `--model_path` carries (e.g. `gemini-2.5-flash`
    # vs. `gemini-3-flash-preview`). Local-checkpoint backends already encode
    # size in the dispatch key (`qwen3_vl_8b`, `internvl_4b`, …) so the bare
    # backend name is specific enough. For API backends, derive the dir from
    # the model string so concurrent runs don't clobber each other.
    case "$BACKEND" in
        gpt|gemini)
            BACKEND_DIR=$(printf '%s' "$(basename "$BACKEND_MODEL_PATH")" \
                | tr -c 'A-Za-z0-9._-' '_')
            ;;
        *)
            BACKEND_DIR="$BACKEND"
            ;;
    esac

    for ABLATION in "${ABLATIONS[@]}"; do
        case "$ABLATION" in
            full)          ABLATION_ARGS=() ;;
            no_vision)     ABLATION_ARGS=(--no_vision) ;;
            no_timeseries) ABLATION_ARGS=(--no_timeseries) ;;
            *)
                echo "[eval_zero_shot.sh] unknown ablation: $ABLATION — skipping" >&2
                continue
                ;;
        esac

        for TEST_JSONL in "${TEST_JSONLS[@]}"; do
            if [ ! -f "$TEST_JSONL" ]; then
                echo "[eval_zero_shot.sh] missing test jsonl: $TEST_JSONL — skipping" >&2
                continue
            fi
            DATASET_STEM=$(jsonl_stem "$TEST_JSONL")
            OUT_DIR="$OUT_ROOT/$BACKEND_DIR/$ABLATION/$DATASET_STEM"
            mkdir -p "$OUT_DIR"
            OUT_JSONL="$OUT_DIR/predictions.jsonl"
            LOG_FILE="$OUT_DIR/eval.log"

            echo "==================================================================" \
                | tee -a "$LOG_FILE"
            echo "[eval_zero_shot.sh] backend=$BACKEND backend_dir=$BACKEND_DIR ablation=$ABLATION dataset=$DATASET_STEM" \
                | tee -a "$LOG_FILE"
            echo "[eval_zero_shot.sh] model_path=$BACKEND_MODEL_PATH" \
                | tee -a "$LOG_FILE"
            echo "[eval_zero_shot.sh] out_dir=$OUT_DIR" | tee -a "$LOG_FILE"

            if [ "$RUN_INFER" = "true" ]; then
                PYTHONPATH=. "${BACKEND_LAUNCHER[@]}" \
                    --backend "$BACKEND" \
                    --model_path "$BACKEND_MODEL_PATH" \
                    $ADAPTER_ARG \
                    --test_jsonl "$TEST_JSONL" \
                    --out_jsonl "$OUT_JSONL" \
                    --batch_size "$BATCH_SIZE" \
                    --dynamic_bs_max_tokens "$DYN_BS_MAX_TOKENS" \
                    --dynamic_bs_max_bs "$DYN_BS_MAX_BS" \
                    --max_new_tokens "$MAX_NEW_TOKENS" \
                    ${EXTRA_BACKEND_ARGS[@]+"${EXTRA_BACKEND_ARGS[@]}"} \
                    ${ABLATION_ARGS[@]+"${ABLATION_ARGS[@]}"} 2>&1 | tee -a "$LOG_FILE"
            else
                echo "[eval_zero_shot.sh] RUN_INFER=false — skipping inference, re-using $OUT_JSONL" \
                    | tee -a "$LOG_FILE"
            fi

            # Score the predictions: accuracy / macro-F1 / weighted-F1 / UAR /
            # confusion + merge in success/failure stats from
            # <out_jsonl>.metrics.json. torchrun runs this whole script on
            # every node; gate on NODE_RANK so scoring + log append happen
            # exactly once.
            if [ "$RUN_EVAL" = "true" ] && [ "${NODE_RANK:-0}" -eq 0 ]; then
                # Pass --test_jsonl so eval.py can read the `task` field from
                # the source dataset. predictions.jsonl rows don't carry
                # `task`, so without this eval falls back to "classification"
                # and scores forecasting datasets (e.g. terra) as 100%
                # unparseable.
                EVAL_ARGS=("$OUT_JSONL" --test_jsonl "$TEST_JSONL")
                if [ "$RUN_REPARSE" = "true" ]; then
                    EVAL_ARGS+=(--reparse)
                fi
                PYTHONPATH=. python -m timeomni_v.inference.eval "${EVAL_ARGS[@]}" \
                    2>&1 | tee -a "$LOG_FILE"
            fi
        done
    done

    # Per-backend rollup: one summary.csv covering every (ablation, dataset)
    # under $OUT_ROOT/$BACKEND/. We summarize per-backend (rather than once at
    # the very end) so a partial sweep of a single backend still writes its
    # CSV — and re-running with BACKENDS=(<X>) refreshes only that file.
    if [ "$RUN_SUMMARY" = "true" ] && [ "${NODE_RANK:-0}" -eq 0 ]; then
        python "$ROOT/scripts/summarize_eval.py" "$OUT_ROOT/$BACKEND_DIR" \
            --out "$OUT_ROOT/$BACKEND_DIR/summary.csv"
    fi

done
