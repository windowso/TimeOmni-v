#!/usr/bin/env bash
# Zero-shot eval: Qwen3-VL-8B-Instruct (single-call apply_chat_template).
# Dynamic-bs disabled (no exposed video_processor patch math); fixed batch.
set -euo pipefail
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export BACKEND=qwen3_vl_8b
export BACKEND_MODEL_PATH_QWEN3_VL_8B=${BACKEND_MODEL_PATH_QWEN3_VL_8B:-$ROOT/ckpts/Qwen3-VL-8B-Instruct}
export RUN_DIR_NAME=${RUN_DIR_NAME:-zero_shot}
export DYN_BS_MAX_TOKENS=${DYN_BS_MAX_TOKENS:-0}
export BATCH_SIZE=${BATCH_SIZE:-1}
# DATASET_GROUP=video|image picks the default test sweep; override at the env.
export DATASET_GROUP=${DATASET_GROUP:-video}
exec bash "$ROOT/scripts/eval_zero_shot.sh" "$@"
