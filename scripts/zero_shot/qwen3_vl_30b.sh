#!/usr/bin/env bash
# Zero-shot eval: Qwen3-VL-30B-A3B-Instruct (MoE; same processor pipeline as
# the 8B). Lower BATCH_SIZE since the activated experts still need real VRAM.
set -euo pipefail
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export BACKEND=qwen3_vl_30b
export BACKEND_MODEL_PATH_QWEN3_VL_30B=${BACKEND_MODEL_PATH_QWEN3_VL_30B:-$ROOT/ckpts/Qwen3-VL-30B-A3B-Instruct}
export RUN_DIR_NAME=${RUN_DIR_NAME:-zero_shot}
export DYN_BS_MAX_TOKENS=${DYN_BS_MAX_TOKENS:-0}
export BATCH_SIZE=${BATCH_SIZE:-1}
# DATASET_GROUP=video|image picks the default test sweep when TEST_JSONLS_STR
# is also unset. Currently TEST_JSONLS_STR is hard-pinned to covla above.
export DATASET_GROUP=${DATASET_GROUP:-video}
exec bash "$ROOT/scripts/eval_zero_shot.sh" "$@"
