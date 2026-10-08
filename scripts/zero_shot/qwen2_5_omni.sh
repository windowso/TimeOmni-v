#!/usr/bin/env bash
# Zero-shot eval: Qwen2.5-Omni-7B (no LoRA, no Chronos tower).
#
# Honors all six video knobs (fps / min/max frames / min/max pixels /
# do_sample_frames=True). Uses dynamic-bs by default; override with
# DYN_BS_MAX_TOKENS=0 BATCH_SIZE=8 for a fixed batch.
set -euo pipefail
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export BACKEND=qwen2_5_omni
export BACKEND_MODEL_PATH_QWEN2_5_OMNI=${BACKEND_MODEL_PATH_QWEN2_5_OMNI:-$ROOT/ckpts/Qwen2.5-Omni-7B}
export RUN_DIR_NAME=${RUN_DIR_NAME:-zero_shot}
export DYN_BS_MAX_TOKENS=${DYN_BS_MAX_TOKENS:-45000}
export DYN_BS_MAX_BS=${DYN_BS_MAX_BS:-50}
# DATASET_GROUP=video|image picks the default test sweep; override at the env.
export DATASET_GROUP=${DATASET_GROUP:-image}
exec bash "$ROOT/scripts/eval_zero_shot.sh" "$@"
