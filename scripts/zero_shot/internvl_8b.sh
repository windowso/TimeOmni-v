#!/usr/bin/env bash
# Zero-shot eval: InternVL3.5-8B (HF re-export, InternVLForConditionalGeneration).
# BATCH_SIZE is hard-fixed to 1 — see backends/internvl.py docstring; the
# stack constraint on pixel_values_videos rules out bs>1 if we want to honor
# the same fps / min_frames / max_frames knobs as the Qwen backends.
set -euo pipefail
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export BACKEND=internvl_8b
export BACKEND_MODEL_PATH_INTERNVL_8B=${BACKEND_MODEL_PATH_INTERNVL_8B:-$ROOT/ckpts/InternVL3_5-8B-HF}
export RUN_DIR_NAME=${RUN_DIR_NAME:-zero_shot}
export DYN_BS_MAX_TOKENS=${DYN_BS_MAX_TOKENS:-0}
export BATCH_SIZE=1
# DATASET_GROUP=video|image picks the default test sweep; override at the env.
export DATASET_GROUP=${DATASET_GROUP:-video}
exec bash "$ROOT/scripts/eval_zero_shot.sh" "$@"
