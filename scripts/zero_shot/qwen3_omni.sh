#!/usr/bin/env bash
# Zero-shot eval: Qwen3-Omni-30B-A3B-Instruct (Thinker only).
#
# Reuses the Qwen2.5-Omni processor pipeline + dynamic-bs path; the only
# differences are the model class and a slightly tighter default token budget
# (the 30B MoE has higher per-token VRAM cost, so cap dynamic batches lower).
set -euo pipefail
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export BACKEND=qwen3_omni
export BACKEND_MODEL_PATH_QWEN3_OMNI=${BACKEND_MODEL_PATH_QWEN3_OMNI:-$ROOT/ckpts/Qwen3-Omni-30B-A3B-Instruct}
export RUN_DIR_NAME=${RUN_DIR_NAME:-zero_shot}
# Tighter dynamic-bs ceiling for the 30B MoE; override at the env if you need
# different memory-vs-throughput trade-offs.
export DYN_BS_MAX_TOKENS=${DYN_BS_MAX_TOKENS:-30000}
export DYN_BS_MAX_BS=${DYN_BS_MAX_BS:-32}
# DATASET_GROUP=video|image picks the default test sweep when TEST_JSONLS_STR
# is also unset. Currently TEST_JSONLS_STR is hard-pinned to holo_assist above.
export DATASET_GROUP=${DATASET_GROUP:-image}
exec bash "$ROOT/scripts/eval_zero_shot.sh" "$@"
