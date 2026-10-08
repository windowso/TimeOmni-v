#!/usr/bin/env bash
# Zero-shot eval: GPT (OpenAI Responses API, via proxy by default).
#
# Single-process only (API backends bypass DDP). Concurrency is provided by
# the orchestrator's ThreadPoolExecutor — tune with API_CONCURRENCY.
# Frames go through the shared decord uniform sampler; per-frame size is
# capped by --video_max_pixels via PIL downscale before base64-JPEG encoding.
set -euo pipefail
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# Hard-coded proxy key — same convention as TS_bench. Override at the env
# (`OPENAI_API_KEY=sk-... bash scripts/zero_shot/gpt.sh`) when needed.
export OPENAI_API_KEY=${OPENAI_API_KEY:-sk-yjKTOBMpL9Ee8pFnzpXRWZY7840d42kLDbdcanJoOFBXa9ha}
export BACKEND=gpt
# Note: for API backends, BACKEND_MODEL_PATH_* carries the model NAME, not a
# filesystem path — e.g. "gpt-5", "gpt-5-mini", "gpt-4o", "o3-mini".
export BACKEND_MODEL_PATH_GPT=${BACKEND_MODEL_PATH_GPT:-gpt-5-mini}
export RUN_DIR_NAME=${RUN_DIR_NAME:-zero_shot}
export API_PROVIDER=${API_PROVIDER:-proxy}
export API_KEY_ENV=${API_KEY_ENV:-OPENAI_API_KEY}
export API_CONCURRENCY=${API_CONCURRENCY:-8}
export API_CHECKPOINT_EVERY=${API_CHECKPOINT_EVERY:-20}
# Force single-process — torchrun makes no sense for the API path.
export NODE_COUNT=1 PROC_PER_NODE=1
# DATASET_GROUP=video|image picks the default test sweep; override at the env.
export DATASET_GROUP=${DATASET_GROUP:-video}
exec bash "$ROOT/scripts/eval_zero_shot.sh" "$@"
