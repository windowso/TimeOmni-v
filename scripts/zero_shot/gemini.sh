#!/usr/bin/env bash
# Zero-shot eval: Gemini (via the OpenAI-compatible proxy).
#
# Same shared frame extractor + retry as the GPT backend, but the chat
# /completions message schema (image_url blocks). Concurrency via
# API_CONCURRENCY. Uses the proxy URL by default — override with
# API_BASE_URL=... for a different gateway.
set -euo pipefail
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# Hard-coded proxy key — same convention as TS_bench. Override at the env
# (`OPENAI_API_KEY=sk-... bash scripts/zero_shot/gemini.sh`) when needed.
export OPENAI_API_KEY=${OPENAI_API_KEY:-sk-yjKTOBMpL9Ee8pFnzpXRWZY7840d42kLDbdcanJoOFBXa9ha}
export BACKEND=gemini
export BACKEND_MODEL_PATH_GEMINI=${BACKEND_MODEL_PATH_GEMINI:-gemini-3-flash-preview}
export RUN_DIR_NAME=${RUN_DIR_NAME:-zero_shot}
export API_PROVIDER=${API_PROVIDER:-proxy}
export API_KEY_ENV=${API_KEY_ENV:-OPENAI_API_KEY}
export API_CONCURRENCY=${API_CONCURRENCY:-8}
export NODE_COUNT=1 PROC_PER_NODE=1
# DATASET_GROUP=video|image picks the default test sweep; override at the env.
export DATASET_GROUP=${DATASET_GROUP:-video}
exec bash "$ROOT/scripts/eval_zero_shot.sh" "$@"
