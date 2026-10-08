#!/usr/bin/env bash
# TimeOmni-v training runner. Edit the constants below to change runs.
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$ROOT/scripts/_jsonl.sh"

# Pick which dataset to train on. Override via positional arg or env var:
#   bash scripts/train.sh agibot
#   DATASET=holo_assist bash scripts/train.sh
#   DATASET=mimic_death bash scripts/train.sh
#   DATASET=all bash scripts/train.sh           # all 10 single datasets merged
#   DATASET=all_no_cuhk bash scripts/train.sh   # ditto, minus cuhk_x_har
# Video sets (under data/video_ts/jsonl):
#   covla | agibot | cuhk_x_har | holo_assist | mixed | smoke
# Image sets (under data/image_ts/jsonl):
#   mimic_death | mimic_disch | pixelrec | sp500 | terra
# Merged sets (under data/, train = concat of all sources, val = 50/dataset):
#   all | all_no_cuhk
# Merged classification-only sets (under data/merged_classification/jsonl/,
# train = concat of all classification sources shuffled, test = 50/dataset):
#   cls_all | cls_no_cuhk
# Merged MIMIC sets (under data/merged_classification/jsonl/,
# train = mimic_death + mimic_disch concat shuffled, val = both test sets):
#   mimic_all
DATASET="${1:-${DATASET:-cls_all}}"

VARIANT=timeomni_v             # timeomni_v | baseline
FUSION_MODE=time_interleave # block_adjacent | time_interleave (timeomni_v only)

# Forecasting head: PRED_LEN > 0 enables it. The per-dataset case block
# below sets defaults for {terra,sp500,pixelrec}_perch (see
# timeomni_v.data.convert_per_channel_forecasting). PRED_LEN starts UNSET so the
# `${PRED_LEN:-N}` default in the inner case actually fires; we coerce to 0
# at the bottom for non-forecasting datasets.
HEAD_WINDOW="${HEAD_WINDOW:-8}"
HEAD_DROPOUT="${HEAD_DROPOUT:-0.0}"

# LR / EPOCHS may be set per-dataset by callers via env var.
LR="${LR:-1e-5}"
EPOCHS="${EPOCHS:-10}"

RUN_NAME="${VARIANT}-${DATASET}-lr${LR}-ep${EPOCHS}"
GRAD_ACCUM=1
PER_DEV_BS=8
MAX_STEPS="${MAX_STEPS:--1}"
WARMUP_RATIO=0.1
# bf16 has no loss scaler — a single grad spike past ~65504 in any attention
# head triggers inf → NaN cascade. With LoRA adapter wake-up + ZeRO-2 allreduce
# this run consistently sees pre-clip grad_norm 100-800. 0.5 (vs HF default 1.0)
# halves the per-step update during spikes; steady-state grads still well above
# the cap so the clip is doing all the work either way.
MAX_GRAD_NORM=1.0

# Mid-training eval over $EVAL_JSONL. "no" disables; "epoch" / "steps" enable.
# Eval forward is heavy (full multimodal sequence) — set per_device_eval_batch_size
# conservatively. DynamicBSTrainer only overrides the train dataloader; eval
# uses fixed bs.
EVAL_STRATEGY="${EVAL_STRATEGY:-epoch}"
EVAL_BS=$PER_DEV_BS

# Token-budget dynamic batching. 0 disables (vanilla bs=PER_DEV_BS).
# >0 enables TokenBudgetBatchSampler: short samples packed up to this many
# tokens, long samples keep bs=1. Set to ~3x median observed sequence.
DYN_BS_MAX_TOKENS=40000
DYN_BS_MAX_BS=32

# Per-step timing (fwd/bwd/opt ms + peak VRAM) + per-batch collator timing.
# true | false. Adds ~1 cuda.synchronize() per step, so leave off for
# high-throughput runs.
LOG_STEP_TIMING=false

# Dataloader tuning — the data pipeline decodes mp4 and reads TS csvs per
# sample; with num_workers=0 (HF default) all of that runs in the main
# process and GPUs starve. Steady-state step is ~100s (long multimodal
# sequences + grad checkpoint), median batch prep is ~2s, so 4x2=8 in-flight
# batches is more than enough to hide prep. Larger values just inflate the
# startup prefetch burst (num_workers*prefetch_factor batches queued before
# step 0) and hold decoded video tensors in RAM longer.
DATALOADER_WORKERS=16
DATALOADER_PREFETCH=4

OUTPUT_DIR="$ROOT/runs/$RUN_NAME"

SCRIPT="timeomni_v/training/train.py"
# Unified jsonl: carries inline TS text AND timeseries_path; both modes read
# the same file (timeomni_v strips inline TS in the collator, baseline ignores
# timeseries_path). See timeomni_v/data/convert_covla.py.
# Video and image datasets live in separate directories; the case maps each
# DATASET to its source dir so callers don't have to.
VIDEO_JSONL_DIR="$ROOT/data/video_ts/jsonl"
IMAGE_JSONL_DIR="$ROOT/data/image_ts/jsonl"
MERGED_JSONL_DIR="$ROOT/data"
MERGED_CLS_JSONL_DIR="$ROOT/data/merged_classification/jsonl"
case "$DATASET" in
    covla|agibot|cuhk_x_har|holo_assist|future_factories|mixed|smoke)
        TRAIN_JSONLS=("$(resolve_jsonl "$VIDEO_JSONL_DIR" "${DATASET}")")
        EVAL_JSONL="$(resolve_jsonl "$VIDEO_JSONL_DIR" "${DATASET}_test")"
        ;;
    mimic_death|mimic_disch|pixelrec|sp500|terra)
        TRAIN_JSONLS=("$(resolve_jsonl "$IMAGE_JSONL_DIR" "${DATASET}")")
        EVAL_JSONL="$(resolve_jsonl "$IMAGE_JSONL_DIR" "${DATASET}_test")"
        ;;
    terra_perch|sp500_perch|pixelrec_perch)
        # Per-channel forecasting datasets produced by
        # timeomni_v.data.convert_per_channel_forecasting. Each row is
        # single-channel; PRED_LEN must be >= the per-dataset target
        # length (terra=20, sp500=12, pixelrec=6).
        BASE="${DATASET%_perch}"
        TRAIN_JSONLS=("$(resolve_jsonl "$IMAGE_JSONL_DIR" "${BASE}.percha")")
        EVAL_JSONL="$(resolve_jsonl "$IMAGE_JSONL_DIR" "${BASE}_test.percha")"
        case "$BASE" in
            terra)    PRED_LEN="${PRED_LEN:-20}" ;;
            sp500)    PRED_LEN="${PRED_LEN:-12}" ;;
            pixelrec) PRED_LEN="${PRED_LEN:-6}" ;;
        esac
        ;;
    all|all_no_cuhk)
        # Cross-dataset merged jsonls produced by hand under data/.
        # Naming convention differs from the per-dataset case (uses
        # _train / _val suffixes) so we set the paths explicitly.
        TRAIN_JSONLS=("$(resolve_jsonl "$MERGED_JSONL_DIR" "${DATASET}_train")")
        EVAL_JSONL="$(resolve_jsonl "$MERGED_JSONL_DIR" "${DATASET}_val")"
        ;;
    cls_all|cls_no_cuhk)
        # Classification-only multi-source training: feed each per-task
        # jsonl directly so TokenBudgetBatchSampler can constrain every
        # batch to one source. The merged file under
        # $MERGED_CLS_JSONL_DIR is no longer used for training (it stays
        # on disk for legacy comparison runs).
        # Eval still uses the merged 50-rows-per-source test set.
        TRAIN_JSONLS=(
            "$(resolve_jsonl "$VIDEO_JSONL_DIR" "covla")"
            "$(resolve_jsonl "$VIDEO_JSONL_DIR" "agibot")"
            "$(resolve_jsonl "$VIDEO_JSONL_DIR" "holo_assist")"
            "$(resolve_jsonl "$VIDEO_JSONL_DIR" "future_factories")"
            "$(resolve_jsonl "$IMAGE_JSONL_DIR" "mimic_death")"
            "$(resolve_jsonl "$IMAGE_JSONL_DIR" "mimic_disch")"
        )
        if [ "$DATASET" = "cls_all" ]; then
            TRAIN_JSONLS=(
                "$(resolve_jsonl "$VIDEO_JSONL_DIR" "cuhk_x_har")"
                "${TRAIN_JSONLS[@]}"
            )
        fi
        EVAL_JSONL="$(resolve_jsonl "$MERGED_CLS_JSONL_DIR" "${DATASET}_test")"
        ;;
    mimic_all)
        # mimic_death + mimic_disch trained jointly with task-pure batches.
        # Final per-dataset eval (post-training) still hits the original
        # mimic_*_test jsonls one at a time — see scripts/train_eval_mimic.sh.
        TRAIN_JSONLS=(
            "$(resolve_jsonl "$IMAGE_JSONL_DIR" "mimic_death")"
            "$(resolve_jsonl "$IMAGE_JSONL_DIR" "mimic_disch")"
        )
        EVAL_JSONL="$(resolve_jsonl "$MERGED_CLS_JSONL_DIR" "mimic_all_test")"
        ;;
    *)
        echo "[train.sh] unknown DATASET=$DATASET (video: covla | agibot | cuhk_x_har | holo_assist | future_factories | mixed | smoke; image: mimic_death | mimic_disch | pixelrec | sp500 | terra; merged: all | all_no_cuhk; merged-cls: cls_all | cls_no_cuhk; merged-mimic: mimic_all)" >&2
        exit 1
        ;;
esac
for f in "${TRAIN_JSONLS[@]}"; do
    [ -f "$f" ] || { echo "[train.sh] missing $f" >&2; exit 1; }
done
[ -f "$EVAL_JSONL" ]  || { echo "[train.sh] missing $EVAL_JSONL"  >&2; exit 1; }
echo "[train.sh] dataset=$DATASET train=(${TRAIN_JSONLS[*]}) eval=$EVAL_JSONL run_name=$RUN_NAME"

PRED_LEN="${PRED_LEN:-20}"
if [ "$VARIANT" = "timeomni_v" ]; then
    EXTRA_ARGS="--mode timeomni_v --chronos_path $ROOT/ckpts/chronos-2 --fusion_mode $FUSION_MODE"
    if [ "$PRED_LEN" -gt 0 ]; then
        EXTRA_ARGS="$EXTRA_ARGS --pred_len $PRED_LEN --head_window $HEAD_WINDOW --head_dropout $HEAD_DROPOUT"
    fi
else
    EXTRA_ARGS="--mode baseline"
fi

mkdir -p "$OUTPUT_DIR"

cd "$ROOT"

# If a conda env name is provided and conda is on PATH, activate it. Otherwise
# we trust whatever Python environment the caller already activated.
if [ -n "${CONDA_ENV:-}" ] && command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
fi

# Triton autotune defaults to ~/.triton/autotune which is on NFS here —
# DeepSpeed warns about NFS slowdowns on exit. Point it at a local tmpfs.
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache_${USER:-$(id -un)}}"
mkdir -p "$TRITON_CACHE_DIR"

# Reduce allocator fragmentation for long multi-modal sequences. Newer
# PyTorch renamed the knob from PYTORCH_CUDA_ALLOC_CONF (now deprecated) to
# PYTORCH_ALLOC_CONF — use the new name; if the caller set the old one,
# mirror it across so legacy launch scripts still work.
if [ -n "${PYTORCH_CUDA_ALLOC_CONF:-}" ] && [ -z "${PYTORCH_ALLOC_CONF:-}" ]; then
    export PYTORCH_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF"
fi
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
unset PYTORCH_CUDA_ALLOC_CONF

# HF tokenizers spins up a Rayon thread pool on first use, then emits a
# warning every time we fork after that (ProcessPoolExecutor in the length
# estimator, DataLoader workers, etc.). Tokenization cost is negligible
# here vs. GPU step time, so just pin it to single-threaded.
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# Multi-node launchers should inject NODE_COUNT, NODE_RANK, PROC_PER_NODE,
# MASTER_ADDR. When unset (single-node / interactive debug), deepspeed silently
# falls back to 1 GPU; fall back to the locally visible CUDA device count
# instead so `bash scripts/train.sh` uses every card on the box by default.
if [ -z "${PROC_PER_NODE:-}" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        PROC_PER_NODE=$(nvidia-smi -L | wc -l)
    else
        PROC_PER_NODE=1
    fi
    echo "[launcher] PROC_PER_NODE not injected — defaulting to $PROC_PER_NODE local GPUs" >&2
fi
: "${NODE_COUNT:=1}"
: "${NODE_RANK:=0}"
: "${MASTER_ADDR:=127.0.0.1}"
# Pick a random high port when MASTER_PORT isn't injected — re-using 29500
# after a failed run gives EADDRINUSE because the prior TCPStore hasn't
# released the socket yet.
: "${MASTER_PORT:=$((20000 + RANDOM % 20000))}"

# ZeRO stage — ZeRO-3 shards weights across ranks but pays bigger allgather
# peaks and generally higher CPU-RAM pressure from partitioning bookkeeping.
# Rolled back to ZeRO-2 after a container-cgroup OOM at 6 GPUs. Flip to
# ds_zero3.json only when CPU RAM is confirmed to have headroom.
DS_CONFIG="timeomni_v/training/configs/ds_zero2.json"

# Multi-node launch path: when the launcher starts train.sh once per node and
# injects NODE_COUNT / NODE_RANK / PROC_PER_NODE / MASTER_ADDR / MASTER_PORT,
# but does NOT permit ssh between nodes (so deepspeed's default pdsh launcher
# can't reach peers), the script does its own rendezvous via shared storage:
#   1. Each node writes its IP to a shared-storage file under $OUTPUT_DIR/_hostfile.
#   2. Rank 0 collects those IPs and writes a hostfile with `slots=PROC_PER_NODE`.
#   3. Workers wait for the hostfile to appear.
#   4. All nodes invoke deepspeed with --hostfile + --no_ssh + --launcher openmpi;
#      deepspeed reads slot counts from the hostfile and skips ssh, ranks
#      rendezvous via MASTER_ADDR / MASTER_PORT.
# Single-node runs (NODE_COUNT=1) skip the rendezvous and use the simple
# --num_nodes/--num_gpus form.
if [ "$NODE_COUNT" -gt 1 ]; then
    HOSTFILE_KEY="${JOB_ID:-$RUN_NAME}"
    HOSTFILE_DIR="$OUTPUT_DIR/_hostfile_${HOSTFILE_KEY}"
    HOSTFILE="$HOSTFILE_DIR/hostfile"
    NODE_INFO_DIR="$HOSTFILE_DIR/node_info"
    mkdir -p "$NODE_INFO_DIR"

    # `hostname -i` may print multiple addresses (RDMA aliases, lo); take the
    # first one — same as the platform template.
    CURRENT_IP=$(hostname -i | awk '{print $1}')
    echo "$CURRENT_IP" > "$NODE_INFO_DIR/node_${NODE_RANK}.ip"
    echo "[hostfile] node_rank=$NODE_RANK ip=$CURRENT_IP wrote $NODE_INFO_DIR/node_${NODE_RANK}.ip"

    if [ "$NODE_RANK" -eq 0 ]; then
        echo "[hostfile] master collecting $NODE_COUNT node IPs..."
        wait_count=0
        while : ; do
            current=$(ls "$NODE_INFO_DIR"/*.ip 2>/dev/null | wc -l)
            if [ "$current" -ge "$NODE_COUNT" ]; then
                echo "[hostfile] collected $current/$NODE_COUNT"
                break
            fi
            if [ "$wait_count" -ge 60 ]; then
                echo "[hostfile] WARN: only got $current/$NODE_COUNT after 5min, proceeding anyway" >&2
                break
            fi
            echo "[hostfile] $current/$NODE_COUNT collected, waiting..."
            sleep 5
            wait_count=$((wait_count + 1))
        done
        : > "$HOSTFILE"
        for ip_file in "$NODE_INFO_DIR"/*.ip; do
            ip=$(cat "$ip_file")
            echo "$ip slots=$PROC_PER_NODE" >> "$HOSTFILE"
        done
        echo "[hostfile] wrote $HOSTFILE:"
        cat "$HOSTFILE"
    else
        echo "[hostfile] worker $NODE_RANK waiting for master to write $HOSTFILE..."
        wait_count=0
        while [ ! -f "$HOSTFILE" ]; do
            if [ "$wait_count" -ge 60 ]; then
                echo "[hostfile] timeout waiting for hostfile" >&2
                exit 1
            fi
            sleep 5
            wait_count=$((wait_count + 1))
        done
        echo "[hostfile] worker $NODE_RANK saw hostfile after ${wait_count} polls"
    fi

    DS_LAUNCHER_ARGS=(
        --hostfile "$HOSTFILE"
        --no_ssh
        --node_rank "$NODE_RANK"
        --master_addr "$MASTER_ADDR"
        --master_port "$MASTER_PORT"
        --launcher openmpi
    )
else
    DS_LAUNCHER_ARGS=(
        --num_nodes "$NODE_COUNT"
        --node_rank "$NODE_RANK"
        --num_gpus "$PROC_PER_NODE"
        --master_addr "$MASTER_ADDR"
        --master_port "$MASTER_PORT"
    )
fi

PYTHONPATH=. deepspeed \
    "${DS_LAUNCHER_ARGS[@]}" \
    "$SCRIPT" \
    --qwen_path "$ROOT/ckpts/Qwen2.5-Omni-7B" \
    $EXTRA_ARGS \
    --train_jsonl "${TRAIN_JSONLS[@]}" \
    --eval_jsonl "$EVAL_JSONL" \
    --output_dir "$OUTPUT_DIR" \
    --per_device_train_batch_size "$PER_DEV_BS" \
    --per_device_eval_batch_size "$EVAL_BS" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --dynamic_bs_max_tokens "$DYN_BS_MAX_TOKENS" \
    --dynamic_bs_max_bs "$DYN_BS_MAX_BS" \
    --log_step_timing "$LOG_STEP_TIMING" \
    --num_train_epochs "$EPOCHS" \
    --max_steps "$MAX_STEPS" \
    --learning_rate "$LR" \
    --warmup_ratio "$WARMUP_RATIO" \
    --max_grad_norm "$MAX_GRAD_NORM" \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --save_strategy epoch \
    --save_total_limit 5 \
    --save_only_model True \
    --eval_strategy "$EVAL_STRATEGY" \
    --metric_for_best_model eval_loss \
    --greater_is_better False \
    --bf16 \
    --gradient_checkpointing \
    --gradient_checkpointing_kwargs '{"use_reentrant": true}' \
    --ddp_find_unused_parameters False \
    --remove_unused_columns False \
    --dataloader_num_workers "$DATALOADER_WORKERS" \
    --dataloader_prefetch_factor "$DATALOADER_PREFETCH" \
    --dataloader_pin_memory True \
    --dataloader_persistent_workers True \
    --report_to tensorboard \
    --logging_dir "$OUTPUT_DIR/tb" \
    --deepspeed "$DS_CONFIG" 2>&1 | tee "$OUTPUT_DIR/train.node${NODE_RANK}.log"
