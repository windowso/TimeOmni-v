# TimeOmni-v

TimeOmni-v extends Qwen2.5-Omni-7B (Thinker half) with a **time-series (TS) modality** by integrating the pretrained Chronos-2 encoder. TS becomes a first-class modality alongside vision and audio: a structured chat element, a placeholder token (`<|ts_placeholder|>`), an encoder branch, and an M-RoPE layout that is time-aligned with co-occurring video. The same checkpoint handles **video + TS** and **image(s) + TS** inputs, and supports both **classification** and **forecasting** outputs.

This is the open-source release of the TimeOmni-v codebase (Python package: `timeomni_v`). It contains the model, processing, training, and inference code; the unified data-loading layer; the train/eval entry points; and the zero-shot baseline harnesses for comparison models. Internal infra (cluster launchers, per-dataset wrapper scripts, supplementary notes) has been stripped — what remains is the core that runs anywhere a single multi-GPU node is available.

This README is organized for a reader who wants to understand: (a) what data was used, (b) what the data files look like on disk, (c) how the system is evaluated, and (d) how TimeOmni-v is trained and evaluated end-to-end.

---

## 1. Datasets

TimeOmni-v is trained and evaluated on **9 datasets** (mimic_death + mimic_disch share modality/format and are sometimes reported jointly as "mimic"). Five pair TS with video; four pair TS with images. The two task types — classification (single-label) and forecasting (numerical sequence) — are scored with separate metric suites.

| Dataset             | Visual modality | Task           | Channels (TS) | Notes |
|---------------------|-----------------|----------------|--------------:|-------|
| **agibot**          | Video + TS      | Classification | 10            | Longest videos / longest TS in the corpus; raw-text TS would consume ~100K tokens per sample — main motivation for the encoder branch. |
| **covla**           | Video + TS      | Classification | 5             | Driving (CoVLA): per-clip vehicle telemetry; brake-reason classification. |
| **cuhk_x_har**      | Video + TS      | Classification | 7             | Sub-second human activity clips; TS sample rate higher than video frame rate. |
| **future_factories**| Video + TS      | Classification | 36            | Industrial sensor stream; only fixed-length TS (16 steps), highest channel count. |
| **holo_assist**     | Video + TS      | Classification | 9             | Egocentric assistive-task video + IMU. |
| **mimic_death**     | Image(s) + TS   | Classification | 6             | MIMIC clinical record: chest X-ray(s) + 6 vital signs → mortality. |
| **mimic_disch**     | Image(s) + TS   | Classification | 6             | Same corpus → discharge label. |
| **pixelrec**        | Image + TS      | Forecasting    | 1             | Univariate, image-conditioned forecasting. |
| **sp500**           | Image + TS      | Forecasting    | 1             | S&P 500 univariate forecasting. |
| **terra**           | Multi-image + TS| Forecasting    | 7             | Multi-view satellite/relief imagery (4–7 images / sample). |

---

## 2. Data format on disk

A **single unified jsonl** drives both training and evaluation. Each row carries:

- `id`, `task` (`"classification"` | `"prediction"`), `prompt`, `answer` — required.
- One of `video_path` (single short mp4 clip) or `image_path` (single string or list of strings).
- `timeseries_path` — points to a per-sample CSV (required for the TimeOmni-v branch).
- The inline `<timeseries>...</timeseries>` block stays in `prompt` so the same file is also consumable by zero-shot models that read TS as text.
- Optional cached `video_meta` / `image_meta` / `ts_shape` so the length estimator skips ffprobe / image-open / csv-scan at startup.

Examples:

```jsonc
// video + TS — CoVLA brake-reason classification
{
  "id": "abc",
  "task": "classification",
  "video_path": "CoVLA-Dataset/video_clips/abc_100_200.mp4",
  "timeseries_path": "CoVLA-Dataset/timeseries/abc_100_200.csv",
  "prompt": "...<video>...</video>...<timeseries>\n25.5: vEgo=6.517, ...\n</timeseries>...",
  "answer": "A",
  "video_meta": {"width": 1928, "height": 1208, "source_fps": 20.0, "nb_frames": 93, "duration": 4.65},
  "ts_shape": [93, 5]
}

// image(s) + TS — terra multi-view forecasting
{
  "id": "terra_test200_1_1ccbc81d48",
  "task": "prediction",
  "image_path": ["…/img_satellite/…png", "…/img_relief/…png", …],
  "timeseries_path": "…/csv/terra_test200_1_1ccbc81d48.csv",
  "prompt": "...<image>...</image>...",
  "answer": "<forecast>\n(2023-12-12: 421.6, 0.7, …)\n…\n</forecast>",
  "image_meta": [{"width": 728, "height": 739, "channels": 3}, …],
  "ts_shape": [100, 8]
}
```

**CSV conventions** (`timeseries_path`):
- Header row = channel names (free-form; not consumed by the model).
- Each subsequent row = one time step; rows assumed uniformly sampled across the visual segment.
- Cells = raw floats (no normalization — Chronos-2 does per-variate instance norm internally).
- Empty cell = NaN; Chronos-2 masks NaN automatically.
- No time column. Channel count and step count vary across samples.

**Layout on disk**

```
data/
├── video_ts/jsonl/                         # video + TS datasets
│   ├── covla.jsonl,        covla_test.jsonl
│   ├── agibot.jsonl,       agibot_test.jsonl
│   ├── cuhk_x_har.jsonl,   cuhk_x_har_test.jsonl
│   ├── holo_assist.jsonl,  holo_assist_test.jsonl
│   └── future_factories.jsonl, future_factories_test.jsonl
├── image_ts/jsonl/                         # image(s) + TS datasets
│   ├── mimic_death.jsonl,  mimic_death_test.jsonl
│   ├── mimic_disch.jsonl,  mimic_disch_test.jsonl
│   ├── pixelrec.jsonl,     pixelrec_test.jsonl
│   ├── sp500.jsonl,        sp500_test.jsonl
│   └── terra.jsonl,        terra_test.jsonl
└── merged_classification/jsonl/           # cross-dataset merged eval splits
    ├── cls_all_test.jsonl          # 50 rows / source, all 7 cls datasets
    └── cls_no_cuhk_test.jsonl
```

A file may also carry one extra dot-tag before `.jsonl` (e.g. `covla_test.<tag>.jsonl`, as in the released MMTA archive); the scripts resolve either form via `scripts/_jsonl.sh`, so downloaded files can be used without renaming.

The collator strips the inline `<timeseries>...</timeseries>` body from the prompt and reads `timeseries_path` → TS-as-tensor (Chronos-2).

---

## 3. Evaluation protocol

Inference produces a **`predictions.jsonl`** (one row per test sample with fields `id`, `task`, `ground_truth`, `prediction`, `raw`); scoring then consumes it. Both stages are wrapped by `scripts/eval.sh`.

### 3.1 Metrics

Decided automatically from each sample's `task` field:

- **Classification** (`timeomni_v/inference/eval.py`): accuracy, per-class precision / recall / F1, macro-F1, weighted-F1, UAR (unweighted average recall), full confusion matrix. Unparseable model outputs go into a dedicated `None` column and lower the parse rate but do not silently fall into a default class. When the strict label match fails, the parser falls back to the longest valid label that prefixes the head of the output (e.g. `"BB\nHuman:"` → `B`, `"3232"` → `32`): with greedy decoding capped by `--max_new_tokens` instead of stopped at EOS, a fine-tuned model's first-token answer can be fused with trailing noise. The fallback is anchored at position 0 and only ever turns `None` into a label, never rewrites a strictly parsed answer; it is applied identically to all models. Pass `prefix_fallback=False` to `build_parser` for strict-only parsing.
- **Forecasting** (`timeomni_v/inference/regression_metrics.py`): MAE, MSE, MAPE, row-wise PCC, plus per-channel breakdowns. The `<forecast>...</forecast>` parser aligns model output to the ground-truth schedule; rows that fail to parse are excluded from numeric averages and reported via `parse_rate`.

### 3.2 Standard evaluation sweep

`scripts/eval.sh` runs **per-(ablation, dataset)** evaluation and rolls everything up into one summary CSV. It honors three ablation flags supported by the inference script:

| Ablation        | What it removes                              | Why it matters                                   |
|-----------------|----------------------------------------------|--------------------------------------------------|
| `full`          | nothing — vision + TS encoder                | Reference configuration.                         |
| `no_vision`     | drops the visual element entirely            | Isolates TS contribution.                        |
| `no_timeseries` | drops the structured TS element & inline TS  | Isolates vision contribution.                    |

For each `(ablation, dataset)` pair, predictions land at `runs/<RUN_NAME>/<ablation>/<dataset>/predictions.jsonl` and metrics at `metrics.json` next to it. After the sweep, `scripts/summarize_eval.py` writes `runs/<RUN_NAME>/summary.csv` with one row per (ablation, dataset) and all metrics flattened. Compare two runs with `python -m timeomni_v.inference.compare`.

### 3.3 Running an evaluation

```bash
# Score one trained run against the standard classification suite, all 3 ablations.
RUN_NAME=timeomni_v-cls_all-lr1e-5-ep10  bash scripts/eval.sh

# Pick a different test list, single ablation:
DATASET_GROUP=image  ABLATIONS=("full")  bash scripts/eval.sh

# Re-score an existing predictions.jsonl after fixing the parser (no re-inference):
RUN_INFER=false  RUN_REPARSE=true  bash scripts/eval.sh
```

Token-budget batching mirrors training (`DYN_BS_MAX_TOKENS` measured in *padded* tokens). Generation length is capped by `MAX_NEW_TOKENS` (8 for letter classification; for image-forecasting tasks like `terra` the answers reach ~1300 Qwen2.5-Omni tokens — set `MAX_NEW_TOKENS=1408` accordingly).

---

## 4. Zero-shot evaluation of base models

A separate entry point, **`scripts/eval_zero_shot.sh`**, evaluates off-the-shelf models on the same test sweep — no LoRA adapter, no Chronos branch, TS injected as inline text inside the prompt. It is the reference for "what does a base model already know how to do here, before any TS-specific training?". Outputs land under `runs/<RUN_DIR_NAME>/<backend>/<ablation>/<dataset>/` and never collide with trained-run artifacts.

### 4.1 Supported backends

| Backend         | Model                              | Wrapper                              |
|-----------------|------------------------------------|--------------------------------------|
| `qwen2_5_omni`  | Qwen2.5-Omni-7B                    | `scripts/zero_shot/qwen2_5_omni.sh`  |
| `qwen3_omni`    | Qwen3-Omni-30B-A3B-Instruct        | `scripts/zero_shot/qwen3_omni.sh`    |
| `qwen3_vl_8b`   | Qwen3-VL-8B-Instruct               | `scripts/zero_shot/qwen3_vl_8b.sh`   |
| `qwen3_vl_30b`  | Qwen3-VL-30B-A3B-Instruct          | `scripts/zero_shot/qwen3_vl_30b.sh`  |
| `internvl_4b`   | InternVL3.5-4B-HF                  | `scripts/zero_shot/internvl_4b.sh`   |
| `internvl_8b`   | InternVL3.5-8B-HF                  | `scripts/zero_shot/internvl_8b.sh`   |
| `gpt`           | GPT (default `gpt-5-mini`, OpenAI-compatible) | `scripts/zero_shot/gpt.sh`     |
| `gemini`        | Gemini (default `gemini-2.5-flash`, OpenAI-compatible proxy) | `scripts/zero_shot/gemini.sh` |

Each wrapper sets the right `BACKEND`, `BACKEND_MODEL_PATH_*`, and a sensible `DATASET_GROUP` default, then `exec`s `eval_zero_shot.sh`. Backend dispatch lives in `timeomni_v/inference/backends/` (one file per family).

### 4.2 Running zero-shot

```bash
# Default: Qwen2.5-Omni-7B over the image test sweep, three ablations.
bash scripts/zero_shot/qwen2_5_omni.sh

# Same backend, video sweep instead.
DATASET_GROUP=video  bash scripts/zero_shot/qwen2_5_omni.sh

# API-backed run (uses OPENAI_API_KEY by default; set API_BASE_URL for a proxy).
bash scripts/zero_shot/gemini.sh

# Sweep multiple backends in one invocation:
BACKENDS=(qwen2_5_omni qwen3_omni gpt) DATASET_GROUP=both bash scripts/eval_zero_shot.sh

# Single-ablation, custom test list:
ABLATIONS=("full") TEST_JSONLS_STR="data/.../terra_test.jsonl" \
  bash scripts/zero_shot/qwen3_vl_8b.sh
```

The same three ablations from §3 (`full | no_vision | no_timeseries`) apply, picked via `ABLATIONS=(...)`. The script writes a per-backend `runs/<RUN_DIR_NAME>/<backend>/summary.csv` after each backend's sweep finishes, so a partial run is still consumable. API backends additionally checkpoint partial predictions every `API_CHECKPOINT_EVERY` rows so a killed run resumes from the last checkpoint.

One operational note: `qwen3_omni` requires `transformers>=5.2`, while every other backend pins `4.57.x`. The script auto-installs `>=5.2` only for that backend's iteration and restores the baseline after (with an `EXIT` trap as a safety net), so you can sweep `qwen3_omni` alongside the others in a single command.

---

## 5. Training and evaluating TimeOmni-v

Training and inference share the same Qwen2.5-Omni base. The TimeOmni-v branch adds:

- **`ts_tower`** — pretrained Chronos-2 encoder (full fine-tune).
- **`ts_adapter`** — 2-layer MLP `768 → 2048 → 2048` projecting Chronos-2 features into Qwen's hidden space (full fine-tune).
- **LLM** — Qwen2.5-Omni Thinker (dense 7B), LoRA on `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`.
- **Vision / audio encoders + adapters** — frozen.
- **(Optional) forecast head** — lightweight projection from a fixed window of last-layer hidden states (anchored at the first TS token) directly to `pred_len × n_channels`, trained against MSE in normalized space. Enabled by `--pred_len > 0`; classification runs leave it off.

TS tokens are scattered into the LLM's input embeddings at positions where `input_ids == ts_placeholder_id`, mirroring how the base model handles vision/audio. M-RoPE for TS is `(t, c, c)` — temporal index plus channel index duplicated into the spatial slots — and when TS coexists with video, the TS time coordinate is rescaled to match the video frame it sits next to.

### 5.1 Fusion modes

- **`time_interleave`** (default): each TS patch (`n_channels` placeholders) is spliced next to the video frame it temporally covers, wrapped by `<|ts_start|>` / `<|ts_end|>`. Mirrors Qwen2.5-Omni's audio-in-video layout. Used for video + TS datasets.
- **`block_adjacent`**: TS placeholders form a contiguous block adjacent to the visual block. Used (automatically) when the visual modality is image(s), since images carry no time axis.

The same trained checkpoint serves both layouts; choice is automatic from the input modality.

### 5.2 Training command

```bash
bash scripts/train.sh <DATASET>
```

`DATASET` is one of:

- Single video set: `covla | agibot | cuhk_x_har | holo_assist | future_factories`
- Single image set: `mimic_death | mimic_disch | pixelrec | sp500 | terra`
- Per-channel forecasting variant: `terra_perch | sp500_perch | pixelrec_perch`
- Cross-dataset merged: `all | all_no_cuhk | cls_all | cls_no_cuhk | mimic_all`

Headline knobs (override via env var on the train.sh command line):

| Env var          | Default            | Meaning                                                                  |
|------------------|--------------------|--------------------------------------------------------------------------|
| `LR`             | `1e-5`             | Learning rate (LoRA + Chronos-2 share the same LR in v1).                |
| `EPOCHS`         | `10`               | Training epochs.                                                         |
| `PRED_LEN`       | `0` (cls)          | >0 enables the forecast head; per-dataset defaults set automatically.    |
| `DYN_BS_MAX_TOKENS` | `40000`         | Token-budget batching ceiling, **measured in padded tokens**.            |
| `EVAL_STRATEGY`  | `epoch`            | `"no" | "epoch" | "steps"` for mid-training eval.                        |

Under the hood `scripts/train.sh` calls `deepspeed -m timeomni_v.training.train` with ZeRO-2 (`timeomni_v/training/configs/ds_zero2.json`), bf16, gradient checkpointing, `--remove_unused_columns False` (mandatory — TS tensors are non-standard columns the HF Trainer would otherwise drop), and the token-budget batch sampler (`timeomni_v/data/dynamic_bs.py::TokenBudgetBatchSampler`) keeping ranks balanced across variable-length samples.

The flagship multi-dataset run is `cls_all` — all seven classification datasets merged, one source per batch (TokenBudgetBatchSampler constrains every batch to one source so collator padding is dominated by intra-source variance, not cross-source).

### 5.3 End-to-end recipe

```bash
# 1. (One-off) convert raw datasets into the unified jsonl + sliced clips.
python -m timeomni_v.data.convert_covla --in-jsonl … --out-jsonl … --csv-dir … --clip-dir …

# 2. Train. Output: runs/<RUN_NAME>/checkpoint-*/ (LoRA adapter + ts_tower + ts_adapter).
bash scripts/train.sh cls_all

# 3. Evaluate. Output: runs/<RUN_NAME>/{full,no_vision,no_timeseries}/<dataset>/{predictions.jsonl,metrics.json}
#                  + runs/<RUN_NAME>/summary.csv
RUN_NAME=timeomni_v-cls_all-lr1e-5-ep10  bash scripts/eval.sh

# 4. (Optional) zero-shot reference run on the same test sweep — see §4.
DATASET_GROUP=classification  bash scripts/zero_shot/qwen2_5_omni.sh

# 5. (Optional) compare two runs side-by-side.
python -m timeomni_v.inference.compare runs/A/summary.csv runs/B/summary.csv
```

---

## 6. Reproducing the headline result

To reproduce the main classification number end-to-end, run in order:

```bash
# Train the cls_all checkpoint (7 classification datasets jointly, ~10 epochs).
DATASET=cls_all  bash scripts/train.sh

# Evaluate on the standard 7-dataset classification test suite, three ablations.
RUN_NAME=timeomni_v-cls_all-lr1e-5-ep10  DATASET_GROUP=classification  bash scripts/eval.sh

# Zero-shot reference (any backend; sweep multiple via BACKENDS=(...)).
DATASET_GROUP=classification  bash scripts/zero_shot/qwen2_5_omni.sh
```

Single-dataset training is just `bash scripts/train.sh <dataset>`; the matching test split is auto-discovered by `eval.sh` from `RUN_NAME`'s training set when `DATASET_GROUP` isn't specified.

The resulting `runs/<RUN_NAME>/summary.csv` is the canonical artifact for paper tables — one row per `(ablation, dataset)` with accuracy / macro-F1 / weighted-F1 / UAR / parse-rate, plus the regression suite for forecasting datasets.

---

## 7. Repository map

```
timeomni_v/                     # library code
├── modeling/               # TimeOmniVConfig, TimeOmniVForConditionalGeneration, ts_tower, ts_adapter, forecast head
├── processing/             # TimeOmniVProcessor (fusion_mode, M-RoPE, placeholder rewriting)
├── data/                   # dataset, collator, jsonl converters, token-budget sampler, length estimator
├── training/               # train.py, DynamicBSTrainer, ZeRO configs
├── inference/              # infer.py + eval.py + parser/regression metrics + compare.py
│   └── backends/           # zero-shot backends: Qwen2.5-Omni / Qwen3-Omni / Qwen3-VL / InternVL / GPT / Gemini
└── utils/                  # TS special-token helpers, frozen-tower wrapping
scripts/
├── train.sh, eval.sh, eval_zero_shot.sh, summarize_eval.py
└── zero_shot/              # one launcher per zero-shot backend
data/                       # converted jsonls + per-sample CSVs (see §2; not under version control)
ckpts/                      # Qwen2.5-Omni-7B and Chronos-2 weights (not under version control)
runs/                       # training outputs, predictions, summary.csv (not under version control)
tests/                      # CPU-only smoke tests for processing, dataset, sampler, parsers, RoPE
```

For the exact M-RoPE math, `masked_scatter` flow in the forward pass, length-estimator contract, and the precise list of frozen parameters, the source of truth is the code itself — `timeomni_v/modeling/`, `timeomni_v/processing/`, and `timeomni_v/data/length_estimate.py` are the relevant entry points.

---

## 8. Citation

If you use MMTA or TimeOmni-v in your research, please cite:

```bibtex
@inproceedings{zhang2026mmta,
  title     = {{MMTA}: Benchmarking Multimodal Temporal Analysis with Time Series, Text, and Vision},
  author    = {Zhang, Ziyang and Li, Shenyi and Wang, Yilin and Cui, Ziyun and Zhou, Bowen and Wu, Wen and Zhang, Chao},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS), Evaluations and Datasets Track},
  year      = {2026}
}
```
