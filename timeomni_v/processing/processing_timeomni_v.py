"""TimeOmniVProcessor: extends Qwen2_5OmniProcessor with a TS modality.

The design mirrors Qwen2.5-Omni's own handling of video/audio/image one-to-one:

1. ``build_conversation`` adds a structured element
   ``{"type": "timeseries", "timeseries": csv_path}`` to user_content
   (alongside the ``{"type":"video"}`` element).
2. ``apply_chat_template`` (overridden here) rewrites that element to a
   text content emitting ``<|ts_start|><|ts_placeholder|><|ts_end|>`` —
   Qwen's Jinja template would not understand a custom ``type`` so we
   rewrite before delegating to the parent. This is the exact analog of
   Qwen emitting ``<|vision_start|><|video_pad|><|vision_end|>`` for a
   ``{"type":"video"}`` element.
3. ``__call__`` loads the CSVs, computes P (patches) × C (channels) per
   sample, and expands each single ``<|ts_placeholder|>`` in the text to
   ``P*C`` copies — directly mirroring ``replace_multimodal_special_tokens``
   in the parent processor (which does ``sample.replace(video_token,
   "<|video_placeholder|>" * N, 1)``).
4. Model forward masked_scatters TS embeddings into positions where
   ``input_ids == ts_placeholder_id`` — same scatter pattern Qwen uses for
   ``video_token_id``.

Fusion modes
------------
* ``block_adjacent`` (default): the P*C-expanded placeholders stay adjacent
  where the TS element sits in user_content (post-video, pre-text by
  default in ``build_conversation``).
* ``time_interleave``: after the parent processor has expanded video
  placeholders, we re-permute ``input_ids`` so each TS patch (``n_channels``
  placeholders) is spliced next to the video time-step it covers. Total
  token count is preserved.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from transformers.models.qwen2_5_omni.processing_qwen2_5_omni import Qwen2_5OmniProcessor

from timeomni_v.utils.tokens import TS_END, TS_PLACEHOLDER, TS_START


def load_and_pad_ts_csvs(
    csv_paths: Sequence[str],
) -> tuple[torch.Tensor, list[int], list[int]]:
    """Load a batch of CSVs and return (ts_values, n_channels_per_sample, n_timesteps_per_sample).

    ts_values: shape (sum(N_channels_i), T_max) with NaN padding on the right.
    Each CSV must have a header row. Channels = columns, time = rows. A
    column literally named ``timestamp`` is dropped — we don't currently
    consume time stamps and keeping the column would force float casting
    of date strings.
    """
    arrays: list[np.ndarray] = []
    n_channels_per: list[int] = []
    n_timesteps_per: list[int] = []
    for p in csv_paths:
        df = pd.read_csv(p)
        if "timestamp" in df.columns:
            df = df.drop(columns=["timestamp"])
        arr = df.to_numpy(dtype=np.float32)  # (T, C)
        n_timesteps_per.append(arr.shape[0])
        n_channels_per.append(arr.shape[1])
        arrays.append(arr.T)  # -> (C, T)

    t_max = max(n_timesteps_per)
    padded_rows: list[np.ndarray] = []
    for arr in arrays:
        c, t = arr.shape
        if t < t_max:
            pad = np.full((c, t_max - t), np.nan, dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=1)
        padded_rows.append(arr)
    stacked = np.concatenate(padded_rows, axis=0)  # (sum C, T_max)
    return torch.from_numpy(stacked), n_channels_per, n_timesteps_per


def interleave_video_ts_placeholders(
    video_positions: Sequence[float],
    ts_positions: Sequence[float],
    video_tokens_per_step: int,
    ts_tokens_per_step: int,
) -> list[str]:
    """Merge video and TS time groups into a single placeholder list, ordered
    by time. Ties go to video. Each group emits ``tokens_per_step``
    placeholders in a row. Returns ``["V"/"T"]`` markers (for testing).
    """
    i = j = 0
    out: list[str] = []
    while i < len(video_positions) or j < len(ts_positions):
        take_video = (
            j >= len(ts_positions)
            or (i < len(video_positions) and video_positions[i] <= ts_positions[j])
        )
        if take_video:
            out.extend(["V"] * video_tokens_per_step)
            i += 1
        else:
            out.extend(["T"] * ts_tokens_per_step)
            j += 1
    return out


FUSION_MODES = ("block_adjacent", "time_interleave")

# Chat-template output of a `{"type":"timeseries"}` element. One placeholder
# per element; __call__ expands it to P*C copies once the CSV shape is known.
TS_MARKER = f"{TS_START}{TS_PLACEHOLDER}{TS_END}"


def _rewrite_ts_elements(conversation):
    """Map every ``{"type":"timeseries", ...}`` content item to a text
    element carrying the canonical marker string. Qwen's baked Jinja
    template does not know about custom content types, so we have to
    translate before calling the parent's ``apply_chat_template``.

    Accepts either a list of messages or a list-of-lists (batched).
    """
    if not conversation:
        return conversation
    # Detect batched vs single (mirror of parent convention)
    first = conversation[0]
    if isinstance(first, list):
        return [_rewrite_ts_elements(c) for c in conversation]
    # Single conversation = list of message dicts
    out = []
    for msg in conversation:
        content = msg.get("content")
        if not isinstance(content, list):
            out.append(msg)
            continue
        new_content = []
        for c in content:
            if isinstance(c, dict) and c.get("type") == "timeseries":
                new_content.append({"type": "text", "text": TS_MARKER})
            else:
                new_content.append(c)
        out.append({**msg, "content": new_content})
    return out


class TimeOmniVProcessor(Qwen2_5OmniProcessor):
    """Extends Qwen2_5OmniProcessor with a timeseries modality.

    Attributes
    ----------
    ts_patch_size : int
        Chronos-2 patch size. Used to compute ``n_patches = ceil(T / patch_size)``.
    fusion_mode : str
        "block_adjacent" (default) keeps TS tokens where the element sits.
        "time_interleave" re-permutes to splice TS next to video time-steps.
    """

    ts_patch_size: int = 16
    fusion_mode: str = "block_adjacent"

    def apply_chat_template(self, conversation, **kwargs):
        return super().apply_chat_template(_rewrite_ts_elements(conversation), **kwargs)

    def __call__(
        self,
        *,
        text: list[str],
        videos: list | None = None,
        images: list | None = None,
        timeseries: list[str] | None = None,
        **kwargs,
    ):
        if self.fusion_mode not in FUSION_MODES:
            raise ValueError(
                f"fusion_mode must be one of {FUSION_MODES}, got {self.fusion_mode!r}"
            )

        ts_values = None
        ts_group_ids = None
        n_patches_list: list[int] = []
        n_channels_list: list[int] = []

        if timeseries is not None:
            if len(timeseries) != len(text):
                raise ValueError("len(timeseries) must equal len(text)")
            ts_values, n_channels_list, n_timesteps_list = load_and_pad_ts_csvs(timeseries)
            ts_group_ids = torch.cat([
                torch.full((c,), i, dtype=torch.long)
                for i, c in enumerate(n_channels_list)
            ])
            n_patches_list = [
                math.ceil(n / self.ts_patch_size) for n in n_timesteps_list
            ]

            # Expand the single-token TS placeholder to P*C copies per sample.
            # Mirror of Qwen's replace_multimodal_special_tokens video_token
            # expansion. A temporary rename avoids accidentally re-matching
            # freshly-inserted copies (harmless here since we use count=1,
            # but keeps the pattern identical).
            TMP = "<|ts_placeholder_tmp|>"
            expanded_text: list[str] = []
            for i, sample in enumerate(text):
                if TS_PLACEHOLDER not in sample:
                    raise RuntimeError(
                        f"sample {i}: expected exactly one {TS_PLACEHOLDER} from the "
                        "chat template's timeseries element, found none. Did "
                        "build_conversation get include_timeseries=True?"
                    )
                n = n_patches_list[i] * n_channels_list[i]
                sample = sample.replace(TS_PLACEHOLDER, TMP * n, 1)
                sample = sample.replace(TMP, TS_PLACEHOLDER)
                expanded_text.append(sample)
            text = expanded_text

        # Forward `images=` straight through to the parent processor — it
        # natively expands `<|image_pad|>` to per-image grid counts and emits
        # `pixel_values_images` / `image_grid_thw`. Only set the kwarg when
        # callers actually pass images so we don't override the parent's None
        # default for video-only calls.
        parent_kwargs = dict(kwargs)
        if images is not None:
            parent_kwargs["images"] = images
        batch = super().__call__(text=text, videos=videos, **parent_kwargs)

        if ts_values is not None:
            batch["ts_values"] = ts_values
            batch["ts_group_ids"] = ts_group_ids
            # Per-sample TS shape info — used by TimeOmniVForConditionalGeneration
            # to compute M-RoPE positions (TS gets 2D (patch_idx, channel_idx,
            # channel_idx)) and to assert placeholder counts at scatter time.
            batch["ts_n_patches"] = torch.tensor(n_patches_list, dtype=torch.long)
            batch["ts_n_channels"] = torch.tensor(n_channels_list, dtype=torch.long)

        if (
            timeseries is not None
            and self.fusion_mode == "time_interleave"
            and "video_grid_thw" in batch
        ):
            self._postprocess_time_interleave(
                batch,
                n_patches_list=n_patches_list,
                n_channels_list=n_channels_list,
            )
        return batch

    def _postprocess_time_interleave(
        self,
        batch: dict,
        n_patches_list: list[int],
        n_channels_list: list[int],
    ) -> None:
        """Re-permute input_ids so each TS patch sits adjacent to the video
        time-step it covers, wrapping the interleaved segment with
        ts_start/ts_end markers (mirroring Qwen2.5-Omni's audio-in-video
        layout where the inner-modality bos/eos brackets the interleaved
        region rather than staying at the original block position).

        Input layout (after block_adjacent expansion):
            [...prefix...] [V V ... V] [...mid...] [ts_start T T ... T ts_end] [...suffix...]

        Output layout:
            [...prefix...] [ts_start V T V T ... V T V ts_end] [...mid...] [...suffix...]
                            └──── interleaved ────┘

        Total token count is preserved → attention_mask is unchanged.

        Image-bearing samples (image+video+TS): images sit between the video
        and TS blocks in the chat template (build_conversation orders video →
        images → timeseries → text). They are non-video / non-TS tokens, so
        they pass through the tail copy untouched. Image+TS samples without
        video skip this method entirely (no `video_grid_thw` in batch).
        """
        input_ids = batch["input_ids"]
        video_grid_thw = batch["video_grid_thw"]  # (B, 3)

        video_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.video_token)
        ts_token_id = self.tokenizer.convert_tokens_to_ids(TS_PLACEHOLDER)
        ts_start_id = self.tokenizer.convert_tokens_to_ids(TS_START)
        ts_end_id = self.tokenizer.convert_tokens_to_ids(TS_END)
        merge_size = self.video_processor.merge_size

        B = input_ids.size(0)
        # `video_grid_thw` only contains rows for video-bearing samples
        # (mixed batches with image+TS samples leave their slot out). Track a
        # separate index that advances only when we consume a video row, so
        # the lookup stays aligned regardless of how many non-video samples
        # are interleaved before / between video samples.
        video_idx = 0
        new_ids_list: list[torch.Tensor] = []
        for b in range(B):
            ids = input_ids[b]
            n_patches = n_patches_list[b]
            n_channels = n_channels_list[b]

            video_positions = (ids == video_token_id).nonzero(as_tuple=True)[0]
            ts_positions = (ids == ts_token_id).nonzero(as_tuple=True)[0]
            if len(video_positions) == 0 or len(ts_positions) == 0:
                # Image-only-with-TS (mimic, pixelrec, sp500, terra) or
                # text-only sample — no video run to interleave. Don't
                # consume a video_grid_thw row for this slot.
                new_ids_list.append(ids)
                continue

            T_v = int(video_grid_thw[video_idx][0])
            H_v = int(video_grid_thw[video_idx][1])
            W_v = int(video_grid_thw[video_idx][2])
            video_idx += 1
            S_v = (H_v // merge_size) * (W_v // merge_size)
            expected_video_tokens = T_v * S_v

            v_start = int(video_positions[0])
            v_end = int(video_positions[-1]) + 1
            t_start = int(ts_positions[0])
            t_end = int(ts_positions[-1]) + 1

            if v_end - v_start != expected_video_tokens:
                raise RuntimeError(
                    f"sample {b}: video run length {v_end - v_start} != T*S "
                    f"= {expected_video_tokens}"
                )
            if t_end - t_start != n_patches * n_channels:
                raise RuntimeError(
                    f"sample {b}: ts run length {t_end - t_start} != P*C "
                    f"= {n_patches * n_channels}"
                )

            video_tokens = ids[v_start:v_end]
            ts_tokens = ids[t_start:t_end]

            patch_to_step = [
                min(T_v - 1, (i * T_v) // n_patches) for i in range(n_patches)
            ]
            step_to_patches: dict[int, list[int]] = {}
            for i, s in enumerate(patch_to_step):
                step_to_patches.setdefault(s, []).append(i)

            interleaved: list[int] = []
            for step in range(T_v):
                interleaved.extend(
                    video_tokens[step * S_v : (step + 1) * S_v].tolist()
                )
                for p_idx in step_to_patches.get(step, []):
                    interleaved.extend(
                        ts_tokens[p_idx * n_channels : (p_idx + 1) * n_channels].tolist()
                    )

            # Rebuild: prefix + ts_start + interleaved + ts_end + (tail w/o old ts_*)
            new_tokens: list[int] = ids[:v_start].tolist()
            new_tokens.append(ts_start_id)
            new_tokens.extend(interleaved)
            new_tokens.append(ts_end_id)
            for idx in range(v_end, len(ids)):
                tok = int(ids[idx])
                if tok in (ts_token_id, ts_start_id, ts_end_id):
                    continue
                new_tokens.append(tok)

            if len(new_tokens) != len(ids):
                raise RuntimeError(
                    f"sample {b}: token count changed {len(ids)} -> {len(new_tokens)}"
                )
            new_ids_list.append(
                torch.tensor(new_tokens, dtype=ids.dtype, device=ids.device)
            )

        batch["input_ids"] = torch.stack(new_ids_list)
        # attention_mask unchanged since total sequence length is preserved
