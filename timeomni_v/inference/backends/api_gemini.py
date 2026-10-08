"""Gemini backend (via the OpenAI-compatible proxy).

Pattern lifted from ``TS_bench/models/gemini.py``: same proxy + OpenAI SDK as
GPT, but the chat/completions message schema with ``image_url`` (not
``input_image``) blocks. Frame extraction is shared with the GPT backend
through ``extract_frames_b64_jpeg``.
"""

from __future__ import annotations

import os
from typing import Any

from timeomni_v.data.collator import strip_ts_block
from timeomni_v.inference.backends.api_base import (
    encode_image_b64_jpeg, extract_frames_b64_jpeg,
    normalize_image_paths, with_retry,
)
from timeomni_v.inference.backends.base import ApiBackend


_DEFAULT_PROXY_URL = "http://35.220.164.252:3888/v1"


class GeminiBackend(ApiBackend):
    name = "gemini"
    supports_pixel_bounds = False
    supports_dense_frames = False

    def __init__(self, *, args, include_vision, include_timeseries):
        super().__init__(
            args=args,
            include_vision=include_vision,
            include_timeseries=include_timeseries,
        )
        self.client = None
        self.model_name = args.model_path

    def load(self, *, device_map=None) -> None:
        from openai import OpenAI

        a = self.args
        api_key = os.getenv(a.api_key_env)
        if not api_key:
            raise SystemExit(
                f"--backend gemini: env var {a.api_key_env!r} is empty",
            )
        base_url = a.api_base_url or _DEFAULT_PROXY_URL
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def _build_content(self, row: dict) -> list[dict[str, Any]]:
        prompt_text = row["prompt"]
        if not self.include_timeseries:
            prompt_text = strip_ts_block(prompt_text)
        content: list[dict[str, Any]] = [
            {"type": "text", "text": prompt_text},
        ]
        if self.include_vision:
            a = self.args
            if row.get("video_path"):
                frames = extract_frames_b64_jpeg(
                    row["video_path"],
                    fps=a.fps, min_frames=a.min_frames, max_frames=a.max_frames,
                    max_pixels=a.video_max_pixels,
                )
                for frame in frames:
                    content.append({
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{frame}",
                    })
            for img_path in normalize_image_paths(row.get("image_path")):
                b64 = encode_image_b64_jpeg(
                    img_path, max_pixels=a.image_max_pixels,
                )
                content.append({
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{b64}",
                })
        return content

    def infer_one(self, row: dict, *, max_new_tokens: int) -> str:
        from openai import (
            APIConnectionError, APITimeoutError,
            InternalServerError, RateLimitError,
        )

        content = self._build_content(row)

        # No max_completion_tokens: API models manage their own answer cap;
        # passing a small one for classification could starve responses on
        # reasoning-style models like GPT-5 (and is harmless to omit on
        # Gemini).
        def _call():
            return self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": content}],
            )

        response = with_retry(
            _call,
            max_attempts=3, base_delay=1.0,
            retry_exceptions=(
                InternalServerError, RateLimitError,
                APIConnectionError, APITimeoutError,
            ),
        )
        return response.choices[0].message.content


__all__ = ["GeminiBackend"]
