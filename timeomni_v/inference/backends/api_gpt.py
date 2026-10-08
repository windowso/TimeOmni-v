"""GPT (OpenAI Responses API) backend.

Pattern lifted from ``TS_bench/models/gpt.py``: an OpenAI client (optionally
against a proxy URL) with image-text Responses API messages. Visual inputs
are either video (base64-encoded JPEG frames sampled by
``extract_frames_b64_jpeg``) OR one-or-more raw images (base64-encoded
directly via ``encode_image_b64_jpeg``); ``--no_vision`` falls back to
text-only ``input_text`` content.

Concurrency is provided by the orchestrator's ``ThreadPoolExecutor``; this
backend is just the per-row HTTP shape.
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


# TS_bench convention: the proxy speaks the OpenAI SDK against this URL.
_DEFAULT_PROXY_URL = "http://35.220.164.252:3888/v1"


class GptBackend(ApiBackend):
    name = "gpt"
    supports_pixel_bounds = False  # max_pixels is enforced via resize, not API knobs.
    supports_dense_frames = False  # dense frames over an API would be wasteful.

    # OpenAI Responses API rejects requests with more than 50 input_image
    # blocks ("Exceeded maximum number of images (50) allowed in the
    # request"). We sample / clip frames to this hard ceiling so a long
    # video with --max_frames=80 doesn't 400 the run.
    MAX_IMAGES_PER_REQUEST = 50

    def __init__(self, *, args, include_vision, include_timeseries):
        super().__init__(
            args=args,
            include_vision=include_vision,
            include_timeseries=include_timeseries,
        )
        self.client = None
        # ``--model_path`` is overloaded for API backends: it carries the
        # model name (e.g. ``gpt-5``).
        self.model_name = args.model_path

    def load(self, *, device_map=None) -> None:
        from openai import OpenAI

        a = self.args
        api_key = os.getenv(a.api_key_env)
        if not api_key:
            raise SystemExit(
                f"--backend gpt: env var {a.api_key_env!r} is empty",
            )
        if a.api_provider == "openai":
            base_url = a.api_base_url  # None ⇒ openai default (api.openai.com)
        else:
            base_url = a.api_base_url or _DEFAULT_PROXY_URL
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def _build_content(self, row: dict) -> list[dict[str, Any]]:
        """Mirror ``GPTImageTextCaller`` content layout: text first, then a
        run of ``input_image`` blocks (one per sampled frame for video, one
        per image for image rows)."""
        prompt_text = row["prompt"]
        if not self.include_timeseries:
            prompt_text = strip_ts_block(prompt_text)
        content: list[dict[str, Any]] = [
            {"type": "input_text", "text": prompt_text},
        ]
        if self.include_vision:
            a = self.args
            cap = self.MAX_IMAGES_PER_REQUEST
            if row.get("video_path"):
                # Clip max_frames to the API's hard 50-image cap (or less if
                # this row also carries side images, since both share the
                # same per-request budget).
                side_images = len(normalize_image_paths(row.get("image_path")))
                video_frame_cap = max(1, cap - side_images)
                frame_max = min(a.max_frames, video_frame_cap)
                frames = extract_frames_b64_jpeg(
                    row["video_path"],
                    fps=a.fps, min_frames=a.min_frames, max_frames=frame_max,
                    max_pixels=a.video_max_pixels,
                )
                for frame in frames:
                    content.append({
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{frame}",
                    })
            # If image-only rows ever exceed the cap, drop the tail rather
            # than 400 the request.
            n_video_imgs = sum(
                1 for c in content if c.get("type") == "input_image"
            )
            remaining = max(0, cap - n_video_imgs)
            for img_path in normalize_image_paths(row.get("image_path"))[:remaining]:
                b64 = encode_image_b64_jpeg(
                    img_path, max_pixels=a.image_max_pixels,
                )
                content.append({
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{b64}",
                })
        return content

    def infer_one(self, row: dict, *, max_new_tokens: int) -> str:
        # Lazy-import retry exceptions so the module imports cleanly even if
        # the openai package is missing (we'd error at load() instead).
        from openai import (
            APIConnectionError, APITimeoutError,
            InternalServerError, RateLimitError,
        )

        content = self._build_content(row)
        # Don't pass ``max_output_tokens`` to the API. For GPT-5 / o-series
        # it covers reasoning AND the answer in one budget, so any small
        # cap (e.g. the 4–16 we'd use for classification) leaves zero
        # headroom for the assistant message and the response comes back
        # with only reasoning items. Letting the model use its default cap
        # is simpler and avoids that whole class of failure.
        is_reasoning_model = (
            self.model_name.startswith("gpt-5")
            or self.model_name.startswith("o")
        )
        kwargs = {
            "model": self.model_name,
            "input": [{"role": "user", "content": content}],
        }
        if is_reasoning_model:
            kwargs["reasoning"] = {"effort": "minimal"}

        def _call():
            return self.client.responses.create(**kwargs)

        response = with_retry(
            _call,
            max_attempts=3, base_delay=1.0,
            retry_exceptions=(
                InternalServerError, RateLimitError,
                APIConnectionError, APITimeoutError,
            ),
        )
        # Per the SDK: ``response.output_text`` is the convenience accessor;
        # fall back to the structured walk for older shapes. The Responses
        # API can interleave reasoning items (whose ``content`` is None)
        # with the assistant message — a blind ``output[-1].content[0].text``
        # explodes with TypeError when the last item is a reasoning item.
        # Scan for the first item whose content carries a text field.
        text = getattr(response, "output_text", None)
        if text:
            return text
        for item in (getattr(response, "output", None) or []):
            for c in (getattr(item, "content", None) or []):
                t = getattr(c, "text", None)
                if t:
                    return t
        status = getattr(response, "status", None)
        incomplete = getattr(response, "incomplete_details", None)
        raise RuntimeError(
            f"GPT response had no extractable text "
            f"(model={self.model_name}, status={status!r}, "
            f"incomplete_details={incomplete!r}, "
            f"output={getattr(response, 'output', None)!r})",
        )


__all__ = ["GptBackend"]
