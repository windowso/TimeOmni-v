"""Build Qwen2.5-Omni chat-template conversations from TimeOmni-v rows.

Follows the official structured-element pattern used by Qwen2.5-Omni for
vision / audio and extends it to timeseries:

    user_content = [
        {"type": "video",      "video": clip_path, "fps": ..., ...},
        {"type": "image",      "image": image_path},  # zero or more
        {"type": "timeseries", "timeseries": csv_path},   # timeomni_v mode only
        {"type": "text",       "text": prompt_text},
    ]

Then:
  1. ``processor.apply_chat_template(conversation, tokenize=False)`` wraps
     turns in ``<|im_start|>role\\n...<|im_end|>\\n`` and inserts
     ``<|vision_start|><|video_pad|><|vision_end|>`` for the video element
     (and ``<|vision_start|><|image_pad|><|vision_end|>`` for each image).
     ``TimeOmniVProcessor`` overrides this so the ``{"type":"timeseries"}``
     element becomes ``<|ts_start|><|ts_placeholder|><|ts_end|>``.
  2. ``qwen_omni_utils.process_mm_info`` decodes the video and loads images.
  3. ``processor(text, videos, images, timeseries, ...)`` expands each
     single-token modality marker to the per-sample placeholder count
     (``video_grid_thw`` / ``image_grid_thw`` → T*H*W/merge² copies; ``P*C``
     copies for timeseries where P=ceil(T/patch_size), C=n_channels).
  4. Model forward does masked_scatter of modality features into positions
     where ``input_ids == <modality>_token_id``.

``prompt_transform`` is applied to ``row["prompt"]`` before it enters the
text element — baseline uses it to downsample inline TS, timeomni_v uses it to
strip the inline TS body entirely (the tensor path provides that signal).
"""

from __future__ import annotations


def _row_image_paths(row: dict) -> list[str]:
    """Normalize ``row["image_path"]`` (str | list[str] | None) to a list."""
    p = row.get("image_path")
    if p is None:
        return []
    if isinstance(p, str):
        return [p]
    return list(p)


def build_conversation(
    row: dict,
    *,
    answer: str | None,
    fps: float,
    max_frames: int,
    min_frames: int,
    video_min_pixels: int,
    video_max_pixels: int,
    image_min_pixels: int | None = None,
    image_max_pixels: int | None = None,
    include_vision: bool = True,
    include_timeseries: bool = False,
    prompt_transform=None,
    prompt_transform_skip_tasks: "frozenset[str] | set[str] | None" = None,
) -> list[dict]:
    """Return a Qwen2.5-Omni conversation list.

    ``answer`` None → user-only (inference-style, caller should then use
    ``add_generation_prompt=True``). Otherwise the assistant turn is
    appended with that exact string.

    ``include_vision=False`` drops the video AND image elements entirely
    (text-only / text+TS prompt). Used by zero-shot ablations to measure how
    much the model relies on the visual stream. The video and image rows in
    the dataset are mutually exclusive in practice — only the field that's
    actually present on the row gets emitted as a structured element.

    ``include_timeseries=True`` inserts a structured
    ``{"type": "timeseries", "timeseries": row["timeseries_path"]}`` element
    between the visual block and the text element — ``TimeOmniVProcessor``
    picks it up and emits the TS placeholder marker exactly the same way
    Qwen emits the video marker for the video element.
    """
    prompt_text = row["prompt"]
    skip_transform = (
        prompt_transform_skip_tasks is not None
        and row.get("task") in prompt_transform_skip_tasks
    )
    if prompt_transform is not None and not skip_transform:
        prompt_text = prompt_transform(prompt_text)

    user_content: list[dict] = []
    if include_vision and row.get("video_path"):
        user_content.append({
            "type": "video",
            "video": row["video_path"],
            "fps": fps,
            "max_frames": max_frames,
            "min_frames": min_frames,
            "max_pixels": video_max_pixels,
            "min_pixels": video_min_pixels,
        })
    if include_vision:
        for img_path in _row_image_paths(row):
            img_ele: dict = {"type": "image", "image": img_path}
            if image_min_pixels is not None:
                img_ele["min_pixels"] = image_min_pixels
            if image_max_pixels is not None:
                img_ele["max_pixels"] = image_max_pixels
            user_content.append(img_ele)
    if include_timeseries:
        user_content.append({"type": "timeseries", "timeseries": row["timeseries_path"]})
    user_content.append({"type": "text", "text": prompt_text})

    conv: list[dict] = [{"role": "user", "content": user_content}]
    if answer is not None:
        conv.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return conv


def apply_template_and_mm(
    processor,
    conversation: list[dict],
    *,
    add_generation_prompt: bool,
    use_audio_in_video: bool = False,
    image_patch_size: int = 14,
):
    """Run apply_chat_template + process_mm_info.

    Returns ``(text, audios, images, videos)``. ``images`` carries the loaded
    image content for any ``{"type":"image"}`` element in the conversation;
    ``videos`` carries decoded clips. ``audios`` is None for our setup.
    """
    from qwen_omni_utils import process_mm_info

    text = processor.apply_chat_template(
        conversation,
        add_generation_prompt=add_generation_prompt,
        tokenize=False,
    )
    audios, images, videos = process_mm_info(
        conversation,
        use_audio_in_video=use_audio_in_video,
        image_patch_size=image_patch_size,
    )
    return text, audios, images, videos
