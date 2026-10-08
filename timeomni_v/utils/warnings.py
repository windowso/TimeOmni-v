"""Project-wide warning suppressions.

`silence_rope_scaling_warning` mutes the noisy "Unrecognized keys in
`rope_scaling` for 'rope_type'='default'" message that transformers emits for
every thinker/audio/vision sub-config of Qwen2.5-Omni. These keys
(`mrope_section`, `mrope_interleaved`, `interleaved`) are genuinely consumed
by Qwen2.5-Omni's modeling code — transformers' generic rope validator just
doesn't know about them. The warning is purely cosmetic and would print 6×
per process.

`silence_qwen_audio_system_prompt_warning` mutes the
"System prompt modified, audio output may not work as expected" message that
Qwen2_5OmniProcessor.apply_chat_template emits via the root `logging` logger
on every single call. We use the Thinker-only path (no audio output), so the
warning is irrelevant — but it fires once per processor invocation (per
sample, per length-estimate pass, per inference call) and floods the log.
"""

from __future__ import annotations

import logging


class _RopeScalingKeysFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "Unrecognized keys in `rope_scaling`" not in record.getMessage()


class _QwenAudioSystemPromptFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "System prompt modified, audio output may not work as expected" not in record.getMessage()


def silence_rope_scaling_warning() -> None:
    logger = logging.getLogger("transformers.modeling_rope_utils")
    if not any(isinstance(f, _RopeScalingKeysFilter) for f in logger.filters):
        logger.addFilter(_RopeScalingKeysFilter())


def silence_qwen_audio_system_prompt_warning() -> None:
    """Filter the root logger — the warning is emitted via `logging.warning(...)`
    on the root logger, not a named logger, so we attach the filter there."""
    logger = logging.getLogger()
    if not any(isinstance(f, _QwenAudioSystemPromptFilter) for f in logger.filters):
        logger.addFilter(_QwenAudioSystemPromptFilter())
