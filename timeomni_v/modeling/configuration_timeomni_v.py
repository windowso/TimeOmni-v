"""Config for TimeOmni-v: Qwen2.5-Omni Thinker + Chronos-2 TS encoder.

We extend the Thinker config (not the top-level OmniConfig) because TimeOmni-v
only reuses the Thinker half of Qwen2.5-Omni — the Talker and Code2Wav stacks
are not part of this architecture.
"""

from __future__ import annotations

from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (
    Qwen2_5OmniThinkerConfig,
)


class TimeOmniVConfig(Qwen2_5OmniThinkerConfig):
    model_type = "timeomni_v"

    def __init__(
        self,
        ts_encoder_path: str = "",
        ts_encoder_hidden_size: int = 768,
        ts_adapter_hidden_size: int = 2048,
        timeseries_token_id: int | None = None,
        timeseries_start_token_id: int | None = None,
        timeseries_end_token_id: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ts_encoder_path = ts_encoder_path
        self.ts_encoder_hidden_size = ts_encoder_hidden_size
        self.ts_adapter_hidden_size = ts_adapter_hidden_size
        self.timeseries_token_id = timeseries_token_id
        self.timeseries_start_token_id = timeseries_start_token_id
        self.timeseries_end_token_id = timeseries_end_token_id
