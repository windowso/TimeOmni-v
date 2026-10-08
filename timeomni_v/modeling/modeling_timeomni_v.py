"""TimeOmni-v = Qwen2.5-Omni Thinker + Chronos-2 TS encoder + TS adapter.

The HF `from_pretrained` on the Qwen checkpoint will emit a warning like
"Some weights of TimeOmniVForConditionalGeneration were not initialized from the
model checkpoint ... and are newly initialized: [ts_tower.*, ts_adapter.*]".
That warning is *expected* — ts_tower isn't in the Qwen checkpoint so HF
random-inits those slots. We then reload Chronos-2 weights into ts_tower.inner
in our overridden `from_pretrained`, and print a verification summary so the
warning is visibly superseded.

Construction: load like the parent, then instantiate ts_tower and ts_adapter.
Forward: compute TS features, scatter into the positions where
`input_ids == config.timeseries_token_id`, then delegate the rest to the
parent class's forward (passing `inputs_embeds` so the parent does not re-embed).

Key findings from reading the parent forward
--------------------------------------------
* Parent `forward` accepts BOTH `input_ids` and `inputs_embeds` at the same time.
  It checks `if inputs_embeds is None` to decide whether to embed; if we pass
  pre-built `inputs_embeds` it skips that step but still uses `input_ids` for
  placeholder-mask computation (`get_placeholder_mask`) and RoPE-delta computation
  (`get_rope_index`).  So passing both is both safe and correct.

* Vision/audio scatter uses `tensor.masked_scatter(mask, features)` where `mask`
  is produced by `get_placeholder_mask(input_ids, inputs_embeds, ...)`.  The mask
  is a 3-D bool tensor `(B, seq_len, D)` obtained by `.unsqueeze(-1).expand_as(inputs_embeds)`.

* The parent calls `self.model(...)` directly (not a deeper `language_model`); our
  subclass never needs to bypass it — we just ensure `inputs_embeds` carries the
  TS features before the parent's first `if inputs_embeds is None` check runs.

TS scatter approach used here
------------------------------
We replicate the same `masked_scatter` pattern used for audio/vision but driven by
`input_ids == config.timeseries_token_id`.  The 1-D boolean mask is expanded to 3-D
before calling `masked_scatter`, mirroring the parent's own logic exactly.
"""

from __future__ import annotations

import torch
from chronos import Chronos2Pipeline
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniThinkerForConditionalGeneration,
)

from timeomni_v.modeling.configuration_timeomni_v import TimeOmniVConfig
from timeomni_v.modeling.forecast_head import ForecastHead
from timeomni_v.modeling.ts_adapter import TsAdapter
from timeomni_v.modeling.ts_encoder import TsEncoder


class TimeOmniVForConditionalGeneration(Qwen2_5OmniThinkerForConditionalGeneration):
    config_class = TimeOmniVConfig

    def __init__(self, config: TimeOmniVConfig):
        super().__init__(config)
        # Build the TS encoder structure empty at __init__ time. Real Chronos-2
        # weights are loaded via load_ts_weights() AFTER from_pretrained has
        # finished — otherwise HF's _init_weights pass re-initializes them.
        self.ts_tower: TsEncoder | None = None
        if getattr(config, "ts_encoder_path", None):
            pipe = Chronos2Pipeline.from_pretrained(config.ts_encoder_path)
            inner = pipe.model
            patch_size = getattr(
                getattr(inner, "chronos_config", inner.config), "input_patch_size", 16
            )
            self.ts_tower = TsEncoder(inner, patch_size=patch_size)
        self.ts_adapter = TsAdapter(
            in_dim=config.ts_encoder_hidden_size,
            hidden_dim=config.ts_adapter_hidden_size,
            out_dim=config.text_config.hidden_size,
        )
        # Forecast head is attached lazily via attach_forecast_head() so
        # classification runs (the majority) don't allocate it. Stored on
        # self.config so save/load roundtrips can reconstruct the head.
        self.forecast_head: ForecastHead | None = None
        head_cfg = getattr(config, "forecast_head_config", None)
        if head_cfg:
            self.attach_forecast_head(
                hidden_size=head_cfg["hidden_size"],
                window=head_cfg["window"],
                pred_len=head_cfg["pred_len"],
                dropout=head_cfg.get("dropout", 0.0),
            )

    def attach_forecast_head(
        self,
        *,
        hidden_size: int,
        window: int,
        pred_len: int,
        dropout: float = 0.0,
    ) -> None:
        """Allocate the forecasting head + record its config on ``self.config``
        so a future ``from_pretrained`` of the saved model rebuilds it.
        """
        head = ForecastHead(
            hidden_size=hidden_size,
            window=window,
            pred_len=pred_len,
            dropout=dropout,
        )
        # Match dtype/device to the LM (so masked_scatter etc don't cast).
        emb = self.get_input_embeddings()
        head = head.to(device=emb.weight.device, dtype=emb.weight.dtype)
        self.forecast_head = head
        self.config.forecast_head_config = {
            "hidden_size": int(hidden_size),
            "window": int(window),
            "pred_len": int(pred_len),
            "dropout": float(dropout),
        }

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """Load the Qwen weights, then re-load Chronos-2 weights into ts_tower
        and re-apply the small ts_adapter.out_norm init.

        HF's `from_pretrained` calls `_init_weights` on parameters whose keys
        aren't in the base checkpoint. Two things break otherwise:

        1. ts_tower's freshly-loaded Chronos weights get re-randomized by HF.
           We reload them here from `config.ts_encoder_path` after the base
           load.

        2. ts_adapter.out_norm.weight gets forced to 1.0 by HF's
           ``"RMSNorm" in module.__class__.__name__`` substring check
           (transformers/modeling_utils.py:2935-2944). That undoes the small
           init we set in TsAdapter.__init__ (~0.013) and makes TS-token
           embeddings ~77× larger than text embeddings, blowing up CE loss
           to thousands. We re-apply the intended init here.
        """
        model = super().from_pretrained(*args, **kwargs)
        ts_encoder_path = getattr(model.config, "ts_encoder_path", None)
        if ts_encoder_path and model.ts_tower is not None:
            pipe = Chronos2Pipeline.from_pretrained(ts_encoder_path)
            # Under device_map="auto" ts_tower's parameters may be scattered
            # across shards. Copy each tensor onto its target param's device/
            # dtype before load_state_dict, so we never collapse shards onto a
            # single device. For single-device loads this is a no-op.
            target_sd = model.ts_tower.inner.state_dict()
            src_sd = {}
            for k, v in pipe.model.state_dict().items():
                if k in target_sd:
                    tgt = target_sd[k]
                    src_sd[k] = v.to(device=tgt.device, dtype=tgt.dtype)
                else:
                    src_sd[k] = v
            missing, unexpected = model.ts_tower.inner.load_state_dict(
                src_sd, strict=True
            )
            print(
                f"[CHRONOS] reloaded ts_tower.inner from {ts_encoder_path} | "
                f"missing={missing} unexpected={unexpected}",
                flush=True,
            )
            del pipe, src_sd
        if hasattr(model, "ts_adapter") and hasattr(model.ts_adapter, "reset_out_norm_init"):
            model.ts_adapter.reset_out_norm_init()
            with torch.no_grad():
                w_mean = model.ts_adapter.out_norm.weight.mean().item()
            print(
                f"[TS_ADAPTER] re-applied out_norm init "
                f"(weight={w_mean:.5f}, undoing HF _init_weights' fill_(1.0))",
                flush=True,
            )
        return model

    def _encode_ts(
        self,
        ts_values: torch.Tensor,
        ts_group_ids: torch.Tensor,
        ts_n_patches: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run Chronos-2 encoder, then flatten per-sample time-first → adapter.

        Input:
            ts_values     (sum_C_batch, T_steps) — NaN-padded to T_max when bs>1
            ts_group_ids  (sum_C_batch,) — each sample's rows share a group id
            ts_n_patches  (B,) — per-sample real patch count; required when
                                 samples have different lengths (bs>1).
        Output:
            (sum_ts_tokens_batch, D_llm) — time-first: [t0_c0,..,t0_cK, t1_c0,...]

        Why we slice by ts_n_patches: the collator right-pads shorter samples
        with NaN to T_max so the encoder sees a rectangular (N, T_max) tensor.
        The encoder therefore emits P_max = ceil(T_max/patch_size) patches per
        variate for *every* sample, but the prompt was built with each sample's
        true P_i = ceil(T_i/patch_size). Without trimming, feats has
        sum(P_max*C_i) tokens while the prompt has sum(P_i*C_i) slots — scatter
        mismatches for bs>1 whenever TS lengths differ.
        """
        # With device_map="auto" the ts_tower may land on a specific shard; move
        # inputs to match. Use the ts_tower's own parameter device.
        ts_device = next(self.ts_tower.parameters()).device
        ts_values = ts_values.to(ts_device)
        ts_group_ids = ts_group_ids.to(ts_device)

        # Pass ts_values positionally: when PEFT wraps ts_tower in
        # AuxiliaryTrainingWrapper (via modules_to_save), its forward signature
        # is `forward(self, x, *args, **kwargs)` — the first arg must be
        # positional. Keyword-only calls raise "missing 1 required positional
        # argument: 'x'".
        feats = self.ts_tower(ts_values, group_ids=ts_group_ids)
        # feats: (sum_C_batch, P_max, 768)
        per_sample_out: list[torch.Tensor] = []
        unique_ids = torch.unique_consecutive(ts_group_ids).tolist()
        for s in unique_ids:
            mask = ts_group_ids == s  # already on ts_device
            sample = feats[mask]                              # (C_s, P_max, 768)
            if ts_n_patches is not None:
                p_true = int(ts_n_patches[s].item())
                sample = sample[:, :p_true, :]                # trim NaN-pad tail
            c, p, d = sample.shape
            flat = sample.transpose(0, 1).reshape(p * c, d)   # time-first
            per_sample_out.append(flat)
        merged = torch.cat(per_sample_out, dim=0)             # (sum_tokens, 768)

        # Adapter may be on yet another device under auto sharding.
        adapter_device = next(self.ts_adapter.parameters()).device
        return self.ts_adapter(merged.to(adapter_device))     # (sum_tokens, D_llm)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        ts_values: torch.Tensor | None = None,
        ts_group_ids: torch.LongTensor | None = None,
        ts_n_patches: torch.LongTensor | None = None,
        ts_n_channels: torch.LongTensor | None = None,
        labels: torch.LongTensor | None = None,
        forecast_target: torch.Tensor | None = None,
        forecast_target_mask: torch.Tensor | None = None,
        forecast_target_lens: torch.LongTensor | None = None,
        **kwargs,
    ):
        """Inject TS features (if present) into inputs_embeds, then delegate to parent.

        The parent's forward:
        1. Skips `get_input_embeddings()(input_ids)` when `inputs_embeds` is not None.
        2. Still uses `input_ids` (when provided) for placeholder-mask computation
           in `get_placeholder_mask` and for RoPE-delta computation in `get_rope_index`.

        Therefore we can safely pass both `input_ids` and our pre-built `inputs_embeds`
        to `super().forward()`.  The parent will use the embeddings we provide (with TS
        features already scattered in) and use `input_ids` for all mask/position logic.

        ts_n_patches / ts_n_channels are produced by `TimeOmniVProcessor` and used by
        our overridden `get_rope_index` to give TS tokens 2D positions (time +
        channel) while keeping V tokens on full 3D vision M-RoPE. We stash them
        on `self` so the parent's call into `self.get_rope_index(...)` can reach
        them without extending the parent's signature.
        """
        # Stash TS shape info for get_rope_index to consume. Always assign (even
        # to None) so a stale value from a previous batch doesn't leak.
        self._ts_n_patches = ts_n_patches
        self._ts_n_channels = ts_n_channels

        # transformers 4.57's Qwen2_5OmniThinkerForConditionalGeneration.forward
        # calls self.loss_function(logits, labels, vocab_size) WITHOUT forwarding
        # **kwargs, so num_items_in_batch (which HF Trainer ships in **kwargs to
        # enable token-weighted loss across ranks) is silently dropped. Result:
        # per-rank `mean` reduction, then Trainer's average_tokens_across_devices
        # multiplies by num_processes → logged loss + grad_norm both inflated by
        # world_size on multi-GPU runs. Rebind loss_function for this call so
        # ForCausalLMLoss receives the global token count and uses `sum / N`.
        nib = kwargs.pop("num_items_in_batch", None)
        if nib is not None:
            from transformers.loss.loss_utils import ForCausalLMLoss

            def _lf(logits, labels, vocab_size, **lf_kwargs):
                lf_kwargs.setdefault("num_items_in_batch", nib)
                return ForCausalLMLoss(
                    logits=logits, labels=labels, vocab_size=vocab_size, **lf_kwargs
                )
            self.loss_function = _lf
        else:
            # Drop any prior per-step rebind so generate() / eval falls back to
            # the default (mean reduction) instead of leaking last step's nib.
            try:
                del self._loss_function
            except AttributeError:
                pass

        # HF generate() ships `ts_values` in model_kwargs to every forward call,
        # but only the prefill step's input_ids contains TS placeholder tokens —
        # subsequent decode steps feed just the freshly-sampled token. So we
        # only run the TS encoder + scatter when the current input_ids actually
        # carries a placeholder. Without this check, step 1+ would encode TS
        # into N features with 0 placeholder slots → mismatch error.
        has_ts_slots = (
            ts_values is not None
            and self.ts_tower is not None
            and input_ids is not None
            and (input_ids == self.config.timeseries_token_id).any().item()
        )
        # Forecast-head mode REQUIRES TS placeholder positions in input_ids
        # (the head reads its window starting at the first one). The
        # `--no_timeseries` ablation strips them — fail loudly rather than
        # falling through to the LM logits path, which silently produces
        # garbage that the `<forecast>(k: v_k)` formatter would then choke on.
        if (
            (forecast_target is not None or kwargs.get("return_forecast", False))
            and not has_ts_slots
        ):
            raise RuntimeError(
                "forecast head requires the input to carry <|ts_placeholder|>; "
                "got an input with no TS tokens. The --no_timeseries ablation "
                "is incompatible with --pred_len > 0."
            )

        if has_ts_slots:
            if ts_group_ids is None:
                raise ValueError(
                    "ts_group_ids must be provided when ts_values is given."
                )
            # 1. Embed token ids if the caller did not supply pre-built embeddings.
            if inputs_embeds is None:
                inputs_embeds = self.get_input_embeddings()(input_ids)

            # 2. Encode the time-series and project to LLM hidden size.
            # ts_n_patches is required when bs>1 with variable-length TS; the
            # encoder sees NaN-padded (N, T_max) tensors and emits P_max
            # patches per variate, but the prompt holds per-sample P_i slots.
            ts_feats = self._encode_ts(
                ts_values, ts_group_ids, ts_n_patches
            )  # (sum_tokens, D_llm)

            # 3. Build a 1-D bool mask over (B, seq_len), then expand to (B, seq_len, D)
            #    to match the convention used by get_placeholder_mask / masked_scatter.
            ts_token_mask_1d = input_ids == self.config.timeseries_token_id  # (B, seq_len)
            n_expected = ts_token_mask_1d.sum().item()
            if n_expected != ts_feats.shape[0]:
                raise RuntimeError(
                    f"TS placeholder count mismatch: prompt has {n_expected} TS token "
                    f"slots but encoder produced {ts_feats.shape[0]} tokens."
                )
            ts_mask_3d = (
                ts_token_mask_1d.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                ts_mask_3d,
                ts_feats.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype),
            )

            # 4a. Forecasting branch — bypass LM head, use the linear forecast
            #     head on top of the last hidden state. Returns MSE loss in
            #     normalized space (statistics from per-sample input history).
            if forecast_target is not None or self._is_forecast_inference(kwargs):
                return self._forward_forecast(
                    input_ids=input_ids,
                    inputs_embeds=inputs_embeds,
                    ts_values=ts_values,
                    ts_group_ids=ts_group_ids,
                    forecast_target=forecast_target,
                    forecast_target_mask=forecast_target_mask,
                    forecast_target_lens=forecast_target_lens,
                    **kwargs,
                )

            # 4b. Delegate to parent.  Pass input_ids intact so the parent can still
            #    compute vision/audio masks and RoPE deltas correctly.
            return super().forward(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                labels=labels,
                **kwargs,
            )

        # No TS data — just forward everything as-is.
        return super().forward(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            labels=labels,
            **kwargs,
        )

    @staticmethod
    def _is_forecast_inference(kwargs: dict) -> bool:
        """At inference time the caller flips ``return_forecast=True`` instead
        of supplying a target. Used by :meth:`forecast`."""
        return bool(kwargs.pop("return_forecast", False))

    def _per_sample_history_stats(
        self,
        ts_values: torch.Tensor,
        ts_group_ids: torch.LongTensor,
        n_samples: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-sample (mu, sigma) over input history, NaN-aware.

        ``ts_values`` is right-padded with NaN to the batch's T_max — so for
        each sample we mask NaNs out before computing the mean and std.
        Returns two ``(B,)`` tensors on ``ts_values.device``, dtype float32.
        """
        device = ts_values.device
        mu = torch.zeros(n_samples, device=device, dtype=torch.float32)
        sigma = torch.ones(n_samples, device=device, dtype=torch.float32)
        vals = ts_values.float()
        for i in range(n_samples):
            sel = (ts_group_ids == i).nonzero(as_tuple=True)[0]
            if sel.numel() == 0:
                continue
            v = vals.index_select(0, sel)  # (C_i, T_max)
            valid = ~torch.isnan(v)
            n = valid.sum().clamp_min(1)
            v_zero = torch.where(valid, v, torch.zeros_like(v))
            mean_i = v_zero.sum() / n
            sq = torch.where(valid, (v - mean_i) ** 2, torch.zeros_like(v)).sum() / n
            mu[i] = mean_i
            sigma[i] = (sq + 1e-6).sqrt()
        return mu, sigma

    def _forward_forecast(
        self,
        *,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.Tensor,
        ts_values: torch.Tensor,
        ts_group_ids: torch.LongTensor,
        forecast_target: torch.Tensor | None,
        forecast_target_mask: torch.Tensor | None,
        forecast_target_lens: torch.LongTensor | None,
        **kwargs,
    ):
        from transformers.modeling_outputs import CausalLMOutputWithPast

        if self.forecast_head is None:
            raise RuntimeError(
                "forecast_target was passed but no forecast_head is attached. "
                "Call model.attach_forecast_head(...) at build time."
            )

        # Drop kwargs that don't apply when we bypass the LM head; keep
        # everything else (attention_mask, position_ids, vision/image grids,
        # past_key_values=None at training, etc).
        kwargs.pop("labels", None)
        kwargs.pop("num_items_in_batch", None)
        kwargs["output_hidden_states"] = True
        kwargs.setdefault("use_cache", False)

        outputs = super().forward(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            labels=None,
            **kwargs,
        )
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError(
                "parent forward returned no hidden_states — "
                "output_hidden_states should have been True."
            )
        last_hidden = hidden_states[-1]  # (B, S, D)
        bsz, seq_len, d_llm = last_hidden.shape

        # Per-sample window: 8 hiddens starting at the FIRST <|ts_placeholder|>.
        ts_id = self.config.timeseries_token_id
        is_ts = input_ids == ts_id
        if not is_ts.any():
            raise RuntimeError(
                "_forward_forecast: input_ids carries no timeseries token; "
                "the forecast collator must always emit one."
            )
        # argmax on a bool tensor returns the position of the first True (or 0
        # if no True; we already gated on is_ts.any() above).
        first_ts_pos = is_ts.int().argmax(dim=1)  # (B,)
        window = self.forecast_head.window
        if (first_ts_pos.max().item() + window) > seq_len:
            raise RuntimeError(
                f"forecast head window={window} exceeds seq end "
                f"(max first_ts_pos={int(first_ts_pos.max())}, seq_len={seq_len})."
            )
        idx = first_ts_pos.unsqueeze(1) + torch.arange(
            window, device=last_hidden.device
        ).unsqueeze(0)  # (B, W)
        gather_idx = idx.unsqueeze(-1).expand(-1, -1, d_llm)  # (B, W, D)
        win_hidden = last_hidden.gather(1, gather_idx)  # (B, W, D)

        head = self.forecast_head
        win_hidden = win_hidden.to(dtype=next(head.parameters()).dtype)
        pred_norm = head(win_hidden)  # (B, pred_len)

        mu, sigma = self._per_sample_history_stats(ts_values, ts_group_ids, bsz)

        if forecast_target is None:
            # Inference path: caller wants denormalized predictions.
            pred_orig = pred_norm.float() * sigma.unsqueeze(1) + mu.unsqueeze(1)
            return CausalLMOutputWithPast(
                loss=None,
                logits=pred_orig,
            )

        # Training path: MSE in normalized space.
        target = forecast_target.to(device=pred_norm.device, dtype=torch.float32)
        if forecast_target_mask is None:
            mask = torch.isfinite(target)
        else:
            mask = forecast_target_mask.to(device=pred_norm.device, dtype=torch.bool)
        target_norm = (target - mu.unsqueeze(1)) / sigma.unsqueeze(1)
        diff = pred_norm.float() - target_norm
        diff = torch.where(mask, diff, torch.zeros_like(diff))
        n = mask.sum().clamp_min(1)
        loss = (diff * diff).sum() / n
        return CausalLMOutputWithPast(loss=loss, logits=pred_norm)

    def forecast(self, **batch_inputs) -> torch.Tensor:
        """Inference helper — runs forward in forecast mode and returns the
        denormalized prediction tensor of shape ``(B, pred_len)``.
        """
        out = self.forward(return_forecast=True, **batch_inputs)
        return out.logits

    def get_rope_index(
        self,
        input_ids: torch.LongTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        use_audio_in_video: bool = False,
        audio_seqlens: torch.LongTensor | None = None,
        second_per_grids: torch.Tensor | None = None,
    ):
        """TimeOmni-v-aware M-RoPE.

        Layout per token kind:
          - text         → (n, n, n)      (1D, parent's convention)
          - video V      → (t·spg·pps, h, w)   (3D, parent's vision convention)
          - timeseries T → (t_val, c, c)  (2D: time + channel, channel duplicated
            into the third dim so all 3 M-RoPE channels carry signal)
          - markers (vision_start/end, ts_start/ts_end) → (n, n, n) tight 1D

        For block_adjacent (TS standalone block, no video alongside) the TS time
        index is the integer patch index. For time_interleave (TS interleaved
        with V inside a single vision span) the TS time index is scaled to align
        with the video time axis: `patch_idx * (T_v · spg · pps) / n_patches`.

        Falls through to the parent implementation when no TS placeholders are
        present in input_ids — important because generation continuation calls
        get_rope_index with single-token input_ids (no TS context).
        """
        ts_token_id = getattr(self.config, "timeseries_token_id", None)
        if input_ids is None or ts_token_id is None:
            return super().get_rope_index(
                input_ids, image_grid_thw, video_grid_thw,
                attention_mask, use_audio_in_video, audio_seqlens, second_per_grids,
            )

        if not (input_ids == ts_token_id).any():
            return super().get_rope_index(
                input_ids, image_grid_thw, video_grid_thw,
                attention_mask, use_audio_in_video, audio_seqlens, second_per_grids,
            )

        n_channels_per_sample = getattr(self, "_ts_n_channels", None)
        if n_channels_per_sample is None:
            raise RuntimeError(
                "TimeOmniVForConditionalGeneration.get_rope_index needs _ts_n_channels "
                "stashed by forward(); the batch passed to forward must include "
                "the `ts_n_channels` field produced by TimeOmniVProcessor."
            )

        # Token id resolution
        cfg = self.config
        spatial_merge = self.spatial_merge_size
        pps = float(cfg.position_id_per_seconds)
        vision_start_id = cfg.vision_start_token_id
        video_token_id = cfg.video_token_id
        ts_start_id = cfg.timeseries_start_token_id
        ts_end_id = cfg.timeseries_end_token_id
        # Qwen2.5-Omni convention: vision_eos follows vision_bos in vocab order.
        # Tests can override by stashing `_vision_end_token_id` on the instance.
        vision_end_id = getattr(self, "_vision_end_token_id", None)
        if vision_end_id is None:
            vision_end_id = vision_start_id + 1  # Qwen2.5-Omni: 151652 → 151653
            self._vision_end_token_id = vision_end_id

        device = input_ids.device
        B, T = input_ids.shape
        # Float during computation, cast to long at the end
        position_ids = torch.zeros(3, B, T, dtype=torch.float32, device=device)
        deltas = []

        video_idx = 0
        for b in range(B):
            if attention_mask is not None:
                valid_mask = attention_mask[b].bool()
                valid_indices = valid_mask.nonzero(as_tuple=True)[0]
            else:
                valid_mask = torch.ones(T, dtype=torch.bool, device=device)
                valid_indices = torch.arange(T, device=device)

            valid_ids = input_ids[b][valid_mask].tolist()
            n = len(valid_ids)
            if n == 0:
                deltas.append(torch.tensor(0.0, device=device))
                continue

            n_chan = max(1, int(n_channels_per_sample[b].item()))
            local = torch.zeros(3, n, dtype=torch.float32, device=device)
            cur_pos = 0.0
            ts_seen = 0
            i = 0

            while i < n:
                tid = valid_ids[i]

                if tid == vision_start_id:
                    # Find matching vision_end
                    j = i + 1
                    while j < n and valid_ids[j] != vision_end_id:
                        j += 1
                    if j >= n:
                        # Unmatched — degrade to text
                        local[:, i] = cur_pos
                        cur_pos += 1.0
                        i += 1
                        continue

                    # Span: [i (vision_start), j (vision_end)] inclusive
                    span_ids = valid_ids[i + 1 : j]
                    v_count = sum(1 for t in span_ids if t == video_token_id)
                    t_count = sum(1 for t in span_ids if t == ts_token_id)
                    n_patches = max(1, t_count // n_chan)

                    if video_grid_thw is not None and video_idx < video_grid_thw.shape[0]:
                        grid_t = int(video_grid_thw[video_idx][0].item())
                        grid_h = int(video_grid_thw[video_idx][1].item())
                        grid_w = int(video_grid_thw[video_idx][2].item())
                        llm_h = max(1, grid_h // spatial_merge)
                        llm_w = max(1, grid_w // spatial_merge)
                        S_v = llm_h * llm_w
                        if second_per_grids is not None and video_idx < len(second_per_grids):
                            spg = float(second_per_grids[video_idx])
                        else:
                            spg = 1.0
                    else:
                        grid_t = llm_h = llm_w = S_v = 0
                        spg = 1.0

                    # Time scale for TS: align with video time axis when both
                    # modalities co-exist in the span; otherwise (no V) use the
                    # integer patch index.
                    if v_count > 0 and t_count > 0 and grid_t > 0:
                        ts_time_per_patch = (grid_t * spg * pps) / n_patches
                    else:
                        ts_time_per_patch = 1.0

                    base = cur_pos
                    local[:, i] = base  # vision_start position
                    inner_max = 0.0
                    v_enc = 0
                    t_enc = 0

                    # Reserve slot for ts_start (if present right after vision_start)
                    has_inner_ts_start = (i + 1 < j) and (valid_ids[i + 1] == ts_start_id)
                    has_inner_ts_end = (j - 1 > i) and (valid_ids[j - 1] == ts_end_id)
                    content_base = base + 1.0
                    if has_inner_ts_start:
                        local[:, i + 1] = base + 1.0
                        content_base = base + 2.0

                    content_first = i + 2 if has_inner_ts_start else i + 1
                    content_last = j - 2 if has_inner_ts_end else j - 1

                    for k in range(content_first, content_last + 1):
                        tk = valid_ids[k]
                        if tk == video_token_id:
                            step = v_enc // S_v if S_v > 0 else 0
                            spatial = v_enc % S_v if S_v > 0 else 0
                            h = spatial // llm_w if llm_w > 0 else 0
                            w = spatial % llm_w if llm_w > 0 else 0
                            t_val = float(step) * spg * pps
                            local[0, k] = content_base + t_val
                            local[1, k] = content_base + float(h)
                            local[2, k] = content_base + float(w)
                            inner_max = max(inner_max, max(t_val, float(h), float(w)))
                            v_enc += 1
                        elif tk == ts_token_id:
                            patch_idx = t_enc // n_chan
                            ch_idx = t_enc % n_chan
                            t_val = float(patch_idx) * ts_time_per_patch
                            local[0, k] = content_base + t_val
                            local[1, k] = content_base + float(ch_idx)
                            local[2, k] = content_base + float(ch_idx)
                            inner_max = max(inner_max, max(t_val, float(ch_idx)))
                            t_enc += 1
                            ts_seen += 1
                        else:
                            # stray text inside the vision span — give it the
                            # next available position
                            local[:, k] = content_base
                            inner_max = max(inner_max, 0.0)

                    end_base = content_base + inner_max + 1.0
                    if has_inner_ts_end:
                        local[:, j - 1] = end_base
                        end_base += 1.0
                    local[:, j] = end_base  # vision_end

                    cur_pos = end_base + 1.0
                    video_idx += 1
                    i = j + 1

                elif tid == ts_start_id:
                    # Standalone TS block (block_adjacent, outside any vision span)
                    j = i + 1
                    while j < n and valid_ids[j] != ts_end_id:
                        j += 1
                    if j >= n:
                        local[:, i] = cur_pos
                        cur_pos += 1.0
                        i += 1
                        continue

                    base = cur_pos
                    local[:, i] = base  # ts_start
                    content_base = base + 1.0
                    inner_max = 0.0
                    t_enc = 0
                    for k in range(i + 1, j):
                        tk = valid_ids[k]
                        if tk == ts_token_id:
                            patch_idx = t_enc // n_chan
                            ch_idx = t_enc % n_chan
                            local[0, k] = content_base + float(patch_idx)
                            local[1, k] = content_base + float(ch_idx)
                            local[2, k] = content_base + float(ch_idx)
                            inner_max = max(inner_max, max(float(patch_idx), float(ch_idx)))
                            t_enc += 1
                            ts_seen += 1
                        else:
                            local[:, k] = content_base
                            inner_max = max(inner_max, 0.0)

                    end_base = content_base + inner_max + 1.0
                    local[:, j] = end_base  # ts_end
                    cur_pos = end_base + 1.0
                    i = j + 1

                else:
                    # Plain text token (incl. unrelated specials)
                    local[:, i] = cur_pos
                    cur_pos += 1.0
                    i += 1

            position_ids[..., b, valid_indices] = local
            # Pad positions left at 0; HF parent uses `position_ids.masked_fill_(
            # attention_mask == 0, 1)` — replicate for consistency
            if attention_mask is not None:
                pad_positions = (~valid_mask).nonzero(as_tuple=True)[0]
                position_ids[..., b, pad_positions] = 1.0
            sample_max = local.max().item()
            deltas.append(torch.tensor(sample_max + 1 - T, device=device))

        deltas = torch.stack(deltas).unsqueeze(1)
        return position_ids.long(), deltas
