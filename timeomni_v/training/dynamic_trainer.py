"""HF Trainer subclass that swaps the train dataloader for a token-budget
batch sampler (variable batch size).

Only active when `max_tokens > 0` AND `lengths` is supplied; otherwise
defers to the parent Trainer's default sampler so the rest of the repo
keeps working unchanged. Accelerate handles distributed sharding of the
batch sampler via `accelerator.prepare(dataloader)`.
"""

from __future__ import annotations

from functools import partial

from torch.utils.data import DataLoader
from transformers import Trainer
from transformers.trainer_utils import seed_worker

from timeomni_v.data.dynamic_bs import TokenBudgetBatchSampler


class DynamicBSTrainer(Trainer):
    """Trainer with an optional token-budget batch sampler.

    Pass ``lengths`` (exact per-sample token counts — see
    ``timeomni_v.data.length_estimate.compute_exact_lengths``) and
    ``max_tokens>0`` to enable packing. Short samples batch together up to
    the budget / ``max_bs``; long samples stay at bs=1.

    Pass ``eval_lengths`` to also pack the eval dataloader (deterministic —
    eval uses fixed seed=epoch 0 so different evaluation calls produce the
    same batch composition, giving a reproducible eval loss curve).
    """

    def __init__(
        self,
        *args,
        lengths: list[int] | None = None,
        eval_lengths: list[int] | None = None,
        max_tokens: int = 0,
        max_bs: int = 8,
        task_ids: list[int] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._dyn_lengths = lengths
        self._dyn_eval_lengths = eval_lengths
        self._dyn_max_tokens = int(max_tokens)
        self._dyn_max_bs = int(max_bs)
        # Optional per-sample task assignment for the train sampler. When
        # supplied, each emitted train batch contains indices from a single
        # task. Eval still mixes tasks (single merged eval jsonl).
        self._dyn_task_ids = list(task_ids) if task_ids is not None else None
        self._dyn_sampler: TokenBudgetBatchSampler | None = None
        self._dyn_eval_sampler: TokenBudgetBatchSampler | None = None

    def _dynamic_enabled(self) -> bool:
        return self._dyn_max_tokens > 0 and self._dyn_lengths is not None

    def _dynamic_eval_enabled(self) -> bool:
        return self._dyn_max_tokens > 0 and self._dyn_eval_lengths is not None

    def _make_dyn_loader(self, dataset, sampler) -> DataLoader:
        rank = int(self.args.process_index)
        worker_init = partial(
            seed_worker,
            num_workers=self.args.dataloader_num_workers,
            rank=rank,
        )
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            prefetch_factor=(
                self.args.dataloader_prefetch_factor
                if self.args.dataloader_num_workers > 0
                else None
            ),
            persistent_workers=self.args.dataloader_persistent_workers,
            worker_init_fn=worker_init,
        )
        # See get_train_dataloader: bypass accelerator.prepare(loader); only
        # flip even_batches=False so HF's gather_for_metrics path tolerates
        # variable-size batches.
        self.accelerator.even_batches = False
        return loader

    def get_train_dataloader(self) -> DataLoader:
        if not self._dynamic_enabled():
            return super().get_train_dataloader()

        rank = int(self.args.process_index)
        world_size = int(self.args.world_size)

        if self._dyn_sampler is None:
            if len(self._dyn_lengths) != len(self.train_dataset):
                raise ValueError(
                    f"lengths list size {len(self._dyn_lengths)} != dataset "
                    f"size {len(self.train_dataset)}; refuse to batch."
                )
            if self._dyn_task_ids is not None and len(self._dyn_task_ids) != len(self.train_dataset):
                raise ValueError(
                    f"task_ids size {len(self._dyn_task_ids)} != dataset "
                    f"size {len(self.train_dataset)}; refuse to batch."
                )
            # Pass rank/world_size so the sampler does its own per-rank
            # slice (and the "truncate to a multiple of world_size" step
            # that keeps NCCL synchronized at epoch end). We then bypass
            # accelerator.prepare(loader) entirely — Accelerate would wrap
            # our batch_sampler in BatchSamplerShard, which (a) raises on
            # variable-size batches unless `even_batches=False` is set,
            # and (b) would round-robin over our already-sharded batches
            # and each rank would see only 1/world_size² of the data.
            self._dyn_sampler = TokenBudgetBatchSampler(
                lengths=self._dyn_lengths,
                max_tokens=self._dyn_max_tokens,
                max_bs=self._dyn_max_bs,
                rank=rank,
                world_size=world_size,
                seed=self.args.seed,
                task_ids=self._dyn_task_ids,
            )
            n_batches = len(self._dyn_sampler)
            avg = sum(self._dyn_lengths) / max(1, len(self._dyn_lengths))
            if rank == 0:
                grouped = "task-grouped " if self._dyn_task_ids is not None else ""
                print(
                    f"[DYN_BS] {grouped}enabled: N={len(self._dyn_lengths)} "
                    f"avg_len={avg:.0f} max_tokens={self._dyn_max_tokens} "
                    f"max_bs={self._dyn_max_bs} world_size={world_size} "
                    f"est_batches_per_rank_per_epoch={n_batches}",
                    flush=True,
                )

        self._dyn_sampler.set_epoch(self.state.epoch or 0)
        return self._make_dyn_loader(self.train_dataset, self._dyn_sampler)

    def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:
        if not self._dynamic_eval_enabled():
            return super().get_eval_dataloader(eval_dataset)

        ds = eval_dataset if eval_dataset is not None else self.eval_dataset
        if ds is None:
            return super().get_eval_dataloader(eval_dataset)
        # Don't pack a caller-supplied dataset we have no lengths for —
        # eval_lengths corresponds 1:1 to self.eval_dataset.
        if eval_dataset is not None and eval_dataset is not self.eval_dataset:
            return super().get_eval_dataloader(eval_dataset)

        rank = int(self.args.process_index)
        world_size = int(self.args.world_size)

        if self._dyn_eval_sampler is None:
            if len(self._dyn_eval_lengths) != len(ds):
                raise ValueError(
                    f"eval_lengths list size {len(self._dyn_eval_lengths)} != "
                    f"eval dataset size {len(ds)}; refuse to batch."
                )
            self._dyn_eval_sampler = TokenBudgetBatchSampler(
                lengths=self._dyn_eval_lengths,
                max_tokens=self._dyn_max_tokens,
                max_bs=self._dyn_max_bs,
                rank=rank,
                world_size=world_size,
                seed=self.args.seed,
            )
            # Eval is deterministic — pin epoch=0 so every evaluation call
            # yields the same batches (otherwise eval loss would jitter
            # purely from re-shuffling).
            self._dyn_eval_sampler.set_epoch(0)
            if rank == 0:
                print(
                    f"[DYN_BS] eval enabled: N={len(self._dyn_eval_lengths)} "
                    f"max_tokens={self._dyn_max_tokens} max_bs={self._dyn_max_bs} "
                    f"world_size={world_size} "
                    f"est_eval_batches_per_rank={len(self._dyn_eval_sampler)}",
                    flush=True,
                )

        return self._make_dyn_loader(ds, self._dyn_eval_sampler)
