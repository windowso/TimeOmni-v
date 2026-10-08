"""Inference backend abstraction.

Backends encapsulate everything model-family-specific (model class, processor,
chat template, video preprocessing, generate signature, decode path) so the
orchestration layer in `timeomni_v/inference/infer.py` stays the same across:

* local HF models (Qwen2.5-Omni, Qwen3-Omni, TimeOmni-v, Qwen3-VL, InternVL) — run
  through a torch DataLoader + DDP scaffold, optionally with the token-budget
  dynamic batching sampler;
* API models (GPT, Gemini) — run through a ThreadPoolExecutor for concurrency;
  no DDP, no DataLoader.

Backend lifecycle (local)::

    backend = SomeBackend(args=args, include_vision=True, include_timeseries=True)
    backend.load(device_map="auto")                # heavy: loads model + processor
    collator = backend.make_collator()             # passed to DataLoader(collate_fn=...)
    lengths = backend.compute_lengths(dataset)     # None ⇒ fixed batch only
    # later, per batch:
    raws = backend.generate(batch["inputs"], max_new_tokens=4)

Backend lifecycle (API)::

    backend = SomeApiBackend(args=args, include_vision=..., include_timeseries=...)
    backend.load()                                  # init HTTP client
    raw = backend.infer_one(row, max_new_tokens=4)  # called concurrently

The `args` reference is the parsed argparse Namespace; backends store it on
``self.args`` so they can read fps / pixel bounds / model paths / api flags
without duplicated plumbing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable


class Backend(ABC):
    """Common base for local + API backends. Subclasses set the class-level
    capability flags so `infer.py` can branch (e.g. enable dynamic-bs only when
    the backend's processor exposes the expected video patch attributes).

    Capability flags:
      * ``is_local`` — True for HF backends, False for API. Drives DataLoader
        vs ThreadPoolExecutor and DDP gate.
      * ``supports_dynamic_bs`` — True only for backends whose processor has the
        Qwen2.5-Omni-shaped ``video_processor.patch_size/temporal_patch_size/
        merge_size`` attributes (Qwen2.5-Omni / Qwen3-Omni / TimeOmni-v).
      * ``supports_pixel_bounds`` — False if the backend can't honor
        ``--video_min_pixels/max_pixels`` (e.g. InternVL uses fixed 448² tiles).
      * ``supports_dense_frames`` — False if the backend cannot run with
        ``do_sample_frames=False`` (InternVL / API backends).
    """

    name: str = ""
    is_local: bool = True
    supports_dynamic_bs: bool = False
    supports_pixel_bounds: bool = True
    supports_dense_frames: bool = True

    def __init__(self, *, args, include_vision: bool, include_timeseries: bool):
        self.args = args
        self.include_vision = include_vision
        self.include_timeseries = include_timeseries
        self._validate_args()

    def _validate_args(self) -> None:
        """Raise SystemExit on incompatible flag combos. Override per-backend.

        The default checks pixel-bound + dense-frame capability so unsupported
        combinations surface up-front instead of mid-run."""
        a = self.args
        if not self.supports_dense_frames and not getattr(a, "do_sample_frames", True):
            raise SystemExit(
                f"--backend {self.name} requires do_sample_frames=True "
                "(it has no dense-frame inference path).",
            )

    @abstractmethod
    def load(self, *, device_map: Any = None) -> None:
        """Prepare for inference (load model+processor for local; init client
        for API). Called exactly once after construction."""


class LocalHFBackend(Backend):
    """Backend that runs locally on GPU(s) via the standard HF generate path."""

    is_local = True

    @abstractmethod
    def make_collator(self) -> Callable[[list[dict]], dict]:
        """Return a callable usable as ``DataLoader(collate_fn=...)``. Output
        must be a dict with at least ``inputs`` (whatever the backend's
        ``generate`` expects), ``answers`` (list[str]), and ``ids``
        (list[str|None])."""

    @abstractmethod
    def generate(self, batch_inputs: Any, *, max_new_tokens: int) -> list[str]:
        """Run a batched forward + decode. ``batch_inputs`` is the value the
        collator placed under the ``inputs`` key. Returns the raw decoded
        strings, one per sample, in the same order the collator emitted them."""

    def compute_lengths(self, dataset) -> list[int] | None:
        """Return per-sample exact post-processor sequence lengths for the
        token-budget sampler, or ``None`` if dynamic batching isn't supported.

        Default: ``None``. Override only when the processor exposes
        Qwen2.5-Omni's video patch attributes."""
        return None


class ApiBackend(Backend):
    """Backend whose forward is an HTTP request. Inference orchestrator skips
    the DataLoader/DDP path and uses a ThreadPoolExecutor instead."""

    is_local = False
    supports_dynamic_bs = False

    @abstractmethod
    def infer_one(self, row: dict, *, max_new_tokens: int) -> str:
        """Per-row inference call. Return the raw model output as a string.

        Implementations should let exceptions bubble up; the orchestrator
        marks the row ``failed=True`` and continues.
        """
