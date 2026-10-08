"""Per-step GPU power / utilization logger.

A background thread polls NVML (power, SM util) every 100ms and at the end
of every training step we print the mean/max/min power and mean util over
that step's window.

Crucially this module does NOT insert any ``torch.cuda.synchronize()`` or
monkey-patch ``training_step``. Doing so serializes kernel launches and
hides MoE-dispatch overlap — fine when you actively need a fwd/bwd split,
harmful for the common "is the GPU hot?" check. If you need phase
breakdown, use ``torch.profiler`` (fully async traces) instead of this.
"""

from __future__ import annotations

import threading
import time

import torch
from transformers import TrainerCallback


class _NvmlPowerSampler:
    """Background thread polling NVML for power (W) + SM util (%).

    Samples every ``interval_s`` into an in-memory ring of (t, power_W, util).
    Cheap — each NVML call is microseconds — but we still keep it opt-in so
    environments without libnvidia-ml.so (login nodes, CPU-only CI) don't
    crash at import time.

    A single process is bound to one CUDA device (DeepSpeed uses 1 GPU per
    rank via LOCAL_RANK), so we sample that device only.
    """

    def __init__(self, interval_s: float = 0.1, ring_size: int = 4096):
        self._ok = False
        self._samples: list[tuple[float, float, float]] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._interval = interval_s
        self._ring = ring_size
        try:
            import pynvml
            pynvml.nvmlInit()
            dev_idx = torch.cuda.current_device() if torch.cuda.is_available() else 0
            self._nvml = pynvml
            self._h = pynvml.nvmlDeviceGetHandleByIndex(dev_idx)
            _ = pynvml.nvmlDeviceGetPowerUsage(self._h)
            _ = pynvml.nvmlDeviceGetUtilizationRates(self._h).gpu
            self._ok = True
        except Exception as e:
            print(f"[STEP_TIMER] NVML unavailable, power logging disabled: {e}", flush=True)
            return
        self._thread = threading.Thread(target=self._run, name="nvml-sampler", daemon=True)
        self._thread.start()

    @property
    def ok(self) -> bool:
        return self._ok

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                p_w = self._nvml.nvmlDeviceGetPowerUsage(self._h) / 1000.0
                u = self._nvml.nvmlDeviceGetUtilizationRates(self._h).gpu
            except Exception:
                self._stop.set()
                return
            t = time.perf_counter()
            with self._lock:
                self._samples.append((t, p_w, float(u)))
                if len(self._samples) > self._ring:
                    self._samples = self._samples[-self._ring // 2 :]
            self._stop.wait(self._interval)

    def window_stats(self, t_begin: float, t_end: float):
        """Return (n, mean_W, max_W, min_W, mean_util) in [t_begin, t_end]."""
        if not self._ok:
            return None
        with self._lock:
            window = [s for s in self._samples if t_begin <= s[0] <= t_end]
        if not window:
            return None
        n = len(window)
        powers = [s[1] for s in window]
        utils = [s[2] for s in window]
        return (
            n,
            sum(powers) / n,
            max(powers),
            min(powers),
            sum(utils) / n,
        )

    def close(self) -> None:
        self._stop.set()
        if self._ok:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass


class StepTimerCallback(TrainerCallback):
    """Print per-logging-step NVML power + SM util + peak VRAM.

    Runs without any ``torch.cuda.synchronize()`` — memory stats are read
    from the caching allocator's CPU-side bookkeeping (no GPU sync needed),
    and NVML is polled from a background thread. Logs on every
    ``logging_steps`` boundary (same cadence as HF Trainer's loss logs) so
    we don't spam the console; the window from last boundary to now gives
    a meaningful long-average power instead of a single-step snapshot.
    """

    def __init__(self, power_sample_interval_s: float = 0.1):
        self._window_start: float | None = None
        self._power = _NvmlPowerSampler(interval_s=power_sample_interval_s)

    def _reset_peak_memory(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_step_begin(self, args, state, control, **kwargs):
        if self._window_start is None:
            self._window_start = time.perf_counter()
            self._reset_peak_memory()

    def on_step_end(self, args, state, control, **kwargs):
        logging_steps = max(1, int(getattr(args, "logging_steps", 1) or 1))
        if state.global_step % logging_steps != 0:
            return
        now = time.perf_counter()
        pwr_stats = self._power.window_stats(self._window_start or now, now)

        mem_str = ""
        if torch.cuda.is_available():
            gib = 1024 ** 3
            peak_alloc = torch.cuda.max_memory_allocated() / gib
            peak_resv = torch.cuda.max_memory_reserved() / gib
            curr_alloc = torch.cuda.memory_allocated() / gib
            mem_str = (
                f" peak_alloc={peak_alloc:.2f}GiB "
                f"peak_reserved={peak_resv:.2f}GiB "
                f"curr_alloc={curr_alloc:.2f}GiB"
            )

        if pwr_stats is not None:
            n, mean_w, max_w, min_w, mean_u = pwr_stats
            pwr_str = (
                f" power_mean={mean_w:.0f}W power_max={max_w:.0f}W "
                f"power_min={min_w:.0f}W util_mean={mean_u:.0f}% "
                f"nvml_samples={n}"
            )
        else:
            pwr_str = ""

        if mem_str or pwr_str:
            print(f"[STEP {state.global_step}]{mem_str}{pwr_str}", flush=True)

        # Reset window for the next logging interval — peak stats reflect
        # the window from last boundary to now, not start-of-training.
        self._window_start = now
        self._reset_peak_memory()

    def on_train_end(self, args, state, control, **kwargs):
        self._power.close()
