"""Backend registry. New backends register themselves at import time via
``register(name, cls)`` and are dispatched by ``--backend <name>`` in
``infer.py``.

The registry is import-lazy: each backend's submodule is only imported when
``get(name)`` is called for the first time, so that (a) optional dependencies
(e.g. google.genai for the gemini backend) don't break callers who never use
that backend, and (b) the heavy HF model classes for backend X aren't pulled
into memory when running backend Y.
"""

from __future__ import annotations

import importlib
from typing import Type

from timeomni_v.inference.backends.base import Backend


# (registry key) → (module path, class name). Imported lazily by ``get``.
#
# Multiple checkpoints of the same backend family register as separate keys
# (e.g. ``qwen3_vl_8b`` vs ``qwen3_vl_30b``) so each one writes into its own
# output subdir under ``runs/<RUN_DIR_NAME>/<key>/``. The class is the same;
# the per-key model path is supplied via the corresponding
# ``BACKEND_MODEL_PATH_<UPPER>`` env var in the eval scripts.
_BACKENDS: dict[str, tuple[str, str]] = {
    "qwen2_5_omni":  ("timeomni_v.inference.backends.qwen2_5_omni", "Qwen2_5OmniBackend"),
    "timeomni_v":        ("timeomni_v.inference.backends.timeomni_v",       "TimeOmniVBackend"),
    "qwen3_omni":    ("timeomni_v.inference.backends.qwen3_omni",   "Qwen3OmniBackend"),
    "qwen3_vl_8b":   ("timeomni_v.inference.backends.qwen3_vl",     "Qwen3VLBackend"),
    "qwen3_vl_30b":  ("timeomni_v.inference.backends.qwen3_vl",     "Qwen3VLBackend"),
    "internvl_4b":   ("timeomni_v.inference.backends.internvl",     "InternVLBackend"),
    "internvl_8b":   ("timeomni_v.inference.backends.internvl",     "InternVLBackend"),
    "gpt":           ("timeomni_v.inference.backends.api_gpt",      "GptBackend"),
    "gemini":        ("timeomni_v.inference.backends.api_gemini",   "GeminiBackend"),
}


def list_names() -> list[str]:
    return sorted(_BACKENDS.keys())


def get(name: str) -> Type[Backend]:
    """Resolve a backend name to its class, importing the submodule lazily."""
    try:
        mod_path, cls_name = _BACKENDS[name]
    except KeyError:
        valid = ", ".join(list_names())
        raise SystemExit(f"unknown --backend {name!r}; valid: {valid}")
    mod = importlib.import_module(mod_path)
    return getattr(mod, cls_name)


__all__ = ["Backend", "get", "list_names"]
