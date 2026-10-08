"""Tests for dataset-jsonl naming: ``scripts/_jsonl.sh`` and per-channel outputs.

Files are ``<name>.jsonl`` or ``<name>.<tag>.jsonl`` (the released MMTA archive
carries a tag); the scripts must resolve either form, and must never confuse a
per-channel variant (``terra.percha.jsonl``) with a tag on the base dataset.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from timeomni_v.data.convert_per_channel_forecasting import _derive_out_path

HELPERS = Path(__file__).resolve().parents[1] / "scripts" / "_jsonl.sh"


def _sh(cmd: str) -> str:
    out = subprocess.run(
        ["bash", "-c", f'set -euo pipefail; source "{HELPERS}"; {cmd}'],
        check=True, capture_output=True, text=True,
    )
    return out.stdout.strip()


def _touch(d: Path, *names: str) -> None:
    for n in names:
        (d / n).write_text("")


def test_resolve_prefers_untagged(tmp_path):
    _touch(tmp_path, "covla_test.jsonl", "covla_test.tagA.jsonl")
    assert _sh(f'resolve_jsonl "{tmp_path}" covla_test') == f"{tmp_path}/covla_test.jsonl"


def test_resolve_accepts_tagged(tmp_path):
    _touch(tmp_path, "covla_test.tagA.jsonl")
    assert _sh(f'resolve_jsonl "{tmp_path}" covla_test') == f"{tmp_path}/covla_test.tagA.jsonl"


def test_resolve_base_never_picks_per_channel_variant(tmp_path):
    # Only per-channel files exist → the base dataset is missing, not "terra.percha".
    _touch(tmp_path, "terra.percha.jsonl", "terra.percha.tagA.jsonl")
    assert _sh(f'resolve_jsonl "{tmp_path}" terra') == f"{tmp_path}/terra.jsonl"
    _touch(tmp_path, "terra.tagA.jsonl")
    assert _sh(f'resolve_jsonl "{tmp_path}" terra') == f"{tmp_path}/terra.tagA.jsonl"


def test_resolve_per_channel_variant(tmp_path):
    _touch(tmp_path, "terra_test.percha.tagA.jsonl", "terra_test.tagA.jsonl")
    assert (
        _sh(f'resolve_jsonl "{tmp_path}" terra_test.percha')
        == f"{tmp_path}/terra_test.percha.tagA.jsonl"
    )


def test_resolve_missing_returns_untagged_path(tmp_path):
    assert _sh(f'resolve_jsonl "{tmp_path}" sp500_test') == f"{tmp_path}/sp500_test.jsonl"


@pytest.mark.parametrize(
    "path,stem",
    [
        ("/d/covla_test.jsonl", "covla_test"),
        ("/d/covla_test.tagA.jsonl", "covla_test"),
        ("/d/terra_test.percha.jsonl", "terra_test.percha"),
        ("/d/terra_test.percha.tagA.jsonl", "terra_test.percha"),
        ("cls_all_test.jsonl", "cls_all_test"),
    ],
)
def test_jsonl_stem(path, stem):
    assert _sh(f'jsonl_stem "{path}"') == stem


@pytest.mark.parametrize(
    "name,out",
    [
        ("terra.jsonl", "terra.percha.jsonl"),
        ("terra_test.tagA.jsonl", "terra_test.percha.tagA.jsonl"),
    ],
)
def test_derive_out_path(name, out):
    assert _derive_out_path(Path("/d") / name, "percha") == Path("/d") / out
