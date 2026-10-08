"""Tests for timeomni_v.data.normalize_jsonl (legacy schema migration)."""

import json

import pytest

from timeomni_v.data.normalize_jsonl import normalize_file, normalize_row


def test_normalize_row_renames_legacy_video_id():
    row = {"video_id": "abc", "video_path": "/v.mp4", "prompt": "P", "answer": "A"}
    normalize_row(row)
    assert row["id"] == "abc"
    assert "video_id" not in row
    assert row["task"] == "classification"


def test_normalize_row_idempotent():
    row = {
        "id": "abc",
        "task": "prediction",
        "image_path": ["/a.png"],
        "prompt": "P",
        "answer": "A",
    }
    before = json.dumps(row, sort_keys=True)
    normalize_row(row)
    after = json.dumps(row, sort_keys=True)
    assert before == after


def test_normalize_row_keeps_existing_task():
    row = {"id": "x", "task": "prediction", "prompt": "P", "answer": "A"}
    normalize_row(row)
    assert row["task"] == "prediction"


def test_normalize_file_writes_clean_rows(tmp_path):
    in_path = tmp_path / "in.jsonl"
    out_path = tmp_path / "out.jsonl"
    rows = [
        {"video_id": "a", "video_path": "/v.mp4", "prompt": "P", "answer": "A"},
        # Image row already conforming — should pass through untouched.
        {"id": "b", "task": "prediction", "image_path": ["/i.png"],
         "prompt": "P", "answer": "<forecast>(t: 1)</forecast>"},
    ]
    in_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    n_rows, n_id, n_task = normalize_file(in_path, out_path)
    assert n_rows == 2 and n_id == 1 and n_task == 1
    out_rows = [json.loads(l) for l in out_path.read_text().splitlines() if l]
    assert out_rows[0]["id"] == "a" and out_rows[0]["task"] == "classification"
    assert "video_id" not in out_rows[0]
    assert out_rows[1] == rows[1]


def test_normalize_file_inplace_idempotent(tmp_path):
    in_path = tmp_path / "in.jsonl"
    rows = [{"video_id": "a", "video_path": "/v.mp4", "prompt": "P", "answer": "A"}]
    in_path.write_text(json.dumps(rows[0]) + "\n")
    normalize_file(in_path, in_path)
    snapshot = in_path.read_text()
    normalize_file(in_path, in_path)
    assert in_path.read_text() == snapshot
