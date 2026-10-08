"""Tests for the task-based routing in timeomni_v.inference.eval."""

import json

from timeomni_v.inference.eval import (
    _detect_task,
    compute_metrics,
    compute_regression_metrics,
)


def test_detect_task_from_test_jsonl(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text(json.dumps({"id": "a", "task": "prediction"}) + "\n")
    assert _detect_task([], p) == "prediction"


def test_detect_task_from_records_when_no_test_jsonl():
    records = [{"id": "a", "task": "prediction", "ground_truth": "x", "prediction": "y"}]
    assert _detect_task(records, None) == "prediction"


def test_detect_task_defaults_to_classification():
    assert _detect_task([], None) == "classification"


def test_classification_metrics_smoke():
    rows = [
        ("A", "A"),
        ("A", "B"),
        ("B", "B"),
        ("B", None),
    ]
    m = compute_metrics(rows, ["A", "B"])
    assert m["task"] == "classification"
    assert m["total"] == 4
    assert m["correct"] == 2
    assert m["unparseable"] == 1
    assert 0 <= m["accuracy"] <= 1


def test_regression_metrics_smoke():
    """Two parseable rows + one unparseable should yield parse_rate=2/3 and
    finite metrics. Use simple inputs so we can sanity-check magnitudes."""
    records = [
        {
            "id": "a",
            "ground_truth": "<forecast>\n(t1: 1, 2)\n(t2: 3, 4)\n</forecast>",
            "raw":          "<forecast>\n(t1: 1.5, 2)\n(t2: 3, 4.5)\n</forecast>",
            "prediction": None, "failed": False,
        },
        {
            "id": "b",
            "ground_truth": "<forecast>\n(t1: 10, 20)\n(t2: 30, 40)\n</forecast>",
            "raw":          "<forecast>\n(t1: 10, 20)\n(t2: 30, 40)\n</forecast>",
            "prediction": None, "failed": False,
        },
        {
            "id": "c",
            "ground_truth": "<forecast>\n(t1: 1)\n</forecast>",
            "raw": "model gave up",
            "prediction": None, "failed": False,
        },
    ]
    metrics, debug = compute_regression_metrics(records)
    assert metrics["task"] == "prediction"
    assert metrics["total"] == 3
    assert metrics["succeeded"] == 2
    assert metrics["failed"] == 1
    assert abs(metrics["parse_rate"] - 2 / 3) < 1e-9
    # Per-row MAE: row a = mean(|0.5|, 0, 0, |0.5|) = 0.25; row b = 0.
    # Dataset MAE = mean over rows = (0.25 + 0) / 2 = 0.125.
    assert abs(metrics["mae"] - 0.125) < 1e-6
    assert metrics["channels"] == 2
    # Debug should have one entry per record.
    assert len(debug) == 3
    assert debug[2]["parsed_pred"] is False
