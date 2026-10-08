"""Tests for the forecast-task answer parser + alignment helper."""

import numpy as np

from timeomni_v.inference.forecast_parse import align_forecast, parse_forecast


SAMPLE_TERRA_ANSWER = """<forecast>
(2023-12-12: 421.6, 0.7, 100789, 69.5, 0.01646, 28.07, 4.77)
(2023-12-13: 404.0, 0.0, 100847, 67.5, 0.01589, 28.00, 5.28)
(2023-12-14: 407.4, 0.2, 100814, 68.0, 0.01603, 28.00, 4.37)
</forecast>"""


def test_parse_terra_answer_round_trips():
    rows = parse_forecast(SAMPLE_TERRA_ANSWER)
    assert rows is not None and len(rows) == 3
    label, vals = rows[0]
    assert label == "2023-12-12"
    assert vals.shape == (7,)
    assert vals[0] == 421.6
    assert vals[-1] == 4.77


def test_parse_returns_none_on_unparseable():
    assert parse_forecast(None) is None
    assert parse_forecast("") is None
    assert parse_forecast("the model rambled but never emitted parens") is None


def test_parse_truncated_close_tag_still_works():
    """Generated outputs frequently lose the closing </forecast> tag."""
    truncated = "<forecast>\n(2024-01-01: 1.0, 2.0)\n(2024-01-02: 3.0, 4.0)"
    rows = parse_forecast(truncated)
    assert rows is not None and len(rows) == 2
    assert rows[1][1].tolist() == [3.0, 4.0]


def test_parse_skips_malformed_lines_between_good_ones():
    body = (
        "<forecast>\n"
        "(2024-01-01: 1.0, 2.0)\n"
        "garbage line\n"
        "(2024-01-02: 3.0, 4.0)\n"
        "</forecast>"
    )
    rows = parse_forecast(body)
    assert [r[0] for r in rows] == ["2024-01-01", "2024-01-02"]


def test_parse_handles_negative_and_scientific_numbers():
    rows = parse_forecast("<forecast>\n(t: -1.5, 1e-3, 2.5e+2)\n</forecast>")
    assert rows is not None
    assert rows[0][1].tolist() == [-1.5, 0.001, 250.0]


def test_align_by_label_when_all_labels_match():
    gt = parse_forecast("<forecast>\n(a: 1, 2)\n(b: 3, 4)\n</forecast>")
    pred = parse_forecast("<forecast>\n(b: 30, 40)\n(a: 10, 20)\n</forecast>")
    g, p, info = align_forecast(gt, pred)
    assert info["alignment"] == "by_label"
    assert not info["length_mismatch"]
    # GT order is preserved; pred is reordered by label.
    np.testing.assert_array_equal(g, np.array([[1.0, 2.0], [3.0, 4.0]]))
    np.testing.assert_array_equal(p, np.array([[10.0, 20.0], [30.0, 40.0]]))


def test_align_falls_back_to_positional_on_label_mismatch():
    gt = parse_forecast("<forecast>\n(2024-01: 1)\n(2024-02: 2)\n</forecast>")
    pred = parse_forecast("<forecast>\n(jan: 10)\n(feb: 20)\n</forecast>")
    g, p, info = align_forecast(gt, pred)
    assert info["alignment"] == "positional"
    assert info["missing_labels"] == 2
    assert g.tolist() == [[1.0], [2.0]]
    assert p.tolist() == [[10.0], [20.0]]


def test_align_length_mismatch_truncates_to_shorter():
    gt = parse_forecast("<forecast>\n(a: 1)\n(b: 2)\n(c: 3)\n</forecast>")
    pred = parse_forecast("<forecast>\n(x: 10)\n(y: 20)\n</forecast>")
    g, p, info = align_forecast(gt, pred)
    assert info["length_mismatch"]
    assert g.shape == p.shape == (2, 1)


def test_align_handles_empty_inputs():
    g, p, info = align_forecast([], [])
    assert g.size == 0 and p.size == 0
    assert info["len_gt"] == 0 and info["len_pred"] == 0
