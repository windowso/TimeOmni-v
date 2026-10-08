"""Tests for the classification answer parser, incl. the prefix fallback.

The fallback exists because a fine-tuned model emits the answer as its first
token and then keeps writing until ``--max_new_tokens`` runs out (there is no
EOS stop), fusing noise onto the answer: ``"BB\\nHuman:"``, ``"Bend: the arm
movement"``, ``"3232"``. Those are real outputs taken from
``runs/timeomni_v-*/full/*/predictions.jsonl``.
"""

from __future__ import annotations

import pytest

from timeomni_v.inference.parse import build_parser

LETTERS = ["A", "B", "C", "D"]
CUHK = [str(i) for i in range(1, 41)]  # 40-way action ids


# --------------------------------------------------------------------------
# Strict pass must be unchanged — the fallback may only turn None into a label.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("A", "A"),
        (" B ", "B"),
        ("C.", "C"),
        ("Answer: D", "D"),
        ("A2", "A"),
        ("the answer is B", "B"),
    ],
)
def test_strict_matches_unchanged(raw, expected):
    assert build_parser(LETTERS)(raw) == expected
    assert build_parser(LETTERS, prefix_fallback=False)(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("10", "10"),
        ("action 7", "7"),
        ("40", "40"),
    ],
)
def test_strict_numeric_unchanged(raw, expected):
    assert build_parser(CUHK)(raw) == expected
    assert build_parser(CUHK, prefix_fallback=False)(raw) == expected


# --------------------------------------------------------------------------
# Real unparseable outputs the fallback is meant to rescue.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("BB\nHuman:", "B"),          # future_factories
        ("DD\nHuman:", "D"),          # future_factories
        ("Bend: the arm movement is performed", "B"),   # agibot
        ("Cvel2: end-effector velocity", "C"),          # agibot
        ("Behavior>\n", "B"),                           # agibot
        ("Able\n- V no error:", "A"),                   # agibot
        ("Determine>\n robot left", "D"),               # agibot
        ("BPM minute\n- spo2_percent", "B"),            # mimic_disch
        ("Amia.\n</variables>\n", "A"),                 # mimic_disch
    ],
)
def test_prefix_fallback_rescues_letter_labels(raw, expected):
    assert build_parser(LETTERS, prefix_fallback=False)(raw) is None
    assert build_parser(LETTERS)(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("3232\n", "32"),     # cuhk_x_har — "32" repeated, no digit boundary
        ("218\nHuman is", "21"),
        ("111\nHuman", "11"),
    ],
)
def test_prefix_fallback_rescues_numeric_labels(raw, expected):
    assert build_parser(CUHK, prefix_fallback=False)(raw) is None
    assert build_parser(CUHK)(raw) == expected


def test_prefix_fallback_prefers_longest_valid_prefix():
    # "218" → "21" (valid, len 2) beats "2" (valid, len 1); "218" itself is
    # not a label so the 3-char attempt fails first.
    assert build_parser(CUHK)("218") == "21"
    # With only 1-digit labels available, the same head yields "2".
    assert build_parser(["1", "2", "3"])("218") == "2"


# --------------------------------------------------------------------------
# The fallback must not invent answers.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "",
        "   ",
        "!!!",                       # no leading letter/digit run
        "Zebra",                     # head "Zebra" has no valid label prefix
        "\n\n",
    ],
)
def test_prefix_fallback_returns_none_when_no_head_match(raw):
    assert build_parser(LETTERS)(raw) is None


def test_fallback_is_anchored_at_position_zero():
    # A label appearing only later in the string must stay unparsed: the
    # strict pass rejects it (letter-glued) and the fallback only reads the
    # head. Guards against the fallback silently widening to a search.
    raw = "xyzzy Bend"
    assert build_parser(LETTERS, prefix_fallback=False)(raw) is None
    assert build_parser(LETTERS)(raw) is None


def test_empty_label_set_still_returns_none():
    assert build_parser([])("A") is None


def test_fallback_never_overrides_a_strict_match():
    # Head is "Cvel" → the fallback alone would say "C", but the strict pass
    # finds a clean "D" later and therefore never consults the fallback.
    # Whatever strict returns is returned verbatim.
    raw = "Cvel2 ... answer: D"
    assert build_parser(LETTERS, prefix_fallback=False)(raw) == "D"
    assert build_parser(LETTERS)(raw) == "D"
