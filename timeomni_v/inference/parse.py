"""Answer parser shared by inference and offline re-scoring.

Kept torch-free so ``timeomni_v.inference.eval --reparse`` can rewrite an
existing predictions.jsonl without dragging in transformers/peft.

The parser auto-adapts to whatever labels the test set actually uses — see
``build_parser``.
"""

from __future__ import annotations

import re
from typing import Callable, Iterable


def _build_prefix_fallback(
    labels: list[str],
) -> Callable[[str | None], str | None]:
    """Return a fallback that reads the answer off the *head* of ``raw``.

    Motivation: a fine-tuned model emits the answer as its very first token
    and then, because generation is capped at ``--max_new_tokens`` rather
    than stopped at EOS, keeps writing noise that fuses onto it —
    ``"BB\\nHuman:"``, ``"Bend: the arm movement"``, ``"3232"``. The strict
    boundary rules in :func:`build_parser` reject all three even though the
    intended answer ("B", "B", "32") is unambiguous from position alone.

    Rule: take the leading run of same-class characters (letters *or*
    digits) after stripping whitespace, then return the **longest label
    that is a prefix of that run**. Anything not anchored at position 0 is
    ignored, so a stray label mention later in the text can never be picked
    up — that case is exactly what the strict parser is for.
    """
    lower_to_canonical = {lbl.lower(): lbl for lbl in labels}
    max_len = max((len(l) for l in labels), default=0)

    def fallback(raw: str | None) -> str | None:
        s = (raw or "").lstrip()
        m = re.match(r"[A-Za-z]+|\d+", s)
        if m is None:
            return None
        head = m.group(0)
        for k in range(min(len(head), max_len), 0, -1):
            hit = lower_to_canonical.get(head[:k].lower())
            if hit is not None:
                return hit
        return None

    return fallback


def build_parser(
    labels: Iterable[str], *, prefix_fallback: bool = True
) -> Callable[[str | None], str | None]:
    """Return a parser closure tuned to the supplied label set.

    Behaviour:

    * Longest-first alternation so e.g. ``"10"`` is preferred over ``"1"``
      when both are valid labels and the model emits ``"10"``.
    * Numeric labels (all digits) require non-digit boundaries — ``"1"``
      inside ``"15 frames"`` won't be returned when ``15`` is also a label.
    * Word labels (all letters) require non-letter boundaries — ``"A"``
      inside ``"Above"`` is rejected, but glued forms like ``"A2"`` or
      ``"A."`` still match (digits/punct are non-letters).
    * Other shapes (rare) are matched verbatim with no boundary.
    * Matching is case-insensitive; the returned string is the canonical
      label from ``labels`` (case preserved).
    * ``prefix_fallback`` (default on): when the strict pass finds nothing,
      re-read the answer off the head of the string — see
      :func:`_build_prefix_fallback`. The strict pass always wins when it
      matches, so this only ever converts ``None`` into a label; it can
      never change an already-parsed answer.

    Empty ``labels`` → a parser that always returns ``None`` (no scoring
    vocabulary, nothing to extract).
    """
    label_list = sorted({str(l) for l in labels}, key=lambda s: (-len(s), s))
    if not label_list:
        return lambda raw: None

    parts: list[str] = []
    for lbl in label_list:
        esc = re.escape(lbl)
        if re.fullmatch(r"\d+", lbl):
            parts.append(rf"(?<!\d){esc}(?!\d)")
        elif re.fullmatch(r"[A-Za-z]+", lbl):
            parts.append(rf"(?<![A-Za-z]){esc}(?![A-Za-z])")
        else:
            parts.append(esc)
    pattern = re.compile("|".join(parts), re.IGNORECASE)

    # Lower → canonical map for case-insensitive recovery. Last writer wins
    # if two labels collapse under lower() — that would only happen with a
    # weird mixed-case label set, which we don't have today.
    canonical = {lbl.lower(): lbl for lbl in label_list}
    fallback = _build_prefix_fallback(label_list) if prefix_fallback else None

    def parser(raw: str | None) -> str | None:
        m = pattern.search(raw or "")
        if m is None:
            return fallback(raw) if fallback is not None else None
        matched = m.group(0)
        return canonical.get(matched.lower(), matched)

    return parser
