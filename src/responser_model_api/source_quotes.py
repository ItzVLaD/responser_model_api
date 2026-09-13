"""Recover exact source spans from formatting-only model transcription changes.

No fuzzy matching, case folding, word replacement, digit conversion, accent
removal or cross-message search. Returned strings always come from the source.
"""

from __future__ import annotations

# Do not fold dashes (ranges), ellipses, symbols, or general Unicode lookalikes.
_QUOTE_FORMS = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})


def _indexed_text(text: str) -> tuple[str, list[tuple[int, int]]]:
    """Collapse whitespace but retain source offsets for every normalized char."""
    normalized: list[str] = []
    offsets: list[tuple[int, int]] = []
    for index, char in enumerate(text):
        if char.isspace():
            if normalized and normalized[-1] == " ":
                offsets[-1] = (offsets[-1][0], index + 1)
                continue
            char = " "
        normalized.append(char.translate(_QUOTE_FORMS))
        offsets.append((index, index + 1))
    return "".join(normalized), offsets


def _word_char(char: str) -> bool:
    return char.isalnum() or char == "_"


def _splits_token(text: str, boundary: int) -> bool:
    """Do not recover partial words, decimals, grouped numbers or numeric ranges."""
    if boundary == 0 or boundary == len(text):
        return False
    left, right = text[boundary - 1], text[boundary]
    if _word_char(left) and _word_char(right):
        return True
    # Punctuation between digits belongs to the number, not a safe quote edge.
    separators = ".,/:+-–—−"
    if left.isdigit() and right in separators:
        return boundary + 1 < len(text) and text[boundary + 1].isdigit()
    if right.isdigit() and left in separators:
        return left in "+-−" or (boundary > 1 and text[boundary - 2].isdigit())
    return False


def formatting_equivalent_spans(source: str, proposed: str) -> set[str]:
    """Return distinct original spans, rejecting matches inside words/numbers.

    The caller must reject multiple distinct spans rather than guess which
    evidence the model meant. Repeated identical spans are interchangeable as
    proof within the same message, whose ID is checked separately.
    """
    normalized, offsets = _indexed_text(source)
    needle, _ = _indexed_text(proposed.strip())
    if not needle:
        return set()
    matches: set[str] = set()
    start = normalized.find(needle)
    while start != -1:
        end = start + len(needle)
        if not _splits_token(normalized, start) and not _splits_token(normalized, end):
            matches.add(source[offsets[start][0]:offsets[end - 1][1]])
        start = normalized.find(needle, start + 1)
    return matches