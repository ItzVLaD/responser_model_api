"""Tests for cleaning chat-template artifacts from model output."""

from __future__ import annotations

from responser_model_api.ollama_client import _clean_completion


def test_strips_trailing_role_word_glued() -> None:
    assert _clean_completion("hey what's up? system") == "hey what's up?"


def test_strips_trailing_role_on_new_line() -> None:
    assert _clean_completion("doing great, you?\nsystem") == "doing great, you?"
    assert _clean_completion("hi there\nassistant") == "hi there"


def test_strips_template_wrapped_role() -> None:
    assert _clean_completion("hello\n<|system|>") == "hello"
    assert _clean_completion("hello\nuser:") == "hello"


def test_strips_multiple_trailing_labels() -> None:
    assert _clean_completion("yo\nsystem\nuser") == "yo"


def test_does_not_touch_real_words() -> None:
    # "ecosystem" ends with "system" but is a real word; must not be truncated.
    assert _clean_completion("i love the ecosystem") == "i love the ecosystem"


def test_leaves_clean_text_unchanged() -> None:
    assert _clean_completion("haha nah, what else is up?") == "haha nah, what else is up?"


def test_collapses_duplicated_reply() -> None:
    doubled = "sorry, my bad! let's move on 🙈\n\nsorry, my bad! let's move on 🙈"
    assert _clean_completion(doubled) == "sorry, my bad! let's move on 🙈"


def test_does_not_collapse_distinct_paragraphs() -> None:
    text = "first thought here\n\na totally different second thought"
    assert _clean_completion(text) == text
