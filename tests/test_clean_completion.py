"""Tests for cleaning chat-template artifacts from model output."""

from __future__ import annotations

from responser_model_api.ollama_client import _clean_completion, _finalize


def test_truncated_reply_is_trimmed_to_last_sentence() -> None:
    response = {
        "message": {"content": "hey! doing great, you? i was just about to"},
        "done_reason": "length",
    }
    assert _finalize(response) == "hey! doing great, you?"


def test_truncated_reply_kept_when_no_sentence_would_remain() -> None:
    response = {"message": {"content": "ok. so basically what happened was that"}, "done_reason": "length"}
    # Trimming to "ok." would drop most of the text; better a clipped line.
    assert _finalize(response) == "ok. so basically what happened was that"


def test_complete_reply_not_trimmed() -> None:
    response = {"message": {"content": "hey! doing great, you? i was just about to"}, "done_reason": "stop"}
    assert _finalize(response) == "hey! doing great, you? i was just about to"


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


def test_drops_echoed_instructions_before_reply_label() -> None:
    # Real leak observed with nous-hermes2: the trailing length note was echoed
    # verbatim, followed by a "Reply:" label and the quoted actual reply.
    leaked = (
        "system\nLength: they asked you something, so answer it properly. A few "
        "sentences is fine.\n\nReply:\n\"hey there 😘 i'm doing pretty good, how "
        "about you?\""
    )
    assert _clean_completion(leaked) == "hey there 😘 i'm doing pretty good, how about you?"


def test_strips_wrapping_quotes() -> None:
    assert _clean_completion('"hey! no worries, what\'s on your mind?"') == (
        "hey! no worries, what's on your mind?"
    )
    assert _clean_completion("«ok, давай»") == "ok, давай"


def test_keeps_inner_quotes() -> None:
    text = 'he literally said "no" and left'
    assert _clean_completion(text) == text


def test_strips_leading_reply_label() -> None:
    assert _clean_completion("Reply: hey you") == "hey you"
    assert _clean_completion("Me: hey you") == "hey you"


def test_does_not_collapse_distinct_paragraphs() -> None:
    text = "first thought here\n\na totally different second thought"
    assert _clean_completion(text) == text
