"""Opt-in compact-summary smoke test with synthetic, non-example facts.

Checks a 20-message batch and a later correction. Passing is not a general
quality or latency guarantee. No browser, persisted context or message sending.
"""

from __future__ import annotations

import os
import re

import pytest

from responser_model_api.config import load_summary_settings
from responser_model_api.schemas import Message, SummarizeContextRequest
from responser_model_api.simple_context import SimpleContextSummarizer

pytestmark = pytest.mark.skipif(os.environ.get("RESPONSER_RUN_SIMPLE_SUMMARY_LIVE_TESTS") != "1", reason="Explicit opt-in local model test")


def test_twenty_message_summary_and_later_correction_keep_valuable_details() -> None:
    messages = [
        Message(sender_type="other", text="I'm 32 and a librarian. I catalogue rare manuscripts.", raw_id="1"),
        Message(sender_type="me", text="I'm 26 and a photographer. I went cycling last weekend.", raw_id="2"),
    ]
    messages += [Message(sender_type="other" if i % 2 else "me", text="Okay, thanks.", raw_id=str(i)) for i in range(3, 21)]
    summarizer = SimpleContextSummarizer(load_summary_settings())
    initial = summarizer.summarize(SummarizeContextRequest(messages=messages)).memory
    other, agent = " ".join(initial.interlocutor).lower(), " ".join(initial.agent).lower()
    assert re.search(r"\b32\b", other) and "librarian" in other and "manuscript" in other
    assert re.search(r"\b26\b", agent) and "photograph" in agent and "cycl" in agent
    # Ages/jobs in instructional examples must not leak into real summaries.
    assert not re.search(r"\b(?:29|24|73)\b", other + agent)
    assert "landscape" not in other + agent and "lab technician" not in other + agent
    assert initial.facts == [] and initial.relationship_source is None
    updated = summarizer.summarize(SummarizeContextRequest(previous=initial, messages=[
        Message(sender_type="other", text="Correction: I'm 33, not 32. Please remember my work and ask follow-up questions.", raw_id="21"),
    ])).memory
    other, agent = " ".join(updated.interlocutor).lower(), " ".join(updated.agent).lower()
    assert re.search(r"\b33\b", other) and "librarian" in other and "manuscript" in other
    assert re.search(r"\b26\b", agent) and "photograph" in agent and "cycl" in agent
    assert "follow" in " ".join([*updated.interlocutor, *updated.interaction]).lower()