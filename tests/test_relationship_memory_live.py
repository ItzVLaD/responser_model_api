"""Opt-in, non-graphic evaluation of shared-event memory and current rapport.

Invented adult conversations only. These checks evaluate a summarizer, not a
reply generator, and never read/write a user's saved context or contact anyone.
"""

from __future__ import annotations

import os
import re

import pytest

from responser_model_api.config import load_summary_settings
from responser_model_api.schemas import MemoryContent, Message, SummarizeContextRequest
from responser_model_api.simple_context import SimpleContextSummarizer

pytestmark = pytest.mark.skipif(
    os.environ.get("RESPONSER_RUN_RELATIONSHIP_MEMORY_LIVE_TESTS") != "1",
    reason="Explicit opt-in local summary evaluation",
)


def test_shared_online_intimacy_and_explicit_repair_are_not_erased() -> None:
    previous = MemoryContent(
        interlocutor=["Age 32; librarian."], agent=["Age 26; photographer."],
        interaction=["They argued earlier about an unanswered message."],
        open_threads=["Interlocutor's real name remains unknown."],
        relationship={"stage": "strained", "evidence": "An earlier disagreement."},
    )
    original = previous.model_dump_json()
    request = SummarizeContextRequest(previous=previous, messages=[
        Message(raw_id="1", sender_type="other", text="We resolved that earlier argument. I trust you and feel comfortable with you now."),
        Message(raw_id="2", sender_type="me", text="I feel the same closeness and trust. We both participated in that intimate online role-play, and I enjoyed sharing it with you."),
        Message(raw_id="3", sender_type="other", text="Yes, I enjoyed our shared online intimacy too. It was virtual, not an in-person meeting. We can still say no next time."),
        Message(raw_id="4", sender_type="me", text="Agreed. We are close, and neither of us owes the other any future participation."),
    ])
    memory = SimpleContextSummarizer(load_summary_settings()).summarize(request).memory
    interactions = " ".join(memory.interaction).casefold()
    assert re.search(r"online|virtual", interactions), "Shared event lost its virtual setting"
    assert re.search(r"intima|role.?play", interactions), "Shared intimacy was generalized away"
    assert memory.relationship.stage == "familiar", "Old strain persisted despite explicit mutual repair and trust"
    assert "librarian" in " ".join(memory.interlocutor).casefold()
    assert "photographer" in " ".join(memory.agent).casefold()
    assert not any("name" in thread.casefold() and "unknown" in thread.casefold() for thread in memory.open_threads)
    assert previous.model_dump_json() == original


def test_one_sided_request_and_current_refusal_do_not_become_shared_closeness() -> None:
    request = SummarizeContextRequest(previous=MemoryContent(
        interlocutor=["Age 32."], agent=["Age 26."],
        relationship={"stage": "acquaintance", "evidence": "They had exchanged hobbies."},
    ), messages=[
        Message(raw_id="1", sender_type="other", text="Would you like to have an intimate online conversation?"),
        Message(raw_id="2", sender_type="me", text="No, I do not want that. Please stop asking; I feel uncomfortable."),
        Message(raw_id="3", sender_type="other", text="I am still asking you to change your mind."),
        Message(raw_id="4", sender_type="me", text="My answer is still no. This pressure is making our conversation tense."),
    ])
    memory = SimpleContextSummarizer(load_summary_settings()).summarize(request).memory
    text = " ".join([*memory.interaction, memory.relationship.evidence]).casefold()
    assert re.search(r"declin|refus|uncomfortable|pressure|boundar", text), "Current refusal/boundary was omitted"
    assert memory.relationship.stage == "strained", "Current pressure was mistaken for mutual closeness"
    assert not re.search(r"both (?:enjoyed|participated)|mutual (?:intimacy|trust)|shared (?:online )?intimacy", text)