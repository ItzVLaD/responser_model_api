"""Opt-in style smoke check using invented chat turns, never user conversations.

One successful response is not a guarantee about long-run phrasing frequency.
Ordinary tests mock inference; enable this check explicitly for local Ollama.
"""

from __future__ import annotations

import os
import re

import pytest

from responser_model_api.config import PERSONALITIES_DIR, load_generation_settings
from responser_model_api.ollama_client import OllamaReplyGenerator
from responser_model_api.personality import load_personality
from responser_model_api.schemas import ChatDescriptor, ChatSnapshot, Message

pytestmark = pytest.mark.skipif(
    os.environ.get("RESPONSER_RUN_REPLY_STYLE_LIVE_TESTS") != "1",
    reason="Explicit opt-in local reply-style model check",
)


def test_reply_does_not_repeat_recent_service_style_or_emoji() -> None:
    snapshot = ChatSnapshot(chat=ChatDescriptor(raw_id="synthetic-style", title="Synthetic conversation"), messages=[
        Message(sender_type="other", text="I started a painting yesterday"),
        Message(sender_type="me", text="Let's talk about something fun 😈"),
        Message(sender_type="other", text="I finished painting the sea this evening"),
    ])
    generator = OllamaReplyGenerator(load_personality("mia", PERSONALITIES_DIR), load_generation_settings())
    reply = generator.generate(snapshot).text
    assert reply.strip()
    assert not re.search(r"\blet['’]s\b|let me know|anything else i can help", reply, re.I)
    assert not reply.endswith(".")
    assert "😈" not in reply