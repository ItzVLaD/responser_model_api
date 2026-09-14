"""Opt-in downstream reply check on synthetic archived evidence, never live chats.

This measures factual use in one reply, not universal retrieval/model quality.
The reader module is optional test-only workspace integration, never a runtime
dependency of the model API.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import pytest

from responser_model_api.config import load_generation_settings
from responser_model_api.ollama_client import OllamaReplyGenerator
from responser_model_api.personality import Personality
from responser_model_api.schemas import ChatSnapshot

pytestmark = pytest.mark.skipif(os.environ.get("RESPONSER_RUN_RETRIEVAL_LIVE_TESTS") != "1", reason="Explicit opt-in local model test")


def test_reply_uses_archived_job_specialty_and_corrected_age(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    reader_source = Path(__file__).resolve().parents[2] / "responser_web_reader" / "src"
    if not reader_source.is_dir():
        pytest.skip("Reader workspace not present")
    monkeypatch.syspath_prepend(str(reader_source))
    from responser_web_reader.history_archive import HistoryArchive
    from responser_web_reader.retrieval_context import RetrievalContextManager
    from responser_web_reader.schemas import ChatDescriptor, ChatSnapshot as ReaderSnapshot, Message

    turns = [("other", "I'm a landscape designer."), ("other", "I design roof gardens."),
             ("other", "I'm 73."), ("other", "Just kidding."), ("other", "I'm 29.")]
    turns += [("me", "Nothing new today.")] * 35
    turns += [("other", "Tell me what you remember about my age, job, and what I design.")]
    messages = [Message.model_validate({"raw_id": str(i), "sender_type": who, "text": text}) for i, (who, text) in enumerate(turns)]
    chat = ChatDescriptor(raw_id="synthetic", title="Synthetic conversation")

    class Reader:
        def account_id(self) -> str:
            return "synthetic-account"

        def read_history(self, chat: ChatDescriptor, until_id: str | None = None) -> ReaderSnapshot:
            return ReaderSnapshot(chat=chat, messages=messages)

    class NoSummary:
        def summarize_context(self, request: object) -> None:
            pytest.fail("Retrieval must not call the summarizer")

    snapshot = RetrievalContextManager(HistoryArchive(tmp_path.resolve() / "archive")).prepare(Reader(), NoSummary(), chat, "synthetic")
    request = ChatSnapshot.model_validate_json(snapshot.model_dump_json())
    generator = OllamaReplyGenerator(Personality(name="Alex", tone="direct and factual"), load_generation_settings())
    scores: dict[str, int] = {}
    for mode, sample in (("recent_only", request.model_copy(update={"retrieval_context": None})), ("retrieval", request)):
        start = time.monotonic()
        reply = generator.generate(sample).text
        score = sum((bool(re.search(r"\b29\b", reply)), "landscape" in reply.lower(), "roof" in reply.lower() and "garden" in reply.lower()))
        scores[mode] = score
        print("RETRIEVAL_REPLY_EVALUATION", json.dumps({"mode": mode, "covered": score, "required": 3, "seconds": round(time.monotonic() - start, 1)}), flush=True)
        if mode == "retrieval":
            assert not re.search(r"\b73\b", reply), "Reply reused a retracted age"
    assert scores["retrieval"] == 3
    assert scores["retrieval"] > scores["recent_only"]