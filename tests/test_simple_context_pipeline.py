"""Workspace-only default endpoint -> reader checkpoint integration, no browser."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from ollama import Client

from responser_model_api import app as app_module
from responser_model_api import simple_context as simple_module
from responser_model_api.config import SummarySettings
from responser_model_api.simple_context import SimpleContextSummarizer


def test_twenty_bootstrap_chains_simple_summaries_and_saves_no_proof_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    reader_source = Path(__file__).resolve().parents[2] / "responser_web_reader" / "src"
    if not reader_source.is_dir():
        pytest.skip("Optional reader workspace is not present")
    monkeypatch.syspath_prepend(str(reader_source))
    from responser_web_reader.conversation_context import ContextStore, ConversationContextManager
    from responser_web_reader.schemas import ChatDescriptor, ChatSnapshot, Message, SummarizeContextRequest, SummarizeContextResponse

    summary = {"interlocutor": ["Librarian; catalogues rare manuscripts."], "agent": ["Photographer."],
               "interaction": [], "open_threads": [], "relationship": {"stage": "unknown", "evidence": ""}}
    inputs: list[dict[str, object]] = []

    def model_response(request: httpx.Request) -> httpx.Response:
        call = json.loads(request.content)
        inputs.append(json.loads(call["messages"][1]["content"]))
        assert set(call["format"]["properties"]) == set(summary)
        return httpx.Response(200, json={"done": True, "done_reason": "stop", "message": {
            "role": "assistant", "content": json.dumps(summary),
        }})

    def model_client(host: str) -> Client:
        return Client(host=host, transport=httpx.MockTransport(model_response))

    monkeypatch.setattr(simple_module, "Client", model_client)
    monkeypatch.setattr(app_module, "_summarizer", SimpleContextSummarizer(SummarySettings()))
    client = TestClient(app_module.app)
    chat = ChatDescriptor(raw_id="synthetic-chat", title="Offline")
    messages = [Message(raw_id=str(i), sender_type="other", text=f"Synthetic message {i}") for i in range(1, 129)]

    class Reader:
        def account_id(self) -> str:
            return "synthetic-account"

        def read_history(self, chat: ChatDescriptor, until_id: str | None = None) -> ChatSnapshot:
            return ChatSnapshot(chat=chat, messages=messages)

    class Api:
        def summarize_context(self, request: SummarizeContextRequest) -> SummarizeContextResponse:
            response = client.post("/summarize_context", json=request.model_dump(mode="json"))
            assert response.status_code == 200
            return SummarizeContextResponse.model_validate(response.json())

    store = ContextStore(tmp_path / "context")
    result = ConversationContextManager(store).prepare(Reader(), Api(), chat, "synthetic")
    assert [len(call["messages"]) for call in inputs] == [20, 20, 20, 20, 20, 18]
    assert inputs[0]["previous"] is None
    assert all(call["previous"] == summary for call in inputs[1:])
    assert result.messages == messages[-10:] and result.retrieval_context is None
    assert result.context is not None and result.context.last_message_id == "118"
    assert result.context.summarized_message_count == 118
    path = store.path_for("synthetic", "synthetic-account", chat.raw_id)
    saved = json.loads(path.read_text())
    assert saved["context"]["memory"] == summary
    assert "facts" not in saved["context"]["memory"] and "relationship_source" not in saved["context"]["memory"]