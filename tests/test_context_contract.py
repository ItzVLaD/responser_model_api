"""Bounded memory validation and optional workspace mirror compatibility checks."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest
from pydantic import BaseModel, ValidationError

from responser_model_api import schemas
from responser_model_api.schemas import (
    ChatDescriptor,
    ChatSnapshot,
    ConversationContext,
    MemoryContent,
    RelationshipState,
    SummarizeContextRequest,
    SummarizeContextResponse,
)


def _checkpoint() -> ConversationContext:
    return ConversationContext(
        memory=MemoryContent(interlocutor=["Enjoys hiking."], agent=["Agent said they enjoy chess."]),
        last_message_id="123", summarized_message_count=30,
        model_name="qwen2.5:7b", updated_at="2026-09-12T12:00:00Z",
    )


def test_memory_defaults_are_independent_and_greetings_can_be_empty() -> None:
    first, second = MemoryContent(), MemoryContent()
    first.agent.append("Agent said hello.")
    assert second.agent == []
    assert second.relationship == RelationshipState(stage="unknown", evidence="")
    snapshot = ChatSnapshot(chat=ChatDescriptor(raw_id="1", title="Test"))
    assert snapshot.context is None


@pytest.mark.parametrize("field", ["interlocutor", "agent", "interaction", "open_threads"])
@pytest.mark.parametrize("value", [[""], [" \n "], ["x" * 201], ["fact"] * 9, [1], None])
def test_memory_list_bounds(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        MemoryContent.model_validate({field: value})


def test_memory_total_size_boundary_and_json_escaping() -> None:
    values = {name: ["x" * 180] * 8 for name in (
        "interlocutor", "agent", "interaction", "open_threads",
    )}
    base = MemoryContent.model_validate(values)
    remaining = schemas.MAX_MEMORY_CHARACTERS - len(base.model_dump_json())
    assert 0 < remaining < 300
    payload = {**values, "relationship": {"evidence": "x" * remaining}}
    assert len(MemoryContent.model_validate(payload).model_dump_json()) == 6000
    payload["relationship"] = {"evidence": "x" * (remaining + 1)}
    with pytest.raises(ValidationError, match="6000 characters"):
        MemoryContent.model_validate(payload)
    # Escaped quotes consume two serialized characters, not one.
    with pytest.raises(ValidationError, match="6000 characters"):
        MemoryContent(interlocutor=['"' * 200] * 8, agent=['"' * 200] * 8)


@pytest.mark.parametrize("field", ["last_message_id", "model_name", "updated_at"])
@pytest.mark.parametrize("value", ["", "  ", None])
def test_checkpoint_strings_must_be_nonempty(field: str, value: object) -> None:
    payload = _checkpoint().model_dump()
    payload[field] = value
    with pytest.raises(ValidationError):
        ConversationContext.model_validate(payload)


@pytest.mark.parametrize("count", [0, -1, 1.5])
def test_checkpoint_count_is_positive_integer(count: float) -> None:
    with pytest.raises(ValidationError):
        ConversationContext.model_validate({**_checkpoint().model_dump(), "summarized_message_count": count})


@pytest.mark.parametrize("model,payload", [
    (RelationshipState, {"extra": 1}),
    (MemoryContent, {"extra": 1}),
    (ConversationContext, {**_checkpoint().model_dump(), "extra": 1}),
    (SummarizeContextRequest, {"messages": [{"sender_type": "other", "text": "hi"}], "extra": 1}),
    (SummarizeContextResponse, {"memory": {}, "model_name": "local", "extra": 1}),
])
def test_new_models_forbid_extra_fields(model: type[BaseModel], payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        model.model_validate(payload)


@pytest.fixture
def mirror(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Load only the mirrored file, never require/install the reader package."""
    path = Path(__file__).resolve().parents[2] / "responser_web_reader/src/responser_web_reader/schemas.py"
    if not path.is_file():
        pytest.skip("reader mirror is not present in this standalone checkout")
    spec = importlib.util.spec_from_file_location("_reader_schema_contract_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", [
    "Message", "ChatDescriptor", "ChatSnapshot", "GenerateReplyRequest", "GeneratedReply",
    "RelationshipState", "MemoryContent", "ConversationContext",
    "SummarizeContextRequest", "SummarizeContextResponse",
])
def test_json_schema_matches_reader_mirror(mirror: ModuleType, name: str) -> None:
    api_model = cast(type[BaseModel], getattr(schemas, name))
    reader_model = cast(type[BaseModel], getattr(mirror, name))
    assert api_model.model_json_schema() == reader_model.model_json_schema()


def test_mirrored_source_and_payloads_are_identical(mirror: ModuleType) -> None:
    assert schemas.__file__ is not None and mirror.__file__ is not None
    api_source = Path(schemas.__file__).read_text(encoding="utf-8")
    reader_source = Path(mirror.__file__).read_text(encoding="utf-8")
    assert api_source.split("from __future__", 1)[1] == reader_source.split("from __future__", 1)[1]
    snapshot = ChatSnapshot(chat=ChatDescriptor(raw_id="1", title="Test"), context=_checkpoint())
    reader_snapshot = mirror.ChatSnapshot.model_validate_json(snapshot.model_dump_json())
    assert reader_snapshot.model_dump_json() == snapshot.model_dump_json()
    request = {"previous": _checkpoint().memory.model_dump(), "messages": [{"sender_type": "other", "text": "hi"}]}
    api_request = SummarizeContextRequest.model_validate(request)
    assert mirror.SummarizeContextRequest.model_validate_json(api_request.model_dump_json()).model_dump_json() == api_request.model_dump_json()
    response = SummarizeContextResponse(memory=_checkpoint().memory, model_name="qwen2.5:7b")
    assert mirror.SummarizeContextResponse.model_validate_json(response.model_dump_json()).model_dump_json() == response.model_dump_json()
    assert json.loads(reader_snapshot.model_dump_json())["context"]["summarized_message_count"] == 30