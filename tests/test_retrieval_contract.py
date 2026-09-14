"""Reference-only retrieval contract, UTF-8 limits, and reader mirror parity."""

from __future__ import annotations

import importlib.util
import json
import sys
import tomllib
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel, ValidationError

from responser_model_api import app as app_module
from responser_model_api import schemas
from responser_model_api.schemas import (
    MAX_RETRIEVAL_BYTES,
    ChatDescriptor,
    ChatSnapshot,
    ConversationContext,
    ConversationStateCandidates,
    HistoryEvidence,
    MemoryContent,
    Message,
    ProfileCandidates,
    RetrievalContext,
    SenderType,
)

PROFILE_FIELDS = tuple(ProfileCandidates.model_fields)
STATE_FIELDS = tuple(ConversationStateCandidates.model_fields)
REFERENCE_FIELDS = [
    *((section, field) for section in ("agent", "interlocutor") for field in PROFILE_FIELDS),
    *(("conversation_state", field) for field in STATE_FIELDS),
    ("", "relevant_message_ids"),
]


def _evidence(
    message_id: str = "private-source-me", sequence: int = 1, sender_type: SenderType = "me",
) -> HistoryEvidence:
    """Use invented source text; no persisted chat data is needed."""
    return HistoryEvidence(message_id=message_id, sequence=sequence, sender_type=sender_type, text="An old statement.")


def _retrieval() -> RetrievalContext:
    return RetrievalContext(
        evidence=[_evidence(), _evidence("private-source-other", 3, "other"), _evidence("private-service", 9, "system")],
        agent=ProfileCandidates(background=["private-source-me"]),
        interlocutor=ProfileCandidates(interests=["private-source-other"]),
        conversation_state=ConversationStateCandidates(reactions=["private-source-me", "private-source-other"]),
        relevant_message_ids=["private-source-other"], archive_message_count=100,
    )


def _checkpoint() -> ConversationContext:
    return ConversationContext(
        memory=MemoryContent(agent=["Old unverified summary."]),
        last_message_id="old-checkpoint", summarized_message_count=30,
        model_name="summary-model", updated_at="2026-09-13T00:00:00Z",
    )


def test_exact_fields_and_independent_defaults() -> None:
    assert set(PROFILE_FIELDS) == {
        "name", "age", "occupation", "specialty", "interests", "preferences", "boundaries", "background",
    }
    assert set(STATE_FIELDS) == {"questions", "commitments", "boundaries", "reactions"}
    assert set(HistoryEvidence.model_fields) == {"message_id", "sequence", "sender_type", "text", "timestamp", "truncated"}
    assert set(RetrievalContext.model_fields) == {
        "evidence", "agent", "interlocutor", "conversation_state", "relevant_message_ids",
        "archive_message_count", "history_complete", "budget_exhausted",
    }
    first = RetrievalContext(evidence=[], relevant_message_ids=[], archive_message_count=0)
    second = RetrievalContext(evidence=[], relevant_message_ids=[], archive_message_count=0)
    assert first.history_complete is True and first.budget_exhausted is False
    assert all(value == [] for value in first.agent.reference_lists().values())
    assert all(value == [] for value in first.interlocutor.reference_lists().values())
    assert all(value == [] for value in first.conversation_state.reference_lists().values())
    first.agent.name.append("not-shared")
    first.conversation_state.questions.append("also-not-shared")
    assert second.agent.name == first.interlocutor.name == []
    assert second.conversation_state.questions == []
    assert _evidence().timestamp is None and _evidence().truncated is False


@pytest.mark.parametrize("field,value", [
    ("message_id", ""), ("message_id", " \n "), ("message_id", "x" * 129),
    ("sequence", 0), ("sequence", -1), ("sequence", 1.5),
    ("sender_type", "assistant"), ("text", ""), ("text", "x" * 1201),
    ("text", None), ("timestamp", 123),
])
def test_evidence_field_bounds(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        HistoryEvidence.model_validate({**_evidence().model_dump(), field: value})


def test_evidence_maximum_and_whitespace_text_follow_the_contract() -> None:
    source = HistoryEvidence(message_id="x" * 128, sequence=1, sender_type="system", text="🙂" * 1200)
    assert len(source.text) == 1200
    assert HistoryEvidence.model_validate({**source.model_dump(), "text": " "}).text == " "


@pytest.mark.parametrize("model,field", [
    *((ProfileCandidates, field) for field in PROFILE_FIELDS),
    *((ConversationStateCandidates, field) for field in STATE_FIELDS),
])
@pytest.mark.parametrize("value", [[""], [" \n"], [1], None, "derived value", ["id", "id"], ["a", "b", "c", "d", "e"]])
def test_candidate_lists_are_unique_bounded_nonempty_references(
    model: type[BaseModel], field: str, value: object,
) -> None:
    with pytest.raises(ValidationError):
        model.model_validate({field: value})
    assert model.model_validate({field: ["a", "b", "c", "d"]}).model_dump()[field] == ["a", "b", "c", "d"]


@pytest.mark.parametrize("model,payload", [
    (HistoryEvidence, {"message_id": "id", "sequence": 1, "sender_type": "me", "text": "hello"}),
    (ProfileCandidates, {}), (ConversationStateCandidates, {}),
    (RetrievalContext, {"evidence": [], "relevant_message_ids": [], "archive_message_count": 0}),
])
def test_retrieval_models_forbid_extra_fields(model: type[BaseModel], payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        model.model_validate({**payload, "inferred_value": "unrequested"})


@pytest.mark.parametrize("section,field", REFERENCE_FIELDS)
@pytest.mark.parametrize("reference,reason", [
    ("missing", "references must identify supplied evidence"),
    ("private-service", "service evidence cannot be referenced"),
])
def test_every_reference_must_exist_and_cannot_point_to_service_evidence(
    section: str, field: str, reference: str, reason: str,
) -> None:
    payload = _retrieval().model_dump()
    if section:
        payload[section] = {field: [reference]}
    else:
        payload[field] = [reference]
    with pytest.raises(ValidationError, match=reason):
        RetrievalContext.model_validate(payload)


@pytest.mark.parametrize("section,reference", [("agent", "private-source-other"), ("interlocutor", "private-source-me")])
@pytest.mark.parametrize("field", PROFILE_FIELDS)
def test_every_profile_reference_must_match_its_speaker(section: str, reference: str, field: str) -> None:
    with pytest.raises(ValidationError, match="profile reference speaker mismatch"):
        RetrievalContext.model_validate({**_retrieval().model_dump(), section: {field: [reference]}})


def test_category_reuse_does_not_derive_values_or_claim_semantic_validation() -> None:
    source = HistoryEvidence(message_id="question", sequence=1, sender_type="me", text="What is your age?")
    context = RetrievalContext(
        evidence=[source], agent=ProfileCandidates(age=[source.message_id], occupation=[source.message_id]),
        conversation_state=ConversationStateCandidates(questions=[source.message_id]),
        relevant_message_ids=[source.message_id], archive_message_count=1,
    )
    # This contract checks links/speakers, NOT whether a question declares age.
    assert context.agent.age == context.agent.occupation == [source.message_id]
    assert context.interlocutor == ProfileCandidates()
    assert context.prompt_view()["candidate_status"] == "UNVERIFIED categorization"


@pytest.mark.parametrize("evidence,reason", [
    ([_evidence("same", 1), _evidence("same", 2)], "message IDs must be unique"),
    ([_evidence("a", 1), _evidence("b", 1)], "sequences must be unique"),
    ([_evidence("a", 2), _evidence("b", 1)], "chronological by sequence"),
])
def test_evidence_is_unique_and_chronological(evidence: list[HistoryEvidence], reason: str) -> None:
    with pytest.raises(ValidationError, match=reason):
        RetrievalContext(evidence=evidence, relevant_message_ids=[], archive_message_count=2)


def test_evidence_and_relevant_list_limits_allow_gaps_and_unreferenced_service_events() -> None:
    evidence = [_evidence(str(index), index * 2 + 1) for index in range(32)]
    valid = RetrievalContext(evidence=evidence, relevant_message_ids=[item.message_id for item in evidence[:12]], archive_message_count=64)
    assert len(valid.evidence) == 32
    with pytest.raises(ValidationError):
        RetrievalContext.model_validate({**valid.model_dump(), "evidence": evidence + [_evidence("33", 99)]})
    with pytest.raises(ValidationError):
        RetrievalContext.model_validate({**valid.model_dump(), "relevant_message_ids": [item.message_id for item in evidence[:13]]})
    for bad in (["0", "0"], [""], [" \n"], [1], None):
        with pytest.raises(ValidationError):
            RetrievalContext.model_validate({**valid.model_dump(), "relevant_message_ids": bad})
    assert _retrieval().evidence[-1].sender_type == "system"


@pytest.mark.parametrize("count", [-1, 1.5, None])
def test_archive_count_is_nonnegative_integer(count: object) -> None:
    with pytest.raises(ValidationError):
        RetrievalContext.model_validate({**_retrieval().model_dump(), "archive_message_count": count})


@pytest.mark.parametrize("fill", ["x", "🙂", '"', "\\", "\n"])
def test_exact_serialized_utf8_byte_boundary_including_json_escaping(fill: str) -> None:
    context = RetrievalContext(
        evidence=[_evidence(str(index), index + 1) for index in range(12)],
        relevant_message_ids=[], archive_message_count=12,
    )
    remaining = MAX_RETRIEVAL_BYTES - len(context.model_dump_json().encode("utf-8"))
    width = len(json.dumps(fill, ensure_ascii=False).encode("utf-8")) - 2
    for item in context.evidence:
        count = min(1200 - len(item.text), remaining // width)
        item.text += fill * count
        remaining -= count * width
    for item in context.evidence:
        count = min(1200 - len(item.text), remaining)
        item.text += "x" * count
        remaining -= count
    assert remaining == 0
    assert len(context.model_dump_json().encode("utf-8")) == MAX_RETRIEVAL_BYTES == 12_000
    valid = RetrievalContext.model_validate_json(context.model_dump_json())
    assert len(json.dumps(valid.prompt_view(), ensure_ascii=False, separators=(",", ":")).encode("utf-8")) <= MAX_RETRIEVAL_BYTES
    next(item for item in context.evidence if len(item.text) < 1200).text += "x"
    with pytest.raises(ValidationError, match="12000 UTF-8 bytes"):
        RetrievalContext.model_validate_json(context.model_dump_json())


@pytest.mark.parametrize("size", [1, 9, 10, 32])
def test_prompt_projection_remains_bounded_with_short_ids_and_dense_references(size: int) -> None:
    evidence = [_evidence(chr(65 + index), index + 1, "me" if index % 2 == 0 else "other") for index in range(size)]
    context = RetrievalContext(
        evidence=evidence,
        agent=ProfileCandidates.model_validate({field: [item.message_id for item in evidence if item.sender_type == "me"][:4] for field in PROFILE_FIELDS}),
        interlocutor=ProfileCandidates.model_validate({field: [item.message_id for item in evidence if item.sender_type == "other"][:4] for field in PROFILE_FIELDS}),
        conversation_state=ConversationStateCandidates.model_validate({field: [item.message_id for item in evidence[:4]] for field in STATE_FIELDS}),
        relevant_message_ids=[item.message_id for item in evidence[:12]], archive_message_count=size,
    )
    # Timestamp has no per-field cap; the whole UTF-8 budget still bounds it.
    context.evidence[0].timestamp = ""
    context.evidence[0].timestamp = "t" * (MAX_RETRIEVAL_BYTES - len(context.model_dump_json().encode("utf-8")))
    valid = RetrievalContext.model_validate_json(context.model_dump_json())
    assert len(valid.model_dump_json().encode("utf-8")) == MAX_RETRIEVAL_BYTES
    assert len(json.dumps(valid.prompt_view(), ensure_ascii=False, separators=(",", ":")).encode("utf-8")) <= MAX_RETRIEVAL_BYTES


@pytest.mark.parametrize("sender_type", ["me", "other", "system"])
def test_recent_message_ids_must_not_overlap_evidence(sender_type: SenderType) -> None:
    with pytest.raises(ValidationError, match="overlaps recent message IDs"):
        ChatSnapshot(
            chat=ChatDescriptor(raw_id="chat", title="Synthetic"), retrieval_context=_retrieval(),
            messages=[Message(raw_id="private-source-me", sender_type=sender_type, text="Different edited text.")],
        )


def test_legacy_and_retrieval_are_mutually_exclusive_even_when_retrieval_is_empty() -> None:
    with pytest.raises(ValidationError, match="mutually exclusive"):
        ChatSnapshot(
            chat=ChatDescriptor(raw_id="chat", title="Synthetic"), context=_checkpoint(),
            retrieval_context=RetrievalContext(evidence=[], relevant_message_ids=[], archive_message_count=0),
        )


def test_old_snapshot_fields_and_optional_legacy_serialization_remain_compatible() -> None:
    payload = {
        "chat": {"raw_id": "chat", "title": "Synthetic", "has_unread": True},
        "messages": [{"sender_type": "other", "text": "hello", "timestamp": None, "raw_id": "recent"}],
        "platform": "Test platform", "account_name": "Test account",
    }
    snapshot = ChatSnapshot.model_validate(payload)
    assert snapshot.context is snapshot.retrieval_context is None
    assert snapshot.model_dump(exclude_unset=True) == payload
    with_legacy = {**payload, "context": _checkpoint().model_dump(mode="json")}
    restored = ChatSnapshot.model_validate_json(json.dumps(with_legacy))
    assert restored.model_dump(exclude={"retrieval_context"}) == with_legacy
    assert restored.context == _checkpoint()
    assert ChatSnapshot.model_validate({**payload, "context": None, "retrieval_context": None}).context is None
    with_retrieval = ChatSnapshot.model_validate({**payload, "retrieval_context": _retrieval().model_dump()})
    assert with_retrieval.messages == snapshot.messages


@pytest.fixture
def reader_schema(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Load the standalone mirror without importing reader runtime modules."""
    path = Path(__file__).resolve().parents[2] / "responser_web_reader/src/responser_web_reader/schemas.py"
    if not path.is_file():
        pytest.skip("reader mirror is absent from this standalone checkout")
    spec = importlib.util.spec_from_file_location("_retrieval_reader_schema_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", [
    "HistoryEvidence", "ProfileCandidates", "ConversationStateCandidates", "RetrievalContext",
    "ChatSnapshot", "GenerateReplyRequest",
])
def test_new_schema_parity_entries(reader_schema: ModuleType, name: str) -> None:
    api_model = cast(type[BaseModel], getattr(schemas, name))
    reader_model = cast(type[BaseModel], getattr(reader_schema, name))
    assert api_model.model_json_schema() == reader_model.model_json_schema()


def test_source_wire_projection_and_constant_parity(reader_schema: ModuleType) -> None:
    assert schemas.__file__ is not None and reader_schema.__file__ is not None
    api_source = Path(schemas.__file__).read_text(encoding="utf-8")
    mirror_source = Path(reader_schema.__file__).read_text(encoding="utf-8")
    assert api_source.split("from __future__", 1)[1] == mirror_source.split("from __future__", 1)[1]
    context = _retrieval()
    mirror_context = reader_schema.RetrievalContext.model_validate_json(context.model_dump_json())
    assert mirror_context.model_dump_json() == context.model_dump_json()
    assert mirror_context.prompt_view() == context.prompt_view()
    assert reader_schema.MAX_RETRIEVAL_BYTES == MAX_RETRIEVAL_BYTES
    snapshot = ChatSnapshot(chat=ChatDescriptor(raw_id="chat", title="Synthetic"), retrieval_context=context)
    assert reader_schema.ChatSnapshot.model_validate_json(snapshot.model_dump_json()).model_dump_json() == snapshot.model_dump_json()


def test_openapi_advertises_optional_retrieval_and_health_stays_exact() -> None:
    client = TestClient(app_module.app)
    contract = client.get("/openapi.json").json()
    assert contract["info"]["version"] == "0.4.0"
    with (Path(__file__).resolve().parents[1] / "pyproject.toml").open("rb") as project_file:
        assert tomllib.load(project_file)["project"]["version"] == "0.4.0"
    models = contract["components"]["schemas"]
    snapshot_schema = models["ChatSnapshot"]
    assert "retrieval_context" not in snapshot_schema["required"]
    assert {"$ref": "#/components/schemas/RetrievalContext"} in snapshot_schema["properties"]["retrieval_context"]["anyOf"]
    assert {"type": "null"} in snapshot_schema["properties"]["retrieval_context"]["anyOf"]
    assert {"/generate_reply", "/summarize_context"} <= contract["paths"].keys()
    assert "ConversationContext" in models and "SummarizeContextResponse" in models
    assert models["RetrievalContext"]["additionalProperties"] is False
    assert models["RetrievalContext"]["properties"]["evidence"]["maxItems"] == 32
    assert models["ProfileCandidates"]["properties"]["name"]["maxItems"] == 4
    assert models["RetrievalContext"]["properties"]["relevant_message_ids"]["maxItems"] == 12
    assert client.get("/health").json() == {"status": "ok", "personality": app_module._personality.name}