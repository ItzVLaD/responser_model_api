"""Inter-module contract shared with the web reader.

These Pydantic models define the request/response shapes of the model API.
FastAPI serves them as an OpenAPI document at /openapi.json, which is the
source-of-truth contract. The web reader mirrors these shapes on its side.
"""

from __future__ import annotations

import json
from typing import Annotated, Literal, Optional, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints, model_validator

SenderType = Literal["me", "other", "system"]
FactSection = Literal["interlocutor", "agent", "interaction", "open_threads"]
FactKind = Literal[
    "age", "name", "occupation", "specialty", "preference", "boundary", "self_report",
    "observation", "story", "interest", "commitment", "question", "other",
]
RelationshipStage = Literal["unknown", "new", "acquaintance", "familiar", "strained"]
FACT_SECTIONS: tuple[FactSection, ...] = ("interlocutor", "agent", "interaction", "open_threads")
# Ollama's structured-output grammar requires anchored patterns. Match a
# non-whitespace character anywhere, including strings containing newlines.
NonEmptyString = Annotated[str, StringConstraints(min_length=1, pattern=r"^[\s\S]*\S[\s\S]*$")]
MemoryItem = Annotated[NonEmptyString, Field(max_length=200)]
MAX_MEMORY_CHARACTERS = 6_000
MAX_SUMMARY_INPUT_BYTES = 12_000
MAX_PERSISTED_MEMORY_BYTES = 48_000
MAX_SUMMARY_WIRE_BYTES = 64_000
MAX_MEMORY_FACTS = 64


class FactEvidence(BaseModel):
    """Verbatim source proof, not a guarantee of truth or complete extraction."""

    model_config = ConfigDict(extra="forbid")

    message_id: NonEmptyString = Field(max_length=128)
    sender_type: Literal["me", "other"]
    quote: MemoryItem


class MemoryFact(BaseModel):
    """A protected quote, or an explicitly unverified legacy record."""

    model_config = ConfigDict(extra="forbid")

    id: NonEmptyString = Field(max_length=128)
    section: FactSection
    kind: FactKind
    text: MemoryItem
    evidence: FactEvidence | None = None

    @model_validator(mode="after")
    def check_quote_matches_text(self) -> Self:
        """Do not allow a source citation to legitimize a free paraphrase."""
        if self.evidence is not None and self.text != self.evidence.quote:
            raise ValueError("fact text must equal evidence.quote")
        return self


class Message(BaseModel):
    sender_type: SenderType
    text: str
    timestamp: Optional[str] = None
    raw_id: Optional[str] = None


class ChatDescriptor(BaseModel):
    raw_id: str
    title: str
    has_unread: bool = False


class RelationshipState(BaseModel):
    """Evidence-based rapport, not a count of messages exchanged."""

    model_config = ConfigDict(extra="forbid")

    stage: RelationshipStage = "unknown"
    evidence: str = Field(default="", max_length=300)


class MemoryContent(BaseModel):
    """Canonical facts with bounded legacy previews; all content is untrusted."""

    model_config = ConfigDict(extra="forbid")

    interlocutor: list[MemoryItem] = Field(
        default_factory=list, max_length=8,
        description="Exact stated ages, jobs/specialties, preferences and self-reports from 'other'; not topic labels.",
    )
    agent: list[MemoryItem] = Field(
        default_factory=list, max_length=8,
        description="Agent said: exact age, job and relevant stories from 'me', retained across updates; claims, not verified facts.",
    )
    interaction: list[MemoryItem] = Field(
        default_factory=list, max_length=8,
        description="Who reacted to what, their trigger and requested adjustment, including apologies; no unsupported diagnoses.",
    )
    open_threads: list[MemoryItem] = Field(
        default_factory=list, max_length=8,
        description="Only unresolved questions or commitments; move concrete answers into the appropriate profile.",
    )
    relationship: RelationshipState = Field(default_factory=RelationshipState)
    facts: list[MemoryFact] = Field(default_factory=list, max_length=MAX_MEMORY_FACTS)
    relationship_source: FactEvidence | None = None

    @model_validator(mode="after")
    def check_serialized_size(self) -> Self:
        """Keep the old preview cap separate from the complete UTF-8 storage cap."""
        previews = self.model_dump_json(exclude={"facts", "relationship_source"})
        if len(previews) > MAX_MEMORY_CHARACTERS:
            raise ValueError("serialized memory exceeds 6000 characters")
        if len(self.model_dump_json().encode("utf-8")) > MAX_PERSISTED_MEMORY_BYTES:
            raise ValueError("serialized memory exceeds 48000 UTF-8 bytes")
        if len({fact.id for fact in self.facts}) != len(self.facts):
            raise ValueError("duplicate fact IDs")
        if self.relationship_source is not None and self.relationship_source.quote != self.relationship.evidence:
            raise ValueError("relationship evidence must equal relationship_source.quote")
        return self

    def prompt_view(self) -> dict[str, JsonValue]:
        """Expose every fact as an attributed quote, without IDs or source proofs."""
        if not self.facts:
            return self.model_dump(mode="json", exclude={"facts", "relationship_source"})
        result: dict[str, JsonValue] = {}
        for section in FACT_SECTIONS:
            records: list[JsonValue] = []
            for fact in self.facts:
                if fact.section != section:
                    continue
                speaker = fact.evidence.sender_type if fact.evidence is not None else {
                    "interlocutor": "other", "agent": "me",
                }.get(section, "unattributed")
                record_type = fact.kind if fact.evidence is not None else f"{fact.kind}; legacy-unverified"
                records.append(f"{speaker} [{record_type}]: {json.dumps(fact.text, ensure_ascii=False)}")
            result[section] = records
        result["relationship"] = self.relationship.model_dump(mode="json")
        return result

    def summary_view(self) -> dict[str, JsonValue]:
        """Compact target registry; callers may migrate legacy memory beforehand."""
        if not self.facts:
            return self.model_dump(mode="json", exclude={"facts", "relationship_source"})
        records: list[JsonValue] = [
            {"id": fact.id, "section": fact.section, "kind": fact.kind, "text": fact.text}
            for fact in self.facts
        ]
        return {"facts": records, "relationship": self.relationship.model_dump(mode="json")}


class ConversationContext(BaseModel):
    """Reader-owned checkpoint metadata wrapped around model-produced memory."""

    model_config = ConfigDict(extra="forbid")

    memory: MemoryContent
    last_message_id: NonEmptyString
    summarized_message_count: int = Field(ge=1)
    model_name: NonEmptyString
    updated_at: NonEmptyString


class SummarizeContextRequest(BaseModel):
    """Chronological batch with separate wire and compact inference budgets."""

    model_config = ConfigDict(extra="forbid")

    previous: MemoryContent | None = None
    messages: list[Message] = Field(min_length=1, max_length=30)

    @model_validator(mode="after")
    def check_serialized_size(self) -> Self:
        """Budget the exact projection passed to inference, not duplicated proofs."""
        if len(self.model_dump_json().encode("utf-8")) > MAX_SUMMARY_WIRE_BYTES:
            raise ValueError("serialized summary wire input exceeds 64000 UTF-8 bytes")
        if len(self.inference_payload().encode("utf-8")) > MAX_SUMMARY_INPUT_BYTES:
            raise ValueError("serialized summary input exceeds 12000 UTF-8 bytes")
        return self

    def inference_payload(self) -> str:
        """Return the exact compact JSON input, using absolute speaker labels."""
        speakers = {"me": "AGENT", "other": "INTERLOCUTOR", "system": "SERVICE_EVENT"}
        return json.dumps({
            "previous": self.previous.summary_view() if self.previous is not None else None,
            "messages": [
                {"speaker": speakers[message.sender_type], "text": message.text,
                 "message_id": message.raw_id, "timestamp": message.timestamp}
                for message in self.messages
            ],
        }, ensure_ascii=False, separators=(",", ":"))


class SummarizeContextResponse(BaseModel):
    """Validated memory and the model selected by the service, not the LLM."""

    model_config = ConfigDict(extra="forbid")

    memory: MemoryContent
    model_name: NonEmptyString


class ChatSnapshot(BaseModel):
    chat: ChatDescriptor
    messages: list[Message] = Field(default_factory=list)
    # Context about where the conversation happens, supplied by the reader.
    # `platform` is the human-readable platform name (e.g. "Telegram").
    # `account_name` is my display name / handle on that platform.
    platform: Optional[str] = None
    account_name: Optional[str] = None
    context: ConversationContext | None = None


class GenerateReplyRequest(BaseModel):
    snapshot: ChatSnapshot
    # `dry_run` lets a caller request a candidate reply without implying it will
    # be sent. Model inference parameters (model, temperature, ...) are NOT part
    # of this contract: they are an internal concern of the model API.
    dry_run: bool = False


class GeneratedReply(BaseModel):
    text: str
    model_name: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    finish_reason: Optional[str] = None
