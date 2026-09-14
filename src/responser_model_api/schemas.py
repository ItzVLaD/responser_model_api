"""Inter-module contract shared with the web reader.

These Pydantic models define the request/response shapes of the model API.
FastAPI serves them as an OpenAPI document at /openapi.json, which is the
source-of-truth contract. The web reader mirrors these shapes on its side.
"""

from __future__ import annotations

import json
from typing import Annotated, Literal, Optional, Self

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, JsonValue, StringConstraints, model_validator

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
MAX_RETRIEVAL_BYTES = 12_000


def _unique_message_references(references: list[str]) -> list[str]:
    """Reject duplicates within a category, not reuse across categories."""
    if len(set(references)) != len(references):
        raise ValueError("message references must be unique within each list")
    return references


MessageReferences = Annotated[list[NonEmptyString], AfterValidator(_unique_message_references)]


class HistoryEvidence(BaseModel):
    """An attributed archive excerpt, not a verified fact or an instruction."""

    model_config = ConfigDict(extra="forbid")

    message_id: NonEmptyString = Field(max_length=128)
    sequence: int = Field(ge=1)
    sender_type: SenderType
    text: str = Field(min_length=1, max_length=1200)
    timestamp: Optional[str] = None
    truncated: bool = False


class ProfileCandidates(BaseModel):
    """UNVERIFIED categorization of message IDs, never extracted profile values."""

    model_config = ConfigDict(extra="forbid")

    name: MessageReferences = Field(default_factory=list, max_length=4)
    age: MessageReferences = Field(default_factory=list, max_length=4)
    occupation: MessageReferences = Field(default_factory=list, max_length=4)
    specialty: MessageReferences = Field(default_factory=list, max_length=4)
    interests: MessageReferences = Field(default_factory=list, max_length=4)
    preferences: MessageReferences = Field(default_factory=list, max_length=4)
    boundaries: MessageReferences = Field(default_factory=list, max_length=4)
    background: MessageReferences = Field(default_factory=list, max_length=4)

    def reference_lists(self) -> dict[str, list[str]]:
        """Expose typed references without deriving or interpreting their values."""
        return {
            "name": self.name, "age": self.age, "occupation": self.occupation,
            "specialty": self.specialty, "interests": self.interests,
            "preferences": self.preferences, "boundaries": self.boundaries,
            "background": self.background,
        }


class ConversationStateCandidates(BaseModel):
    """UNVERIFIED references; questions/commitments are NOT guaranteed unresolved."""

    model_config = ConfigDict(extra="forbid")

    questions: MessageReferences = Field(default_factory=list, max_length=4)
    commitments: MessageReferences = Field(default_factory=list, max_length=4)
    boundaries: MessageReferences = Field(default_factory=list, max_length=4)
    reactions: MessageReferences = Field(default_factory=list, max_length=4)

    def reference_lists(self) -> dict[str, list[str]]:
        """Keep candidate labels separate from any claim about current state."""
        return {
            "questions": self.questions, "commitments": self.commitments,
            "boundaries": self.boundaries, "reactions": self.reactions,
        }


class RetrievalContext(BaseModel):
    """Bounded chronological evidence and unverified reference-only candidates."""

    model_config = ConfigDict(extra="forbid")

    evidence: list[HistoryEvidence] = Field(max_length=32)
    agent: ProfileCandidates = Field(default_factory=ProfileCandidates)
    interlocutor: ProfileCandidates = Field(default_factory=ProfileCandidates)
    conversation_state: ConversationStateCandidates = Field(default_factory=ConversationStateCandidates)
    relevant_message_ids: MessageReferences = Field(max_length=12)
    archive_message_count: int = Field(ge=0)
    history_complete: bool = True
    budget_exhausted: bool = False

    @model_validator(mode="after")
    def check_evidence_and_references(self) -> Self:
        """Validate structure and attribution only; do not infer semantic claims."""
        if len(self.model_dump_json().encode("utf-8")) > MAX_RETRIEVAL_BYTES:
            raise ValueError("serialized retrieval context exceeds 12000 UTF-8 bytes")
        by_id = {item.message_id: item for item in self.evidence}
        if len(by_id) != len(self.evidence):
            raise ValueError("retrieval evidence message IDs must be unique")
        sequences = [item.sequence for item in self.evidence]
        if len(set(sequences)) != len(sequences):
            raise ValueError("retrieval evidence sequences must be unique")
        if sequences != sorted(sequences):
            raise ValueError("retrieval evidence must be chronological by sequence")

        groups: list[tuple[dict[str, list[str]], SenderType | None]] = [
            (self.agent.reference_lists(), "me"),
            (self.interlocutor.reference_lists(), "other"),
            (self.conversation_state.reference_lists(), None),
            ({"relevant_message_ids": self.relevant_message_ids}, None),
        ]
        for fields, expected_speaker in groups:
            for references in fields.values():
                for message_id in references:
                    source = by_id.get(message_id)
                    if source is None:
                        raise ValueError("references must identify supplied evidence")
                    if source.sender_type == "system":
                        raise ValueError("service evidence cannot be referenced")
                    if expected_speaker is not None and source.sender_type != expected_speaker:
                        raise ValueError("profile reference speaker mismatch")
        return self

    def prompt_view(self) -> dict[str, JsonValue]:
        """Show text once, oldest first, with local aliases and no archive metadata.

        Removing source IDs, absolute sequences and storage counts keeps this
        projection bounded without dropping any evidence or filling unknowns.
        Reply instructions supply precedence and unresolved-state caveats.
        """
        aliases = {item.message_id: f"e{index}" for index, item in enumerate(self.evidence, 1)}
        evidence: list[JsonValue] = [
            {"alias": aliases[item.message_id], "speaker": item.sender_type,
             "text": item.text, "timestamp": item.timestamp, "truncated": item.truncated}
            for item in self.evidence
        ]
        result: dict[str, JsonValue] = {
            "candidate_status": "UNVERIFIED categorization",
            "limited": self.budget_exhausted,
            "complete": self.history_complete,
            "evidence": evidence,
        }
        for section, candidates in (
            ("agent", self.agent), ("interlocutor", self.interlocutor),
            ("conversation_state", self.conversation_state),
        ):
            fields: dict[str, JsonValue] = {}
            for field, references in candidates.reference_lists().items():
                fields[field] = [aliases[message_id] for message_id in references]
            result[section] = fields
        result["relevant"] = [aliases[message_id] for message_id in self.relevant_message_ids]
        return result


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
        """Budget the shared compact projection; API-only selection has its own cap."""
        if len(self.model_dump_json().encode("utf-8")) > MAX_SUMMARY_WIRE_BYTES:
            raise ValueError("serialized summary wire input exceeds 64000 UTF-8 bytes")
        if len(self.inference_payload().encode("utf-8")) > MAX_SUMMARY_INPUT_BYTES:
            raise ValueError("serialized summary input exceeds 12000 UTF-8 bytes")
        return self

    def inference_payload(self) -> str:
        """Return the shared compact budgeting projection, using absolute speaker labels."""
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
    retrieval_context: RetrievalContext | None = None

    @model_validator(mode="after")
    def check_context_sources(self) -> Self:
        """Never mix legacy memory with retrieval or repeat recent source IDs."""
        if self.retrieval_context is None:
            return self
        if self.context is not None:
            raise ValueError("context and retrieval_context are mutually exclusive")
        recent_ids = {message.raw_id for message in self.messages if message.raw_id is not None}
        if any(item.message_id in recent_ids for item in self.retrieval_context.evidence):
            raise ValueError("retrieval evidence overlaps recent message IDs")
        return self


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
