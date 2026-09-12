"""Inter-module contract shared with the web reader.

These Pydantic models define the request/response shapes of the model API.
FastAPI serves them as an OpenAPI document at /openapi.json, which is the
source-of-truth contract. The web reader mirrors these shapes on its side.
"""

from __future__ import annotations

from typing import Annotated, Literal, Optional, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

SenderType = Literal["me", "other", "system"]
# Ollama's structured-output grammar requires anchored patterns. Match a
# non-whitespace character anywhere, including strings containing newlines.
NonEmptyString = Annotated[str, StringConstraints(min_length=1, pattern=r"^[\s\S]*\S[\s\S]*$")]
MemoryItem = Annotated[NonEmptyString, Field(max_length=200)]
MAX_MEMORY_CHARACTERS = 6_000
MAX_SUMMARY_INPUT_BYTES = 12_000


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

    stage: Literal["unknown", "new", "acquaintance", "familiar", "strained"] = "unknown"
    evidence: str = Field(default="", max_length=300)


class MemoryContent(BaseModel):
    """Bounded conversation evidence; never trusted instructions or metadata."""

    model_config = ConfigDict(extra="forbid")

    interlocutor: list[MemoryItem] = Field(default_factory=list, max_length=8)
    agent: list[MemoryItem] = Field(default_factory=list, max_length=8)
    interaction: list[MemoryItem] = Field(default_factory=list, max_length=8)
    open_threads: list[MemoryItem] = Field(default_factory=list, max_length=8)
    relationship: RelationshipState = Field(default_factory=RelationshipState)

    @model_validator(mode="after")
    def check_serialized_size(self) -> Self:
        """Include JSON escaping and field overhead in the persisted size cap."""
        if len(self.model_dump_json()) > MAX_MEMORY_CHARACTERS:
            raise ValueError("serialized memory exceeds 6000 characters")
        return self


class ConversationContext(BaseModel):
    """Reader-owned checkpoint metadata wrapped around model-produced memory."""

    model_config = ConfigDict(extra="forbid")

    memory: MemoryContent
    last_message_id: NonEmptyString
    summarized_message_count: int = Field(ge=1)
    model_name: NonEmptyString
    updated_at: NonEmptyString


class SummarizeContextRequest(BaseModel):
    """Chronological batch plus previous memory, with a UTF-8 input budget."""

    model_config = ConfigDict(extra="forbid")

    previous: MemoryContent | None = None
    messages: list[Message] = Field(min_length=1, max_length=30)

    @model_validator(mode="after")
    def check_serialized_size(self) -> Self:
        """Bound the entire canonical request, including message metadata."""
        if len(self.model_dump_json().encode("utf-8")) > MAX_SUMMARY_INPUT_BYTES:
            raise ValueError("serialized summary input exceeds 12000 UTF-8 bytes")
        return self


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
