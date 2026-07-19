"""Inter-module contract shared with the web reader.

These Pydantic models define the request/response shapes of the model API.
FastAPI serves them as an OpenAPI document at /openapi.json, which is the
source-of-truth contract. The web reader mirrors these shapes on its side.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

SenderType = Literal["me", "other", "system"]


class Message(BaseModel):
    sender_type: SenderType
    text: str
    timestamp: Optional[str] = None
    raw_id: Optional[str] = None


class ChatDescriptor(BaseModel):
    raw_id: str
    title: str
    has_unread: bool = False


class ChatSnapshot(BaseModel):
    chat: ChatDescriptor
    messages: list[Message] = Field(default_factory=list)


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
