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


class GenerationConfig(BaseModel):
    model_name: str = "llama3.2:3b"
    temperature: float = 0.3
    max_output_tokens: int = 256
    top_p: float = 0.9


class GenerateReplyRequest(BaseModel):
    snapshot: ChatSnapshot
    config: Optional[GenerationConfig] = None
    dry_run: bool = False


class GeneratedReply(BaseModel):
    text: str
    model_name: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    finish_reason: Optional[str] = None
