"""Thin wrapper around the Ollama chat API."""

from __future__ import annotations

from ollama import Client

from .config import OLLAMA_HOST, SYSTEM_PROMPT
from .schemas import ChatSnapshot, GeneratedReply, GenerationConfig


def _snapshot_to_messages(snapshot: ChatSnapshot) -> list[dict[str, str]]:
    """Map a ChatSnapshot into an Ollama/OpenAI-style messages array."""
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    for msg in snapshot.messages:
        if msg.sender_type == "other":
            role = "user"
        elif msg.sender_type == "me":
            role = "assistant"
        else:
            role = "system"
        messages.append({"role": role, "content": msg.text})

    messages.append({"role": "user", "content": "Write my next reply to this conversation."})
    return messages


class OllamaReplyGenerator:
    def __init__(self, host: str = OLLAMA_HOST) -> None:
        self._client = Client(host=host)

    def generate(self, snapshot: ChatSnapshot, config: GenerationConfig) -> GeneratedReply:
        messages = _snapshot_to_messages(snapshot)
        response = self._client.chat(
            model=config.model_name,
            messages=messages,
            options={
                "temperature": config.temperature,
                "top_p": config.top_p,
                "num_predict": config.max_output_tokens,
            },
        )

        return GeneratedReply(
            text=response["message"]["content"].strip(),
            model_name=config.model_name,
            prompt_tokens=response.get("prompt_eval_count"),
            completion_tokens=response.get("eval_count"),
            finish_reason=response.get("done_reason"),
        )
