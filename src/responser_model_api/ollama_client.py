"""Thin wrapper around the Ollama chat API."""

from __future__ import annotations

from ollama import Client

from .config import OLLAMA_HOST, RESPONSE_FORMAT_INSTRUCTIONS, GenerationSettings
from .personality import Personality
from .schemas import ChatSnapshot, GeneratedReply


def _system_prompt(personality: Personality) -> str:
    """Combine the fixed format contract with the selected persona."""
    return f"{RESPONSE_FORMAT_INSTRUCTIONS}\n\n{personality.to_system_prompt()}"


def _context_note(snapshot: ChatSnapshot) -> str | None:
    """Describe where the conversation happens, if the reader provided it."""
    parts: list[str] = []
    if snapshot.platform:
        parts.append(f"You are chatting on {snapshot.platform}.")
    if snapshot.account_name:
        parts.append(f"Your name on this platform is {snapshot.account_name}.")
    if snapshot.chat.title:
        parts.append(f"This conversation is with {snapshot.chat.title}.")
    return " ".join(parts) if parts else None


def _snapshot_to_messages(
    snapshot: ChatSnapshot, personality: Personality
) -> list[dict[str, str]]:
    """Map a ChatSnapshot into an Ollama/OpenAI-style messages array."""
    messages: list[dict[str, str]] = [
        {"role": "system", "content": _system_prompt(personality)}
    ]

    # Platform / account context, so the model knows where it is and who it is.
    context = _context_note(snapshot)
    if context:
        messages.append({"role": "system", "content": context})

    # Few-shot examples from the persona, shown as prior user/assistant turns so
    # the model learns the desired style before seeing the real conversation.
    for example in personality.examples:
        messages.append({"role": "user", "content": example.incoming})
        messages.append({"role": "assistant", "content": example.reply})

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
    def __init__(
        self,
        personality: Personality,
        settings: GenerationSettings,
        host: str = OLLAMA_HOST,
    ) -> None:
        self._personality = personality
        self._settings = settings
        self._client = Client(host=host)

    def generate(self, snapshot: ChatSnapshot) -> GeneratedReply:
        messages = _snapshot_to_messages(snapshot, self._personality)
        settings = self._settings
        response = self._client.chat(
            model=settings.model_name,
            messages=messages,
            options={
                "temperature": settings.temperature,
                "top_p": settings.top_p,
                "num_predict": settings.max_output_tokens,
            },
        )

        return GeneratedReply(
            text=response["message"]["content"].strip(),
            model_name=settings.model_name,
            prompt_tokens=response.get("prompt_eval_count"),
            completion_tokens=response.get("eval_count"),
            finish_reason=response.get("done_reason"),
        )
