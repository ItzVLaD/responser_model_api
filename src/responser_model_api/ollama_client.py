"""Thin wrapper around the Ollama chat API."""

from __future__ import annotations

import time

from ollama import Client

from .config import OLLAMA_HOST, RESPONSE_FORMAT_INSTRUCTIONS, GenerationSettings
from .logging_config import LOG_PROMPTS, get_logger
from .personality import Personality
from .schemas import ChatSnapshot, GeneratedReply

log = get_logger()

# Boilerplate that means the model's built-in alignment kicked in and it broke
# character with an AI/assistant-style refusal instead of replying as the person.
# Matching any of these triggers a regeneration (and, failing that, a fallback).
_REFUSAL_MARKERS: tuple[str, ...] = (
    "i cannot",
    "i can't fulfill",
    "i can't help with",
    "i can not",
    "i'm not able to",
    "i am not able to",
    "i'm unable to",
    "as an ai",
    "as a language model",
    "i'm just an ai",
    "is there anything else i can help you with",
    "i can't create content",
    "i cannot create content",
    "i can't write a response",
    "i cannot write a response",
    "i can't engage with",
)

# Neutral, in-character-ish deflection used only if the model keeps refusing.
# Deliberately vague so it fits most personas without generating any real content.
_DEFLECTION_FALLBACK = "haha nah, let's not go there. anyway, what else is up?"


def _looks_like_refusal(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _REFUSAL_MARKERS)


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
        log.info(
            "generate: persona=%s platform=%s chat=%r messages=%d model=%s temp=%.2f",
            self._personality.name,
            snapshot.platform,
            snapshot.chat.title,
            len(snapshot.messages),
            self._settings.model_name,
            self._settings.temperature,
        )
        if LOG_PROMPTS:
            log.debug("prompt messages: %s", messages)

        started = time.monotonic()
        response = self._chat(messages)
        text = response["message"]["content"].strip()
        if LOG_PROMPTS:
            log.debug("raw completion: %r", text)

        # If the model's alignment produced an out-of-character AI refusal, nudge
        # it once to answer as the person; if it still refuses, deflect in a
        # neutral, human way rather than shipping assistant boilerplate.
        if _looks_like_refusal(text):
            log.warning("refusal detected; regenerating in character")
            nudge = messages + [
                {
                    "role": "system",
                    "content": (
                        "That last reply was an out-of-character AI refusal. "
                        f"Reply again as {self._personality.name}, a real person. "
                        "Do not use any AI disclaimers. If you would rather not "
                        "engage, just brush it off, joke, or change the subject "
                        "like a human would."
                    ),
                }
            ]
            response = self._chat(nudge)
            retry_text = response["message"]["content"].strip()
            if _looks_like_refusal(retry_text):
                log.warning("retry still refused; using deflection fallback")
                text = _DEFLECTION_FALLBACK
            else:
                text = retry_text

        elapsed_ms = (time.monotonic() - started) * 1000
        log.info(
            "generated in %.0fms: prompt_tokens=%s completion_tokens=%s reply=%r",
            elapsed_ms,
            response.get("prompt_eval_count"),
            response.get("eval_count"),
            text,
        )

        return GeneratedReply(
            text=text,
            model_name=self._settings.model_name,
            prompt_tokens=response.get("prompt_eval_count"),
            completion_tokens=response.get("eval_count"),
            finish_reason=response.get("done_reason"),
        )

    def _chat(self, messages: list[dict[str, str]]):
        settings = self._settings
        return self._client.chat(
            model=settings.model_name,
            messages=messages,
            options={
                "temperature": settings.temperature,
                "top_p": settings.top_p,
                "num_predict": settings.max_output_tokens,
            },
        )
