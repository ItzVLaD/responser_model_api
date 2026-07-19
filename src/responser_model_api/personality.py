"""Personality loading.

A *personality* describes how the model should behave when writing replies. It
lives in a YAML file inside the ``personalities/`` folder and is selected once at
API startup (never per request). The fixed output-format instructions stay in
code (see ``config.RESPONSE_FORMAT_INSTRUCTIONS``); the personality is layered on
top of them, so a persona edit can shape tone/behaviour but cannot break the
reply contract.

Example ``personalities/friendly.yaml``::

    name: Alex
    identity: a 28-year-old game developer from Dublin
    background: Works remotely, loves indie games and late-night coding.
    tone: warm and casual
    speech_style: short lowercase sentences, minimal punctuation
    language: English
    emoji_usage: often, especially 😄 and 🔥
    signature_phrases:
      - "haha"
      - "for real"
    interests:
      - video games
      - coffee
    avoid:
      - being formal or stiff
    rules:
      - Keep replies short and upbeat.
      - Never be rude, even to rude messages.
    examples:
      - incoming: "How are you?"
        reply: "doing great haha, you? 😄"
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class PersonalityExample(BaseModel):
    """A single few-shot example: an incoming message and the desired reply."""

    incoming: str
    reply: str


class Personality(BaseModel):
    """A behaviour profile for reply generation.

    The fields aim to capture *who* is replying richly enough to mimic a
    specific person's voice: their identity, background, how they speak, and the
    little habits that make their messages recognizable.
    """

    # Who the persona is.
    name: str
    # A short identity line, e.g. "Alex, a 28-year-old game developer from Dublin".
    identity: str = ""
    # Free-form background/biography the model can draw on for context.
    background: str = ""

    # How the persona communicates.
    tone: str = ""
    # Concrete description of speech style: sentence length, punctuation habits,
    # capitalization, slang, formality, etc.
    speech_style: str = ""
    # Language(s) the persona writes in (e.g. "English", "casual Irish English").
    language: str = ""
    # How the persona uses emojis (e.g. "rarely", "loves 😂 and 🔥").
    emoji_usage: str = ""
    # Recognizable catchphrases or filler words the persona often uses.
    signature_phrases: list[str] = Field(default_factory=list)
    # Topics/interests the persona talks about comfortably.
    interests: list[str] = Field(default_factory=list)
    # Things the persona avoids saying or doing.
    avoid: list[str] = Field(default_factory=list)

    # Explicit behaviour rules and few-shot examples.
    rules: list[str] = Field(default_factory=list)
    examples: list[PersonalityExample] = Field(default_factory=list)

    def to_system_prompt(self) -> str:
        """Render the persona as a human-readable system-prompt fragment."""
        parts: list[str] = []

        # Identity block: establish who is speaking.
        identity_line = f"You are {self.name}"
        if self.identity:
            identity_line += f", {self.identity}"
        identity_line += "."
        parts.append(identity_line)

        if self.background:
            parts.append(f"Background: {self.background}")

        # Voice block: how this person writes. Framed as *tendencies* so the
        # model adapts them to context instead of applying them mechanically.
        voice: list[str] = []
        if self.tone:
            voice.append(f"Tone: {self.tone}.")
        if self.speech_style:
            voice.append(f"Speech style (a tendency, not a fixed rule): {self.speech_style}.")
        if self.language:
            voice.append(f"Write in {self.language}.")
        if self.emoji_usage:
            voice.append(f"Emoji use (only when it fits): {self.emoji_usage}.")
        if voice:
            parts.append(" ".join(voice))

        if self.signature_phrases:
            phrases = ", ".join(f'"{p}"' for p in self.signature_phrases)
            parts.append(f"Occasionally use signature phrases like {phrases}.")

        if self.interests:
            parts.append(f"You are comfortable talking about: {', '.join(self.interests)}.")

        if self.avoid:
            avoid = "\n".join(f"- {item}" for item in self.avoid)
            parts.append(f"Avoid the following:\n{avoid}")

        if self.rules:
            rules = "\n".join(f"- {rule}" for rule in self.rules)
            parts.append(f"Follow these rules:\n{rules}")

        # Adaptability note: the persona describes who you are, not a rigid
        # template. Real people vary message to message depending on context.
        parts.append(
            "Adapt naturally to the flow of the conversation. The traits above "
            "describe your general character, not a strict formula: vary your "
            "message length, emoji use, and phrasing to fit what is actually "
            "being said. A short style does not mean every message is short, and "
            "liking emojis does not mean using them in every message - match the "
            "moment like a real person would."
        )

        return "\n\n".join(parts)


class PersonalityError(RuntimeError):
    """Raised when a personality cannot be found or parsed."""


def available_personalities(directory: Path) -> list[str]:
    """Return the names (file stems) of personalities in ``directory``."""
    if not directory.is_dir():
        return []
    return sorted(p.stem for p in directory.glob("*.yaml"))


def load_personality(name: str, directory: Path) -> Personality:
    """Load a personality by name (its YAML file stem) from ``directory``.

    Raises PersonalityError with an actionable message if the file is missing or
    malformed, so a bad startup configuration fails loudly.
    """
    path = directory / f"{name}.yaml"
    if not path.is_file():
        found = available_personalities(directory)
        raise PersonalityError(
            f"Personality {name!r} not found at {path}. "
            f"Available: {', '.join(found) if found else '(none)'}."
        )

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise PersonalityError(f"Personality {name!r} is not valid YAML: {exc}") from exc

    try:
        return Personality.model_validate(raw)
    except ValueError as exc:
        raise PersonalityError(f"Personality {name!r} has invalid fields: {exc}") from exc
