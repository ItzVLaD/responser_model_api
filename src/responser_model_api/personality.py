"""Personality loading.

A *personality* describes how the model should behave when writing replies. It
lives in a YAML file inside the ``personalities/`` folder and is selected once at
API startup (never per request). The fixed output-format instructions stay in
code (see ``config.RESPONSE_FORMAT_INSTRUCTIONS``); the personality is layered on
top of them, so a persona edit can shape tone/behaviour but cannot break the
reply contract.

Example ``personalities/friendly.yaml``::

    name: Friendly
    tone: warm and casual
    rules:
      - Keep replies short and upbeat.
      - Never be rude, even to rude messages.
    examples:
      - incoming: "How are you?"
        reply: "Doing great, thanks! How about you?"
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
    """A behaviour profile for reply generation."""

    name: str
    tone: str = ""
    rules: list[str] = Field(default_factory=list)
    examples: list[PersonalityExample] = Field(default_factory=list)

    def to_system_prompt(self) -> str:
        """Render the persona as a human-readable system-prompt fragment."""
        parts: list[str] = [f"Your persona is '{self.name}'."]
        if self.tone:
            parts.append(f"Tone: {self.tone}.")
        if self.rules:
            rules = "\n".join(f"- {rule}" for rule in self.rules)
            parts.append(f"Follow these rules:\n{rules}")
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
