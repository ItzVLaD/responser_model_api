"""Runtime configuration for the model API."""

from __future__ import annotations

import os
from pathlib import Path

from .personality import Personality, load_personality
from .schemas import GenerationConfig

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("RESPONSER_MODEL", "llama3.2:3b")

# Fixed output-contract instructions. These live in code (not the personality
# file) so a persona edit can never break the reply format the web reader relies
# on. The selected personality is layered on top of these.
RESPONSE_FORMAT_INSTRUCTIONS = (
    "You are replying to chat messages on my behalf. "
    "Write a single reply in my voice. "
    "Do not include quotation marks, labels, or explanations - output only the "
    "reply text itself."
)

# Directory holding personality YAML files, and which one to use. Both are read
# once at startup; the personality is NOT selectable per request.
PERSONALITIES_DIR = Path(
    os.environ.get(
        "RESPONSER_PERSONALITIES_DIR",
        str(Path(__file__).resolve().parents[2] / "personalities"),
    )
)
PERSONALITY_NAME = os.environ.get("RESPONSER_PERSONALITY", "friendly")


def default_generation_config() -> GenerationConfig:
    return GenerationConfig(model_name=DEFAULT_MODEL)


def load_active_personality() -> Personality:
    """Load the personality selected for this process (via env)."""
    return load_personality(PERSONALITY_NAME, PERSONALITIES_DIR)
