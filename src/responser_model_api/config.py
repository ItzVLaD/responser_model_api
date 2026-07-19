"""Runtime configuration for the model API."""

from __future__ import annotations

import os

from .schemas import GenerationConfig

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("RESPONSER_MODEL", "llama3.2:3b")

SYSTEM_PROMPT = (
    "You are helping me reply to messages in a chat. "
    "Write a single concise, polite reply in my voice. "
    "Do not include quotation marks or explanations, only the reply text."
)


def default_generation_config() -> GenerationConfig:
    return GenerationConfig(model_name=DEFAULT_MODEL)
