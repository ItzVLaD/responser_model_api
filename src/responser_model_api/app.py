"""FastAPI application exposing the reply-generation endpoint.

Run locally with:
    uvicorn responser_model_api.app:app --reload --port 8000
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from .config import load_active_personality, load_generation_settings
from .ollama_client import OllamaReplyGenerator
from .schemas import GenerateReplyRequest, GeneratedReply

app = FastAPI(
    title="Responser Model API",
    version="0.1.0",
    description="Turns chat snapshots into replies using an Ollama-hosted LLM.",
)

# The personality and generation settings are selected once at startup (via env)
# and fixed for the life of the process; they are never chosen per request.
_personality = load_active_personality()
_settings = load_generation_settings()
_generator = OllamaReplyGenerator(personality=_personality, settings=_settings)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "personality": _personality.name}


@app.post("/generate_reply", response_model=GeneratedReply)
def generate_reply(request: GenerateReplyRequest) -> GeneratedReply:
    try:
        reply = _generator.generate(request.snapshot)
    except Exception as exc:  # surface Ollama/model errors as 502
        raise HTTPException(status_code=502, detail=f"generation failed: {exc}") from exc

    return reply
