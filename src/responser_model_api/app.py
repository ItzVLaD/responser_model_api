"""FastAPI application exposing independent reply and context-summary endpoints.

Run locally with:
    uvicorn responser_model_api.app:app --reload --port 8000
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from .config import load_active_personality, load_generation_settings, load_summary_settings
from .context_summarizer import ContextSummaryError, OllamaContextSummarizer
from .logging_config import configure_logging
from .ollama_client import OllamaReplyGenerator
from .schemas import (
    GenerateReplyRequest,
    GeneratedReply,
    SummarizeContextRequest,
    SummarizeContextResponse,
)

log = configure_logging()

app = FastAPI(
    title="Responser Model API",
    version="0.3.0",
    description="Generates replies and structured conversation memory using Ollama.",
)

# The personality and generation settings are selected once at startup (via env)
# and fixed for the life of the process; they are never chosen per request.
_personality = load_active_personality()
_settings = load_generation_settings()
_generator = OllamaReplyGenerator(personality=_personality, settings=_settings)
_summary_settings = load_summary_settings()
_summarizer = OllamaContextSummarizer(
    settings=_summary_settings, reply_model_name=_settings.model_name,
)
log.info(
    "model API ready: personality=%s model=%s temperature=%.2f context_model=%s",
    _personality.name,
    _settings.model_name,
    _settings.temperature,
    _summary_settings.model_name,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "personality": _personality.name}


@app.post("/generate_reply", response_model=GeneratedReply)
def generate_reply(request: GenerateReplyRequest) -> GeneratedReply:
    try:
        reply = _generator.generate(request.snapshot)
    except Exception as exc:  # surface Ollama/model errors as 502
        log.exception("generation failed")
        raise HTTPException(status_code=502, detail=f"generation failed: {exc}") from exc

    return reply


@app.post("/summarize_context", response_model=SummarizeContextResponse)
def summarize_context(request: SummarizeContextRequest) -> SummarizeContextResponse:
    """Fail closed: never substitute a reply or old memory for a failed summary."""
    try:
        return _summarizer.summarize(request)
    except ContextSummaryError as exc:
        log.error(
            "context summary failed: error_type=%s reason=%s summary_id=%s",
            type(exc).__name__, exc.reason, exc.summary_id,
        )
        raise HTTPException(
            status_code=502,
            detail=f"context summary failed: {exc.reason}; no memory/checkpoint update was accepted",
        ) from None
    except Exception as exc:  # Ollama transport, missing model, or invalid output
        # Exception text/tracebacks can include chat data or model JSON. Keep
        # both the logs and public error free of that sensitive content.
        log.error("context summary failed: error_type=%s", type(exc).__name__)
        raise HTTPException(
            status_code=502,
            detail="context summary failed; verify the local summary model and its budgets",
        ) from None
