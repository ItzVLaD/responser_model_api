# responser_model_api

Local HTTP API that turns chat snapshots into replies using an Ollama-hosted LLM.
It owns the inter-module contract (Pydantic schemas + OpenAPI at `/openapi.json`).

## Setup

```bash
# 1. Install Ollama (https://ollama.com) and pull a small CPU-friendly model
ollama pull llama3.2:3b

# 2. Create a virtual environment and install the package
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run

```bash
uvicorn responser_model_api.app:app --reload --port 8000
```

- Health check: `GET http://localhost:8000/health`
- Generate reply: `POST http://localhost:8000/generate_reply`
- Contract / docs: `http://localhost:8000/docs` and `http://localhost:8000/openapi.json`

## Personalities

A *personality* describes how the model should behave (identity, background, tone,
speech style, quirks, and few-shot examples). It can mimic a specific person's
voice. Personalities are YAML files in `personalities/` and are selected once at
**startup** via `RESPONSER_PERSONALITY` — never per request. This keeps the API
contract stable while letting you swap behaviour.

```bash
# use the professional persona instead of the default "friendly"
RESPONSER_PERSONALITY=professional uvicorn responser_model_api.app:app --port 8000
```

Add a new persona by dropping a file in `personalities/`, e.g. `personalities/alex.yaml`:

```yaml
name: Alex
identity: a 28-year-old game developer from Kyiv
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
  - sounding formal or stiff
rules:
  - Keep replies short and upbeat.
  - Never claim to be a bot or AI; you are just Alex.
examples:
  - incoming: "How are you?"
    reply: "doing great haha, you? 😄"
```

All fields except `name` are optional. The fixed output-format instructions —
including "reply only with the reply text" and "always act human, never admit to
being an AI" — live in code (`config.RESPONSE_FORMAT_INSTRUCTIONS`), so a persona
edit can shape tone but cannot break the reply contract or the human-acting rule.

## Configuration (env vars)

| Variable                     | Default                  | Purpose                                  |
|------------------------------|--------------------------|------------------------------------------|
| `OLLAMA_HOST`                | `http://localhost:11434` | Where Ollama is listening                |
| `RESPONSER_MODEL`            | `llama3.2:3b`            | Default model name                       |
| `RESPONSER_TEMPERATURE`      | `0.7`                    | Sampling temperature (higher = more varied) |
| `RESPONSER_TOP_P`            | `0.9`                    | Nucleus sampling cutoff                   |
| `RESPONSER_MAX_OUTPUT_TOKENS`| `256`                    | Max tokens per reply                      |
| `RESPONSER_PERSONALITY`      | `friendly`               | Which `personalities/<name>.yaml` to use |
| `RESPONSER_PERSONALITIES_DIR`| `./personalities`        | Folder holding personality YAML files    |
| `RESPONSER_LOG_LEVEL`        | `INFO`                   | Log level (DEBUG, INFO, WARNING, ...)     |
| `RESPONSER_LOG_FILE`         | *(unset)*                | Also write logs to this file              |
| `RESPONSER_LOG_PROMPTS`      | `false`                  | DEBUG-log full prompts + raw output       |

> Model inference parameters (model, temperature, top-p, max tokens) are an
> internal concern of this service and are **not** part of the HTTP contract; the
> web reader never sends them.

## Logging & debugging

The service logs to stderr (and optionally a file). This is the main tool for
debugging and tuning how the model responds:

```bash
# see full prompts + raw completions while iterating on personas/params
RESPONSER_LOG_LEVEL=DEBUG RESPONSER_LOG_PROMPTS=true \
  uvicorn responser_model_api.app:app --port 8000
```

- **INFO** logs each request (persona, platform, chat, message count, model,
  temperature), the generation time, token counts, and the final reply. Refusal
  detection and fallbacks are logged as **WARNING**.
- **DEBUG** with `RESPONSER_LOG_PROMPTS=true` additionally logs the full assembled
  prompt and the raw model output — invaluable for understanding *why* the model
  answered a certain way.

> Privacy: prompts contain private chat content, so prompt logging is **off by
> default** and must be explicitly enabled.

## Test

```bash
pytest
```
