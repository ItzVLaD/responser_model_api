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

## Choosing a model

The model is selected via `RESPONSER_MODEL` (see the config table). The default,
`llama3.2:3b`, is fast on CPU but its built-in safety training makes it refuse
"taboo" topics with robotic, out-of-character boilerplate. For personas that need
freer conversation, an uncensored model works far better.

Top 5 models to consider (tuned for a CPU-only, 16 GB machine):

| Model | Size | Speed (CPU) | Refusals | Notes |
|-------|------|-------------|----------|-------|
| `dolphin-mistral` | 7B | medium | very low | Best overall balance; strong instruction-following. |
| `dolphin3` (Llama 3.1) | 8B | slow | very low | Highest quality / most natural; slowest here. |
| llama3.2 **abliterated** 3B | 3B | fast | low | Same speed as default; refusals removed (check hub for exact tag). |
| `nous-hermes2` (Mistral) | 7B | medium | low | Most personable / roleplay-friendly tone. |
| `gemma2:2b` | 2B | fast | high (aligned) | Fast, but still censored — does *not* fix refusals. |

Recommendation: start with the **3B abliterated** model (no speed penalty, fixes
most refusals); step up to **`dolphin-mistral`** if you want better quality.

> First request after loading a new model is slow (it loads into RAM); on 16 GB,
> run only one 7–8B model at a time. Only adult, legal content is in scope.

### Check which models you have pulled

```bash
ollama list
```

This lists every model available locally (name, size, and modified date). Use it
to confirm a `RESPONSER_MODEL` value exists before starting the API.

### Pull and switch

```bash
ollama pull dolphin-mistral            # download once (a few GB)
ollama list                            # confirm it is available

# switch by pointing RESPONSER_MODEL at it (no code change needed)
RESPONSER_MODEL=dolphin-mistral uvicorn responser_model_api.app:app --port 8000

ollama rm llama3.2:3b                  # optional: remove a model you no longer need
```

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
| `RESPONSER_LOG_FILE`         | `logs/model_api.log`     | Log file path (empty string disables it)  |
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

- Logs are also **written to a file by default** (`logs/model_api.log`). Change the
  path with `RESPONSER_LOG_FILE`, or set it to an empty string to disable file
  logging.
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
