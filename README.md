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

A *personality* describes how the model should behave (tone, rules, few-shot
examples). Personalities are YAML files in `personalities/` and are selected once
at **startup** via `RESPONSER_PERSONALITY` — never per request. This keeps the API
contract stable while letting you swap behaviour.

```bash
# use the professional persona instead of the default "friendly"
RESPONSER_PERSONALITY=professional uvicorn responser_model_api.app:app --port 8000
```

Add a new persona by dropping a file in `personalities/`, e.g. `personalities/sarcastic.yaml`:

```yaml
name: Sarcastic
tone: dry and witty
rules:
  - Keep replies short.
  - Be playful, never mean.
examples:
  - incoming: "Are you awake?"
    reply: "Nope, texting you in my sleep."
```

The fixed output-format instructions ("reply only with the reply text") live in
code (`config.RESPONSE_FORMAT_INSTRUCTIONS`), so a persona edit can shape tone but
cannot break the reply contract.

## Configuration (env vars)

| Variable                     | Default                  | Purpose                                  |
|------------------------------|--------------------------|------------------------------------------|
| `OLLAMA_HOST`                | `http://localhost:11434` | Where Ollama is listening                |
| `RESPONSER_MODEL`            | `llama3.2:3b`            | Default model name                       |
| `RESPONSER_PERSONALITY`      | `friendly`               | Which `personalities/<name>.yaml` to use |
| `RESPONSER_PERSONALITIES_DIR`| `./personalities`        | Folder holding personality YAML files    |

## Test

```bash
pytest
```
