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

## Configuration (env vars)

| Variable          | Default                  | Purpose                         |
|-------------------|--------------------------|---------------------------------|
| `OLLAMA_HOST`     | `http://localhost:11434` | Where Ollama is listening       |
| `RESPONSER_MODEL` | `llama3.2:3b`            | Default model name              |

## Test

```bash
pytest
```
