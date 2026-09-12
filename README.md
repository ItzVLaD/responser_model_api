# responser_model_api

Local HTTP API that turns chat snapshots into replies and structured conversation
memory using Ollama-hosted models. Replies and summaries are written in English.
It owns the inter-module contract (Pydantic schemas + OpenAPI at `/openapi.json`),
version **0.2.0**. The API itself is stateless; the reader owns context persistence.

## Setup

```bash
# 1. Install Ollama (https://ollama.com) and pull the configured models once
ollama pull nous-hermes2:latest
ollama pull qwen2.5:7b

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
- Summarize context: `POST http://localhost:8000/summarize_context`
- Contract / docs: `http://localhost:8000/docs` and `http://localhost:8000/openapi.json`

## Choosing a model

The reply model is selected via `RESPONSER_MODEL` (see the config table). Its
default remains `nous-hermes2:latest`; enabling summaries does not change it.
The smaller alternative `llama3.2:3b` is fast on CPU but its built-in safety training makes it refuse
"taboo" topics with robotic, out-of-character boilerplate. For personas that need
freer conversation, an uncensored model works far better.

Top 5 models to consider (tuned for a CPU-only, 16 GB machine):

| Model | Size | Speed (CPU) | Refusals | Notes |
|-------|------|-------------|----------|-------|
| `dolphin-mistral` | 7B | medium | very low | Best overall balance; strong instruction-following. |
| `dolphin3` (Llama 3.1) | 8B | slow | very low | Highest quality / most natural; slowest here. |
| llama3.2 **abliterated** 3B | 3B | fast | low | Small alternative; refusals removed (check hub for exact tag). |
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
The fixed contract requires English; a persona's `language` field describes its
English voice or dialect. Summaries do not use the selected personality.

## Persisted conversation context

The two endpoints are independent. A client that uses persisted context must
**finish summarization successfully before requesting a reply**:

1. Send `POST /summarize_context` with `previous` (the last `MemoryContent`, or
  `null`) and `messages` (a chronological batch of **1–30** `Message` objects).
  The response is `{memory, model_name}`: a complete replacement memory, not a
  delta. Pass the previous memory on every subsequent batch so relevant older
  facts, agent claims, and pending topics can be retained.
2. Only after success, the reader persists a `ConversationContext` containing
  `memory`, `last_message_id`, `summarized_message_count` (at least 1),
  `model_name`, and `updated_at`. Metadata strings must be nonempty. The reader,
  not the summarizing model, supplies checkpoint IDs, counts, and timestamps;
  the API supplies the configured model name.
3. Include that checkpoint as `snapshot.context` in `POST /generate_reply`,
  alongside **only raw messages after `last_message_id`**, excluding the checkpoint
  message and everything already summarized. After an update this tail contains
  10 messages, growing to 30 before the next update. Without context, send all
  available messages up to 30. Recent messages take precedence over memory;
  summarized messages are not repeated as raw turns. Relationship guidance uses persisted evidence
  rather than the visible message count, with current boundaries taking priority.

`MemoryContent` has four lists: `interlocutor`, `agent`, `interaction`, and
`open_threads`. Each defaults to empty and holds at most **8 nonempty strings of
200 characters**. `relationship` contains `stage` (`unknown`, `new`, `acquaintance`,
`familiar`, or `strained`; default `unknown`) and `evidence` (at most **300
characters**, default empty). All new contract models reject unknown fields.
Complete memory serialized with `model_dump_json()` is limited to **6,000
characters**, including JSON escaping and field overhead.

The complete summary request serialized with `model_dump_json()` is limited to
**12,000 UTF-8 bytes**, including previous memory, message metadata, JSON escaping,
and field overhead. A batch with 6,000 text bytes plus a 6,000-character memory
may exceed that cap: clients must measure the whole request and reduce the batch,
not silently discard old memory. The output schema and fixed prompt additionally
consume model context, outside this request byte limit. The defaults reserve room
for them and a 2,048-token completion in an 8,192-token window; byte limits are not
exact tokenizer limits, so reduce batch sizes before shrinking the window.

Invalid/oversized requests return **422** before inference. Missing models,
inference errors, malformed or schema-invalid output, and length-truncated output
return **502**. No partial summary, old-memory fallback, reply fallback, or retry
is substituted. Valid empty structured memory is allowed for greetings. On any
failure, the reader must keep its previous checkpoint and **not generate/send a
reply**; this ordering is a client responsibility, not an implicit endpoint call.

The default summarizer is **`qwen2.5:7b`**, independent of the reply model. Download
it through Ollama before using context (see setup); the service never downloads
models automatically. Use a current Ollama server with JSON-schema structured
output support. Summaries use temperature `0.1` and their own output/context
budgets. A separate summary model uses `keep_alive=0` to unload after extraction;
if the configured model names are identical, it uses the normal `5m` residency.
Alternating separate 7B models can incur substantial loading and swapping costs,
particularly on a 16 GB machine; initial history batches may be slow.

Privacy: reader-side context is sensitive local chat data. Protect it like chat
history, exclude it from version control, and keep deletion/retention under the
reader's control. The summary prompt excludes secrets, passwords, OTP/access codes,
filler, and embedded instructions, but model extraction is not a guaranteed secret
filter or fact checker. Review stored memory as needed. Both previous memory and
messages are treated as untrusted evidence; summaries never receive a reply persona.

## Configuration (env vars)

| Variable                     | Default                  | Purpose                                  |
|------------------------------|--------------------------|------------------------------------------|
| `OLLAMA_HOST`                | `http://localhost:11434` | Where Ollama is listening                |
| `RESPONSER_MODEL`            | `nous-hermes2:latest`    | Reply model name                         |
| `RESPONSER_TEMPERATURE`      | `0.7`                    | Sampling temperature (higher = more varied) |
| `RESPONSER_TOP_P`            | `0.9`                    | Nucleus sampling cutoff                   |
| `RESPONSER_MAX_OUTPUT_TOKENS`| `256`                    | Max tokens per reply                      |
| `RESPONSER_REPLY_CONTEXT_WINDOW` | `8192`              | Reply input/output window for memory plus up to 30 unsummarized messages |
| `RESPONSER_CONTEXT_MODEL`   | `qwen2.5:7b`             | Independent structured summary model      |
| `RESPONSER_CONTEXT_MAX_TOKENS` | `2048`                 | Summary output token budget, not the reply cap |
| `RESPONSER_CONTEXT_WINDOW`  | `8192`                   | Summary context window (`num_ctx`)        |
| `RESPONSER_PERSONALITY`      | `friendly`               | Which `personalities/<name>.yaml` to use |
| `RESPONSER_PERSONALITIES_DIR`| `./personalities`        | Folder holding personality YAML files    |
| `RESPONSER_LOG_LEVEL`        | `INFO`                   | Log level (DEBUG, INFO, WARNING, ...)     |
| `RESPONSER_LOG_FILE`         | `logs/model_api.log`     | Log file path (empty string disables it)  |
| `RESPONSER_LOG_PROMPTS`      | `false`                  | DEBUG-log reply prompts/output; context-bearing system prompts are redacted |

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
  reply prompt and raw model output, except that system prompts containing
  persisted context are redacted.
- Summarization logs **metadata only**: model, message count, input size, timing,
  token counts, and memory size; failures record only their exception type. It
  never logs raw memory, summary prompts/output, or error tracebacks, even when
  reply prompt logging is enabled.

> Privacy: prompts contain private chat content, so prompt logging is **off by
> default** and must be explicitly enabled. Ordinary reply text is still logged
> at INFO and may itself refer to private facts; protect and rotate those logs.

## Test

```bash
pytest
```
