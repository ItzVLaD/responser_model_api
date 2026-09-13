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
  The response is `{memory, model_name}`: the complete memory **merged by code**.
  Internally the model produces only evidence-backed operations, not a rewritten
  summary. Pass previous memory on every batch; unrelated facts remain unchanged.
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

### Protected fact records (API 0.3.0)

`MemoryContent.facts` is the canonical collection, up to **64** records. Each has
a code-generated ID, section, kind, exact text, and evidence containing a source
message ID, speaker and verbatim quote (up to 200 characters). Stored text must
equal that quote. This avoids turning a speaker's words into an invented paraphrase.

The model proposes `add`, `replace`, or `remove` operations. Code checks citations
against the current input, checks speaker attribution, validates targets, and
applies the entire delta atomically. Replacements must target a previous fact of
the same section/kind and speaker with fresh evidence. Removals are restricted to
resolved open-thread questions/commitments; profile facts cannot be silently
deleted. Omitted facts are preserved. Repeated normalized quotes are deduplicated
without rewriting their original provenance; semantic paraphrase deduplication is
not guaranteed. When no previous facts exist, constrained decoding allows only
additions. Later targets are constrained to existing IDs.

The four old lists (`interlocutor`, `agent`, `interaction`, `open_threads`) remain
bounded **previews**: up to 8 strings of 200 characters per section, 6,000 serialized
characters in total including relationship. Reply prompts use **all canonical
facts**, not just these previews, with attribution and without IDs/duplicated source
proofs. `relationship_source` stores the quote supporting a relationship change;
an omitted relationship update keeps the existing stage.

Existing contexts load without resetting checkpoints. At their next summary update,
legacy strings become deterministic `legacy-unverified` records without fabricated
citations. This preserves existing mistakes too; migration is not a factual repair.
Do not edit preview lists expecting to change canonical facts. Both services must
be upgraded together: older readers do not understand the added fields.

There is **no silent eviction** when memory fills. The 64-fact limit, **48,000-byte**
persisted-memory cap, or model-input budget can require operator review. An update
that exceeds capacity fails and leaves the old file/checkpoint unchanged.

Memory extraction prioritizes **actual values**, not topic labels: both speakers'
stated ages, exact jobs and specialties, explicit communication preferences,
boundaries, and reactions with their triggers. Retracted jokes must not replace
corrected ages; questions must not be stored as facts about the questioner.
Behavior observations and self-reports are distinct—do not infer an anxiety
diagnosis from impatience. Unrelated later batches must retain earlier facts.

The API converts `me`/`other`/`system` to absolute `AGENT`/`INTERLOCUTOR`/
`SERVICE_EVENT` labels in the summarizer's model input. This does not change the
HTTP contract. The summary prompt contains no invented conversation examples:
small models can mistake such examples for facts to retain.

**Valid JSON does not guarantee correct memory.** A live Qwen check found speaker
attribution and detail-retention errors that mocked tests cannot detect. The
opt-in semantic evaluation in `tests/test_context_quality_live.py` uses invented
profiles across several updates; enable `RESPONSER_RUN_LIVE_CONTEXT_TESTS=1` to
run it against local Ollama. It is deliberately stricter than schema validation
and may expose remaining model limitations. Prompt changes cannot restore facts
already dropped before a saved checkpoint; those require a from-scratch rebuild
from the original messages, not another update of the lossy memory.

The evaluations separate **retention** (seeded, source-verified facts survive an
unrelated real-model update) from **extraction completeness** (all useful details
must be found in new messages). The first passed with local Qwen2.5:7b; the second
still exposed missing job/specialty details. Evidence validation rejects fabricated
quotes and wrong-speaker profile entries but cannot prove every fact was found or
that the model selected the correct fact to replace. The strict completeness test
is left enabled within the opt-in suite; it is not weakened to conceal omissions.

Wire requests may contain up to **64,000 UTF-8 bytes** including stored proofs.
The **12,000-byte** limit applies separately to `inference_payload()`: compact
previous fact IDs/text plus the current messages, without duplicate previews or
proofs. Reader batching validates both budgets. Legacy migration adds target IDs,
so a near-limit legacy request can be rejected after migration rather than silently
truncated. Schema/prompt tokens are additional; byte caps are not exact tokenizer
limits. Defaults use a 2,048-token completion in an 8,192-token model window.

Invalid/oversized requests return **422** before inference. Missing models,
inference errors, invalid evidence/targets, malformed or schema-invalid output, and length-truncated output
return **502**. No partial summary, old-memory fallback, reply fallback, or retry
is substituted. A valid empty delta preserves all old facts. On any
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
reader's control. Exact source excerpts are now persisted as proof; they are private
chat content, not anonymized metadata. The prompt excludes secrets, passwords, OTP/access codes,
filler, and embedded instructions, but model extraction is not a guaranteed secret
filter or fact checker. A real quote proves its source, not its truth, correct
semantic classification, or completeness. Review stored memory as needed. Both previous memory and
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
  token counts, and memory size; failures record exception types and allowlisted
  reason codes, never arbitrary exception text. It
  never logs raw memory, summary prompts/output, or error tracebacks, even when
  reply prompt logging is enabled.

### Diagnosing context-generation 502 responses

A 502 from `/summarize_context` can mean **model output was rejected**, not that
the API or Ollama was unreachable. Context errors now expose an allowlisted
`reason` in logs and the HTTP detail without printing message text or evidence:

- `output_truncated`: the completion was cut off or incomplete.
- `delta_schema_invalid` / `output_empty`: unusable model JSON or no text.
- `citation_message_missing`: the proposed source ID was not in the current batch.
- `citation_quote_mismatch`: the model's quote was not a verbatim substring of
  the message it cited.
- `profile_speaker_mismatch`: a quote was assigned to the wrong participant.
- `age_declaration_invalid`: the age quote is not a supported unambiguous
  declaration (for example a question, retracted joke, or conflicting numbers).
  First-person age sentences can include ordinary trailing clauses or emoji;
  they no longer have to consist solely of a number-bearing declaration. Source
  quotes are never rewritten to make them pass. This is a conservative syntax
  check, not proof of age or exhaustive natural-language understanding.
- `target_missing`, `target_kind_mismatch`, or another target code: invalid
  correction/removal, not a token-budget issue.
- `memory_validation_or_capacity`: the merged result exceeded limits or failed
  validation. Facts are not evicted to force the update through.

Rejected-output logs also include prompt/completion token counts. Smaller batches
can reduce extraction complexity or output truncation, but cannot guarantee valid
citations and must not bypass validation. The context checkpoint commits only
after all batches succeed: a failed second batch leaves no new saved context on
initial creation, so the next cycle starts again from batch one. Restart the model
API after updating code to see the detailed reason codes.

> Privacy: prompts contain private chat content, so prompt logging is **off by
> default** and must be explicitly enabled. Ordinary reply text is still logged
> at INFO and may itself refer to private facts; protect and rotate those logs.

## Test

```bash
pytest
```
