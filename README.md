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
additions. Later replacements/removals require compatible existing targets.

### Source-excerpt selection (internal extraction)

The summarizer now asks the model to **select evidence**, not copy it. Code splits
the current messages into exact bounded excerpts, labelled `s0`, `s1`, etc.; all
original message text stays visible in chronological order as context. Excerpts
are limited to 200 characters and never joined across messages. Missing source
IDs, service events and oversized indivisible tokens are visible context but
cannot supply selectable proof. Some supported age declarations additionally get
a shorter, verbatim age-only excerpt. This is not semantic age inference.

The model can emit only:
- `add`: `source_id`, `scope` (`profile`, `interaction`, `open_threads`) and `kind`.
  For `profile`, **code derives agent/interlocutor ownership from the source**.
- `replace`: `source_id`, `kind` and a required target alias (`t0`, `t1`, etc.).
  Decoding permits only combinations compatible with the prior fact's section,
  kind, speaker and evidence-freshness rules. Legacy unverified `other` records
  may gain a concrete kind, as before.
- `remove`: `source_id` and a required target alias for an open question/commitment.
- Optional relationship change: `stage` and `source_id`.

Quotes, original message IDs, speakers and profile sections are **not output
fields**. The resolver supplies them from the selected source. Targets are
mandatory for replace/remove; null targets and unknown choices are rejected.
Age selections are restricted to excerpts accepted by the existing syntax
validator, so a standalone retraction cannot be selected as a replacement age.
Model-decoding constraints are checked again locally, then the existing atomic
merger independently validates all operations. Old free-quote model output is
not accepted as a fallback, and no extra inference retries are added.

This is an **API-internal redesign**: the HTTP contract, persisted fact/evidence
shape, model selection and reader checkpoint policy remain unchanged. Existing
contexts are not rebuilt or rewritten automatically. Restart the model API;
start logs will include `extraction=source_selection` and `excerpts`.

#### Typed-source guards (v2)

Inspection of four complete diagnostic batches (118 messages) showed that raw
selections, resolved quotes and saved memory matched exactly. No input text was
lost and all completions stopped normally. The schema itself still allowed the
bad choices: age-as-occupation, question-as-occupation, and replacing a factual
value with an acknowledgement. The model also omitted available work details.
This was not a storage/preview-list bug or a token-limit failure.

The source selector now shares one allowed-use table between its grammar and
resolver:
- Bare supported ages and short age excerpts can only be `profile/age`; they
  cannot evade age checks by selecting `self_report` or `occupation` instead.
- Age, name, occupation and specialty kinds belong only to profiles.
- Excerpts containing `?` or supported direct-question prefixes can only use
  `question` under interaction/open threads, never supply a profile statement.
- A small exact-match list of standalone acknowledgements/fragments remains
  context-only: it cannot add/replace facts or change relationship state. Longer
  meaningful statements containing those words are not removed.
- The adjacent **same-speaker age → explicit retraction → corrected age** pattern
  excludes the earlier source from all selection choices, without hiding text.
  This intentionally does not infer arbitrary distant or cross-batch retractions.
- Non-age replacements require an explicit change/correction cue (for example
  `now`, `instead`, or `correction`) in the excerpt or an immediately preceding
  correction prefix within the same message. Freshness alone is insufficient.
  A cue is necessary but not sufficient proof: unrelated statements with a cue
  can still be misinterpreted. Legitimate unmarked corrections may be missed.

Ordinary commas no longer fragment sentences; a narrow comma boundary still
separates a retraction prefix from a corrected age. Sentence/size limits remain.
This preserves subjects and qualifiers and avoids treating fragments as complete
claims. All source characters remain in chronological model context. Grammar
field order chooses kind/scope (or target) before source ID; that is a usability
change for decoding, not proof of improved semantic reasoning.

Trace sources include `use`, `allowed_adds`, `excluded_reason`, and
`correction_signalled`. Startup/request metadata includes
`constraints=typed_sources_v2`. These are **negative guards**, not a complete
semantic validator: compound age-and-job sentences remain eligible for multiple
kinds so work details are not discarded, and the model must still extract all
useful details. Repeated/conflicting targets still reject the whole delta.

An intermediate read-only replay with typed sources populated the interlocutor's
occupation and the agent's age, but omitted the work specialty and failed batch
three on repeated replacements. The additional correction-cue requirement was
then added. That replay is not a successful complete rebuild or evidence that
all semantic problems are fixed. Existing saved mistakes/checkpoints have not
been edited: repairing past omissions still requires a separately authorized
rebuild from source history after extraction quality is verified.

Final v2 evaluation on the unchanged live synthetic tests still returned
**1 passed, 1 failed**: retention passed, while initial extraction selected the
interlocutor's age/occupation/specialty but omitted the agent profile entirely.
The test failed before later updates. This establishes a remaining coverage
problem, not another citation/storage bug; v2 must not be described as reliable
complete extraction. Further work needs explicit per-participant coverage
evaluation and/or a more capable extractor, not weaker validation or repeated
full-history retries. The current model and one-call design remain unchanged.

**Remaining limitations:** selecting a genuine excerpt does not prove its
meaning. The model must still distinguish a self-report from a statement about
someone else, choose a corrected age rather than a retracted earlier one, decide
whether a thread was resolved, and extract every relevant detail. Full nearby
context is retained for those decisions. Excerpt selection reduces transcription
and ownership errors; it does not make a 7B model a reliable semantic fact checker.

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
must be found in new messages). Before source selection, the first passed with
local Qwen2.5:7b; the second exposed missing job/specialty details. Evidence validation rejects fabricated
quotes and wrong-speaker profile entries but cannot prove every fact was found or
that the model selected the correct fact to replace. The strict completeness test
is left enabled within the opt-in suite; it is not weakened to conceal omissions.

After the source-selection redesign, a short real-Qwen smoke check extracted both
speakers' ages/jobs plus a work specialty and correctly replaced only one age in a
later update. The unchanged, more demanding live suite returned **1 passed,
1 failed**: seeded-fact retention passed; initial extraction still omitted the
garden specialty and kayaking detail in the multi-clause/retraction conversation.
That failure occurs before its later-update assertions, so those later steps
were not verified by that run. Successful decoding or a 200 response must not be
presented as proof of complete extraction or production-history reliability.

Wire requests may contain up to **64,000 UTF-8 bytes** including stored proofs.
The **12,000-byte** limit applies separately to `inference_payload()`: compact
previous fact IDs/text plus the current messages, without duplicate previews or
proofs. This shared method is a budgeting projection, not the new model prompt.
Reader batching validates both budgets. The API additionally bounds the actual
selection input at **12,000 UTF-8 bytes**, the source registry at **256 excerpts**,
and the decoding schema at **256,000 bytes**. Exceeding any internal limit returns
`selection_input_capacity` before inference; excerpts are never silently evicted.
The reader does not currently shrink/retry batches automatically for this extra
overhead; `RESPONSER_CONTEXT_BATCH_SIZE` can be reduced if this limit is reached.
Legacy migration adds target IDs,
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
| `RESPONSER_CONTEXT_TRACE`    | `false`                  | Explicit opt-in full private summary trace, independent of DEBUG/reply logging |
| `RESPONSER_CONTEXT_TRACE_DIR` | project `diagnostics/context` | One owner-only JSONL trace per summary call; not ordinary logs |

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
- Ordinary summarization logs contain **metadata only**: model, message count, input size, timing,
  token counts, and memory size; failures record exception types and allowlisted
  reason codes, never arbitrary exception text. Raw memory, summary prompts/output
  and error tracebacks never enter these handlers. Full tracing below uses a
  separate, explicitly opted-in file sink, even when reply prompt logging is on.

### Full private context trace mode

Use this when a summary succeeds but contains wrong classifications, omitted
facts, or an empty profile. The trace exposes decisions; it does **not** fix or
certify semantic quality.

**Warning:** traces contain unredacted chat text, previous memory, message IDs,
exact prompts, selected/unselected excerpts and model output. Secrets present
in that data can also be captured. Keep traces local; review and redact before
sharing, and disable the mode after investigation.

Start the API with `RESPONSER_CONTEXT_TRACE=true`, retaining your other settings:

```bash
RESPONSER_CONTEXT_TRACE=true RESPONSER_MODEL=nous-hermes2 \
  RESPONSER_PERSONALITY=mia RESPONSER_TEMPERATURE=0.6 \
  .venv/bin/python -m uvicorn responser_model_api.app:app --port 8000
```

Settings are read at startup. DEBUG, `RESPONSER_LOG_PROMPTS` and HTTP callers
cannot enable tracing. `RESPONSER_LOG_FILE=''` disables ordinary file logging,
not an explicitly enabled private trace.

Each summary call creates `diagnostics/context/summary-<summary_id>.jsonl` under
this project, independent of the default working directory. Override it with
`RESPONSER_CONTEXT_TRACE_DIR` (relative overrides use the process working
directory). Match `summary_id` to console logs. Separate batches/attempts have
separate files; existing files are never overwritten.

Each JSONL line has a sequence number, UTC timestamp, elapsed time, stage and
data. Events are flushed and fsynced immediately:

| Stage | Captured data |
|-------|---------------|
| `request_received` | Incoming messages, full previous memory and model settings |
| `previous_prepared` | Previous memory after legacy migration |
| `selection_plan` | Excerpts with proofs/age eligibility, target aliases, compatible changes |
| `inference_request` | Exact prompt messages, schema, model and options |
| `inference_response` | SDK-decoded response, including raw completion text; not transport headers |
| `selection_parsed` | Model selections and unselected excerpts in plain text |
| `resolved_delta` | Quotes, speakers, sections and kinds supplied to the merger |
| `merge_started` / `merge_result` | Merge boundary and full resulting memory |
| `review` | All facts grouped as kind/text, empty sections, added/removed/preserved/modified IDs |
| `response` / `completed` | Returned response data and completion of the summary pipeline |
| `failed` | Error type and safe reason where available; no exception text/tracebacks |

Rejections also record `selection_schema_rejected` or `merge_rejected` details
where applicable. Raw invalid completion text stays in `inference_response`.
Unselected text may be irrelevant or omitted; the trace does not judge which.
The complete `review` is not limited to eight preview items, and explicitly says
`semantic_correctness_and_completeness=NOT_VALIDATED`.

For empty `interlocutor`, start with `review`, then compare `selection_parsed`
and `resolved_delta`: were the other speaker's excerpts omitted, put under
`interaction`, or used to replace an earlier fact? The request/prepared stages
show whether a bad fact already existed before this call.

**Storage:** macOS/Linux directories must be owned by the running user with no
group/other access (`0700` when created); files are `0600`. Symlink destinations
are refused. `diagnostics/` and `summary-*.jsonl` are gitignored. Each file is
capped at **4 MiB**. Permission/disk/size errors abort the summary with
`trace_write_failed`; earlier events remain. A hard exit or disk error may leave
no terminal event or an incomplete final line—ignore that line when inspecting.
There is no automatic deletion or global retention cap; monitor disk space.

`completed` is **not** proof of reader checkpoint persistence or message sending.
Browser scraping, reader persistence and reply sending remain outside this
trace. FastAPI 422s before the summarizer produce no trace; network/SDK failures
may leave `inference_request` followed by `failed`, without a response stage.

Enabling tracing does not rebuild saved context or reconstruct past runs. With
an existing checkpoint, the reader normally submits only the unsummarized tail.
A full-history diagnostic rebuild requires a separate explicit operation.
Prefer reader `dry_run` while testing: it prevents sending, but can still save
context after a valid summary. Tracing itself never edits reader context files.

### Diagnosing context-generation 502 responses

A 502 from `/summarize_context` can mean **model output was rejected**, not that
the API or Ollama was unreachable. Context errors now expose an allowlisted
`reason` in logs and the HTTP detail without printing message text or evidence:

- `output_truncated`: the completion was cut off or incomplete.
- `delta_schema_invalid` / `output_empty`: unusable model JSON or no text.
- `selection_source_unknown` / `selection_target_unknown`: an output ID was not
  in the request-local excerpt/target registry.
- `selection_choice_invalid`: a source/target/kind combination violated the
  locally checked selection rules (even if the model server ignored its grammar).
- `selection_input_capacity`: excerpt count, actual prompt bytes or decoding
  schema size exceeded an internal bound before inference.
- `trace_write_failed`: explicitly enabled private tracing could not safely
  create or persist its file. Check the trace directory permissions/disk space;
  do not expect complete diagnostic output after this failure.
- `citation_message_missing`: the proposed source ID was not in the current batch.
- `citation_quote_mismatch`: the quote did not match its cited message exactly
  and could not be recovered through the formatting-only rules below.
- `citation_quote_ambiguous`: formatting recovery matched multiple distinct
  original source spans; the API refuses to guess which quote was intended.
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

Rejected deltas now also emit a `summary validation detail` warning at the default
log level. It reports the **first actual failure** without retrying, skipping an
operation, or dumping the model output:

- `summary_id`: a fresh server-generated ID linking the start, success or
  validation rejection, and API validation-error logs for one summary call. It
  is not a chat identifier or a batch number.
- `component`: `operation`, `relationship`, or `merge` for registry-wide failures.
- `operation_index`: **zero-based** operation position (`None` for relationship).
- `action`, `section`, `kind`, `target_state`: validated enum labels, never fact IDs.
- `source_positions`: **zero-based** positions of messages matching the cited ID
  in this request; multiple positions can represent fragments. `source_speakers`
  lists those rows' `me`/`other`/`system` labels, not account names.
- `source_match`: `missing_message`, `no_match`, `exact`, `formatting_only`, or
  `ambiguous_formatting`; exact provenance is not proof of correct meaning.
- `quote_chars`: proposed quote length, not its content.
- `age_check`: the specific existing age rule that rejected the verified source
  quote: `question_mark`, `uncertainty_or_retraction`,
  `unsupported_declaration_form`, `additional_numeric_claim`, or
  `unsupported_age_continuation`. Other failures use `not_applicable`.

For example, `kind=age source_match=exact age_check=unsupported_declaration_form`
means the quote exists but its syntax is unsupported, whereas
`source_match=no_match age_check=not_applicable` means source validation failed
before any age check. These are validator rules, not semantic diagnoses. Age
acceptance rules are unchanged by this logging update.

Start logs include previous fact count and configured token/context budgets;
rejection logs include elapsed time. No chat text, actual ages, source/fact IDs,
raw model output or tracebacks are added, even with DEBUG or reply prompt logging
enabled. These guarantees apply to ordinary logs; explicitly opted-in private
trace files intentionally include full content. No new environment setting is
needed for metadata diagnostics; restart the API to apply code changes. Old
failures cannot be reconstructed retroactively from ordinary logs.

The independent merger retains citation recovery for direct delta callers and
defense-in-depth tests; normal source-selection output already uses original
source text and does not require this repair. Recovery first checks the exact
substring. If that fails, it allows only
whitespace collapsing and curly/straight single or double quotation marks, within
the **same cited message ID**. A unique match restores the original source span,
including its original punctuation and whitespace, into the fact and evidence.
It never stores the model's reformatted version, joins fragments, searches other
message IDs, changes case/words/numbers, or uses fuzzy similarity. Multiple distinct
spans, ambiguous speakers, oversized restored quotes and unsupported mismatches
still fail the entire update. Restored spans cannot cut through words or numeric
tokens. Existing attribution, age, freshness and capacity checks still apply.
Recovery uses no additional inference call and logs only a metadata event;
it does not mean the batch has committed. A later invalid operation still aborts
everything. Source matching remains provenance validation, not semantic proof or
a guarantee that the model extracted every relevant fact.

> Privacy: prompts contain private chat content, so prompt logging is **off by
> default** and must be explicitly enabled. Ordinary reply text is still logged
> at INFO and may itself refer to private facts; protect and rotate those logs.

## Test

`tests/test_summary_trace.py` uses synthetic data and mocked Ollama transport to
test phase coverage, faithful wrong-classification traces, disabled-mode privacy,
no trace leakage into ordinary logs, file permissions, concurrency and I/O errors.
The test fixtures disable tracing during collection and use temporary trace paths
so a developer's environment cannot write private traces during ordinary pytest.

`tests/test_source_selection.py` exercises the real selection parser/resolver and
HTTP endpoint with only Ollama transport stubbed. `tests/test_context_summarizer.py`
also keeps historical malformed-delta tests through an explicitly named
post-selection test seam: these verify independent merger defenses, not real
model extraction. JSON-schema compatibility is checked with the dev-only
`jsonschema` dependency. The unchanged opt-in live tests check semantic quality.

```bash
pytest
```
