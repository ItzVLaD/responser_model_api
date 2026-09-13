"""Deterministic, fail-closed merging of evidence-backed memory deltas.

Source matching proves only that a quote was supplied by that speaker, not its
truth, semantic classification, resolution of a thread, or extraction completeness.
No model-proposed prose, IDs, persistence, checkpoints, or network calls live here.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from responser_model_api.schemas import (
    FACT_SECTIONS,
    FactEvidence,
    FactKind,
    FactSection,
    MemoryContent,
    MemoryFact,
    MemoryItem,
    Message,
    NonEmptyString,
    RelationshipStage,
    RelationshipState,
    SenderType,
)
from .source_quotes import formatting_equivalent_spans

log = logging.getLogger("responser.model_api")

MAX_MEMORY_OPERATIONS = 40
MAX_PREVIEW_ITEMS = 8
# Minimal declarations remain supported. A fullmatch alone rejects valid source
# sentences with conversational continuations, so also recognize a conservative
# first-person sentence form below. Neither pattern is a semantic fact checker.
_AGE_DECLARATION = re.compile(
    r"(?:i(?:\s+am|['’]m)\s+\d{1,3}(?:\s+years\s+old)?"
    r"|\d{1,3}\s+years\s+old|(?:my\s+)?age\s*:\s*\d{1,3})"
    r"(?:,?\s+not\s+\d{1,3})?[.!]?",
    re.IGNORECASE,
)

_AGE_SENTENCE_START = re.compile(
    r"^(?:(?:(?:and\s+)?as\s+for\s+my\s+age|well|actually|honestly|by\s+the\s+way)[,:]?\s+)?"
    r"i(?:\s+am|['’]m)\s+(?:now\s+)?[0-9]{1,3}"
    r"(?:\s+years\s+old)?(?!\w|\.[0-9])",
    re.IGNORECASE,
)
_AGE_CORRECTION_TAIL = re.compile(r"^,?\s*not\s+[0-9]{1,3}(?!\w|\.[0-9])", re.IGNORECASE)
_AGE_CONTINUATION = re.compile(r"^(?:[,;.!]\s*|(?:and|but|so)\b)", re.IGNORECASE)
_AGE_UNCERTAINTY = re.compile(
    r"\b(?:kidding|joking|joke|pretend(?:ing)?|maybe|perhaps|probably|possibly)\b",
    re.IGNORECASE,
)


AgeCheck = Literal[
    "question_mark", "uncertainty_or_retraction", "unsupported_declaration_form",
    "additional_numeric_claim", "unsupported_age_continuation",
]


def _age_declaration_issue(quote: str) -> AgeCheck | None:
    """Return the rejecting rule, not source text or an inferred age.

    Do not truncate/repair evidence to make a quote pass. Questions, explicit
    uncertainty/retractions and multiple numbers remain conservatively rejected
    (apart from a direct 'not N' correction). Unsupported phrasing should be
    quoted as a shorter declaration by the extractor, not guessed here.
    """
    text = quote.strip()
    if "?" in text:
        return "question_mark"
    if _AGE_UNCERTAINTY.search(text):
        return "uncertainty_or_retraction"
    if _AGE_DECLARATION.fullmatch(text):
        return None
    head = _AGE_SENTENCE_START.match(text)
    if head is None:
        return "unsupported_declaration_form"
    tail = text[head.end():].strip()
    tail = _AGE_CORRECTION_TAIL.sub("", tail, count=1).strip()
    if any(char.isdigit() for char in tail):
        # Prevent accepting an age range or two conflicting age assertions.
        return "additional_numeric_claim"
    if not tail or all(unicodedata.category(char)[0] in "PSMZ" for char in tail):
        # Terminal punctuation and emoji do not invalidate the declaration.
        return None
    # A word directly after the number may be a unit (months, dollars, plants)
    # rather than an age. Only allow an explicit clause boundary/connector.
    return None if _AGE_CONTINUATION.match(tail) is not None else "unsupported_age_continuation"


def _is_age_declaration(quote: str) -> bool:
    """Check the same conservative age rules used by rejection diagnostics."""
    return _age_declaration_issue(quote) is None


# Allowlisted codes, not exception text, may be logged or returned over HTTP.
# Pydantic causes can contain private quotes; unknown messages map to a generic
# code rather than being echoed while diagnosing failed extraction.
MEMORY_FAILURE_REASONS: dict[str, str] = {
    "legacy memory violates validation or capacity limits": "legacy_validation_or_capacity",
    "citation message ID is not in the supplied batch": "citation_message_missing",
    "citation quote must exactly match the supplied message": "citation_quote_mismatch",
    "citation formatting matches multiple distinct source quotes": "citation_quote_ambiguous",
    "citation has ambiguous source speakers": "citation_speaker_ambiguous",
    "service events cannot provide fact evidence": "citation_service_event",
    "profile section does not match source speaker": "profile_speaker_mismatch",
    "target section mismatch": "target_section_mismatch",
    "target kind mismatch": "target_kind_mismatch",
    "target speaker mismatch": "target_speaker_mismatch",
    "replace/remove requires new evidence, not a previous source": "target_evidence_not_new",
    "only open_threads questions or commitments may be removed": "removal_not_allowed",
    "add must not have a target ID": "unexpected_add_target",
    "replace/remove requires an existing target ID": "target_missing",
    "repeated target ID conflicts within one delta": "target_repeated",
    "age quote must contain an unambiguous age declaration, not a question": "age_declaration_invalid",
    "relationship change requires new evidence": "relationship_evidence_not_new",
    "replacement conflicts with another fact": "replacement_conflict",
    "add conflicts with a replaced or removed fact": "addition_conflict",
    "memory update violates validation or capacity limits": "memory_validation_or_capacity",
}


@dataclass(frozen=True)
class UpdateDiagnostics:
    """Metadata from validated fields only; never hold free text or source IDs."""

    component: Literal["operation", "relationship"]
    operation_index: int | None
    action: Literal["add", "replace", "remove"] | None
    section: FactSection | None
    kind: FactKind | None
    source_positions: tuple[int, ...]
    source_speakers: tuple[SenderType, ...]
    source_match: Literal["missing_message", "exact", "formatting_only", "ambiguous_formatting", "no_match"]
    quote_chars: int
    target_state: Literal["none", "present", "missing", "not_applicable"]


class MemoryUpdateError(ValueError):
    """An invalid delta; safe reason codes never expose its private source text."""

    def __init__(self, message: str, *, age_check: AgeCheck | None = None) -> None:
        super().__init__(message)
        self.age_check = age_check
        self.diagnostics: UpdateDiagnostics | None = None

    @property
    def reason(self) -> str:
        return MEMORY_FAILURE_REASONS.get(str(self), "invalid_memory_update")


class MemoryOperation(BaseModel):
    """An operation names a target and a verbatim source, never invented prose."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["add", "replace", "remove"] = Field(
        description="add for new facts; replace/remove ONLY an existing previous fact named by target_id.",
    )
    section: FactSection = Field(description="Owner: INTERLOCUTOR -> interlocutor, AGENT -> agent; reactions -> interaction.")
    kind: FactKind = Field(description="Use age/occupation/specialty for exact profile values; a question is NOT a self_report.")
    target_id: str | None = Field(default=None, description="Existing previous fact ID for replace/remove; null for add.")
    message_id: NonEmptyString = Field(description="Exact message_id of the CURRENT source message, not a fact ID.")
    quote: MemoryItem = Field(description="Exact short source substring, not a paraphrase. One fact per quote; ages exclude any retracted joke.")


class RelationshipChange(BaseModel):
    """A proposed rapport stage with a source quote, not arbitrary evidence."""

    model_config = ConfigDict(extra="forbid")

    stage: RelationshipStage
    message_id: NonEmptyString
    quote: MemoryItem


class MemoryDelta(BaseModel):
    """Internal model output contract; absent changes preserve existing memory."""

    model_config = ConfigDict(extra="forbid")

    operations: list[MemoryOperation] = Field(default_factory=list, max_length=MAX_MEMORY_OPERATIONS)
    relationship: RelationshipChange | None = None


def _update_diagnostics(
    change: MemoryOperation | RelationshipChange, operation_index: int | None,
    messages: list[Message], memory: MemoryContent,
) -> UpdateDiagnostics:
    """Locate failed evidence without printing private message/fact identifiers."""
    positions = tuple(index for index, message in enumerate(messages) if message.raw_id == change.message_id)
    sources = [messages[index] for index in positions]
    source_match: Literal["missing_message", "exact", "formatting_only", "ambiguous_formatting", "no_match"]
    if not sources:
        source_match = "missing_message"
    elif any(change.quote in source.text for source in sources):
        source_match = "exact"
    else:
        candidates = {
            (source.sender_type, span)
            for source in sources for span in formatting_equivalent_spans(source.text, change.quote)
        }
        source_match = "no_match" if not candidates else "formatting_only" if len(candidates) == 1 else "ambiguous_formatting"
    operation = change if isinstance(change, MemoryOperation) else None
    target_state: Literal["none", "present", "missing", "not_applicable"] = "not_applicable"
    if operation is not None:
        if operation.target_id is None:
            target_state = "none"
        else:
            target_state = "present" if any(fact.id == operation.target_id for fact in memory.facts) else "missing"
    return UpdateDiagnostics(
        component="operation" if operation is not None else "relationship",
        operation_index=operation_index,
        action=operation.action if operation is not None else None,
        section=operation.section if operation is not None else None,
        kind=operation.kind if operation is not None else None,
        source_positions=positions,
        source_speakers=tuple(sorted({source.sender_type for source in sources})),
        source_match=source_match, quote_chars=len(change.quote), target_state=target_state,
    )


@contextmanager
def _diagnose_change(
    change: MemoryOperation | RelationshipChange, operation_index: int | None,
    messages: list[Message], memory: MemoryContent,
) -> Iterator[None]:
    """Attach metadata at the actual failure; never rerun/skip failed operations."""
    try:
        yield
    except MemoryUpdateError as exc:
        exc.diagnostics = _update_diagnostics(change, operation_index, messages, memory)
        raise
    except ValidationError as exc:
        error = MemoryUpdateError("memory update violates validation or capacity limits")
        error.diagnostics = _update_diagnostics(change, operation_index, messages, memory)
        raise error from exc


def _normalized_quote(quote: str) -> str:
    """Normalize identity only; stored text and proof remain byte-for-byte exact."""
    return " ".join(unicodedata.normalize("NFKC", quote).split()).casefold()


def _fact_id(section: FactSection, speaker: str, quote: str) -> str:
    canonical = json.dumps(
        [section, speaker, _normalized_quote(quote)], ensure_ascii=False, separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def migrate_legacy(memory: MemoryContent) -> MemoryContent:
    """Deterministically migrate previews once without manufacturing sources."""
    try:
        # Revalidate and detach nested lists even if callers mutated a model.
        result = MemoryContent.model_validate(memory.model_dump())
        if result.facts:
            return result
        facts: dict[str, MemoryFact] = {}
        for section in FACT_SECTIONS:
            for text in getattr(result, section):
                fact_id = _fact_id(section, "legacy-unverified", text)
                facts.setdefault(fact_id, MemoryFact(
                    id=fact_id, section=section, kind="other", text=text,
                ))
        return MemoryContent.model_validate({**result.model_dump(), "facts": list(facts.values())})
    except ValidationError as exc:
        raise MemoryUpdateError("legacy memory violates validation or capacity limits") from exc


def _source(message_id: str, quote: str, messages: list[Message]) -> FactEvidence:
    """Validate exact evidence, recovering only unique formatting-equivalent spans."""
    sources = [message for message in messages if message.raw_id == message_id]
    if not sources:
        raise MemoryUpdateError("citation message ID is not in the supplied batch")
    matches = [message for message in sources if quote in message.text]
    if not matches:
        candidates = {
            (message.sender_type, span)
            for message in sources
            for span in formatting_equivalent_spans(message.text, quote)
        }
        if not candidates:
            raise MemoryUpdateError("citation quote must exactly match the supplied message")
        if len({speaker for speaker, _ in candidates}) != 1:
            raise MemoryUpdateError("citation has ambiguous source speakers")
        if len(candidates) != 1:
            raise MemoryUpdateError("citation formatting matches multiple distinct source quotes")
        speaker, original = next(iter(candidates))
        if speaker == "system":
            raise MemoryUpdateError("service events cannot provide fact evidence")
        evidence = FactEvidence(message_id=message_id, sender_type=speaker, quote=original)
        # No quote, source ID or other private data in recovery diagnostics.
        log.info("citation formatting recovered: original source span restored")
        return evidence
    speakers = {message.sender_type for message in matches}
    if len(speakers) != 1:
        raise MemoryUpdateError("citation has ambiguous source speakers")
    speaker = matches[0].sender_type
    if speaker == "system":
        raise MemoryUpdateError("service events cannot provide fact evidence")
    return FactEvidence(message_id=message_id, sender_type=speaker, quote=quote)


def _check_attribution(section: FactSection, evidence: FactEvidence) -> None:
    expected = {"interlocutor": "other", "agent": "me"}.get(section)
    if expected is not None and evidence.sender_type != expected:
        raise MemoryUpdateError("profile section does not match source speaker")


def _same_source(old: FactEvidence, new: FactEvidence) -> bool:
    return old.sender_type == new.sender_type and (
        old.message_id == new.message_id or _normalized_quote(old.quote) == _normalized_quote(new.quote)
    )


def _check_target(
    operation: MemoryOperation, evidence: FactEvidence, target: MemoryFact,
    previous_facts: list[MemoryFact],
) -> None:
    """Destructive operations need a matching target and fresh source evidence."""
    if target.section != operation.section:
        raise MemoryUpdateError("target section mismatch")
    if target.kind != operation.kind and not (target.evidence is None and target.kind == "other"):
        raise MemoryUpdateError("target kind mismatch")
    # A response from the other speaker can close a question. Replacement of a
    # speaker's claim, however, cannot use the other person's words as theirs.
    if operation.action == "replace" and target.evidence is not None and target.evidence.sender_type != evidence.sender_type:
        raise MemoryUpdateError("target speaker mismatch")
    if _normalized_quote(target.text) == _normalized_quote(evidence.quote) or any(
        fact.evidence is not None and _same_source(fact.evidence, evidence) for fact in previous_facts
    ):
        raise MemoryUpdateError("replace/remove requires new evidence, not a previous source")
    if operation.action == "remove" and (
        operation.section != "open_threads" or operation.kind not in ("question", "commitment")
    ):
        raise MemoryUpdateError("only open_threads questions or commitments may be removed")


def _duplicate(facts: list[MemoryFact], candidate: MemoryFact) -> MemoryFact | None:
    """Replay never upgrades legacy provenance or overwrites an earlier citation."""
    for fact in facts:
        if fact.section != candidate.section or _normalized_quote(fact.text) != _normalized_quote(candidate.text):
            continue
        if fact.evidence is None or candidate.evidence is None or fact.evidence.sender_type == candidate.evidence.sender_type:
            return fact
    return None


def _merge(
    previous: MemoryContent | None, delta: MemoryDelta, messages: list[Message],
) -> MemoryContent:
    memory = migrate_legacy(previous) if previous is not None else MemoryContent()
    delta = MemoryDelta.model_validate(delta.model_dump())
    targets = {fact.id: fact for fact in memory.facts}
    touched: set[str] = set()
    additions: list[MemoryFact] = []
    replacements: dict[str, MemoryFact] = {}
    removals: set[str] = set()

    # Validate the entire plan against the original registry before applying it.
    # New facts cannot become targets during the same batch.
    for index, operation in enumerate(delta.operations):
        with _diagnose_change(operation, index, messages, memory):
            if operation.action == "add":
                if operation.target_id is not None:
                    raise MemoryUpdateError("add must not have a target ID")
            else:
                if not operation.target_id or operation.target_id not in targets:
                    raise MemoryUpdateError("replace/remove requires an existing target ID")
                if operation.target_id in touched:
                    raise MemoryUpdateError("repeated target ID conflicts within one delta")
                touched.add(operation.target_id)
            evidence = _source(operation.message_id, operation.quote, messages)
            _check_attribution(operation.section, evidence)
            age_check = _age_declaration_issue(evidence.quote) if operation.kind == "age" else None
            if age_check is not None:
                raise MemoryUpdateError(
                    "age quote must contain an unambiguous age declaration, not a question", age_check=age_check,
                )
            if operation.target_id is not None:
                _check_target(operation, evidence, targets[operation.target_id], memory.facts)
            if operation.action == "remove":
                assert operation.target_id is not None
                removals.add(operation.target_id)
                continue
            fact = MemoryFact(
                id=_fact_id(operation.section, evidence.sender_type, evidence.quote),
                section=operation.section, kind=operation.kind, text=evidence.quote, evidence=evidence,
            )
            if operation.action == "replace":
                assert operation.target_id is not None
                replacements[operation.target_id] = fact
            else:
                additions.append(fact)

    relationship = memory.relationship
    relationship_source = memory.relationship_source
    if delta.relationship is not None:
        change = delta.relationship
        with _diagnose_change(change, None, messages, memory):
            source = _source(change.message_id, change.quote, messages)
            # A repeated rapport observation preserves its original provenance.
            if change.stage != relationship.stage or _normalized_quote(source.quote) != _normalized_quote(relationship.evidence):
                if relationship_source is not None and _same_source(relationship_source, source):
                    raise MemoryUpdateError("relationship change requires new evidence")
                relationship = RelationshipState(stage=change.stage, evidence=source.quote)
                relationship_source = source

    # Reject replacement collisions instead of silently losing another target.
    retained = [fact for fact in memory.facts if fact.id not in touched]
    for fact in replacements.values():
        if _duplicate(retained, fact) is not None:
            raise MemoryUpdateError("replacement conflicts with another fact")
        retained.append(fact)
    facts = [replacements.get(fact.id, fact) for fact in memory.facts if fact.id not in removals]
    for fact in additions:
        # An add may not resurrect a fact removed/replaced in the same plan.
        old = _duplicate(memory.facts, fact)
        if old is not None and old.id in touched:
            raise MemoryUpdateError("add conflicts with a replaced or removed fact")
        if _duplicate(facts, fact) is None:
            facts.append(fact)
    if facts == memory.facts and relationship == memory.relationship and relationship_source == memory.relationship_source:
        return memory

    payload = memory.model_dump()
    payload.update(facts=facts, relationship=relationship, relationship_source=relationship_source)
    for section in FACT_SECTIONS:
        payload[section] = [fact.text for fact in facts if fact.section == section][:MAX_PREVIEW_ITEMS]
    # Pydantic enforces preview, fact-count and full UTF-8 caps; never evict facts.
    return MemoryContent.model_validate(payload)


def merge_memory_delta(
    previous: MemoryContent | None, delta: MemoryDelta, messages: list[Message],
) -> MemoryContent:
    """Return a detached, validated memory or raise without changing any input.

    Only open-thread questions/commitments can be removed. Other corrections
    require replacement with a fresh quote from the same speaker and section.
    An exact source match does not prove the model chose the right semantic fact.
    """
    try:
        return _merge(previous, delta, messages)
    except ValidationError as exc:
        raise MemoryUpdateError("memory update violates validation or capacity limits") from exc