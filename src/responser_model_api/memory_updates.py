"""Pure, fail-closed merging of evidence-backed memory deltas.

Source matching proves only that a quote was supplied by that speaker, not its
truth, semantic classification, resolution of a thread, or extraction completeness.
No model-proposed prose, IDs, persistence, checkpoints, or network calls live here.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
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
)

MAX_MEMORY_OPERATIONS = 40
MAX_PREVIEW_ITEMS = 8
# Deliberately narrow declaration syntax, not a semantic age extractor. A model
# must select the corrected clause rather than a question or multi-clause joke.
_AGE_DECLARATION = re.compile(
    r"(?:i(?:\s+am|['’]m)\s+\d{1,3}(?:\s+years\s+old)?"
    r"|\d{1,3}\s+years\s+old|(?:my\s+)?age\s*:\s*\d{1,3})"
    r"(?:,?\s+not\s+\d{1,3})?[.!]?",
    re.IGNORECASE,
)


class MemoryUpdateError(ValueError):
    """An invalid or oversized delta leaves the caller's memory unchanged."""


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
    """Repeated-ID fragments are permitted only when quote attribution is unique."""
    matches = [message for message in messages if message.raw_id == message_id and quote in message.text]
    if not matches:
        raise MemoryUpdateError("citation must exactly match a supplied message ID and quote")
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
    for operation in delta.operations:
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
        if operation.kind == "age" and _AGE_DECLARATION.fullmatch(operation.quote.strip()) is None:
            raise MemoryUpdateError("age quote must be a standalone age declaration, not a question")
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
        source = _source(change.message_id, change.quote, messages)
        # A repeated rapport observation preserves its original provenance.
        if change.stage != relationship.stage or _normalized_quote(change.quote) != _normalized_quote(relationship.evidence):
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