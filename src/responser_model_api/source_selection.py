"""Internal, bounded selection of code-owned source excerpts and fact targets.

The HTTP and persisted-memory contracts remain unchanged. The model can choose
evidence, but cannot author its quote, speaker, message ID or replacement target
metadata. Selection proves provenance, not truth or semantic completeness.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from .memory_updates import (
    MAX_MEMORY_OPERATIONS, MemoryDelta, MemoryOperation, MemoryUpdateError,
    RelationshipChange, _AGE_DECLARATION, _check_target, _is_age_declaration,
)
from .schemas import (
    FactEvidence, FactKind, FactSection, MemoryContent, MemoryFact, Message,
    MAX_SUMMARY_INPUT_BYTES, RelationshipStage, SenderType, SummarizeContextRequest,
)

MAX_EXCERPT_CHARS = 200
MAX_EXCERPTS = 256
MAX_SELECTION_INPUT_BYTES = MAX_SUMMARY_INPUT_BYTES
MAX_SELECTION_SCHEMA_BYTES = 256_000
SourceId = Annotated[str, Field(min_length=1, max_length=16)]
Scope = Literal["profile", "interaction", "open_threads"]
_KINDS: tuple[FactKind, ...] = (
    "age", "name", "occupation", "specialty", "preference", "boundary", "self_report",
    "observation", "story", "interest", "commitment", "question", "other",
)
# Keep numeric punctuation and explicit 'not N' corrections intact. Splitting
# a sentence is not evidence that its neighbouring sentences are irrelevant.
_CLAUSE_BOUNDARY = re.compile(
    r"[.!?;](?=\s)|(?<!age):(?=\s)|\n+"
    r"|(?:(?<=kidding)|(?<=joking)),(?=\s*i(?:['’]m|\s+am)\s+\d)",
    re.IGNORECASE,
)
_AGE_HEAD = re.compile(
    r"\bi(?:\s+am|['’]m)\s+(?:now\s+)?\d{1,3}(?:\s+years\s+old)?(?:,?\s+not\s+\d{1,3})?",
    re.IGNORECASE,
)
_QUESTION_START = re.compile(
    r"^(?:(?:what|where|when|why|who|how)\s+(?:is|are|do|does|did|can|would|will|was|were|have|has)\b"
    r"|(?:do|does|did|are|is|can|could|would|will|have|has)\s+(?:you|u|we|he|she|they)\b)",
    re.IGNORECASE,
)
_FILLERS = frozenset({"yes", "yeah", "yep", "ok", "okay", "sure", "for sure", "i think", "just kidding", "just joking"})
_RETRACTIONS = frozenset({"just kidding", "just joking", "i was kidding", "i was joking"})
_CORRECTION_CUE = re.compile(r"\b(?:correction|actually|instead|now|changed|no\s+longer|not\s+anymore)\b", re.IGNORECASE)
_CORRECTION_PREFIX = re.compile(r"^(?:correction|actually)\s*[:,]\s*$", re.IGNORECASE)
_PROFILE_KINDS: tuple[FactKind, ...] = tuple(kind for kind in _KINDS if kind not in ("question", "observation"))
_INTERACTION_KINDS: tuple[FactKind, ...] = (
    "preference", "boundary", "self_report", "observation", "story", "interest", "commitment", "question", "other",
)


def _plain_marker(text: str) -> str:
    """Match standalone phrases only, never delete words from longer statements."""
    return " ".join(text.casefold().split()).strip(".,!?;: ")


def _question(text: str) -> bool:
    """Conservative negative guard, not a general natural-language classifier."""
    return "?" in text or _QUESTION_START.match(text.lstrip()) is not None


class SelectionError(ValueError):
    """Allowlisted metadata only; never include a supplied ID or excerpt."""

    def __init__(self, reason: Literal[
        "selection_input_capacity", "selection_source_unknown", "selection_target_unknown",
        "selection_choice_invalid",
    ]) -> None:
        super().__init__(reason)
        self.reason = reason


class AddSelection(BaseModel):
    """A new detail whose profile owner is derived from its selected source."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["add"]
    source_id: SourceId
    scope: Scope
    kind: FactKind


class ReplaceSelection(BaseModel):
    """A correction naming an existing target, never a nullable target."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["replace"]
    source_id: SourceId
    target_id: SourceId
    kind: FactKind


class RemoveSelection(BaseModel):
    """A resolved open thread; code derives its section and kind from the target."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["remove"]
    source_id: SourceId
    target_id: SourceId


class RelationshipSelection(BaseModel):
    """A rapport proposal supported by a current source excerpt."""

    model_config = ConfigDict(extra="forbid")

    stage: RelationshipStage
    source_id: SourceId


class SelectionDelta(BaseModel):
    """Local validation remains mandatory even with constrained decoding."""

    model_config = ConfigDict(extra="forbid")

    operations: list[Annotated[AddSelection | ReplaceSelection | RemoveSelection, Field(discriminator="action")]] = Field(
        default_factory=list, max_length=MAX_MEMORY_OPERATIONS,
    )
    relationship: RelationshipSelection | None = None


@dataclass(frozen=True)
class Excerpt:
    """Exact source proof and syntactic eligibility, not an assertion of truth."""

    evidence: FactEvidence
    age_eligible: bool
    age_only: bool = False
    excluded_reason: Literal["explicit_age_retraction"] | None = None
    correction_signalled: bool = False

    @property
    def usage(self) -> Literal["general", "age", "question", "context_only"]:
        """Disjoint source categories prevent escaping age checks with another kind."""
        if self.excluded_reason or _plain_marker(self.evidence.quote) in _FILLERS:
            return "context_only"
        if self.age_only or (self.age_eligible and _AGE_DECLARATION.fullmatch(self.evidence.quote.strip())):
            return "age"
        if _question(self.evidence.quote):
            return "question"
        return "general"

    def allowed_adds(self) -> tuple[tuple[Scope, FactKind], ...]:
        """One rule table is used by both decoding and local selection validation."""
        usage = self.usage
        if usage == "context_only":
            return ()
        if usage == "age":
            return (("profile", "age"),)
        if usage == "question":
            return (("interaction", "question"), ("open_threads", "question"))
        return tuple(
            [("profile", kind) for kind in _PROFILE_KINDS if kind != "age" or self.age_eligible]
            + [("interaction", kind) for kind in _INTERACTION_KINDS]
            + [("open_threads", kind) for kind in ("question", "commitment")]
        )

    @property
    def relationship_eligible(self) -> bool:
        return self.usage == "general" and not self.age_eligible


def _pieces(text: str) -> list[str]:
    """Partition without losing a character or cutting a word/numeric token.

    Oversized indivisible tokens remain visible as context without a selectable
    ID. We never truncate one to turn it into plausible evidence.
    """
    clauses: list[str] = []
    start = 0
    for boundary in _CLAUSE_BOUNDARY.finditer(text):
        # A space after numeric punctuation does not necessarily end the value.
        before, after = text[:boundary.start()].rstrip(), text[boundary.end():].lstrip()
        if before and after and before[-1].isdigit() and after[0].isdigit():
            continue
        clauses.append(text[start:boundary.end()])
        start = boundary.end()
    clauses.append(text[start:])
    pieces: list[str] = []
    for clause in clauses:
        while len(clause.strip()) > MAX_EXCERPT_CHARS:
            # Do not consume leading whitespace as an empty chunk repeatedly.
            limit = len(clause) - len(clause.lstrip()) + MAX_EXCERPT_CHARS
            boundaries = [
                match for match in re.finditer(r"\s+", clause)
                if not (
                    match.start() > 0 and match.end() < len(clause)
                    and clause[match.start() - 1].isdigit() and clause[match.end()].isdigit()
                )
            ]
            before = [match.end() for match in boundaries if 0 < match.start() <= limit]
            end = before[-1] if before else next((match.end() for match in boundaries if match.start() > 0), len(clause))
            pieces.append(clause[:end])
            clause = clause[end:]
        if clause:
            pieces.append(clause)
    return pieces


def _object(properties: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """Use explicit alternatives rather than conditionals ignored by grammars."""
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def _const(value: str) -> dict[str, JsonValue]:
    return {"type": "string", "const": value}


def _enum(values: list[str]) -> dict[str, JsonValue]:
    return {"type": "string", "enum": values}


@dataclass(frozen=True)
class SelectionPlan:
    """Request-local registries; never cached across accounts, chats or batches."""

    payload: str
    sources: dict[str, Excerpt]
    targets: dict[str, MemoryFact]
    # (action, target, kind) -> eligible source IDs. Shared by grammar and resolver.
    choices: dict[tuple[str, str, FactKind], tuple[str, ...]]

    def schema(self) -> dict[str, JsonValue]:
        """Only compatible action/target/source combinations can be decoded."""
        variants: list[JsonValue] = []
        additions: dict[tuple[Scope, FactKind], list[str]] = {}
        for source_id, source in self.sources.items():
            for scope, kind in source.allowed_adds():
                additions.setdefault((scope, kind), []).append(source_id)
        grouped_additions: dict[tuple[Scope, tuple[str, ...]], list[str]] = {}
        for (scope, kind), ids in additions.items():
            grouped_additions.setdefault((scope, tuple(ids)), []).append(kind)
        for (scope, ids), kinds in grouped_additions.items():
            # Choose the type/owner before evidence, rather than commit to an
            # arbitrary source ID first and then invent a compatible label.
            variants.append(_object({
                "action": _const("add"), "scope": _const(scope), "kind": _enum(kinds),
                "source_id": _enum(list(ids)),
            }))
        grouped: dict[tuple[str, str, tuple[str, ...]], list[str]] = {}
        for (action, target, kind), ids in self.choices.items():
            grouped.setdefault((action, target, ids), []).append(kind)
        for (action, target, ids), kinds in grouped.items():
            properties: dict[str, JsonValue] = {
                "action": _const(action), "target_id": _const(target),
            }
            if action == "replace":
                properties["kind"] = _enum(kinds)
            properties["source_id"] = _enum(list(ids))
            variants.append(_object(properties))
        operations: dict[str, JsonValue] = {"type": "array", "maxItems": MAX_MEMORY_OPERATIONS}
        if variants:
            operations["items"] = {"anyOf": variants}
        else:
            operations["maxItems"] = 0
            operations["items"] = {"type": "object"}
        relationship: dict[str, JsonValue] = {"type": "null"}
        rapport_sources = [key for key, source in self.sources.items() if source.relationship_eligible]
        if rapport_sources:
            relationship = {"anyOf": [{"type": "null"}, _object({
                "stage": _enum(["unknown", "new", "acquaintance", "familiar", "strained"]),
            "source_id": _enum(rapport_sources),
            })]}
        schema = _object({"operations": operations, "relationship": relationship})
        if len(json.dumps(schema).encode("utf-8")) > MAX_SELECTION_SCHEMA_BYTES:
            raise SelectionError("selection_input_capacity")
        return schema

    def resolve(self, selection: SelectionDelta) -> MemoryDelta:
        """Supply exact code-owned evidence and ownership; never repair choices."""
        operations: list[MemoryOperation] = []
        for selected in selection.operations:
            source = self.sources.get(selected.source_id)
            if source is None:
                raise SelectionError("selection_source_unknown")
            evidence = source.evidence
            target_id: str | None = None
            if isinstance(selected, AddSelection):
                if (selected.scope, selected.kind) not in source.allowed_adds():
                    raise SelectionError("selection_choice_invalid")
                section: FactSection = (
                    "agent" if evidence.sender_type == "me" else "interlocutor"
                ) if selected.scope == "profile" else selected.scope
                kind = selected.kind
            else:
                target = self.targets.get(selected.target_id)
                if target is None:
                    raise SelectionError("selection_target_unknown")
                kind = selected.kind if isinstance(selected, ReplaceSelection) else target.kind
                if selected.source_id not in self.choices.get((selected.action, selected.target_id, kind), ()):
                    raise SelectionError("selection_choice_invalid")
                section, target_id = target.section, target.id
            operations.append(MemoryOperation(
                action=selected.action, section=section, kind=kind, target_id=target_id,
                message_id=evidence.message_id, quote=evidence.quote,
            ))
        relationship: RelationshipChange | None = None
        if selection.relationship is not None:
            source = self.sources.get(selection.relationship.source_id)
            if source is None:
                raise SelectionError("selection_source_unknown")
            if not source.relationship_eligible:
                raise SelectionError("selection_choice_invalid")
            relationship = RelationshipChange(
                stage=selection.relationship.stage,
                message_id=source.evidence.message_id, quote=source.evidence.quote,
            )
        return MemoryDelta(operations=operations, relationship=relationship)


def build_selection_plan(request: SummarizeContextRequest) -> SelectionPlan:
    """Build exact excerpts while keeping every message visible in chronological order."""
    sources: dict[str, Excerpt] = {}
    rows: list[JsonValue] = []
    units: list[tuple[SenderType, str, list[str]]] = []
    source_parts: dict[str, dict[str, JsonValue]] = {}

    def add_source(message: Message, quote: str, *, age_only: bool = False) -> str:
        if len(sources) >= MAX_EXCERPTS:
            raise SelectionError("selection_input_capacity")
        assert message.raw_id is not None and message.sender_type != "system"
        key = f"s{len(sources)}"
        sources[key] = Excerpt(
            FactEvidence(message_id=message.raw_id, sender_type=message.sender_type, quote=quote),
            age_eligible=_is_age_declaration(quote), age_only=age_only,
        )
        return key

    for message in request.messages:
        parts: list[JsonValue] = []
        correction_prefix = False
        for piece in _pieces(message.text):
            quote = piece.strip()
            part: dict[str, JsonValue] = {"text": piece}
            source_ids: list[str] = []
            if (
                message.sender_type != "system" and message.raw_id is not None
                and message.raw_id.strip() and len(message.raw_id) <= 128
                and 0 < len(quote) <= MAX_EXCERPT_CHARS
            ):
                source_id = add_source(message, quote)
                part["id"] = source_id
                source_ids.append(source_id)
                source_parts[source_id] = part
                # Only shorten an already-valid declaration. Do not cut a question
                # or retraction down until it passes the existing age validator.
                head = _AGE_HEAD.search(quote) if sources[source_id].age_eligible else None
                if head is not None and head.group() != quote and _is_age_declaration(head.group()):
                    age_id = add_source(message, head.group(), age_only=True)
                    age_part: dict[str, JsonValue] = {"id": age_id, "text": head.group()}
                    part["age_excerpt"] = age_part
                    source_ids.append(age_id)
                    source_parts[age_id] = age_part
            parts.append(part)
            if quote:
                signalled = correction_prefix or _CORRECTION_CUE.search(quote) is not None
                for key in source_ids:
                    sources[key] = replace(sources[key], correction_signalled=signalled)
                # Only inherit an explicit adjacent prefix within this message,
                # not an unrelated 'now' or a different speaker's suggestion.
                correction_prefix = _CORRECTION_PREFIX.fullmatch(quote) is not None
                units.append((message.sender_type, quote, source_ids))
        rows.append({"speaker": {"me": "AGENT", "other": "INTERLOCUTOR", "system": "SERVICE_EVENT"}[message.sender_type], "parts": parts})

    # Only an adjacent, explicit same-speaker age/retraction/corrected-age
    # sequence is resolved by code. Other retractions still require judgement.
    for first, retraction, correction in zip(units, units[1:], units[2:]):
        if (
            first[0] == retraction[0] == correction[0] != "system"
            and _is_age_declaration(first[1]) and _is_age_declaration(correction[1])
            and _plain_marker(retraction[1]) in _RETRACTIONS
        ):
            for key in first[2]:
                sources[key] = replace(sources[key], excluded_reason="explicit_age_retraction")
    for key, source in sources.items():
        if source.usage != "general":
            source_parts[key]["use"] = source.usage

    previous = request.previous or MemoryContent()
    targets = {f"t{index}": fact for index, fact in enumerate(previous.facts)}
    choices: dict[tuple[str, str, FactKind], tuple[str, ...]] = {}
    target_rows: list[JsonValue] = []
    for target_id, target in targets.items():
        kinds = _KINDS if target.evidence is None and target.kind == "other" else (target.kind,)
        for action in ("replace", "remove"):
            if action == "remove" and (target.section != "open_threads" or target.kind not in ("question", "commitment")):
                continue
            for kind in kinds if action == "replace" else (target.kind,):
                ids: list[str] = []
                for source_id, source in sources.items():
                    scope: Scope = "profile" if target.section in ("agent", "interlocutor") else target.section
                    if action == "replace":
                        if (scope, kind) not in source.allowed_adds():
                            continue
                        # Same speaker/kind and a newer message do NOT prove a
                        # correction. For non-age claims require a visible cue.
                        # This is a necessary heuristic, not semantic proof.
                        if kind != "age" and not source.correction_signalled:
                            continue
                    elif source.usage in ("question", "context_only") or source.age_only:
                        continue
                    owner = {"agent": "me", "interlocutor": "other"}.get(target.section)
                    if owner is not None and source.evidence.sender_type != owner:
                        continue
                    operation = MemoryOperation(
                        action=action, section=target.section, kind=kind, target_id=target.id,
                        message_id=source.evidence.message_id, quote=source.evidence.quote,
                    )
                    try:
                        _check_target(operation, source.evidence, target, previous.facts)
                    except MemoryUpdateError:
                        continue
                    ids.append(source_id)
                if ids:
                    choices[(action, target_id, kind)] = tuple(ids)
        target_rows.append({
            "id": target_id, "section": target.section, "kind": target.kind, "text": target.text,
            "speaker": target.evidence.sender_type if target.evidence is not None else "legacy-unverified",
        })
    payload = json.dumps({
        "previous": {"facts": target_rows, "relationship": previous.relationship.model_dump(mode="json")},
        "messages": rows,
    }, ensure_ascii=False, separators=(",", ":"))
    if len(payload.encode("utf-8")) > MAX_SELECTION_INPUT_BYTES:
        raise SelectionError("selection_input_capacity")
    return SelectionPlan(payload, sources, targets, choices)