"""Test-only participant extraction prototype; never imported by the API.

Three focused passes and one coverage review are a hypothesis to evaluate, not
a proven fix. The original source, target, capacity and atomic-merge checks are
retained. No browser, persistence, private transcript or live replies are used.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, replace
from typing import Literal

from ollama import Client
from pydantic import JsonValue, ValidationError

from responser_model_api.config import OLLAMA_HOST, SummarySettings
from responser_model_api.context_summarizer import ContextSummaryError, SUMMARY_INSTRUCTIONS, _completion_content
from responser_model_api.memory_updates import MemoryDelta, merge_memory_delta, migrate_legacy
from responser_model_api.schemas import (
    FactKind, MemoryContent, SummarizeContextRequest, SummarizeContextResponse,
)
from responser_model_api.source_selection import (
    Excerpt, MAX_SELECTION_INPUT_BYTES, Scope, SelectionDelta, SelectionPlan, build_selection_plan,
)

Focus = Literal["interlocutor", "agent", "interaction"]
PROFILE_FOCI: tuple[Focus, ...] = ("interlocutor", "agent")


@dataclass(frozen=True)
class FocusedExcerpt(Excerpt):
    """Narrow decoding AND local validation to the same pass scope."""

    focus: Focus = "interaction"

    def allowed_adds(self) -> tuple[tuple[Scope, FactKind], ...]:
        return tuple((scope, kind) for scope, kind in super().allowed_adds()
                     if (scope == "profile") == (self.focus != "interaction"))

    @property
    def relationship_eligible(self) -> bool:
        return self.focus == "interaction" and super().relationship_eligible


def focused_plan(plan: SelectionPlan, focus: Focus) -> SelectionPlan:
    """Keep all conversation text, but only let the pass select owned evidence."""
    speaker = {"agent": "me", "interlocutor": "other"}.get(focus)
    sources = {
        key: FocusedExcerpt(
            evidence=source.evidence, age_eligible=source.age_eligible, age_only=source.age_only,
            excluded_reason=source.excluded_reason, correction_signalled=source.correction_signalled,
            focus=focus,
        ) for key, source in plan.sources.items()
        if speaker is None or source.evidence.sender_type == speaker
    }
    targets = {key: target for key, target in plan.targets.items() if (
        target.section in ("interaction", "open_threads") if focus == "interaction" else target.section == focus
    )}
    choices = {}
    for (action, target, kind), ids in plan.choices.items():
        eligible = tuple(key for key in ids if key in sources)
        if target in targets and eligible:
            choices[(action, target, kind)] = eligible
    return replace(plan, sources=sources, targets=targets, choices=choices)


def _focus_instructions(focus: Focus) -> str:
    if focus == "interaction":
        return (
            "CURRENT PASS: interaction and open threads ONLY. Profiles are handled separately. "
            "Preserve explicit reactions, boundaries, apologies and requests for attentive follow-up. "
            "No profile facts or resolved questions. Return a no-op if no relevant changes exist."
        )
    speaker = "INTERLOCUTOR" if focus == "interlocutor" else "AGENT"
    return (
        f"CURRENT PASS: ONLY {speaker}'s own profile, regardless of which person speaks more. "
        "The other participant's messages are context, never this pass's evidence. "
        "Inspect each relevant excerpt for age, name, occupation, work specialty, interests, "
        "preferences, boundaries, behavioral self-reports, and stories. Extract all distinct "
        "durable details, not a representative sample. A job and its specialty are distinct. "
        "A hobby/story is not replaced by an age fact. Respect explicit retractions and corrections. "
        "Return no-op only if none of these categories changed."
    )


def _review_payload(plan: SelectionPlan) -> str:
    """Explicit participant/category coverage inventory for the final review."""
    data = json.loads(plan.payload)
    data["coverage_review"] = {
        "categories": ["age", "name", "occupation", "specialty", "interest", "preference", "boundary", "self_report", "story"],
        "current_profiles": {
            focus: [{"kind": target.kind, "text": target.text} for target in plan.targets.values() if target.section == focus]
            for focus in PROFILE_FOCI
        },
        "instruction": "Compare each participant's evidence against each category; add only durable omissions not already represented.",
    }
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    if len(payload.encode("utf-8")) > MAX_SELECTION_INPUT_BYTES:
        raise ContextSummaryError("coverage input capacity", reason="selection_input_capacity")
    return payload


class ParticipantExperiment:
    """At most four calls; all results stay local until every check succeeds."""

    def __init__(self, settings: SummarySettings, host: str = OLLAMA_HOST, client: Client | None = None) -> None:
        self.settings = settings
        self.client = client if client is not None else Client(host=host)
        self.calls = 0
        self.pass_metrics: list[dict[str, JsonValue]] = []

    def _extract(self, plan: SelectionPlan, instructions: str, name: str) -> MemoryDelta:
        self.calls += 1
        started = time.monotonic()
        # Test-only progress is metadata, not transcript/output. A four-call
        # experiment on CPU must not appear stalled until its final score.
        print("PARTICIPANT_PASS", json.dumps({"pass": name, "state": "started", "call": self.calls}), flush=True)
        response = self.client.chat(
            model=self.settings.model_name,
            messages=[{"role": "system", "content": SUMMARY_INSTRUCTIONS + "\n\n" + instructions},
                      {"role": "user", "content": plan.payload}],
            format=plan.schema(), stream=False, tools=[], keep_alive=0,
            options={"temperature": 0.1, "num_predict": self.settings.max_output_tokens, "num_ctx": self.settings.context_window},
        )
        self.pass_metrics.append({"pass": name, "prompt_tokens": response.prompt_eval_count, "output_tokens": response.eval_count})
        try:
            selection = SelectionDelta.model_validate_json(_completion_content(response), strict=True)
        except ValidationError:
            raise ContextSummaryError("invalid prototype selection", reason="delta_schema_invalid") from None
        delta = plan.resolve(selection)
        print("PARTICIPANT_PASS", json.dumps({
            "pass": name, "state": "validated", "call": self.calls,
            "elapsed_seconds": round(time.monotonic() - started, 1), "operations": len(delta.operations),
        }), flush=True)
        return delta

    def summarize(self, request: SummarizeContextRequest) -> SummarizeContextResponse:
        previous = migrate_legacy(request.previous) if request.previous is not None else None
        prepared = SummarizeContextRequest(previous=previous, messages=request.messages)
        plan = build_selection_plan(prepared)
        combined = MemoryDelta()
        for focus in (*PROFILE_FOCI, "interaction"):
            scoped = focused_plan(plan, focus)
            if not any(source.allowed_adds() or source.relationship_eligible for source in scoped.sources.values()):
                continue
            delta = self._extract(scoped, _focus_instructions(focus), focus)
            combined = MemoryDelta(
                operations=[*combined.operations, *delta.operations],
                relationship=delta.relationship if focus == "interaction" else combined.relationship,
            )
        # Validate the combined plan against the original registry, not partly
        # updated targets from another pass. A failure returns no partial memory.
        memory = merge_memory_delta(previous, combined, request.messages)
        review = build_selection_plan(SummarizeContextRequest(previous=memory, messages=request.messages))
        review = replace(review, payload=_review_payload(review))
        delta = self._extract(
            review,
            "FINAL COVERAGE REVIEW: current facts are the provisional result of focused passes. "
            "Check both profiles category by category against the visible messages. "
            "Add missing durable facts without repeating existing ones. Preserve all unrelated facts. "
            "Do not replace a correctly extracted fact with different wording. "
            "If nothing is missing, return operations: [] and relationship: null.",
            "coverage",
        )
        memory = merge_memory_delta(memory, delta, request.messages)
        return SummarizeContextResponse(memory=memory, model_name=self.settings.model_name)