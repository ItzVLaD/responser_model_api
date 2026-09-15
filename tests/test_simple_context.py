"""Compact-summary boundaries and busy-request regression tests; no live model."""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import httpx
import pytest
from fastapi.testclient import TestClient
from ollama import Client

from responser_model_api import app as app_module
from responser_model_api import simple_context as simple_module
from responser_model_api.config import SummarySettings
from responser_model_api.context_summarizer import ContextSummaryError
from responser_model_api.schemas import ChatDescriptor, ChatSnapshot, GeneratedReply, MemoryContent, MemoryFact, Message, SummarizeContextRequest, SummarizeContextResponse
from responser_model_api.simple_context import SIMPLE_SUMMARY_INSTRUCTIONS, SimpleContextSummarizer, SimpleSummary, plain_memory_view, simple_input


def _summary() -> dict[str, object]:
    return {"interlocutor": ["Age 32; librarian.", "Prefers clear follow-up questions."],
            "agent": ["Age 26; photographer."], "interaction": [], "open_threads": [],
            "relationship": {"stage": "unknown", "evidence": ""}}


def _request(previous: MemoryContent | None = None) -> SummarizeContextRequest:
    return SummarizeContextRequest(previous=previous, messages=[
        Message(sender_type="other", raw_id="PRIVATE_SOURCE", text="I'm 32, a librarian. Please ask clear follow-up questions."),
        Message(sender_type="me", raw_id="PRIVATE_AGENT", text="I'm 26 and a photographer."),
    ])


def _stub(monkeypatch: pytest.MonkeyPatch, content: str, *, done_reason: str = "stop",
          network_error: httpx.TransportError | None = None) -> tuple[SimpleContextSummarizer, list[httpx.Request]]:
    calls: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if network_error is not None:
            raise network_error
        return httpx.Response(200, json={"message": {"role": "assistant", "content": content},
                                       "done": True, "done_reason": done_reason, "prompt_eval_count": 100, "eval_count": 80})

    def client(host: str) -> Client:
        return Client(host=host, transport=httpx.MockTransport(handle))

    monkeypatch.setattr(simple_module, "Client", client)
    summarizer = SimpleContextSummarizer(SummarySettings())
    monkeypatch.setattr(app_module, "_summarizer", summarizer)
    return summarizer, calls


def test_simple_contract_has_only_four_lists_and_relationship() -> None:
    schema = SimpleSummary.model_json_schema()
    assert set(schema["properties"]) == {"interlocutor", "agent", "interaction", "open_threads", "relationship"}
    assert set(schema["required"]) == set(schema["properties"])
    for field in ("interlocutor", "agent", "interaction", "open_threads"):
        assert schema["properties"][field]["maxItems"] == 8
        assert schema["properties"][field]["items"]["maxLength"] == 200


def test_endpoint_uses_plain_summary_with_previous_and_original_roles(monkeypatch: pytest.MonkeyPatch) -> None:
    previous = MemoryContent(agent=["Existing hobby"], interlocutor=["Existing occupation"])
    request = _request(previous)
    before = request.model_dump_json()
    _, calls = _stub(monkeypatch, json.dumps(_summary()))
    response = TestClient(app_module.app).post("/summarize_context", json=request.model_dump())
    assert response.status_code == 200
    result = SummarizeContextResponse.model_validate(response.json())
    assert result.memory.model_dump(exclude={"facts", "relationship_source"}) == _summary()
    assert result.memory.facts == [] and result.memory.relationship_source is None
    assert request.model_dump_json() == before and len(calls) == 1
    model_request = json.loads(calls[0].content)
    assert model_request["messages"][1]["content"] == simple_input(request)
    payload = json.loads(model_request["messages"][1]["content"])
    assert payload["previous"] == plain_memory_view(previous)
    assert payload["messages"] == [{"speaker": "INTERLOCUTOR", "text": request.messages[0].text},
                                    {"speaker": "AGENT", "text": request.messages[1].text}]
    assert "PRIVATE_SOURCE" not in json.dumps(model_request)
    assert "operations" not in model_request["format"]["properties"]
    assert model_request["stream"] is False
    assert model_request["options"] == {"temperature": 0.1, "num_predict": 2048, "num_ctx": 8192}


def test_all_legacy_fact_text_reaches_input_without_ids_proof_or_stale_previews() -> None:
    memory = MemoryContent(agent=["STALE_PREVIEW"], facts=[
        MemoryFact(id=f"PRIVATE_FACT_{i}", section="interlocutor", kind="other", text=f"Old detail {i}") for i in range(12)
    ])
    before = memory.model_dump_json()
    payload = simple_input(_request(memory))
    assert "STALE_PREVIEW" not in payload and "PRIVATE_" not in payload
    assert len(json.loads(payload)["previous"]["interlocutor"]) == 12
    assert memory.model_dump_json() == before


@pytest.mark.parametrize("content", ["", "{}", "[]", "not json", '{"operations":[]}',
                                        json.dumps({**_summary(), "facts": []}),
                                        json.dumps({**_summary(), "agent": [" "]}),
                                        json.dumps({**_summary(), "agent": ["x" * 201]}),
                                        json.dumps({**_summary(), "agent": ["x"] * 9}),
                                        json.dumps({**_summary(), "interlocutor": ["x" * 200] * 8, "agent": ["y" * 200] * 8})])
def test_invalid_or_oversized_summary_fails_without_reset_retry_or_private_leak(
    monkeypatch: pytest.MonkeyPatch, content: str, caplog: pytest.LogCaptureFixture,
) -> None:
    _, calls = _stub(monkeypatch, content)
    previous = MemoryContent(interlocutor=["PRIVATE_OLD_MEMORY"])
    original = previous.model_dump_json()
    monkeypatch.setattr(simple_module.log, "propagate", True)
    caplog.set_level(logging.DEBUG, logger=simple_module.log.name)
    response = TestClient(app_module.app).post("/summarize_context", json=_request(previous).model_dump())
    assert response.status_code == 502 and set(response.json()) == {"detail"}
    assert previous.model_dump_json() == original and len(calls) == 1
    assert "PRIVATE_" not in response.text + caplog.text


def test_truncated_valid_json_is_still_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _, calls = _stub(monkeypatch, json.dumps(_summary()), done_reason="length")
    response = TestClient(app_module.app).post("/summarize_context", json=_request().model_dump())
    assert response.status_code == 502 and "output_truncated" in response.text and len(calls) == 1


def test_no_change_summary_keeps_previous_simple_lists(monkeypatch: pytest.MonkeyPatch) -> None:
    previous = MemoryContent.model_validate(_summary())
    summarizer, _ = _stub(monkeypatch, json.dumps(_summary()))
    assert summarizer.summarize(_request(previous)).memory == previous


def test_prompt_has_criteria_without_fictional_profile_examples() -> None:
    assert "ILLUSTRATIVE EXAMPLES" not in SIMPLE_SUMMARY_INSTRUCTIONS
    for fictional in ("Age 24", "lab technician", "landscape designer", "roof gardens", "kayaking", "I'm 29", "I'm 73"):
        assert fictional.casefold() not in SIMPLE_SUMMARY_INSTRUCTIONS.casefold()
    for concept in ("job title AND what they work on", "BOTH participants", "not an empty reset", "Questions about a job are NOT occupations"):
        assert concept in SIMPLE_SUMMARY_INSTRUCTIONS
    assert "landscape" not in simple_input(_request())


def test_prompt_prioritizes_current_declarations_without_erasing_unrelated_notes() -> None:
    prompt = " ".join(SIMPLE_SUMMARY_INSTRUCTIONS.split())
    for rule in (
        "previous summary is fallible generated notes, NOT verified source evidence",
        "current self-declaration overrides conflicting previous notes",
        "even without an explicit correction word",
        "do not retain both values as if both were current",
        "Preserve useful previous details that are NOT contradicted or retracted",
        "does not replace unrelated work details",
    ):
        assert rule in prompt


def test_conflicting_previous_notes_are_separate_from_current_source(monkeypatch: pytest.MonkeyPatch) -> None:
    previous = MemoryContent(agent=["Age 24; lab technician.", "Enjoys cycling."], interlocutor=["Librarian."])
    original = previous.model_dump_json()
    updated = {**_summary(), "agent": ["Age 26; photographer.", "Enjoys cycling."]}
    summarizer, calls = _stub(monkeypatch, json.dumps(updated))
    result = summarizer.summarize(_request(previous))
    sent = json.loads(calls[0].content)["messages"]
    assert "lab technician" not in sent[0]["content"]
    source = json.loads(sent[1]["content"])
    assert source["previous"]["agent"] == previous.agent
    assert source["messages"][1] == {"speaker": "AGENT", "text": "I'm 26 and a photographer."}
    assert result.memory.agent == updated["agent"]
    assert previous.model_dump_json() == original


def test_summary_guidance_preserves_shared_events_without_moralizing_or_forcing_closeness() -> None:
    instructions = " ".join(SIMPLE_SUMMARY_INSTRUCTIONS.split())
    for rule in (
        "significant shared events", "adult virtual sexual role-play", "non-graphic",
        "Do not turn virtual events into real-world encounters",
        "A request is not mutual participation", "not ongoing consent",
        "Reassess stage on every update", "Earlier conflict alone must not freeze",
        "Sexual content alone is not evidence of strain",
        "closeness and tension can coexist", "do not invent a reconciliation",
        "unknown name is not an open thread",
    ):
        assert rule.casefold() in instructions.casefold()


@pytest.mark.parametrize("stage", ["familiar", "strained"])
def test_neutral_intimacy_summary_is_preserved_with_evidence_based_stage(
    monkeypatch: pytest.MonkeyPatch, stage: str,
) -> None:
    summary = {
        **_summary(),
        "interaction": [
            "Both adults acknowledged participating in virtual sexual role-play; this was online, not an in-person encounter.",
            "A stated boundary remains in effect; past participation does not establish ongoing consent.",
        ],
        "relationship": {"stage": stage, "evidence": "Shared closeness and a specific unresolved disagreement coexist."},
    }
    previous = MemoryContent.model_validate(summary)
    before = previous.model_dump_json()
    summarizer, calls = _stub(monkeypatch, json.dumps(summary))
    result = summarizer.summarize(SummarizeContextRequest(previous=previous, messages=[
        Message(sender_type="other", text="Thanks for remembering what we discussed.", raw_id="PRIVATE_CURRENT_ID"),
    ]))
    assert result.memory.interaction == summary["interaction"]
    assert result.memory.relationship.stage == stage  # No keyword-forced stage rewrite.
    sent = json.loads(json.loads(calls[0].content)["messages"][1]["content"])
    assert sent["previous"] == plain_memory_view(previous)
    assert previous.model_dump_json() == before and len(calls) == 1


def test_existing_strained_summary_can_be_revised_without_changing_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    previous = MemoryContent(interlocutor=["Librarian."], interaction=["An earlier disagreement."],
                             relationship={"stage": "strained", "evidence": "They argued earlier."})
    updated = {**_summary(), "interaction": ["They acknowledged the earlier disagreement and expressed renewed mutual trust."],
               "relationship": {"stage": "familiar", "evidence": "Both explicitly expressed renewed trust after resolving the disagreement."}}
    summarizer, _ = _stub(monkeypatch, json.dumps(updated))
    result = summarizer.summarize(_request(previous))
    assert result.memory.relationship.stage == "familiar"
    assert previous.relationship.stage == "strained"


def test_simple_private_trace_records_real_model_stages(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    directory = tmp_path.resolve() / "private-traces"
    monkeypatch.setenv("RESPONSER_CONTEXT_TRACE", "true")
    monkeypatch.setenv("RESPONSER_CONTEXT_TRACE_DIR", str(directory))
    _, calls = _stub(monkeypatch, json.dumps(_summary()))
    response = TestClient(app_module.app).post("/summarize_context", json=_request().model_dump())
    assert response.status_code == 200
    events = [json.loads(line) for line in next(directory.glob("*.jsonl")).read_text().splitlines()]
    assert [event["stage"] for event in events] == ["started", "request_received", "previous_prepared", "inference_request",
                                                  "inference_response", "summary_validated", "review", "response", "completed"]
    assert events[3]["data"] == json.loads(calls[0].content)
    assert events[6]["data"]["sections"]["interlocutor"] == _summary()["interlocutor"]


def test_active_generation_rejects_new_model_work_but_not_health(monkeypatch: pytest.MonkeyPatch) -> None:
    entered, release = Event(), Event()

    def generate(snapshot: ChatSnapshot) -> GeneratedReply:
        entered.set()
        assert release.wait(timeout=10)
        return GeneratedReply(text="Synthetic result", model_name="stub")

    monkeypatch.setattr(app_module._generator, "generate", generate)
    request = {"snapshot": ChatSnapshot(chat=ChatDescriptor(raw_id="1", title="Test")).model_dump(), "dry_run": True}
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(lambda: TestClient(app_module.app).post("/generate_reply", json=request))
        try:
            assert entered.wait(timeout=5)
            client = TestClient(app_module.app)
            for endpoint, body in (("/generate_reply", request), ("/summarize_context", _request().model_dump())):
                busy = client.post(endpoint, json=body)
                assert busy.status_code == 503 and busy.headers["Retry-After"] == "30"
            assert client.get("/health").status_code == 200
        finally:
            release.set()
        assert first.result(timeout=10).status_code == 200
    assert TestClient(app_module.app).post("/generate_reply", json=request).status_code == 200


def test_inference_gate_released_after_summary_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub(monkeypatch, "", network_error=httpx.ReadTimeout("PRIVATE_TRANSPORT_ERROR"))
    client = TestClient(app_module.app)
    for _ in range(2):
        response = client.post("/summarize_context", json=_request().model_dump())
        assert response.status_code == 502 and "PRIVATE_" not in response.text