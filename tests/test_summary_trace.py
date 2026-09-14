"""Opt-in private tracing tests with synthetic data and no live inference."""

from __future__ import annotations

import json
import logging
import os
import stat
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from ollama import Client
from pydantic import JsonValue

from responser_model_api import app as app_module
from responser_model_api import context_summarizer as summary_module
from responser_model_api import summary_trace as trace_module
from responser_model_api.config import SummarySettings
from responser_model_api.context_summarizer import OllamaContextSummarizer
from responser_model_api.schemas import MemoryContent, Message, SummarizeContextRequest
from responser_model_api.summary_trace import SummaryTrace, SummaryTraceError, SummaryTraceSettings, load_summary_trace_settings


def _request() -> SummarizeContextRequest:
    return SummarizeContextRequest(previous=MemoryContent(agent=["PRIVATE_OLD_FACT"]), messages=[
        Message(raw_id="PRIVATE_AGENT", sender_type="me", text="I visited PRIVATE_MUSEUM."),
        Message(raw_id="PRIVATE_OTHER", sender_type="other", text="I design PRIVATE_GARDENS."),
        Message(raw_id="PRIVATE_STORY", sender_type="me", text="I enjoy quiet weekends."),
    ])


def _output() -> dict[str, JsonValue]:
    # A museum visit misclassified as occupation is still semantically wrong.
    # Negative source guards do not prove every non-question states a job.
    return {"operations": [
        {"action": "add", "source_id": "s0", "scope": "profile", "kind": "occupation"},
        {"action": "add", "source_id": "s2", "scope": "interaction", "kind": "interest"},
    ], "relationship": None}


def _setup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, enabled: bool = True,
    output: dict[str, JsonValue] | None = None, content: str | None = None,
    done_reason: str = "stop", network_error: httpx.TransportError | None = None,
    inspect_before_response: Callable[[], None] | None = None,
) -> tuple[TestClient, list[httpx.Request], Path]:
    directory = tmp_path.resolve() / "traces"
    monkeypatch.setenv("RESPONSER_CONTEXT_TRACE", "true" if enabled else "false")
    monkeypatch.setenv("RESPONSER_CONTEXT_TRACE_DIR", str(directory))
    monkeypatch.setenv("RESPONSER_LOG_PROMPTS", "true")
    calls: list[httpx.Request] = []
    response_content = content if content is not None else json.dumps(_output() if output is None else output)

    def handle(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if inspect_before_response is not None:
            inspect_before_response()
        if network_error is not None:
            raise network_error
        return httpx.Response(200, json={
            "model": "qwen2.5:7b", "message": {"role": "assistant", "content": response_content},
            "done": True, "done_reason": done_reason, "prompt_eval_count": 100, "eval_count": 50,
        })

    def client(host: str) -> Client:
        return Client(host=host, transport=httpx.MockTransport(handle))

    monkeypatch.setattr(summary_module, "Client", client)
    monkeypatch.setattr(app_module, "_summarizer", OllamaContextSummarizer(SummarySettings()))
    return TestClient(app_module.app), calls, directory


def _events(directory: Path) -> list[dict[str, JsonValue]]:
    files = list(directory.glob("summary-*.jsonl"))
    assert len(files) == 1
    return [json.loads(line) for line in files[0].read_text().splitlines()]


def test_full_trace_contains_every_phase_and_readable_wrong_classification(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    client, calls, directory = _setup(monkeypatch, tmp_path)
    monkeypatch.setattr(summary_module.log, "propagate", True)
    caplog.set_level(logging.DEBUG, logger=summary_module.log.name)
    request = _request()
    original = request.model_dump_json()
    response = client.post("/summarize_context", json=request.model_dump())
    assert response.status_code == 200 and len(calls) == 1
    events = _events(directory)
    assert [event["stage"] for event in events] == [
        "started", "request_received", "previous_prepared", "selection_plan", "inference_request",
        "inference_response", "selection_parsed", "resolved_delta", "merge_started", "merge_result",
        "review", "response", "completed",
    ]
    assert [event["sequence"] for event in events] == list(range(len(events)))
    summary_id = events[0]["summary_id"]
    assert all(event["summary_id"] == summary_id and event["trace_version"] == 1 for event in events)
    data = {event["stage"]: event["data"] for event in events}
    assert data["request_received"]["request"] == request.model_dump(mode="json")
    assert data["previous_prepared"]["previous"]["facts"][0]["evidence"] is None
    assert data["selection_plan"]["sources"]["s1"]["evidence"]["message_id"] == "PRIVATE_OTHER"
    assert data["selection_plan"]["targets"]["t0"]["text"] == "PRIVATE_OLD_FACT"
    assert data["inference_request"] == json.loads(calls[0].content)
    assert json.loads(data["inference_response"]["response"]["message"]["content"]) == _output()
    assert data["selection_parsed"]["selection"] == _output()
    assert "s1" in data["selection_parsed"]["unselected_source_ids"]
    assert {"source_id": "s1", "speaker": "other", "text": "I design PRIVATE_GARDENS."} in data["selection_parsed"]["unselected_excerpts"]
    assert data["resolved_delta"]["delta"]["operations"][0]["quote"] == "I visited PRIVATE_MUSEUM."
    assert data["resolved_delta"]["delta"]["operations"][1]["section"] == "interaction"
    assert data["review"]["sections"]["interlocutor"] == []
    assert "interlocutor" in data["review"]["empty_sections"]
    assert data["review"]["sections"]["agent"][-1] == {"kind": "occupation", "text": "I visited PRIVATE_MUSEUM."}
    assert data["review"]["semantic_correctness_and_completeness"] == "NOT_VALIDATED"
    assert len(data["review"]["added_fact_ids"]) == 2
    assert len(data["review"]["preserved_fact_ids"]) == 1
    assert data["response"]["response"] == response.json()
    assert data["response"]["reader_checkpoint"] == "not_written_by_api"
    assert request.model_dump_json() == original
    assert f"summary_id={summary_id}" in caplog.text
    assert all(text not in caplog.text for text in ("PRIVATE_", "I enjoy quiet weekends.", str(directory)))
    assert all(record.exc_info is None for record in caplog.records)
    trace_file = next(directory.glob("*.jsonl"))
    assert stat.S_IMODE(trace_file.stat().st_mode) == 0o600
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700


def test_disabled_tracing_creates_nothing_even_with_debug_and_prompt_logging(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    client, calls, directory = _setup(monkeypatch, tmp_path, enabled=False)
    monkeypatch.setattr(summary_module.log, "propagate", True)
    caplog.set_level(logging.DEBUG, logger=summary_module.log.name)
    response = client.post("/summarize_context", json=_request().model_dump())
    assert response.status_code == 200 and len(calls) == 1 and not directory.exists()
    assert "PRIVATE_" not in caplog.text


def test_enabling_trace_does_not_change_model_request_or_response(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    request = _request()
    plain, plain_calls, _ = _setup(monkeypatch, tmp_path / "disabled", enabled=False)
    before = plain.post("/summarize_context", json=request.model_dump())
    traced, trace_calls, _ = _setup(monkeypatch, tmp_path / "enabled")
    after = traced.post("/summarize_context", json=request.model_dump())
    assert before.json() == after.json()
    assert plain_calls[0].content == trace_calls[0].content


@pytest.mark.parametrize("failure,last_stage,reason", [
    ("malformed", "selection_schema_rejected", "delta_schema_invalid"),
    ("truncated", "inference_response", "output_truncated"),
    ("unknown", "selection_parsed", "selection_source_unknown"),
    ("wrong-kind", "selection_parsed", "selection_choice_invalid"),
])
def test_rejected_output_is_preserved_but_never_marked_successful(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    failure: str, last_stage: str, reason: str,
) -> None:
    output: dict[str, JsonValue] = {"operations": [
        {"action": "add", "source_id": "PRIVATE_ID" if failure == "unknown" else "s0", "scope": "profile", "kind": "age"},
    ], "relationship": None}
    client, _, directory = _setup(
        monkeypatch, tmp_path, output=output,
        content="PRIVATE_INVALID_JSON\n{not json" if failure == "malformed" else None,
        done_reason="length" if failure == "truncated" else "stop",
    )
    monkeypatch.setattr(summary_module.log, "propagate", True)
    caplog.set_level(logging.DEBUG, logger=summary_module.log.name)
    response = client.post("/summarize_context", json=_request().model_dump())
    assert response.status_code == 502 and reason in response.text
    events = _events(directory)
    assert events[-2]["stage"] == last_stage
    assert events[-1]["stage"] == "failed" and events[-1]["data"]["reason"] == reason
    assert not any(event["stage"] in {"completed", "merge_result", "response"} for event in events)
    assert any(event["stage"] == "inference_response" for event in events)
    assert "PRIVATE" not in caplog.text + response.text


def test_network_failure_keeps_flushed_input_without_exception_secrets(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    directory = tmp_path.resolve() / "traces"

    def inspect() -> None:
        assert _events(directory)[-1]["stage"] == "inference_request"

    client, _, _ = _setup(monkeypatch, tmp_path, network_error=httpx.ReadTimeout("PRIVATE_TRANSPORT_TOKEN"), inspect_before_response=inspect)
    response = client.post("/summarize_context", json=_request().model_dump())
    assert response.status_code == 502
    events = _events(directory)
    assert events[-1]["data"] == {"error_type": "ReadTimeout", "last_completed_stage": "inference_request"}
    assert not any(event["stage"] == "inference_response" for event in events)
    assert "PRIVATE_TRANSPORT_TOKEN" not in json.dumps(events) + response.text


def test_merge_rejection_keeps_actual_failing_operation_details(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    output: dict[str, JsonValue] = {"operations": [
        {"action": "replace", "target_id": "t0", "source_id": "s0", "kind": "occupation"},
        {"action": "replace", "target_id": "t0", "source_id": "s0", "kind": "occupation"},
    ], "relationship": None}
    client, _, directory = _setup(monkeypatch, tmp_path, output=output)
    request = _request()
    request.messages[0].text = "I now work in a PRIVATE_MUSEUM."
    original = request.model_dump_json()
    response = client.post("/summarize_context", json=request.model_dump())
    assert response.status_code == 502 and "target_repeated" in response.text
    events = _events(directory)
    detail = next(event["data"] for event in events if event["stage"] == "merge_rejected")
    assert detail["diagnostics"]["operation_index"] == 1
    assert detail["reason"] == "target_repeated"
    assert events[-1]["stage"] == "failed" and request.model_dump_json() == original


def test_preparation_failure_keeps_input_trace_without_inference(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from responser_model_api import source_selection

    monkeypatch.setattr(source_selection, "MAX_EXCERPTS", 1)
    client, calls, directory = _setup(monkeypatch, tmp_path)
    response = client.post("/summarize_context", json=_request().model_dump())
    assert response.status_code == 502 and not calls
    events = _events(directory)
    assert events[1]["stage"] == "request_received"
    assert events[-1]["data"]["reason"] == "selection_input_capacity"


@pytest.mark.parametrize("value", ["", "yes", "1", "enable", "PRIVATE_BAD_SETTING"])
def test_trace_opt_in_requires_explicit_boolean(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("RESPONSER_CONTEXT_TRACE", value)
    with pytest.raises(ValueError, match="must be true or false"):
        load_summary_trace_settings()


def test_trace_settings_loaded_at_startup_and_default_off(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("RESPONSER_CONTEXT_TRACE", raising=False)
    assert not load_summary_trace_settings().enabled
    client, _, directory = _setup(monkeypatch, tmp_path, enabled=False)
    monkeypatch.setenv("RESPONSER_CONTEXT_TRACE", "true")
    assert client.post("/summarize_context", json=_request().model_dump()).status_code == 200
    assert not directory.exists()


def test_empty_directory_setting_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESPONSER_CONTEXT_TRACE_DIR", " ")
    with pytest.raises(ValueError, match="must not be empty"):
        load_summary_trace_settings()


def test_insecure_directory_fails_before_inference_without_chmod_or_path_leak(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    client, calls, directory = _setup(monkeypatch, tmp_path)
    directory.mkdir(mode=0o755)
    directory.chmod(0o755)
    monkeypatch.setattr(summary_module.log, "propagate", True)
    caplog.set_level(logging.INFO, logger=summary_module.log.name)
    response = client.post("/summarize_context", json=_request().model_dump())
    assert response.status_code == 502 and "trace_write_failed" in response.text
    assert not calls and not list(directory.iterdir())
    assert stat.S_IMODE(directory.stat().st_mode) == 0o755
    assert str(directory) not in caplog.text + response.text


@pytest.mark.parametrize("parent_symlink", [False, True])
def test_trace_refuses_symlink_destinations_before_creating_data(tmp_path: Path, parent_symlink: bool) -> None:
    root = tmp_path.resolve()
    target = root / "target"
    target.mkdir(mode=0o700)
    link = root / "link"
    link.symlink_to(target, target_is_directory=True)
    path = link / "child" if parent_symlink else link
    with pytest.raises(SummaryTraceError):
        with SummaryTrace(SummaryTraceSettings(True, path), "a" * 32):
            pytest.fail("symlink was accepted")
    assert not list(target.iterdir())


def test_existing_trace_is_not_overwritten_even_if_symlinked(tmp_path: Path) -> None:
    root = tmp_path.resolve() / "traces"
    root.mkdir(mode=0o700)
    destination = root / ("summary-" + "a" * 32 + ".jsonl")
    destination.write_text("KEEP_EXISTING")
    settings = SummaryTraceSettings(True, root)
    with pytest.raises(SummaryTraceError):
        with SummaryTrace(settings, "a" * 32):
            pytest.fail("existing trace was accepted")
    assert destination.read_text() == "KEEP_EXISTING"
    (root / ("summary-" + "b" * 32 + ".jsonl")).symlink_to(destination)
    with pytest.raises(SummaryTraceError):
        with SummaryTrace(settings, "b" * 32):
            pytest.fail("symlinked trace was accepted")
    assert destination.read_text() == "KEEP_EXISTING"


def test_trace_capacity_aborts_without_silent_truncation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    client, calls, directory = _setup(monkeypatch, tmp_path)
    monkeypatch.setattr(trace_module, "MAX_TRACE_BYTES", 900)
    response = client.post("/summarize_context", json=_request().model_dump())
    assert response.status_code == 502 and "trace_write_failed" in response.text
    assert not calls
    events = _events(directory)
    assert events[-1]["stage"] == "started"
    assert all(event["stage"] != "completed" for event in events)


def test_trace_disk_failure_is_sanitized_and_blocks_result(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    client, calls, directory = _setup(monkeypatch, tmp_path)
    original_record = SummaryTrace.record

    def fail_result(self: SummaryTrace, stage: trace_module.TraceStage, data: dict[str, JsonValue]) -> None:
        if stage == "merge_result":
            raise SummaryTraceError("PRIVATE_DISK_PATH")
        original_record(self, stage, data)

    monkeypatch.setattr(SummaryTrace, "record", fail_result)
    response = client.post("/summarize_context", json=_request().model_dump())
    assert response.status_code == 502 and "trace_write_failed" in response.text
    assert "PRIVATE" not in response.text and len(calls) == 1
    assert _events(directory)[-1]["stage"] == "failed"


def test_actual_write_failure_closes_descriptor_without_retrying_private_data(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    settings = SummaryTraceSettings(True, tmp_path.resolve() / "traces")
    trace = SummaryTrace(settings, "a" * 32)
    trace.__enter__()
    fd = trace._fd
    assert fd is not None
    writes = 0

    def fail_write(descriptor: int, data: bytes) -> int:
        nonlocal writes
        writes += 1
        raise OSError("PRIVATE_DISK_ERROR")

    monkeypatch.setattr(trace_module.os, "write", fail_write)
    with pytest.raises(SummaryTraceError, match="cannot persist") as error:
        trace.record("review", {"private": "PRIVATE_CONTENT"})
    trace.__exit__(type(error.value), error.value, None)
    assert writes == 1  # Do not repeatedly write a failure record onto a broken disk.
    with pytest.raises(OSError):
        os.fstat(fd)
    assert "PRIVATE" not in str(error.value)


def test_interrupted_trace_keeps_completed_stages_and_closes_file(tmp_path: Path) -> None:
    directory = tmp_path.resolve() / "traces"
    with pytest.raises(KeyboardInterrupt):
        with SummaryTrace(SummaryTraceSettings(True, directory), "a" * 32) as trace:
            trace.record("inference_request", {"prompt": "PRIVATE_PROMPT"})
            raise KeyboardInterrupt()
    events = _events(directory)
    assert events[-1]["data"] == {"error_type": "KeyboardInterrupt", "last_completed_stage": "inference_request"}
    assert trace._fd is None


def test_fsync_failure_blocks_success_and_closes_trace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    trace = SummaryTrace(SummaryTraceSettings(True, tmp_path.resolve() / "traces"), "a" * 32)
    trace.__enter__()
    fd = trace._fd
    assert fd is not None

    def fail_fsync(descriptor: int) -> None:
        raise OSError("PRIVATE_SYNC_ERROR")

    monkeypatch.setattr(trace_module.os, "fsync", fail_fsync)
    with pytest.raises(SummaryTraceError) as error:
        trace.record("review", {"private": "PRIVATE_CONTENT"})
    trace.__exit__(type(error.value), error.value, None)
    assert "PRIVATE" not in str(error.value) and trace._fd is None
    with pytest.raises(OSError):
        os.fstat(fd)


@pytest.mark.parametrize("directory_failure", [False, True])
def test_close_failures_are_sanitized_and_release_trace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, directory_failure: bool,
) -> None:
    trace = SummaryTrace(SummaryTraceSettings(True, tmp_path.resolve() / "traces"), "a" * 32)
    original_close = os.close

    def fail_close(descriptor: int) -> None:
        is_directory = stat.S_ISDIR(os.fstat(descriptor).st_mode)
        original_close(descriptor)
        if is_directory == directory_failure:
            raise OSError("PRIVATE_CLOSE_ERROR")

    monkeypatch.setattr(trace_module.os, "close", fail_close)
    with pytest.raises(SummaryTraceError) as error:
        with trace:
            trace.record("review", {"private": "PRIVATE_CONTENT"})
    assert "PRIVATE" not in str(error.value) and trace._fd is None


def test_distinct_concurrent_traces_never_interleave(tmp_path: Path) -> None:
    directory = tmp_path.resolve() / "traces"
    settings = SummaryTraceSettings(True, directory)

    def write(index: int) -> None:
        with SummaryTrace(settings, f"{index:032x}") as trace:
            trace.record("review", {"private_marker": str(index), "text": "line1\nline2\r\nforged-stage\u001b"})

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(write, range(8)))
    files = list(directory.glob("*.jsonl"))
    assert len(files) == 8
    for file in files:
        events = [json.loads(line) for line in file.read_text().splitlines()]
        assert len(events) == 3 and len({event["summary_id"] for event in events}) == 1
        assert events[1]["data"]["text"] == "line1\nline2\r\nforged-stage\u001b"
        assert stat.S_IMODE(file.stat().st_mode) == 0o600