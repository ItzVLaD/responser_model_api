"""Explicitly opted-in, private per-summary traces, separate from normal logs.

Trace records contain untrusted chat/model content, potentially including secrets.
Nothing here redacts that content or proves semantic correctness. Never enable
implicitly through DEBUG/LOG_PROMPTS, and never overwrite an existing trace.
"""

from __future__ import annotations

import json
import os
import stat
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Literal

from pydantic import JsonValue

from .schemas import FACT_SECTIONS, MemoryContent

DEFAULT_TRACE_DIRECTORY = Path(__file__).resolve().parents[2] / "diagnostics" / "context"
MAX_TRACE_BYTES = 4 * 1024 * 1024
TraceStage = Literal[
    "started", "request_received", "previous_prepared", "selection_plan", "inference_request",
    "inference_response", "selection_parsed", "selection_schema_rejected", "resolved_delta",
    "merge_started", "merge_rejected", "merge_result", "review", "response", "failed", "completed",
    "summary_schema_rejected", "summary_validated",
]


@dataclass(frozen=True)
class SummaryTraceSettings:
    """Read once on summarizer startup; never selectable by an HTTP caller."""

    enabled: bool = False
    directory: Path = DEFAULT_TRACE_DIRECTORY


def load_summary_trace_settings() -> SummaryTraceSettings:
    """Require an explicit true/false opt-in and a nonempty destination."""
    value = os.environ.get("RESPONSER_CONTEXT_TRACE", "false").strip().lower()
    if value not in {"true", "false"}:
        raise ValueError("RESPONSER_CONTEXT_TRACE must be true or false")
    directory = os.environ.get("RESPONSER_CONTEXT_TRACE_DIR", str(DEFAULT_TRACE_DIRECTORY))
    if not directory.strip():
        raise ValueError("RESPONSER_CONTEXT_TRACE_DIR must not be empty")
    return SummaryTraceSettings(enabled=value == "true", directory=Path(directory).expanduser())


class SummaryTraceError(RuntimeError):
    """Sanitized trace storage failure; paths and data never escape via HTTP."""


class SummaryTrace:
    """Append JSONL stages durably; enabled tracing fails closed on I/O errors.

    Each call owns its own descriptor, avoiding cross-request interleaving. The
    final directory must be owned by this user and inaccessible to others. We
    refuse insecure existing paths rather than chmod somebody else's directory.
    """

    def __init__(self, settings: SummaryTraceSettings, summary_id: str) -> None:
        if len(summary_id) != 32 or any(char not in "0123456789abcdef" for char in summary_id):
            raise SummaryTraceError("invalid trace identifier")
        self.enabled = settings.enabled
        self._directory = settings.directory
        self.summary_id = summary_id
        self._fd: int | None = None
        self._bytes = 0
        self._sequence = 0
        self._started = time.monotonic()
        self._broken = False
        self._last_stage: TraceStage | None = None

    def __enter__(self) -> SummaryTrace:
        if not self.enabled:
            return self
        directory_fd: int | None = None
        try:
            # Do not traverse operator-supplied symlinks to unexpected storage.
            for path in (self._directory, *self._directory.parents):
                if path.is_symlink():
                    raise SummaryTraceError("trace directory must not use symlinks")
            self._directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            directory_fd = os.open(self._directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            info = os.fstat(directory_fd)
            if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
                raise SummaryTraceError("trace directory must be owner-only")
            self._fd = os.open(
                f"summary-{self.summary_id}.jsonl",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600, dir_fd=directory_fd,
            )
            os.fchmod(self._fd, 0o600)
            self.record("started", {
                "content_policy": "FULL_PRIVATE_CONTENT_UNREDACTED",
                "reader_checkpoint": "not_written_by_api",
            })
            os.fsync(directory_fd)
            return self
        except (OSError, SummaryTraceError):
            self._close()
            raise SummaryTraceError("cannot create protected summary trace") from None
        finally:
            if directory_fd is not None:
                try:
                    os.close(directory_fd)
                except OSError:
                    # __enter__ has not returned: __exit__ will not clean up.
                    self._close()
                    raise SummaryTraceError("cannot close trace directory") from None

    def record(self, stage: TraceStage, data: dict[str, JsonValue]) -> None:
        """Persist one complete escaped JSON record, never console-log its data."""
        if not self.enabled:
            return
        if self._fd is None or self._broken:
            raise SummaryTraceError("summary trace is unavailable")
        event = {
            "trace_version": 1, "summary_id": self.summary_id, "sequence": self._sequence,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_ms": round((time.monotonic() - self._started) * 1000, 3),
            "stage": stage, "data": data,
        }
        # JSON escaping ensures a newline/control character in a chat cannot
        # masquerade as another stage. Raw output is data, never an instruction.
        encoded = (json.dumps(event, ensure_ascii=True, separators=(",", ":")) + "\n").encode("utf-8")
        if self._bytes + len(encoded) > MAX_TRACE_BYTES:
            self._broken = True
            raise SummaryTraceError("summary trace size limit exceeded")
        try:
            offset = 0
            while offset < len(encoded):
                written = os.write(self._fd, encoded[offset:])
                if written == 0:
                    raise OSError("trace write made no progress")
                offset += written
            os.fsync(self._fd)
        except OSError:
            self._broken = True
            raise SummaryTraceError("cannot persist summary trace") from None
        self._bytes += len(encoded)
        self._sequence += 1
        self._last_stage = stage

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            if not self._broken:
                if exc is None:
                    self.record("completed", {"outcome": "summary_pipeline_finished_not_a_checkpoint_commit"})
                elif self._last_stage != "failed":
                    # Exception strings can include transport credentials. Exact
                    # model output has its own stage; never serialize tracebacks.
                    self.record("failed", {"error_type": type(exc).__name__, "last_completed_stage": self._last_stage})
        finally:
            self._close()

    def _close(self) -> None:
        if self._fd is not None:
            fd, self._fd = self._fd, None
            try:
                os.close(fd)
            except OSError:
                raise SummaryTraceError("cannot close summary trace") from None


def memory_review(previous: MemoryContent | None, current: MemoryContent) -> dict[str, JsonValue]:
    """Readable, complete section view plus factual diff, not a quality score."""
    old = {fact.id: fact for fact in previous.facts} if previous is not None else {}
    new = {fact.id: fact for fact in current.facts}
    sections: dict[str, JsonValue] = {
        section: [{"kind": fact.kind, "text": fact.text} for fact in current.facts if fact.section == section]
        for section in FACT_SECTIONS
    }
    return {
        "sections": sections,
        "empty_sections": [section for section in FACT_SECTIONS if not sections[section]],
        "added_fact_ids": [key for key in new if key not in old],
        "removed_fact_ids": [key for key in old if key not in new],
        "preserved_fact_ids": [key for key in new if key in old and new[key] == old[key]],
        "modified_fact_ids": [key for key in new if key in old and new[key] != old[key]],
        "structural_validation": "passed",
        "semantic_correctness_and_completeness": "NOT_VALIDATED",
    }