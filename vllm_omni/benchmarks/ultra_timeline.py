# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Opt-in, host-side JSONL diagnostics for the Ultra benchmark workflow.

This module deliberately stays outside the request result and model data paths:
it emits only host timestamps and metadata after callers opt in with
``VLLM_OMNI_ULTRA_TIMELINE=1``.  In particular, it must never materialize a
tensor, synchronize a device, or change the benchmark's official metrics.

The recorder stores one JSON object per line.  The required event fields are
present on every line so an evidence consumer can concatenate files from
multiple benchmark client processes without a schema migration.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import os
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)

ULTRA_TIMELINE_ENV = "VLLM_OMNI_ULTRA_TIMELINE"
ULTRA_TIMELINE_PATH_ENV = "VLLM_OMNI_ULTRA_TIMELINE_PATH"
ULTRA_TIMELINE_DIR_ENV = "VLLM_OMNI_ULTRA_TIMELINE_DIR"
ULTRA_TIMELINE_CAPTURE_RAW_ENV = "VLLM_OMNI_ULTRA_TIMELINE_CAPTURE_RAW"
ULTRA_TIMELINE_SCHEMA_VERSION = 1

_TRUTHY_ENV_VALUES = frozenset({"1", "true", "yes", "on"})
_PROCESS_EVENT_SEQUENCE = itertools.count()
_WRITE_LOCK = threading.Lock()
_RAW_KIND_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789_-")


class TimelineRecorder(Protocol):
    """Small recorder contract used by benchmark request functions."""

    @property
    def enabled(self) -> bool: ...

    def next_chunk_id(self) -> int: ...

    def emit(
        self,
        event: str,
        *,
        stage: str | int = "client",
        chunk_id: str | int | None = None,
        stream: str | None = None,
        shape: object = None,
        num_bytes: int | None = None,
        error: object | None = None,
        details: Mapping[str, object] | None = None,
        payload: bytes | bytearray | memoryview | str | None = None,
        raw_kind: str | None = None,
    ) -> None: ...

    def close(self) -> None: ...


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY_ENV_VALUES


def ultra_timeline_enabled() -> bool:
    """Return whether Ultra timeline diagnostics are explicitly enabled."""
    return _env_enabled(ULTRA_TIMELINE_ENV)


def ultra_timeline_capture_raw_enabled() -> bool:
    """Return whether separately stored raw SSE/PCM evidence is requested."""
    return ultra_timeline_enabled() and _env_enabled(ULTRA_TIMELINE_CAPTURE_RAW_ENV)


def _default_output_path(pid: int) -> Path:
    return Path.cwd() / "ultra-timeline" / f"events.{pid}.jsonl"


def resolve_ultra_timeline_path(*, pid: int | None = None) -> Path:
    """Resolve the per-process JSONL sink without opening or creating it.

    ``VLLM_OMNI_ULTRA_TIMELINE_PATH`` wins over
    ``VLLM_OMNI_ULTRA_TIMELINE_DIR``.  The latter produces ``events.<pid>.jsonl``
    so concurrent benchmark clients do not contend for a shared file.  When no
    destination is supplied, an enabled recorder uses ``./ultra-timeline``;
    production benchmark runs should normally set ``*_DIR`` to their evidence
    bundle instead.
    """
    process_id = os.getpid() if pid is None else int(pid)
    raw_path = os.environ.get(ULTRA_TIMELINE_PATH_ENV, "").strip()
    if raw_path:
        path = Path(raw_path).expanduser()
        if path.exists() and path.is_dir():
            return path / f"events.{process_id}.jsonl"
        return path

    raw_dir = os.environ.get(ULTRA_TIMELINE_DIR_ENV, "").strip()
    if raw_dir:
        return Path(raw_dir).expanduser() / f"events.{process_id}.jsonl"
    return _default_output_path(process_id)


def _normalize_shape(shape: object) -> object:
    if shape is None or isinstance(shape, (str, int, float, bool)):
        return shape
    if isinstance(shape, Sequence) and not isinstance(shape, (bytes, bytearray, memoryview)):
        return [_normalize_shape(item) for item in shape]
    return str(shape)


def _safe_details(details: Mapping[str, object] | None) -> dict[str, object] | None:
    """Keep diagnostic metadata JSON-safe and bounded, without payload values."""
    if not details:
        return None

    def convert(value: object, *, depth: int = 0) -> object:
        if depth >= 3:
            return "<truncated>"
        if value is None or isinstance(value, (str, int, float, bool)):
            return value if not isinstance(value, str) else value[:512]
        if isinstance(value, Mapping):
            return {str(key)[:128]: convert(item, depth=depth + 1) for key, item in list(value.items())[:32]}
        if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, memoryview)):
            return [convert(item, depth=depth + 1) for item in list(value)[:32]]
        return str(value)[:512]

    return {str(key)[:128]: convert(value) for key, value in list(details.items())[:32]}


def _coerce_payload(payload: bytes | bytearray | memoryview | str) -> bytes:
    if isinstance(payload, str):
        return payload.encode("utf-8")
    return bytes(payload)


def _safe_raw_kind(raw_kind: str | None) -> str:
    candidate = (raw_kind or "payload").lower()
    cleaned = "".join(char if char in _RAW_KIND_CHARS else "_" for char in candidate)
    return cleaned[:48] or "payload"


class _NullTimelineRecorder:
    """Disabled recorder that intentionally performs no clock or I/O work."""

    @property
    def enabled(self) -> bool:
        return False

    def next_chunk_id(self) -> int:
        return 0

    def emit(
        self,
        event: str,
        *,
        stage: str | int = "client",
        chunk_id: str | int | None = None,
        stream: str | None = None,
        shape: object = None,
        num_bytes: int | None = None,
        error: object | None = None,
        details: Mapping[str, object] | None = None,
        payload: bytes | bytearray | memoryview | str | None = None,
        raw_kind: str | None = None,
    ) -> None:
        return

    def close(self) -> None:
        return


_NULL_TIMELINE_RECORDER = _NullTimelineRecorder()


class UltraTimelineRecorder:
    """Process-local recorder that flushes one request's records at its terminal event."""

    def __init__(
        self,
        *,
        request_id: str | None,
        turn_id: str | int | None = None,
        output_path: Path | None = None,
        capture_raw: bool | None = None,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self._request_id = request_id
        self._turn_id = turn_id
        self._pid = os.getpid()
        self._output_path = output_path or resolve_ultra_timeline_path(pid=self._pid)
        self._capture_raw = ultra_timeline_capture_raw_enabled() if capture_raw is None else bool(capture_raw)
        self._clock_ns = clock_ns
        self._chunk_sequence = itertools.count()
        self._events: list[dict[str, object]] = []
        self._write_failed = False
        self._closed = False

    @property
    def enabled(self) -> bool:
        return True

    @property
    def output_path(self) -> Path:
        return self._output_path

    def next_chunk_id(self) -> int:
        return next(self._chunk_sequence)

    def emit(
        self,
        event: str,
        *,
        stage: str | int = "client",
        chunk_id: str | int | None = None,
        stream: str | None = None,
        shape: object = None,
        num_bytes: int | None = None,
        error: object | None = None,
        details: Mapping[str, object] | None = None,
        payload: bytes | bytearray | memoryview | str | None = None,
        raw_kind: str | None = None,
    ) -> None:
        if self._write_failed or self._closed:
            return

        payload_bytes: bytes | None = None
        if payload is not None:
            payload_bytes = _coerce_payload(payload)
            if num_bytes is None:
                num_bytes = len(payload_bytes)

        sequence = next(_PROCESS_EVENT_SEQUENCE)
        record: dict[str, object] = {
            "schema_version": ULTRA_TIMELINE_SCHEMA_VERSION,
            "pid": self._pid,
            "seq": sequence,
            "request_id": self._request_id,
            "turn_id": self._turn_id,
            "chunk_id": chunk_id,
            "stage": str(stage),
            "event": str(event),
            "monotonic_ns": int(self._clock_ns()),
            "stream": stream,
            "shape": _normalize_shape(shape),
            "bytes": int(num_bytes) if num_bytes is not None else None,
            "error": str(error)[:512] if error is not None else None,
        }
        safe_details = _safe_details(details)
        if safe_details is not None:
            record["details"] = safe_details
        if payload_bytes is not None:
            record["sha256"] = hashlib.sha256(payload_bytes).hexdigest()
            raw_path = self._write_raw_payload(payload_bytes, raw_kind=raw_kind, sequence=sequence)
            if raw_path is not None:
                record["raw_path"] = raw_path
        self._events.append(record)

    def _write_raw_payload(self, payload: bytes, *, raw_kind: str | None, sequence: int) -> str | None:
        if not self._capture_raw:
            return None
        raw_dir = self._output_path.parent / "raw"
        raw_name = f"{self._output_path.stem}.{self._pid}.{sequence}.{_safe_raw_kind(raw_kind)}.bin"
        raw_path = raw_dir / raw_name
        try:
            raw_dir.mkdir(parents=True, exist_ok=True)
            with _WRITE_LOCK:
                raw_path.write_bytes(payload)
            return str(raw_path.relative_to(self._output_path.parent))
        except OSError:
            logger.warning("Unable to write raw Ultra timeline evidence to %s", raw_path, exc_info=True)
            return None

    def close(self) -> None:
        """Flush this request's host records after its terminal benchmark event."""
        if self._closed:
            return
        self._closed = True
        if self._write_failed or not self._events:
            return
        try:
            encoded = b"".join(
                (json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")
                for record in self._events
            )
            self._output_path.parent.mkdir(parents=True, exist_ok=True)
            with _WRITE_LOCK:
                with self._output_path.open("ab") as sink:
                    sink.write(encoded)
        except OSError:
            self._write_failed = True
            logger.warning("Ultra timeline disabled after failing to write %s", self._output_path, exc_info=True)
        finally:
            self._events.clear()


def create_ultra_timeline_recorder(
    *,
    request_id: str | None,
    turn_id: str | int | None = None,
) -> TimelineRecorder:
    """Create an enabled recorder only after an explicit environment opt-in."""
    if not ultra_timeline_enabled():
        return _NULL_TIMELINE_RECORDER
    return UltraTimelineRecorder(request_id=request_id, turn_id=turn_id)


def emit_ultra_timeline_event(
    event: str,
    *,
    request_id: str | None,
    turn_id: str | int | None = None,
    stage: str | int,
    chunk_id: str | int | None = None,
    stream: str | None = None,
    shape: object = None,
    num_bytes: int | None = None,
    error: object | None = None,
    details: Mapping[str, object] | None = None,
) -> None:
    """Append one process-local server event without touching device state.

    Model stages and connector workers do not share a reliable request-terminal
    callback, so server events are flushed immediately instead of being held in
    a request-local buffer.  The helper deliberately accepts metadata only: a
    caller cannot accidentally materialize or hash a device tensor through this
    interface.  Diagnostic failures never affect inference.
    """
    if not ultra_timeline_enabled():
        return
    try:
        recorder = UltraTimelineRecorder(
            request_id=request_id,
            turn_id=turn_id,
            capture_raw=False,
        )
        recorder.emit(
            event,
            stage=stage,
            chunk_id=chunk_id,
            stream=stream,
            shape=shape,
            num_bytes=num_bytes,
            error=error,
            details=details,
        )
        recorder.close()
    except Exception:
        logger.warning("Unable to emit Ultra server timeline event %s", event, exc_info=True)
