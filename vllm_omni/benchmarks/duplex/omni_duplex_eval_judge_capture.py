# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Bounded failed-judge request evidence; independent of scoring and transport."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)
_MAX_REQUEST_BYTES = 8 * 1024 * 1024
_MAX_CAPTURE_BYTES = 16 * 1024 * 1024  # Per split's owned judge_requests directory.
_MAX_FRAME_BYTES = 1024 * 1024
_MAX_FRAMES = 64
_CAPTURE_LOCK = Lock()


@dataclass
class JudgeRequestCapture:
    payload: dict[str, Any] | None = None
    calls: int = 0


_CURRENT_CAPTURE: ContextVar[JudgeRequestCapture | None] = ContextVar("duplex_judge_request_capture", default=None)


@contextmanager
def capture_judge_request() -> Iterator[JudgeRequestCapture]:
    captured = JudgeRequestCapture()
    token = _CURRENT_CAPTURE.set(captured)
    try:
        yield captured
    finally:
        _CURRENT_CAPTURE.reset(token)


def record_judge_request(payload: dict[str, Any]) -> None:
    if (captured := _CURRENT_CAPTURE.get()) is not None:
        captured.calls += 1
        # The owned client builds this dictionary for this synchronous call.
        # Keep a reference until parsing finishes; successes never serialize it.
        if captured.calls == 1:
            captured.payload = payload


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _atomic_write(path: Path, data: bytes) -> None:
    temporary = path.with_name(f"{path.name}.{uuid4().hex}.part")
    try:
        temporary.write_bytes(data)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_failed_judge_request(captured: JudgeRequestCapture, trace_path: Path, call_index: int) -> None:
    """Retain only failed calls; never replace the original exception on I/O failure.

    Records end in .judge.jsonl for the existing bounded DFX export. They cannot
    enter the *.json score scan. No temporary media directory is copied.
    """
    try:
        manifest: dict[str, Any] = {
            "kind": "duplex_judge_request_v1",
            "trace_file": trace_path.name,
            "call_index": call_index,
            "capture_status": "unavailable",
            "replayable": False,
        }
        files: dict[str, bytes] = {}
        if captured.calls == 1 and captured.payload is not None:
            encoded = _json_bytes(captured.payload)
            manifest["request_body_sha256"] = hashlib.sha256(encoded).hexdigest()
            if len(encoded) > _MAX_REQUEST_BYTES:
                manifest["capture_status"] = "request_byte_limit"
            else:
                # Failure-only copy; never mutate the dictionary passed to requests.post.
                body = json.loads(encoded)
                refs: list[dict[str, Any]] = []
                external: list[dict[str, Any]] = []

                def externalize(value: object, path: list[str | int]) -> None:
                    if isinstance(value, dict):
                        for key, item in value.items():
                            current = [*path, key]
                            if key == "url" and isinstance(item, str):
                                if item.startswith("data:image/") and ";base64," in item:
                                    raw = item.encode()
                                    if len(raw) > _MAX_FRAME_BYTES or len(refs) >= _MAX_FRAMES:
                                        raise ValueError("judge frame capture limit")
                                    digest = hashlib.sha256(raw).hexdigest()
                                    name = f"frame-{digest}.judge.jsonl"
                                    files[name] = _json_bytes({"data_url": item})
                                    refs.append({"path": current, "file": name, "data_url_sha256": digest})
                                    value[key] = f"judge-frame:{digest}"
                                else:
                                    # Preserve the exact URL but do not copy file-backed videos or fetch remote media.
                                    external.append({"path": current, "url": item})
                            else:
                                externalize(item, current)
                    elif isinstance(value, list):
                        for index, item in enumerate(value):
                            externalize(item, [*path, index])

                try:
                    externalize(body, [])
                except ValueError:
                    files.clear()
                    manifest["capture_status"] = "frame_limit"
                else:
                    manifest.update(
                        capture_status="captured",
                        replayable=not external,
                        request_body=body,
                        frame_refs=refs,
                        external_media_refs=external,
                    )
        else:
            manifest["reason"] = "Owned request JSON was not observed exactly once"

        encoded_manifest = _json_bytes(manifest)
        digest = hashlib.sha256(encoded_manifest).hexdigest()
        files[f"request-{trace_path.stem}-{call_index}-{digest}.judge.jsonl"] = encoded_manifest
        destination = trace_path.parent / "judge_requests"
        # Evaluation workers share threads, not processes; serialize only failed-call writes.
        with _CAPTURE_LOCK:
            destination.mkdir(parents=True, exist_ok=True)
            pending: dict[str, bytes] = {}
            for name, data in files.items():
                path = destination / name
                if path.exists():
                    if path.read_bytes() != data:
                        raise OSError("Conflicting judge request artifact")
                else:
                    pending[name] = data
            used = sum(path.stat().st_size for path in destination.glob("*.judge.jsonl") if path.is_file())
            if used + sum(map(len, pending.values())) > _MAX_CAPTURE_BYTES:
                logger.warning("Failed judge request capture reached its artifact byte limit")
                return
            # Commit referenced frames first, then the request manifest.
            for name, data in pending.items():
                _atomic_write(destination / name, data)
    except Exception as exc:
        # Diagnostic errors must not mask the judge/parser failure. Exception
        # text can contain request details, so report its type only.
        logger.warning("Failed to retain judge request (%s)", type(exc).__name__)
