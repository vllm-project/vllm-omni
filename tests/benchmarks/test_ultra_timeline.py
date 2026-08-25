# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU coverage for the opt-in Ultra benchmark timeline recorder."""

from __future__ import annotations

import hashlib
import json

import pytest

from vllm_omni.benchmarks.ultra_timeline import (
    ULTRA_TIMELINE_DIR_ENV,
    ULTRA_TIMELINE_ENV,
    ULTRA_TIMELINE_PATH_ENV,
    ULTRA_TIMELINE_SCHEMA_VERSION,
    UltraTimelineRecorder,
    create_ultra_timeline_recorder,
    emit_ultra_timeline_event,
    resolve_ultra_timeline_path,
)

pytestmark = [pytest.mark.core_model, pytest.mark.benchmark, pytest.mark.cpu]

_REQUIRED_EVENT_FIELDS = {
    "schema_version",
    "pid",
    "seq",
    "request_id",
    "turn_id",
    "chunk_id",
    "stage",
    "event",
    "monotonic_ns",
    "stream",
    "shape",
    "bytes",
    "error",
}


def _read_events(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_default_disabled_does_not_create_file_or_read_clock(tmp_path, monkeypatch):
    monkeypatch.delenv(ULTRA_TIMELINE_ENV, raising=False)
    monkeypatch.delenv(ULTRA_TIMELINE_PATH_ENV, raising=False)
    monkeypatch.delenv(ULTRA_TIMELINE_DIR_ENV, raising=False)

    recorder = create_ultra_timeline_recorder(request_id="request-1")
    assert recorder.enabled is False
    recorder.emit("request_start", payload=b"must-not-be-written")
    recorder.close()

    assert list(tmp_path.iterdir()) == []


def test_enabled_recorder_writes_complete_payload_free_schema(tmp_path):
    output_path = tmp_path / "events.jsonl"
    timestamps = iter((101, 202))
    recorder = UltraTimelineRecorder(
        request_id="request-1",
        turn_id="turn-7",
        output_path=output_path,
        capture_raw=False,
        clock_ns=lambda: next(timestamps),
    )

    recorder.emit(
        "sse_fragment_received",
        stage="client",
        chunk_id=3,
        stream="sse",
        shape=(2, 4),
        payload=b"secret-response-fragment",
        details={"attempt": 0},
    )
    recorder.emit("request_finished", stage="client", stream="http")
    recorder.close()

    events = _read_events(output_path)
    assert len(events) == 2
    first = events[0]
    assert _REQUIRED_EVENT_FIELDS <= first.keys()
    assert first["schema_version"] == ULTRA_TIMELINE_SCHEMA_VERSION
    assert first["request_id"] == "request-1"
    assert first["turn_id"] == "turn-7"
    assert first["chunk_id"] == 3
    assert first["monotonic_ns"] == 101
    assert first["shape"] == [2, 4]
    assert first["bytes"] == len(b"secret-response-fragment")
    assert first["sha256"] == hashlib.sha256(b"secret-response-fragment").hexdigest()
    assert "secret-response-fragment" not in output_path.read_text(encoding="utf-8")
    assert events[1]["monotonic_ns"] == 202
    assert events[1]["seq"] > first["seq"]


def test_factory_uses_per_process_directory_sink(tmp_path, monkeypatch):
    monkeypatch.setenv(ULTRA_TIMELINE_ENV, "true")
    monkeypatch.setenv(ULTRA_TIMELINE_DIR_ENV, str(tmp_path))

    recorder = create_ultra_timeline_recorder(request_id="request-2")
    assert recorder.enabled is True
    recorder.emit("request_start")
    recorder.close()

    output_path = resolve_ultra_timeline_path()
    assert output_path.parent == tmp_path
    assert [event["event"] for event in _read_events(output_path)] == ["request_start"]


def test_path_takes_precedence_over_directory(tmp_path, monkeypatch):
    explicit_path = tmp_path / "explicit" / "events.jsonl"
    monkeypatch.setenv(ULTRA_TIMELINE_PATH_ENV, str(explicit_path))
    monkeypatch.setenv(ULTRA_TIMELINE_DIR_ENV, str(tmp_path / "ignored"))

    assert resolve_ultra_timeline_path() == explicit_path


def test_server_event_is_default_off(tmp_path, monkeypatch):
    output_path = tmp_path / "events.jsonl"
    monkeypatch.delenv(ULTRA_TIMELINE_ENV, raising=False)
    monkeypatch.setenv(ULTRA_TIMELINE_PATH_ENV, str(output_path))

    emit_ultra_timeline_event(
        "connector_put",
        request_id="request-off",
        stage=1,
        shape=(4, 8),
        num_bytes=128,
    )

    assert not output_path.exists()


def test_server_event_flushes_metadata_only_record(tmp_path, monkeypatch):
    output_path = tmp_path / "events.jsonl"
    monkeypatch.setenv(ULTRA_TIMELINE_ENV, "1")
    monkeypatch.setenv(ULTRA_TIMELINE_PATH_ENV, str(output_path))

    emit_ultra_timeline_event(
        "cfm_end",
        request_id="request-server",
        turn_id=3,
        stage=2,
        chunk_id=7,
        stream="compute",
        shape=(1, 80, 50),
        num_bytes=8000,
        details={"dtype": "torch.float16"},
    )

    events = _read_events(output_path)
    assert len(events) == 1
    assert events[0]["event"] == "cfm_end"
    assert events[0]["request_id"] == "request-server"
    assert events[0]["turn_id"] == 3
    assert events[0]["chunk_id"] == 7
    assert events[0]["shape"] == [1, 80, 50]
    assert events[0]["bytes"] == 8000
    assert "sha256" not in events[0]


def test_raw_capture_writes_sidecar_not_json_payload(tmp_path):
    output_path = tmp_path / "events.jsonl"
    recorder = UltraTimelineRecorder(
        request_id="request-3",
        output_path=output_path,
        capture_raw=True,
        clock_ns=lambda: 1,
    )
    recorder.emit("first_nonempty_decodable_pcm", payload=b"raw-pcm", raw_kind="pcm")
    recorder.close()

    event = _read_events(output_path)[0]
    assert event["raw_path"].startswith("raw/")
    raw_path = output_path.parent / event["raw_path"]
    assert raw_path.read_bytes() == b"raw-pcm"
    assert "raw-pcm" not in output_path.read_text(encoding="utf-8")


def test_write_failure_is_diagnostic_only(tmp_path):
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("block", encoding="utf-8")
    output_path = blocker / "events.jsonl"
    recorder = UltraTimelineRecorder(
        request_id="request-4",
        output_path=output_path,
        capture_raw=False,
        clock_ns=lambda: 1,
    )

    recorder.emit("request_start")
    recorder.emit("request_finished")
    recorder.close()

    assert not output_path.exists()
