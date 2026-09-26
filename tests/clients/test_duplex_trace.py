# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Metadata retention and offline interpretation of duplex wire traces."""

import json

import pytest

from vllm_omni.clients.duplex_trace import DuplexTrace

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_trace_omits_payloads_and_credentials(tmp_path, monkeypatch):
    monkeypatch.setattr("vllm_omni.clients.duplex_trace.time.monotonic", lambda: 10.0)
    trace = DuplexTrace()
    trace.record(
        "send",
        {
            "type": "session.update",
            "session": {"id": "s1", "instructions": "SECRET", "ref_audio": "SECRET"},
            "resume_token": "SECRET",
        },
    )
    trace.record(
        "receive",
        {
            "type": "response.output_audio.delta",
            "response_id": "r1",
            "delta": "SECRET",
            "audio": "SECRET",
            "transcript": "SECRET",
            "response": {"output": [{"content": "SECRET"}]},
            "sample_rate_hz": 24000,
        },
    )
    trace.record(
        "receive",
        {
            "type": "error",
            "error": {"code": "invalid_request", "event_id": "e1", "message": "SECRET"},
        },
    )
    path = tmp_path / "trace.json"
    trace.write_json(path)
    assert "SECRET" not in path.read_text()
    data = json.loads(path.read_text())
    assert data["schema_version"] == 1
    assert data["events"][0]["session_id"] == "s1"
    assert data["events"][1]["sample_rate_hz"] == 24000
    assert data["events"][2]["error_code"] == "invalid_request"
    assert data["events"][2]["related_event_id"] == "e1"
    assert data["event_counts"]["receive:error"] == 1
    assert data["responses"] == [{"response_id": "r1", "first_audio_s": 0.0}]


def test_trace_reports_partial_turns_and_cancel_without_inventing_completion(monkeypatch):
    clock = iter([10.0, 10.5, 11.0, 11.1, 11.2, 12.0, 13.0])
    monkeypatch.setattr("vllm_omni.clients.duplex_trace.time.monotonic", lambda: next(clock))
    trace = DuplexTrace()
    trace.record("receive", {"type": "response.created", "response": {"id": "r1"}})
    trace.record("receive", {"type": "response.output_audio.delta", "response_id": "r1"})
    trace.record("send", {"type": "response.cancel", "response_id": "r1"})
    trace.record("receive", {"type": "response.output_audio.delta", "response_id": "r1"})
    trace.record("receive", {"type": "response.done", "response": {"id": "r1", "status": "cancelled"}})
    trace.record("receive", {"type": "response.created", "response": {"id": "r2"}})
    trace.record("send", {"type": "response.cancel"})  # no response identity: do not guess
    responses = trace.snapshot()["responses"]
    assert responses[0] == {
        "response_id": "r1",
        "created_s": 0.0,
        "first_audio_s": 0.5,
        "cancel_sent_s": 1.0,
        "done_s": pytest.approx(1.2),
        "status": "cancelled",
    }
    assert responses[1] == {"response_id": "r2", "created_s": 2.0}


def test_trace_bounds_retention_and_snapshots_are_independent():
    trace = DuplexTrace(max_events=2)
    response = {"id": "r1"}
    event = {"type": "response.created", "response": response}
    trace.record("receive", event)
    response["id"] = "mutated"
    snapshot = trace.snapshot()
    snapshot["events"][0]["response_id"] = "changed"
    assert trace.snapshot()["events"][0]["response_id"] == "r1"
    trace.record("receive", {"type": "response.output_audio.delta", "response_id": "r1"})
    trace.record("receive", {"type": "response.done", "response": {"id": "r1"}})
    snapshot = trace.snapshot()
    assert snapshot["total_events"] == 3
    assert snapshot["dropped_events"] == 1
    assert [row["index"] for row in snapshot["events"]] == [1, 2]
    assert "created_s" not in snapshot["responses"][0]


@pytest.mark.parametrize("capacity", [0, -1, True, 1.5])
def test_trace_rejects_invalid_capacity(capacity):
    with pytest.raises(ValueError, match="positive integer"):
        DuplexTrace(max_events=capacity)


def test_trace_bounds_strings_and_omits_nonfinite_metadata():
    trace = DuplexTrace()
    trace.record("receive", {"type": "x" * 1000, "played_ms": float("nan"), "server_event_seq": False})
    row = trace.snapshot()["events"][0]
    assert "type" not in row
    assert "played_ms" not in row
    assert "server_event_seq" not in row


def test_trace_write_error_is_visible(tmp_path):
    with pytest.raises(OSError):
        DuplexTrace().write_json(tmp_path / "absent" / "trace.json")
