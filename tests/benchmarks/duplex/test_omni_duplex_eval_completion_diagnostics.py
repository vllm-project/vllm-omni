# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Failure diagnostics retain bounded timing and ownership already collected."""

import copy
import json
from types import SimpleNamespace

import pytest

from vllm_omni.benchmarks.duplex import omni_duplex_eval_runner as runner
from vllm_omni.experimental.fullduplex.client import RealtimeEventCollector

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.benchmark]


def _diagnostics(collector, monkeypatch, *, start=0):
    monkeypatch.setattr(runner, "time", SimpleNamespace(monotonic=lambda: 100.0))
    return runner._completion_diagnostics(
        collector.events, start, response_ids=collector.response_ids, stable_since=99.9
    )


def test_completion_trace_retains_collected_listen_timing_and_ownership(monkeypatch):
    collector = RealtimeEventCollector()
    collector.add({"type": "session.created"}, received_at_s=1.0)
    collector.add({"type": "input_audio_buffer.committed", "server_event_seq": 2}, received_at_s=10.0)
    collector.add({"type": "response.created", "response": {"id": "r1"}}, received_at_s=11.0)
    collector.add({"type": "response.done", "response": {"id": "r1", "status": "completed"}}, received_at_s=12.0)
    for index in range(4):
        collector.add(
            {
                "type": "response.listen",
                "server_event_seq": 5 + index,
                "session_id": "s1",
                "epoch": 0,
                "response": {
                    "status": "listening",
                    "metadata": {
                        "reason": "model_listen",
                        "model_listen": True,
                        "vllm_omni": {"model_turn_id": 2},
                    },
                },
            },
            received_at_s=13.0 + index,
        )
    original = copy.deepcopy(collector.events)
    received_at = list(collector.event_received_at_s)

    diagnostics = _diagnostics(collector, monkeypatch, start=1)

    assert diagnostics["unfinished_response_ids"] == []
    assert diagnostics["normal_terminal_decision"] is True
    assert diagnostics["quiet_window_satisfied"] is False
    trace = diagnostics["completion_event_trace"]
    assert trace["event_count"] == 7
    assert trace["omitted_event_count"] == 0
    assert trace["events"][-1] == {
        "event_index": 7,
        "type": "response.listen",
        "server_event_seq": 8,
        "session_id": "s1",
        "epoch": 0,
        "client_received_at_s": 16.0,
        "reason": "model_listen",
        "model_turn_id": 2,
        "model_listen": True,
    }
    assert collector.events == original
    assert collector.event_received_at_s == received_at


def test_completion_trace_bounds_tail_and_strings_without_copying_payloads(monkeypatch):
    collector = RealtimeEventCollector()
    large_value = "x" * 32_768
    count = 300
    for index in range(count):
        collector.add(
            {
                "type": "response.output_text.delta",
                "response_id": large_value,
                "delta": large_value,
                "metadata": {"reason": large_value, "unrelated": large_value},
                "server_event_seq": index,
            },
            received_at_s=float(index),
        )

    trace = _diagnostics(collector, monkeypatch)["completion_event_trace"]

    assert trace["limit"] == 256
    assert len(trace["events"]) == 256
    assert trace["event_count"] == count
    assert trace["omitted_event_count"] == 44
    assert trace["events"][0]["event_index"] == 44
    assert trace["events"][-1]["server_event_seq"] == 299
    assert trace["events"][-1]["client_received_at_s"] == 299.0
    assert all(len(row["response_id"]) == len(row["reason"]) == 128 for row in trace["events"])
    assert all("delta" not in row and "metadata" not in row for row in trace["events"])
    assert len(json.dumps(trace)) < 140_000
    assert all(event["delta"] is large_value for event in collector.events)
    assert all(event["response_id"] is large_value for event in collector.events)


@pytest.mark.parametrize("invalid_timestamp", [None, True, "1.0", float("nan"), float("inf")])
def test_completion_trace_omits_invalid_timing_and_identity_fields(monkeypatch, invalid_timestamp):
    collector = RealtimeEventCollector()
    collector.add(
        {
            "type": "response.listen",
            "_client_received_at_s": invalid_timestamp,
            "server_event_seq": True,
            "epoch": "0",
            "response": {"metadata": {"vllm_omni": {"model_turn_id": False}}},
        },
        received_at_s=1.0,
    )

    trace = _diagnostics(collector, monkeypatch)["completion_event_trace"]

    assert trace["events"] == [{"event_index": 0, "type": "response.listen"}]
    assert collector.event_received_at_s == [1.0]
