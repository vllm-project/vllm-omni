# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json
from dataclasses import dataclass, field

import pytest

from vllm_omni.entrypoints.client_request_state import ClientRequestState
from vllm_omni.entrypoints.omni_base import OmniBase
from vllm_omni.metrics.stream_edge import RequestStreamEdgeEvents, StreamEdgeEvent, StreamEdgeKey
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def observe(recorder, event, elapsed_ms, *, clock_domain="orchestrator.segment-0", meaningful=True, chunk_seq=None):
    return recorder.record(
        event, elapsed_ms=elapsed_ms, clock_domain=clock_domain, meaningful=meaningful, chunk_seq=chunk_seq
    )


@dataclass
class SyntheticTwoStagePipeline:
    """Model-neutral producer, connector and consumer contract proof."""

    send_success: bool = True
    accept_success: bool = True
    output_ready: bool = True

    def deliver(self, edge, payload: tuple[int, ...]) -> tuple[int, ...] | None:
        meaningful = bool(payload)
        observe(edge, StreamEdgeEvent.PRODUCER_EMIT, 10.0, meaningful=meaningful, chunk_seq=3)
        if not self.send_success:
            return None
        observe(edge, StreamEdgeEvent.SEND_COMPLETE, 12.0, meaningful=meaningful)
        if not self.accept_success:
            return None
        observe(edge, StreamEdgeEvent.CONSUMER_ACCEPT, 14.0, meaningful=meaningful)
        output = payload if self.output_ready else ()
        observe(edge, StreamEdgeEvent.CONSUMER_OUTPUT, 25.0, meaningful=bool(output))
        return output


def test_synthetic_two_stage_flow_records_each_event_once():
    request = ClientRequestState("internal", external_request_id="external")
    assert request.stream_edge_events is None
    events = request.enable_stream_edge_metrics()
    key = StreamEdgeKey(0, 1)
    edge = events.start_segment(key)
    pipeline = SyntheticTwoStagePipeline()
    assert pipeline.deliver(edge, (7,)) == (7,)
    assert pipeline.deliver(edge, (8,)) == (8,)
    assert not observe(edge, StreamEdgeEvent.PRODUCER_EMIT, 99.0, chunk_seq=9)
    assert not observe(edge, StreamEdgeEvent.CONSUMER_OUTPUT, 100.0)
    record = events.snapshot()["0->1:0"]
    assert record["producer_first_nonempty_emit_ms"] == 10.0
    assert record["consumer_first_nonempty_output_ms"] == 25.0
    assert record["first_chunk_seq"] == 3
    assert events.elapsed_between(key, StreamEdgeEvent.PRODUCER_EMIT, StreamEdgeEvent.SEND_COMPLETE) == 2.0
    assert "internal" not in json.dumps(events.snapshot())
    assert "external" not in json.dumps(events.snapshot())


@pytest.mark.parametrize("failure", ["send", "admission"])
def test_failed_handoff_leaves_unreached_events_null(failure):
    events = RequestStreamEdgeEvents()
    edge = events.start_segment(StreamEdgeKey(0, 1))
    pipeline = SyntheticTwoStagePipeline(send_success=failure != "send", accept_success=failure != "admission")
    assert pipeline.deliver(edge, (7,)) is None
    record = events.snapshot()["0->1:0"]
    assert record["edge_first_send_complete_ms"] == (12.0 if failure == "admission" else None)
    assert record["consumer_first_accept_ms"] is None
    assert record["consumer_first_nonempty_output_ms"] is None


def test_empty_terminal_does_not_create_first_events():
    events = RequestStreamEdgeEvents()
    edge = events.start_segment(StreamEdgeKey(0, 1))
    assert SyntheticTwoStagePipeline().deliver(edge, ()) == ()
    assert events.snapshot() == {}


def test_consumer_buffering_does_not_count_as_first_output():
    events = RequestStreamEdgeEvents()
    edge = events.start_segment(StreamEdgeKey(0, 1))
    assert SyntheticTwoStagePipeline(output_ready=False).deliver(edge, (7,)) == ()
    record = events.snapshot()["0->1:0"]
    assert record["consumer_first_accept_ms"] == 14.0
    assert record["consumer_first_nonempty_output_ms"] is None
    assert SyntheticTwoStagePipeline().deliver(edge, (8,)) == (8,)
    assert events.snapshot()["0->1:0"]["consumer_first_nonempty_output_ms"] == 25.0


def test_edges_requests_and_segments_are_isolated_and_old_segments_are_fenced():
    first = RequestStreamEdgeEvents()
    second = RequestStreamEdgeEvents()
    old = first.start_segment(StreamEdgeKey(0, 1, 0))
    observe(old, StreamEdgeEvent.PRODUCER_EMIT, 10.0)
    current = first.start_segment(StreamEdgeKey(0, 1, 1))
    observe(current, StreamEdgeEvent.PRODUCER_EMIT, 5.0, clock_domain="segment-1")
    assert not observe(old, StreamEdgeEvent.SEND_COMPLETE, 12.0)
    other_edge = first.start_segment(StreamEdgeKey(1, 2))
    observe(other_edge, StreamEdgeEvent.CONSUMER_ACCEPT, 20.0, clock_domain="worker")
    other_request = second.start_segment(StreamEdgeKey(0, 1))
    observe(other_request, StreamEdgeEvent.PRODUCER_EMIT, 1.0)
    assert first.snapshot()["0->1:0"]["edge_first_send_complete_ms"] is None
    assert first.snapshot()["0->1:1"]["producer_first_nonempty_emit_ms"] == 5.0
    assert first.snapshot()["1->2:0"]["consumer_first_accept_ms"] == 20.0
    assert second.snapshot()["0->1:0"]["producer_first_nonempty_emit_ms"] == 1.0
    with pytest.raises(ValueError, match="retired"):
        first.start_segment(StreamEdgeKey(0, 1, 0))


def test_cross_clock_deltas_are_unset_and_snapshots_are_detached():
    events = RequestStreamEdgeEvents()
    key = StreamEdgeKey(0, 1)
    edge = events.start_segment(key)
    observe(edge, StreamEdgeEvent.PRODUCER_EMIT, 100.0, clock_domain="sender")
    observe(edge, StreamEdgeEvent.SEND_COMPLETE, 103.0, clock_domain="sender")
    observe(edge, StreamEdgeEvent.CONSUMER_ACCEPT, 5.0, clock_domain="receiver")
    assert events.elapsed_between(key, StreamEdgeEvent.SEND_COMPLETE, StreamEdgeEvent.CONSUMER_ACCEPT) is None
    assert events.elapsed_between(key, StreamEdgeEvent.CONSUMER_ACCEPT, StreamEdgeEvent.CONSUMER_OUTPUT) is None
    snapshot = events.snapshot()
    snapshot["0->1:0"]["clock_domains"].clear()
    snapshot["0->1:0"]["producer_first_nonempty_emit_ms"] = 999.0
    assert events.snapshot()["0->1:0"]["producer_first_nonempty_emit_ms"] == 100.0
    assert events.elapsed_between(key, StreamEdgeEvent.PRODUCER_EMIT, StreamEdgeEvent.SEND_COMPLETE) == 3.0


def test_zero_elapsed_time_is_observed_and_negative_order_has_no_delta():
    events = RequestStreamEdgeEvents()
    key = StreamEdgeKey(0, 1)
    edge = events.start_segment(key)
    observe(edge, StreamEdgeEvent.PRODUCER_EMIT, 10.0)
    observe(edge, StreamEdgeEvent.SEND_COMPLETE, 0.0)
    assert events.snapshot()["0->1:0"]["edge_first_send_complete_ms"] == 0.0
    assert events.elapsed_between(key, StreamEdgeEvent.PRODUCER_EMIT, StreamEdgeEvent.SEND_COMPLETE) is None


def test_finalization_fences_late_events_and_external_id_reuse():
    request = ClientRequestState("old", external_request_id="external")
    events = request.enable_stream_edge_metrics()
    edge = events.start_segment(StreamEdgeKey(0, 1))
    observe(edge, StreamEdgeEvent.PRODUCER_EMIT, 10.0)
    snapshot = events.snapshot()
    request.release_stream_edge_metrics()
    request.release_stream_edge_metrics()
    assert not observe(edge, StreamEdgeEvent.SEND_COMPLETE, 12.0)
    assert events.snapshot() == {}
    assert snapshot["0->1:0"]["producer_first_nonempty_emit_ms"] == 10.0
    with pytest.raises(RuntimeError, match="released"):
        request.enable_stream_edge_metrics()
    replacement = ClientRequestState("new", external_request_id="external")
    assert replacement.enable_stream_edge_metrics().snapshot() == {}


@pytest.mark.parametrize("reason", ["normal", "client_disconnect", "timeout", "stage_error"])
def test_canonical_cleanup_releases_events_even_without_stage_metrics(mocker, reason):
    omni = object.__new__(OmniBase)
    request = ClientRequestState("internal")
    events = request.enable_stream_edge_metrics()
    edge = events.start_segment(StreamEdgeKey(0, 1))
    observe(edge, StreamEdgeEvent.PRODUCER_EMIT, 10.0)
    omni.request_states = {"internal": request}
    omni._publish_request_gauges = mocker.Mock()
    omni._log_summary_and_cleanup("internal", reason=reason)
    assert omni.request_states == {}
    assert events.snapshot() == {}
    assert not observe(edge, StreamEdgeEvent.SEND_COMPLETE, 12.0)
    with pytest.raises(RuntimeError, match="released"):
        request.enable_stream_edge_metrics()


@dataclass
class StageMetadata:
    final_output: bool = True
    final_output_type: str = "text"


@dataclass
class EngineOutput:
    finished: bool = False
    final_output_type: str = "text"
    stage_durations: dict = field(default_factory=dict)


@dataclass
class Result:
    request_id: str = "internal"
    engine_outputs: EngineOutput = field(default_factory=EngineOutput)
    stage_submit_ts: float = 0.0
    metrics: None = None
    replica_id: int = 0


def test_response_snapshot_is_incremental_detached_and_absent_by_default(mocker):
    omni = object.__new__(OmniBase)
    request = ClientRequestState("internal")
    omni.request_states = {"internal": request}
    omni._enable_ar_profiler = False
    omni.engine = mocker.Mock()
    omni.engine.get_stage_metadata.return_value = StageMetadata()
    omni._publish_request_gauges = mocker.Mock()
    metrics = mocker.Mock(stage_first_ts=[None], stage_last_ts=[None], stage_events={}, e2e_done=set())
    factory = mocker.patch.object(OmniRequestOutput, "from_stage_output")

    def snapshot():
        omni._process_single_result(Result(), 0, metrics, {}, 0.0, 0)
        return factory.call_args.kwargs["metrics"]

    assert "stream_edge_metrics" not in snapshot()
    events = request.enable_stream_edge_metrics()
    edge = events.start_segment(StreamEdgeKey(0, 1))
    observe(edge, StreamEdgeEvent.PRODUCER_EMIT, 10.0)
    first = snapshot()["stream_edge_metrics"]
    assert first["0->1:0"]["edge_first_send_complete_ms"] is None
    observe(edge, StreamEdgeEvent.SEND_COMPLETE, 12.0)
    second = snapshot()["stream_edge_metrics"]
    assert second["0->1:0"]["producer_first_nonempty_emit_ms"] == 10.0
    assert second["0->1:0"]["edge_first_send_complete_ms"] == 12.0
    assert first["0->1:0"]["edge_first_send_complete_ms"] is None


@pytest.mark.parametrize("elapsed_ms", [-1.0, float("inf"), float("nan")])
def test_invalid_elapsed_time_is_rejected_without_state(elapsed_ms):
    events = RequestStreamEdgeEvents()
    edge = events.start_segment(StreamEdgeKey(0, 1))
    with pytest.raises(ValueError, match="finite"):
        observe(edge, StreamEdgeEvent.PRODUCER_EMIT, elapsed_ms)
    assert events.snapshot() == {}
