# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_omni.engine.messages import OutputMessage, StageMetricsMessage
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.client_request_state import ClientRequestState
from vllm_omni.entrypoints.omni_base import OmniBase
from vllm_omni.experimental.fullduplex.engine.contracts import duplex_resource_request_id
from vllm_omni.experimental.fullduplex.engine.messages import DuplexFence
from vllm_omni.experimental.fullduplex.openai.protocol import DuplexSession, DuplexSessionConfig, DuplexTurnState
from vllm_omni.experimental.fullduplex.output import attach_duplex_output_decision
from vllm_omni.experimental.fullduplex.request_client import DuplexRequestClient, DuplexRequestOutputPort
from vllm_omni.metrics.duplex_turn import (
    DuplexTurnMetrics,
    accumulate_turn_stage_metrics,
    finalize_duplex_turn_metrics,
    finished_reason_for_cancel,
    is_duplex_resource_request_id,
    make_turn_aggregator,
)
from vllm_omni.metrics.stats import OrchestratorAggregator, StageRequestStats, StageStats
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _fence(session_id: str = "sid-metrics", *, epoch: int = 0) -> DuplexFence:
    return DuplexFence(session_id, epoch=epoch)


def _request_id(fence: DuplexFence | None = None) -> str:
    return duplex_resource_request_id(fence or _fence(), "stage0")


def _stage_stats(**overrides) -> StageRequestStats:
    values = {
        "batch_id": 1,
        "batch_size": 1,
        "num_tokens_in": 0,
        "num_tokens_out": 0,
        "stage_gen_time_ms": 0.0,
        "rx_transfer_bytes": 0,
        "rx_decode_time_ms": 0.0,
        "rx_in_flight_time_ms": 0.0,
        "stage_stats": StageStats(),
    }
    values.update(overrides)
    return StageRequestStats(**values)


def _omni_base(*, log_stats: bool = True, num_stages: int = 3) -> OmniBase:
    obj = object.__new__(OmniBase)
    obj.log_stats = log_stats
    obj.request_states = {}
    obj._consumed_metric_messages = {}
    obj.engine = SimpleNamespace(
        num_stages=num_stages,
        get_stage_metadata=lambda stage_id: SimpleNamespace(
            final_output_type="text" if stage_id == 0 else "audio",
            final_output=stage_id == num_stages - 1,
        ),
        _running_counter=None,
    )
    obj.prom_metrics = SimpleNamespace(set_running=lambda n: None, set_waiting=lambda n: None)
    return obj


def _client(*, log_stats: bool = True, num_stages: int = 3, request_states: dict | None = None) -> DuplexRequestClient:
    return DuplexRequestClient(
        SimpleNamespace(),
        DuplexRequestOutputPort(
            request_states={} if request_states is None else request_states,
            num_stages=num_stages,
            log_stats=log_stats,
            start_output_handler=lambda: None,
            process_single_result=lambda *a, **k: None,
        ),
    )


def test_is_duplex_resource_request_id_matches_engine_format() -> None:
    assert is_duplex_resource_request_id(_request_id())
    assert not is_duplex_resource_request_id("chatcmpl-abc")
    assert not is_duplex_resource_request_id("duplex-sid-e0-stage0")
    assert not is_duplex_resource_request_id(None)


def test_accumulate_appends_segments_collapse_at_finalize() -> None:
    turn = DuplexTurnMetrics(
        request_id=_request_id(),
        response_id="resp-1",
        turn_id=3,
        arrival_ts=1.0,
        aggregator=make_turn_aggregator(num_stages=3, log_stats=False, wall_start_ts=1.0),
    )
    accumulate_turn_stage_metrics(
        turn,
        0,
        _stage_stats(num_tokens_in=10, num_tokens_out=4, stage_gen_time_ms=15.0, vllm_ttft_ms=12.0, vllm_tpot_ms=10.0),
        final_output_type="text",
    )
    accumulate_turn_stage_metrics(
        turn,
        0,
        _stage_stats(
            num_tokens_in=8,
            num_tokens_out=6,
            stage_gen_time_ms=20.0,
            vllm_ttft_ms=99.0,
            vllm_tpot_ms=20.0,
            vllm_itls_ms=[8.0, 9.0],
        ),
        final_output_type="text",
    )
    accumulate_turn_stage_metrics(
        turn,
        1,
        _stage_stats(num_tokens_out=2, stage_gen_time_ms=40.0, audio_duration_s=0.5, audio_generated_frames=8000),
        final_output_type="audio",
    )
    accumulate_turn_stage_metrics(
        turn,
        1,
        _stage_stats(num_tokens_out=1, stage_gen_time_ms=10.0, audio_duration_s=0.25, audio_generated_frames=4000),
        final_output_type="audio",
    )

    events = turn.aggregator.stage_events[turn.response_id]
    assert [event.stage_id for event in events] == [0, 0, 1, 1]

    assert finalize_duplex_turn_metrics(turn, reason="stop") is True
    collapsed = turn.aggregator.stage_events[turn.response_id]
    assert [event.stage_id for event in collapsed] == [0, 1]
    stage0 = collapsed[0]
    assert stage0.num_tokens_in == 18
    assert stage0.num_tokens_out == 10
    assert stage0.stage_gen_time_ms == 35.0
    assert stage0.vllm_ttft_ms == 12.0
    assert stage0.vllm_itls_ms == [8.0, 9.0]
    assert stage0.vllm_tpot_ms == pytest.approx((10.0 * 3 + 20.0 * 5) / 8.0)
    assert collapsed[1].audio_duration_s == 0.75
    assert collapsed[1].audio_generated_frames == 12000


def test_finalize_logs_once_with_identity_line(mocker) -> None:
    request_id = _request_id()
    aggregator = make_turn_aggregator(num_stages=2, log_stats=True, wall_start_ts=1.0)
    spy = mocker.spy(aggregator, "build_and_log_summary")
    logged = mocker.patch("vllm_omni.metrics.stats.logger.info")
    turn = DuplexTurnMetrics(
        request_id=request_id,
        response_id="resp-turn",
        turn_id=3,
        arrival_ts=1.0,
        aggregator=aggregator,
    )
    accumulate_turn_stage_metrics(turn, 0, _stage_stats(num_tokens_out=3, stage_gen_time_ms=11.0))

    assert finalize_duplex_turn_metrics(turn, reason="stop") is True
    assert finalize_duplex_turn_metrics(turn, reason="abort") is False
    spy.assert_called_once_with()
    timing = [call for call in logged.call_args_list if call.args and call.args[0] == "[OmniTiming] %s"]
    assert len(timing) == 1
    identity = timing[0].args[1]
    assert request_id in identity
    assert "response=resp-turn" in identity
    assert "turn=3" in identity
    assert "reason=stop" in identity
    assert "total=" not in identity
    assert "engine=" not in identity
    assert "resp-turn" in aggregator.e2e_done
    assert turn.finished_reason == "stop"


def test_finished_reason_passthrough(mocker) -> None:
    logged = mocker.patch("vllm_omni.metrics.stats.logger.info")
    for reason in ("barge_in", "cancel", "close", "abort", "error"):
        turn = DuplexTurnMetrics(
            request_id=_request_id(),
            response_id=f"resp-{reason}",
            arrival_ts=1.0,
            aggregator=make_turn_aggregator(num_stages=1, log_stats=True, wall_start_ts=1.0),
        )
        assert finalize_duplex_turn_metrics(turn, reason=reason) is True
        assert turn.finished_reason == reason
    barge_lines = [
        call.args[1]
        for call in logged.call_args_list
        if call.args and call.args[0] == "[OmniTiming] %s" and "reason=barge_in" in call.args[1]
    ]
    assert barge_lines
    unknown = DuplexTurnMetrics(
        request_id=_request_id(),
        response_id="resp-unknown",
        arrival_ts=1.0,
        aggregator=make_turn_aggregator(num_stages=1, log_stats=False, wall_start_ts=1.0),
    )
    assert finalize_duplex_turn_metrics(unknown, reason="oops") is True
    assert unknown.finished_reason == "abort"


def test_finished_reason_for_cancel_mapping() -> None:
    assert finished_reason_for_cancel("barge_in") == "barge_in"
    assert finished_reason_for_cancel("session_close") == "close"
    assert finished_reason_for_cancel("disconnect") == "close"
    assert finished_reason_for_cancel("disconnect_grace_expired") == "close"
    assert finished_reason_for_cancel("timeout") == "cancel"
    assert finished_reason_for_cancel("new_response") == "cancel"
    assert finished_reason_for_cancel("output_audio_buffer_clear") == "cancel"
    assert finished_reason_for_cancel("client_cancelled") == "cancel"
    assert finished_reason_for_cancel("stop") == "stop"


def test_accumulate_prefers_stage_submit_ts(mocker) -> None:
    mocker.patch("vllm_omni.metrics.duplex_turn.time.time", return_value=200.0)
    turn = DuplexTurnMetrics(
        request_id=_request_id(),
        response_id="resp-ts",
        arrival_ts=1.0,
        aggregator=make_turn_aggregator(num_stages=2, log_stats=False, wall_start_ts=1.0),
    )
    accumulate_turn_stage_metrics(
        turn,
        0,
        _stage_stats(num_tokens_out=1, stage_gen_time_ms=5.0),
        stage_submit_ts=100.0,
    )
    accumulate_turn_stage_metrics(turn, 1, _stage_stats(num_tokens_out=1, stage_gen_time_ms=5.0))
    assert turn.aggregator.stage_first_ts[0] == 100.0
    assert turn.aggregator.stage_last_ts[0] == 200.0
    assert turn.aggregator.stage_first_ts[1] == 200.0


def test_stage_metrics_message_forwards_submit_ts() -> None:
    obj = _omni_base()
    request_id = _request_id()
    client = _client(request_states=obj.request_states, num_stages=3)
    req_state = ClientRequestState(request_id)
    obj.request_states[request_id] = req_state
    client.begin_turn_metrics(request_id, response_id="resp-submit", turn_id=1, arrival_ts=1.0)

    obj._handle_output_message(
        StageMetricsMessage(
            request_id=request_id,
            stage_id=0,
            metrics=_stage_stats(num_tokens_out=2, stage_gen_time_ms=8.0),
            stage_submit_ts=50.0,
        )
    )
    turn = req_state.duplex_turn
    assert turn is not None
    assert turn.aggregator.stage_first_ts[0] == 50.0


def test_metrics_before_begin_are_flushed_on_begin() -> None:
    obj = _omni_base()
    request_id = _request_id()
    client = _client(request_states=obj.request_states, num_stages=3)
    req_state = ClientRequestState(request_id)
    req_state.metrics = OrchestratorAggregator(3, True, 0.0, 2)
    obj.request_states[request_id] = req_state

    obj._handle_output_message(
        StageMetricsMessage(
            request_id=request_id,
            stage_id=0,
            metrics=_stage_stats(num_tokens_out=7, stage_gen_time_ms=30.0),
            stage_submit_ts=50.0,
        )
    )

    assert req_state.duplex_turn is None
    assert req_state.metrics.stage_events[request_id][0].num_tokens_out == 7
    assert len(req_state.duplex_turn_pending) == 1
    assert req_state.duplex_turn_arrival_ts == 50.0

    turn = client.begin_turn_metrics(request_id, response_id="resp-auto")
    assert turn is not None
    assert turn.arrival_ts == 50.0
    events = turn.aggregator.stage_events[turn.response_id]
    assert len(events) == 1
    assert events[0].num_tokens_out == 7
    assert events[0].stage_gen_time_ms == 30.0
    assert req_state.duplex_turn_pending == []
    assert req_state.duplex_turn_arrival_ts is None


def test_finalize_noop_when_log_stats_off(mocker) -> None:
    aggregator = make_turn_aggregator(num_stages=1, log_stats=False, wall_start_ts=1.0)
    spy = mocker.spy(aggregator, "build_and_log_summary")
    turn = DuplexTurnMetrics(
        request_id=_request_id(),
        response_id="resp-quiet",
        arrival_ts=1.0,
        aggregator=aggregator,
    )
    assert finalize_duplex_turn_metrics(turn, reason="stop") is True
    spy.assert_not_called()


def test_client_two_turns_emit_two_summaries(mocker) -> None:
    client = _client()
    request_id = _request_id()
    logged = []

    def fake_summary(self):
        logged.append(self.stage_events)
        return {}

    mocker.patch("vllm_omni.metrics.stats.OrchestratorAggregator.build_and_log_summary", fake_summary)

    turn_a = client.begin_turn_metrics(request_id, response_id="resp-a", turn_id=1, arrival_ts=1.0)
    accumulate_turn_stage_metrics(turn_a, 0, _stage_stats(num_tokens_out=2, stage_gen_time_ms=10.0))
    assert client.finalize_turn_metrics(request_id, reason="stop") is True

    turn_b = client.begin_turn_metrics(request_id, response_id="resp-b", turn_id=2, arrival_ts=2.0)
    accumulate_turn_stage_metrics(turn_b, 0, _stage_stats(num_tokens_out=5, stage_gen_time_ms=20.0))
    assert client.finalize_turn_metrics(request_id, reason="stop") is True

    assert len(logged) == 2
    assert "resp-a" in logged[0]
    assert "resp-b" in logged[1]
    req_state = client.output_port.request_states[request_id]
    assert req_state.duplex_turn is None


def test_begin_preregisters_and_ignores_chat_ids() -> None:
    client = _client()
    request_id = _request_id()
    turn = client.begin_turn_metrics(request_id, response_id="resp-pre", turn_id=1)
    assert turn is not None
    assert request_id in client.output_port.request_states
    assert client.output_port.request_states[request_id].duplex_turn is turn
    assert client.begin_turn_metrics("chatcmpl-1", response_id="resp-x") is None


def test_stage_metrics_message_records_into_open_turn() -> None:
    obj = _omni_base()
    request_id = _request_id()
    client = _client(request_states=obj.request_states, num_stages=3)
    req_state = ClientRequestState(request_id)
    req_state.metrics = OrchestratorAggregator(3, True, 0.0, 2)
    obj.request_states[request_id] = req_state
    client.begin_turn_metrics(request_id, response_id="resp-msg", turn_id=1, arrival_ts=1.0)

    obj._handle_output_message(
        StageMetricsMessage(
            request_id=request_id,
            stage_id=0,
            metrics=_stage_stats(num_tokens_out=7, stage_gen_time_ms=30.0, vllm_ttft_ms=4.0),
        )
    )
    obj._handle_output_message(
        OutputMessage(
            request_id=request_id,
            stage_id=1,
            engine_outputs=OmniRequestOutput(request_id=request_id, stage_id=1, finished=True),
            metrics=_stage_stats(num_tokens_out=1, stage_gen_time_ms=50.0, audio_duration_s=1.0),
            finished=True,
        )
    )

    turn = req_state.duplex_turn
    assert turn is not None
    events = turn.aggregator.stage_events[turn.response_id]
    assert [event.stage_id for event in events] == [0, 1]
    assert events[0].num_tokens_out == 7
    assert events[1].audio_duration_s == 1.0
    assert req_state.metrics.stage_events[request_id][0].num_tokens_out == 7


def test_chat_log_summary_and_cleanup_does_not_use_duplex_turn(mocker) -> None:
    obj = _omni_base()
    req_state = ClientRequestState("chatcmpl-1")
    summary = mocker.Mock(return_value={})
    req_state.metrics = SimpleNamespace(e2e_done={"chatcmpl-1"}, build_and_log_summary=summary)
    obj.request_states["chatcmpl-1"] = req_state

    obj._log_summary_and_cleanup("chatcmpl-1")

    summary.assert_called_once_with()
    assert "chatcmpl-1" not in obj.request_states


def test_session_begin_end_hooks_capture_ids_before_clear() -> None:
    session = DuplexSession(session_id="s", config=DuplexSessionConfig())
    events: list[tuple] = []
    session.on_response_begin = lambda s: events.append(("begin", s.active_response_id, s.active_request_id))
    session.on_response_end = lambda s, **kwargs: events.append(("end", kwargs))
    session.bind_request("rid-1")
    response_id = session.begin_response(turn_id=4)
    session.end_response(finished_reason="abort")
    assert events[0] == ("begin", response_id, "rid-1")
    assert events[1][1]["request_id"] == "rid-1"
    assert events[1][1]["response_id"] == response_id
    assert events[1][1]["reason"] == "abort"
    assert session.active_request_id is None
    assert session.active_response_id is None


@pytest.mark.asyncio
async def test_signal_barge_in_finalizes_before_pop(mocker) -> None:
    fence = _fence()
    request_id = _request_id(fence)
    logged = []

    def fake_barge_in_summary(self):
        logged.append(self.stage_events.copy())
        return {}

    mocker.patch(
        "vllm_omni.metrics.stats.OrchestratorAggregator.build_and_log_summary",
        fake_barge_in_summary,
    )
    client = _client()
    turn = client.begin_turn_metrics(request_id, response_id="resp-barge", turn_id=1, arrival_ts=1.0)
    accumulate_turn_stage_metrics(turn, 0, _stage_stats(num_tokens_out=3, stage_gen_time_ms=9.0))
    reasons: list[str] = []
    real_finalize = finalize_duplex_turn_metrics

    def capture_finalize(captured_turn, *, reason):
        emitted = real_finalize(captured_turn, reason=reason)
        reasons.append(captured_turn.finished_reason or "")
        return emitted

    mocker.patch("vllm_omni.experimental.fullduplex.request_client.finalize_duplex_turn_metrics", capture_finalize)

    async def signal_duplex_turn_async(session_id, **kwargs):
        return {"ok": True}

    client.engine = SimpleNamespace(signal_duplex_turn_async=signal_duplex_turn_async)
    await client.signal(
        "sid-metrics",
        event="barge_in",
        fence=fence,
        next_fence=DuplexFence("sid-metrics", epoch=1),
        session_config=None,
        runtime_config=None,
        timeout=1.0,
    )

    assert request_id not in client.output_port.request_states
    assert len(logged) == 1
    assert "resp-barge" in logged[0]
    assert reasons == ["barge_in"]


@pytest.mark.asyncio
async def test_close_without_open_turn_does_not_log(mocker) -> None:
    fence = _fence()
    request_id = _request_id(fence)
    client = _client()
    client.output_port.request_states[request_id] = ClientRequestState(request_id)
    spy = mocker.spy(OrchestratorAggregator, "build_and_log_summary")

    async def close_duplex_session_async(session_id, **kwargs):
        return {"ok": True}

    client.engine = SimpleNamespace(close_duplex_session_async=close_duplex_session_async)
    await client.close("sid-metrics", reason="done", fence=fence, timeout=1.0)
    spy.assert_not_called()
    assert request_id not in client.output_port.request_states


@pytest.mark.asyncio
async def test_close_open_turn_emits_close(mocker) -> None:
    fence = _fence()
    request_id = _request_id(fence)
    reasons: list[str] = []
    real_finalize = finalize_duplex_turn_metrics

    def capture_finalize(captured_turn, *, reason):
        emitted = real_finalize(captured_turn, reason=reason)
        reasons.append(captured_turn.finished_reason or "")
        return emitted

    mocker.patch("vllm_omni.experimental.fullduplex.request_client.finalize_duplex_turn_metrics", capture_finalize)
    client = _client()
    turn = client.begin_turn_metrics(request_id, response_id="resp-close", turn_id=1, arrival_ts=1.0)
    accumulate_turn_stage_metrics(turn, 0, _stage_stats(num_tokens_out=1, stage_gen_time_ms=4.0))

    async def close_duplex_session_async(session_id, **kwargs):
        return {"ok": True}

    client.engine = SimpleNamespace(close_duplex_session_async=close_duplex_session_async)
    await client.close("sid-metrics", reason="done", fence=fence, timeout=1.0)
    assert reasons == ["close"]
    assert request_id not in client.output_port.request_states


@pytest.mark.asyncio
async def test_session_close_cancel_then_close_logs_once(mocker) -> None:
    fence = _fence()
    request_id = _request_id(fence)
    logged = []

    def fake_summary(self):
        logged.append(self.timing_identity)
        return {}

    mocker.patch("vllm_omni.metrics.stats.OrchestratorAggregator.build_and_log_summary", fake_summary)
    client = _client()
    turn = client.begin_turn_metrics(request_id, response_id="resp-close-once", turn_id=1, arrival_ts=1.0)
    accumulate_turn_stage_metrics(turn, 0, _stage_stats(num_tokens_out=1, stage_gen_time_ms=4.0))
    assert client.finalize_turn_metrics(request_id, reason="close") is True

    async def close_duplex_session_async(session_id, **kwargs):
        return {"ok": True}

    client.engine = SimpleNamespace(close_duplex_session_async=close_duplex_session_async)
    await client.close("sid-metrics", reason="session_close", fence=fence, timeout=1.0)
    assert len(logged) == 1
    assert logged[0]["reason"] == "close"


def test_session_transfer_tx_copied_into_turn() -> None:
    request_id = _request_id()
    client = _client(num_stages=2)
    req_state = ClientRequestState(request_id)
    req_state.metrics = OrchestratorAggregator(2, False, 1.0, 1)
    client.output_port.request_states[request_id] = req_state
    req_state.metrics.on_forward(0, 1, request_id, 512, 3.0, False)

    turn = client.begin_turn_metrics(request_id, response_id="resp-tx", turn_id=1, arrival_ts=1.0)
    req_state.metrics.on_forward(0, 1, request_id, 1024, 7.5, True)
    accumulate_turn_stage_metrics(
        turn,
        1,
        _stage_stats(
            num_tokens_out=1,
            stage_gen_time_ms=5.0,
            rx_transfer_bytes=1024,
            rx_decode_time_ms=1.0,
            rx_in_flight_time_ms=0.5,
        ),
        final_output_type="audio",
    )
    assert client.finalize_turn_metrics(request_id, reason="stop") is True
    evt = turn.aggregator.transfer_events[(0, 1, "resp-tx")]
    assert evt.tx_time_ms == pytest.approx(7.5)
    assert evt.rx_decode_time_ms == pytest.approx(1.0)
    assert evt.in_flight_time_ms == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_abort_emits_table_once_then_close_is_noop(mocker) -> None:
    request_id = _request_id()
    obj = object.__new__(AsyncOmni)
    obj.log_stats = True
    obj.request_states = {}
    obj._duplex_request_client = None
    obj._final_output_handler = lambda: None
    obj._process_single_result = lambda *a, **k: None
    obj.prom_metrics = SimpleNamespace(
        request_failed=lambda: None,
        inc_requests_failed=lambda reason: None,
        set_running=lambda n: None,
        set_waiting=lambda n: None,
    )
    obj.engine = SimpleNamespace(
        num_stages=2,
        get_stage_metadata=lambda stage_id: SimpleNamespace(final_output_type="text"),
        abort_async=mocker.AsyncMock(return_value=[]),
        _running_counter=None,
    )
    logged = []

    def fake_abort_summary(self):
        logged.append(True)
        return {}

    mocker.patch(
        "vllm_omni.metrics.stats.OrchestratorAggregator.build_and_log_summary",
        fake_abort_summary,
    )
    obj.begin_duplex_turn_metrics(request_id, response_id="resp-abort", turn_id=1, arrival_ts=1.0)
    turn = obj.request_states[request_id].duplex_turn
    accumulate_turn_stage_metrics(turn, 0, _stage_stats(num_tokens_out=2, stage_gen_time_ms=8.0))

    await obj._abort([request_id])
    assert len(logged) == 1
    assert request_id in obj.request_states
    assert obj.request_states[request_id].duplex_turn is None
    assert obj.finalize_duplex_turn_metrics(request_id, reason="abort") is False


def test_duplex_direct_output_does_not_double_count_stage0() -> None:
    obj = _omni_base()
    request_id = _request_id()
    client = _client(request_states=obj.request_states, num_stages=3)
    req_state = ClientRequestState(request_id)
    obj.request_states[request_id] = req_state
    client.begin_turn_metrics(request_id, response_id="resp-listen", turn_id=1, arrival_ts=1.0)

    stats = _stage_stats(num_tokens_in=10, num_tokens_out=4, stage_gen_time_ms=15.0)
    obj._handle_output_message(
        StageMetricsMessage(
            request_id=request_id,
            stage_id=0,
            metrics=stats,
        )
    )
    engine_outputs = attach_duplex_output_decision(
        OmniRequestOutput(request_id=request_id, stage_id=0, finished=True),
        SimpleNamespace(action="direct_response", final_output_type="text"),
    )
    obj._handle_output_message(
        OutputMessage(
            request_id=request_id,
            stage_id=0,
            engine_outputs=engine_outputs,
            metrics=stats,
            finished=True,
        )
    )

    turn = req_state.duplex_turn
    assert turn is not None
    events = turn.aggregator.stage_events[turn.response_id]
    assert len(events) == 1
    assert events[0].num_tokens_in == 10
    assert events[0].num_tokens_out == 4
    assert events[0].stage_gen_time_ms == 15.0


def test_chat_request_does_not_buffer_duplex_pending() -> None:
    obj = _omni_base()
    req_state = ClientRequestState("chatcmpl-1")
    obj.request_states["chatcmpl-1"] = req_state
    obj._handle_output_message(
        StageMetricsMessage(
            request_id="chatcmpl-1",
            stage_id=0,
            metrics=_stage_stats(num_tokens_out=2, stage_gen_time_ms=8.0),
        )
    )
    assert req_state.duplex_turn_pending == []
    assert req_state.duplex_turn_arrival_ts is None


def test_mark_turn_arrival_stamps_before_begin(mocker) -> None:
    mocker.patch("vllm_omni.experimental.fullduplex.request_client.time.time", return_value=42.0)
    client = _client()
    request_id = _request_id()
    ts = client.mark_turn_arrival(request_id)
    assert ts == 42.0
    mocker.patch("vllm_omni.experimental.fullduplex.request_client.time.time", return_value=99.0)
    assert client.mark_turn_arrival(request_id) == 42.0
    turn = client.begin_turn_metrics(request_id, response_id="resp-arr")
    assert turn is not None
    assert turn.arrival_ts == 42.0
    assert client.mark_turn_arrival("chatcmpl-1") is None


def test_duplex_dump_titles_hide_ttfo_and_skip_total_engine(mocker) -> None:
    logged = mocker.patch("vllm_omni.metrics.stats.logger.info")
    request_id = _request_id()
    turn = DuplexTurnMetrics(
        request_id=request_id,
        response_id="resp-dump",
        turn_id=4,
        arrival_ts=1.0,
        aggregator=make_turn_aggregator(num_stages=2, log_stats=True, wall_start_ts=1.0),
    )
    accumulate_turn_stage_metrics(
        turn,
        0,
        _stage_stats(
            num_tokens_out=3,
            stage_gen_time_ms=11.0,
            serving_time_to_first_output_ms=1_757_000_000.0,
        ),
    )
    assert finalize_duplex_turn_metrics(turn, reason="stop") is True
    blobs = []
    for call in logged.call_args_list:
        if not call.args:
            continue
        if call.args[0] == "[OmniTiming] %s":
            blobs.append(call.args[1])
        elif call.args[0] == "\n%s":
            blobs.append(call.args[1])
    joined = "\n".join(blobs)
    assert f"request_id={request_id}" in joined
    assert "response=resp-dump" in joined
    assert "turn=4" in joined
    assert "reason=stop" in joined
    assert "StageRequestStats" in joined
    assert "serving_time_to_first_output_ms" not in joined
    timing = next(text for text in blobs if text.startswith("req="))
    assert "total=" not in timing
    assert "engine=" not in timing


def test_response_hooks_do_not_raise_into_begin_end() -> None:
    session = DuplexSession(session_id="s", config=DuplexSessionConfig())

    def boom_begin(sess: DuplexSession) -> None:
        raise RuntimeError("begin metrics failed")

    def boom_end(sess: DuplexSession, **kwargs) -> None:
        raise RuntimeError("end metrics failed")

    session.on_response_begin = boom_begin
    session.on_response_end = boom_end
    session.bind_request("rid-1")
    response_id = session.begin_response(turn_id=4)
    assert session.active_response_id == response_id
    assert session.turn_state == DuplexTurnState.ASSISTANT_GENERATING
    session.end_response(finished_reason="stop")
    assert session.active_response_id is None
    assert session.turn_state == DuplexTurnState.IDLE


def test_finalize_survives_summary_failure(mocker) -> None:
    aggregator = make_turn_aggregator(num_stages=1, log_stats=True, wall_start_ts=1.0)
    mocker.patch(
        "vllm_omni.metrics.stats.OrchestratorAggregator.build_and_log_summary",
        side_effect=RuntimeError("log boom"),
    )
    turn = DuplexTurnMetrics(
        request_id=_request_id(),
        response_id="resp-fail",
        arrival_ts=1.0,
        aggregator=aggregator,
    )
    accumulate_turn_stage_metrics(turn, 0, _stage_stats(num_tokens_out=2, stage_gen_time_ms=8.0))
    assert finalize_duplex_turn_metrics(turn, reason="stop") is True
    assert turn.finalized is True
