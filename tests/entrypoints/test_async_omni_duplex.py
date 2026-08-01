from types import SimpleNamespace

import pytest

from vllm_omni.engine.messages import OutputMessage
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.client_request_state import ClientRequestState
from vllm_omni.experimental.fullduplex.engine.duplex_runtime import (
    DuplexOutputAction,
    DuplexOutputDecision,
    duplex_resource_request_id,
)
from vllm_omni.experimental.fullduplex.engine.lease import DuplexLeaseActivity
from vllm_omni.experimental.fullduplex.engine.messages import DuplexFence
from vllm_omni.experimental.fullduplex.output import attach_duplex_output_decision
from vllm_omni.experimental.fullduplex.request_client import DuplexRequestClient
from vllm_omni.metrics.stats import OrchestratorAggregator, StageRequestStats, StageStats
from vllm_omni.outputs import OmniRequestOutput


def test_async_omni_uses_extracted_duplex_request_client():
    app = object.__new__(AsyncOmni)
    app.engine = SimpleNamespace(num_stages=2)
    app.request_states = {}
    app._duplex_request_client = None

    client = app._get_duplex_request_client()

    assert isinstance(client, DuplexRequestClient)
    assert not hasattr(client, "_async_omni")


pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.asyncio
async def test_async_omni_open_duplex_forwards_session_config_and_timeout():
    calls = []
    output_handler_starts = 0

    async def open_duplex_session_async(session_id, **kwargs):
        calls.append((session_id, kwargs))
        return {"ok": True}

    app = object.__new__(AsyncOmni)
    app.engine = SimpleNamespace(open_duplex_session_async=open_duplex_session_async)

    def start_output_handler():
        nonlocal output_handler_starts
        output_handler_starts += 1

    app._final_output_handler = start_output_handler

    result = await app.open_duplex_session_async(
        "sid",
        fence=DuplexFence("sid"),
        capabilities={"implementation_level": "model_native_duplex"},
        session_config={"instructions": "Be brief.", "idle_timeout_s": 30},
        timeout=7.5,
    )

    assert result == {"ok": True}
    assert output_handler_starts == 1
    assert calls == [
        (
            "sid",
            {
                "session_mode": "duplex",
                "capabilities": {"implementation_level": "model_native_duplex"},
                "session_config": {"instructions": "Be brief.", "idle_timeout_s": 30},
                "fence": DuplexFence("sid"),
                "timeout": 7.5,
            },
        )
    ]


@pytest.mark.asyncio
async def test_async_omni_duplex_runtime_controls_forward_timeout():
    calls = []
    cancelled_fence = DuplexFence("sid", epoch=2, turn_id=3)
    next_fence = DuplexFence("sid", epoch=3, turn_id=3)

    async def append_duplex_input_async(session_id, **kwargs):
        calls.append(("append", session_id, kwargs))
        return {"ok": True}

    async def signal_duplex_turn_async(session_id, **kwargs):
        calls.append(("signal", session_id, kwargs))
        return {"ok": True}

    async def close_duplex_session_async(session_id, **kwargs):
        calls.append(("close", session_id, kwargs))
        return {"ok": True}

    async def touch_duplex_session_async(session_id, **kwargs):
        calls.append(("touch", session_id, kwargs))
        return {"ok": True}

    async def resume_duplex_session_async(session_id, **kwargs):
        calls.append(("resume", session_id, kwargs))
        return {"ok": True}

    app = object.__new__(AsyncOmni)
    app.request_states = {}
    app._final_output_handler = lambda: None
    app.engine = SimpleNamespace(
        append_duplex_input_async=append_duplex_input_async,
        signal_duplex_turn_async=signal_duplex_turn_async,
        close_duplex_session_async=close_duplex_session_async,
        touch_duplex_session_async=touch_duplex_session_async,
        resume_duplex_session_async=resume_duplex_session_async,
    )

    await app.append_duplex_input_async(
        "sid",
        mode="append_audio_chunk",
        payload={},
        fence=cancelled_fence,
        timeout=12.5,
    )
    await app.signal_duplex_turn_async(
        "sid",
        event="barge_in",
        fence=cancelled_fence,
        next_fence=next_fence,
        timeout=13.5,
    )
    await app.close_duplex_session_async("sid", reason="done", fence=next_fence, timeout=14.5)
    await app.touch_duplex_session_async(
        "sid",
        fence=next_fence,
        activity=DuplexLeaseActivity.HEARTBEAT,
        timeout=15.5,
    )
    await app.resume_duplex_session_async(
        "sid",
        fence=next_fence,
        expected_lease_generation=3,
        timeout=16.5,
    )

    assert calls == [
        (
            "append",
            "sid",
            {
                "mode": "append_audio_chunk",
                "payload": {},
                "final": False,
                "expected_epoch": None,
                "fence": cancelled_fence,
                "timeout": 12.5,
            },
        ),
        (
            "signal",
            "sid",
            {
                "event": "barge_in",
                "fence": cancelled_fence,
                "next_fence": next_fence,
                "timeout": 13.5,
            },
        ),
        (
            "close",
            "sid",
            {
                "reason": "done",
                "fence": next_fence,
                "timeout": 14.5,
            },
        ),
        (
            "touch",
            "sid",
            {
                "fence": next_fence,
                "activity": DuplexLeaseActivity.HEARTBEAT,
                "timeout": 15.5,
            },
        ),
        (
            "resume",
            "sid",
            {
                "fence": next_fence,
                "expected_lease_generation": 3,
                "timeout": 16.5,
            },
        ),
    ]


@pytest.mark.asyncio
async def test_async_omni_duplex_append_can_defer_data_plane_output_collection():
    app = object.__new__(AsyncOmni)
    request_id = duplex_resource_request_id(DuplexFence("sid"), "stage0")

    async def append_duplex_input_async(session_id, **kwargs):
        del session_id, kwargs
        request_state = app.request_states.get(request_id)
        assert request_state is not None
        assert request_state.metrics is not None
        return {
            "stage_results": [
                {
                    "result": {
                        "data_plane_append": True,
                        "request_id": request_id,
                        "response_stage_id": 1,
                    }
                }
            ]
        }

    app.engine = SimpleNamespace(append_duplex_input_async=append_duplex_input_async)
    app.request_states = {}
    app._final_output_handler = lambda: None

    result = await app.append_duplex_input_async(
        "sid",
        mode="append_audio_chunk",
        payload={},
        fence=DuplexFence("sid"),
        collect_outputs=False,
    )

    assert "data_plane_outputs" not in result


@pytest.mark.asyncio
async def test_async_omni_duplex_append_forwards_fence():
    calls = []

    async def append_duplex_input_async(session_id, **kwargs):
        calls.append((session_id, kwargs))
        return {"ok": True}

    app = object.__new__(AsyncOmni)
    app.request_states = {}
    app._final_output_handler = lambda: None
    app.engine = SimpleNamespace(append_duplex_input_async=append_duplex_input_async)
    fence = DuplexFence("sid", epoch=1, turn_id=2)

    await app.append_duplex_input_async(
        "sid",
        mode="append_audio_chunk",
        payload={},
        fence=fence,
        collect_outputs=False,
    )

    assert calls[0][1]["fence"] is fence


@pytest.mark.asyncio
async def test_async_omni_duplex_collect_waits_for_response_stage():
    app = object.__new__(AsyncOmni)
    req_state = ClientRequestState("duplex-sid-e0-stage0")
    stage0_output = OmniRequestOutput(
        request_id="duplex-sid-e0-stage0",
        stage_id=0,
        finished=False,
    )
    stage1_output = OmniRequestOutput(
        request_id="duplex-sid-e0-stage0",
        stage_id=1,
        final_output_type="audio",
        finished=False,
    )
    await req_state.queue.put(
        OutputMessage(
            request_id="duplex-sid-e0-stage0",
            stage_id=0,
            engine_outputs=stage0_output,
            finished=False,
        )
    )
    await req_state.queue.put(
        OutputMessage(
            request_id="duplex-sid-e0-stage0",
            stage_id=1,
            engine_outputs=stage1_output,
            finished=False,
        )
    )

    outputs = await app._collect_duplex_data_plane_outputs(
        "duplex-sid-e0-stage0",
        req_state,
        response_stage_id=1,
        timeout=1.0,
    )

    assert outputs == [stage1_output]


@pytest.mark.asyncio
async def test_duplex_request_client_retains_output_route_at_segment_terminal():
    request_id = "duplex-sid-e0-stage0"
    request_state = ClientRequestState(request_id)
    request_states = {request_id: request_state}
    output = OmniRequestOutput(
        request_id=request_id,
        stage_id=1,
        final_output_type="audio",
        finished=True,
    )
    await request_state.queue.put(
        OutputMessage(
            request_id=request_id,
            stage_id=1,
            engine_outputs=output,
            finished=True,
        )
    )
    client = DuplexRequestClient(
        SimpleNamespace(),
        SimpleNamespace(
            request_states=request_states,
            num_stages=2,
            log_stats=False,
        ),
    )

    outputs = await client.collect_outputs(
        request_id,
        request_state,
        response_stage_id=1,
        timeout=1.0,
    )

    assert outputs == [output]
    assert request_states[request_id] is request_state


@pytest.mark.asyncio
async def test_async_omni_duplex_direct_output_keeps_stage_metrics():
    app = object.__new__(AsyncOmni)
    app._duplex_request_client = None
    app._final_output_handler = lambda: None
    request_id = "duplex-sid-e0-stage0"
    req_state = ClientRequestState(request_id)
    decision = DuplexOutputDecision(
        action=DuplexOutputAction.DIRECT_RESPONSE,
        metadata={
            "duplex_direct_response": True,
            "duplex_native_decision": "speak",
        },
    )
    direct_output = attach_duplex_output_decision(
        OmniRequestOutput(
            request_id=request_id,
            stage_id=0,
            final_output_type="text",
            finished=True,
        ),
        decision,
    )
    await req_state.queue.put(
        OutputMessage(
            request_id=request_id,
            stage_id=0,
            engine_outputs=direct_output,
            metrics=StageRequestStats(
                batch_id=1,
                batch_size=1,
                num_tokens_in=160,
                num_tokens_out=8,
                stage_gen_time_ms=80.0,
                rx_transfer_bytes=0,
                rx_decode_time_ms=0.0,
                rx_in_flight_time_ms=0.0,
                stage_stats=StageStats(total_token=8, total_gen_time_ms=80.0),
                vllm_ttft_ms=12.5,
                vllm_tpot_ms=9.0,
                vllm_itl_ms=8.75,
                vllm_itls_ms=[8.0, 9.5],
            ),
            finished=True,
        )
    )

    outputs = await app._collect_duplex_data_plane_outputs(
        request_id,
        req_state,
        response_stage_id=1,
        timeout=1.0,
    )

    assert outputs == [direct_output]
    assert direct_output.metrics["stage_metrics"]["0"]["vllm_ttft_ms"] == 12.5
    assert direct_output.metrics["stage_metrics"]["0"]["vllm_tpot_ms"] == 9.0
    assert direct_output.metrics["stage_metrics"]["0"]["vllm_itls_ms"] == [8.0, 9.5]


@pytest.mark.asyncio
async def test_duplex_metrics_cursor_isolates_resumable_collect_windows():
    app = object.__new__(AsyncOmni)
    app._duplex_request_client = None
    app._final_output_handler = lambda: None
    request_id = "duplex-sid-e0-stage0"
    req_state = ClientRequestState(request_id)
    req_state.metrics = OrchestratorAggregator(
        num_stages=2,
        log_stats=False,
        wall_start_ts=0.0,
        final_stage_id_for_e2e=1,
    )

    def stage_message(*, batch_id: int, tokens: int, ttft_ms: float, itls: list[float]) -> OutputMessage:
        decision = DuplexOutputDecision(
            action=DuplexOutputAction.DIRECT_RESPONSE,
            metadata={"duplex_direct_response": True, "duplex_native_decision": "speak"},
        )
        output = attach_duplex_output_decision(
            OmniRequestOutput(
                request_id=request_id,
                stage_id=0,
                final_output_type="text",
                finished=True,
            ),
            decision,
        )
        return OutputMessage(
            request_id=request_id,
            stage_id=0,
            engine_outputs=output,
            metrics=StageRequestStats(
                batch_id=batch_id,
                batch_size=1,
                num_tokens_in=160,
                num_tokens_out=tokens,
                stage_gen_time_ms=80.0,
                rx_transfer_bytes=0,
                rx_decode_time_ms=0.0,
                rx_in_flight_time_ms=0.0,
                stage_stats=StageStats(total_token=tokens, total_gen_time_ms=80.0),
                vllm_ttft_ms=ttft_ms,
                vllm_tpot_ms=9.0,
                vllm_itl_ms=sum(itls) / len(itls),
                vllm_itls_ms=itls,
            ),
            finished=True,
        )

    await req_state.queue.put(stage_message(batch_id=1, tokens=71, ttft_ms=62.5, itls=[10.0, 11.0]))
    first = await app._collect_duplex_data_plane_outputs(
        request_id,
        req_state,
        response_stage_id=1,
        timeout=1.0,
    )
    await req_state.queue.put(stage_message(batch_id=2, tokens=155, ttft_ms=81.0, itls=[12.0, 13.0, 14.0]))
    second = await app._collect_duplex_data_plane_outputs(
        request_id,
        req_state,
        response_stage_id=1,
        timeout=1.0,
    )

    assert first[0].metrics["stage_metrics"]["0"]["num_tokens_out"] == 71
    second_stage0 = second[0].metrics["stage_metrics"]["0"]
    assert second_stage0["num_tokens_out"] == 155
    assert second_stage0["vllm_ttft_ms"] == 81.0
    assert second_stage0["vllm_itls_ms"] == [12.0, 13.0, 14.0]


def test_async_omni_duplex_direct_output_prefers_outer_control_metadata():
    decision = DuplexOutputDecision(
        action=DuplexOutputAction.DIRECT_RESPONSE,
        metadata={
            "duplex_direct_response": True,
            "duplex_native_decision": "listen",
        },
    )
    output = attach_duplex_output_decision(
        OmniRequestOutput(
            request_id="duplex-sid-e0-stage0",
            stage_id=0,
            request_output=SimpleNamespace(
                outputs=[],
                multimodal_output=SimpleNamespace(kind="processed-payload"),
                _custom_output={"inner": "completion"},
            ),
        ),
        decision,
    )

    assert AsyncOmni._is_direct_duplex_data_plane_response(output)


def test_async_omni_drops_unregistered_duplex_prefixed_output():
    app = object.__new__(AsyncOmni)
    app.request_states = {}
    output = OmniRequestOutput(request_id="duplex-unregistered-e0-stage0", stage_id=0)

    keep_processing, request_id, stage_id, request_state = app._handle_output_message(
        OutputMessage(
            request_id=output.request_id,
            stage_id=0,
            engine_outputs=output,
            finished=False,
        )
    )

    assert keep_processing is True
    assert request_id is stage_id is request_state is None
    assert app.request_states == {}


@pytest.mark.asyncio
async def test_async_omni_duplex_collect_wraps_raw_response_stage_output():
    app = object.__new__(AsyncOmni)
    app.log_stats = False
    app._enable_ar_profiler = False
    app.request_states = {}
    app.prom_metrics = SimpleNamespace(
        set_running=lambda value: None,
        set_waiting=lambda value: None,
    )
    app.engine = SimpleNamespace(
        num_stages=2,
        get_stage_metadata=lambda stage_id: SimpleNamespace(
            final_output=stage_id == 1,
            final_output_type="audio",
        ),
    )
    req_state = ClientRequestState("duplex-sid-e0-stage0")
    raw_stage1_output = SimpleNamespace(
        finished=False,
        outputs=[],
        stage_durations={},
        peak_memory_mb=0.0,
        final_output_type="audio",
    )
    await req_state.queue.put(
        OutputMessage(
            request_id="duplex-sid-e0-stage0",
            stage_id=1,
            engine_outputs=raw_stage1_output,
            finished=False,
        )
    )

    outputs = await app._collect_duplex_data_plane_outputs(
        "duplex-sid-e0-stage0",
        req_state,
        response_stage_id=1,
        timeout=1.0,
    )

    assert len(outputs) == 1
    assert outputs[0].request_id == "duplex-sid-e0-stage0"
    assert outputs[0].finished is False
    assert outputs[0].stage_id == 1
    assert outputs[0].final_output_type == "audio"
    assert outputs[0].request_output is raw_stage1_output


def test_async_omni_duplex_request_info_includes_response_stage():
    request_id, response_stage_id = AsyncOmni._duplex_data_plane_request_info(
        {
            "stage_results": [
                {
                    "result": {
                        "data_plane_append": True,
                        "request_id": "duplex-sid-e0-stage0",
                        "response_stage_id": 1,
                    }
                }
            ]
        }
    )

    assert request_id == "duplex-sid-e0-stage0"
    assert response_stage_id == 1


@pytest.mark.asyncio
async def test_duplex_request_client_retains_output_route_when_close_fails():
    fence = DuplexFence("sid-close-route")
    request_id = duplex_resource_request_id(fence, "stage0")
    request_states = {request_id: ClientRequestState(request_id)}

    async def close_duplex_session_async(session_id, **kwargs):
        del session_id, kwargs
        raise RuntimeError("close failed")

    client = DuplexRequestClient(
        SimpleNamespace(close_duplex_session_async=close_duplex_session_async),
        SimpleNamespace(request_states=request_states),
    )

    with pytest.raises(RuntimeError, match="close failed"):
        await client.close(fence.session_id, reason="test", fence=fence, timeout=1.0)

    assert request_id in request_states
