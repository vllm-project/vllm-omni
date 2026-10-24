# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real orchestrator and StagePool logic with lightweight engine clients (L1)."""

import asyncio
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreOutputs, FinishReason

from tests.engine.test_async_omni_engine_input import _FakeHub, _replica, _RoundRobinLB
from tests.engine.test_orchestrator import (
    FakeOutputProcessor,
    FakePromptRequest,
    FakeRunningCounter,
    FakeStageClient,
    _build_request_output,
    _build_stage_pools,
    _engine_core_outputs,
    _enqueue_add_request,
    _get_output_message,
    _get_rpc_message,
    _processed_request,
    _shutdown_orchestrator,
    _wait_for,
    orchestrator_factory,
)
from tests.engine.test_orchestrator_stage_input_bridge import FakeInputProcessor
from vllm_omni.config.stage_config import PipelineConfig, StagePipelineConfig
from vllm_omni.config.stage_routing import StageRouting
from vllm_omni.engine import OmniEngineCoreOutput
from vllm_omni.engine.messages import AbortRequestMessage, AbortResultMessage, ErrorMessage, StageSubmissionMessage
from vllm_omni.engine.omni_engine_base import OmniEngineBase
from vllm_omni.engine.orchestrator import Orchestrator, OrchestratorRequestState, StreamingSegmentState
from vllm_omni.engine.stage_pool import RequestClosedError
from vllm_omni.entrypoints.async_omni_base import AsyncOmniBase
from vllm_omni.entrypoints.client_request_state import ClientRequestState
from vllm_omni.entrypoints.omni_base import OmniBase
from vllm_omni.entrypoints.utils import get_final_stage_id_for_e2e
from vllm_omni.metrics.stats import OrchestratorAggregator
from vllm_omni.outputs.output_processor import MultimodalOutputProcessor

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
configured_orchestrator_factory = orchestrator_factory


def _direct_orchestrator(transitions=((0, 2),), *, async_chunk=False):
    final_stage_id = StageRouting.from_transitions(3, transitions).stage_order[-1]
    clients = [FakeStageClient(final_output=(sid == final_stage_id)) for sid in range(3)]
    pools = _build_stage_pools([[client] for client in clients])
    orch = Orchestrator(
        request_async_queue=asyncio.Queue(),
        output_async_queue=asyncio.Queue(),
        rpc_async_queue=asyncio.Queue(),
        stage_pools=pools,
        stage_transitions=transitions,
        async_chunk=async_chunk,
    )
    state = OrchestratorRequestState(
        request_id="r",
        sampling_params_list=[SamplingParams(max_tokens=1) for _ in pools],
        final_stage_id=final_stage_id,
        final_output_stage_ids={final_stage_id},
        prompt={"prompt_token_ids": [1, 2]},
    )
    orch.request_states["r"] = state
    return orch, state, clients


def _submission(**overrides):
    defaults = dict(
        type="add_request",
        request_id="r",
        prompt=FakePromptRequest(request_id="r", prompt_token_ids=[1], resumable=False),
        original_prompt={},
        output_prompt_text=None,
        request_timestamp=1.0,
        preprocess_ms=0.0,
        enqueue_ts=0.0,
        sampling_params_list=[SamplingParams(max_tokens=1) for _ in range(3)],
        final_stage_id=2,
        final_output_stage_ids=[2],
    )
    return StageSubmissionMessage(**(defaults | overrides))


@pytest.mark.asyncio
@pytest.mark.parametrize("transitions", [None, ((0, 1), (1, 2)), ((0, 2), (2, 1))])
async def test_bridge_sender_is_bound_before_receiver_prewarm(mocker, transitions):
    order = StageRouting.from_transitions(3, transitions).stage_order
    bridge, receiver = order[1:]
    clients = [[FakeStageClient(next_inputs=[{"prompt_token_ids": [11]}])] for _ in range(3)]
    clients[bridge].append(FakeStageClient(next_inputs=[{"prompt_token_ids": [11]}]))
    clients[receiver][0].final_output = True
    for rid, client in enumerate(clients[bridge]):
        mocker.patch.object(
            client, "get_payload_sender_info", create=True, return_value={"host": "127.0.0.1", "zmq_port": 50051 + rid}
        )
    configs: list[object] = [
        SimpleNamespace(
            model_config=SimpleNamespace(
                max_model_len=64,
                stage_connector_config={"extra": {"role": "receiver" if sid == receiver else "sender"}},
            )
        )
        for sid in range(3)
    ]
    pools = _build_stage_pools(clients, stage_vllm_configs=configs)
    pools[bridge].mark_replica_unavailable(0)
    orch = Orchestrator(
        request_async_queue=asyncio.Queue(),
        output_async_queue=asyncio.Queue(),
        rpc_async_queue=asyncio.Queue(),
        stage_pools=pools,
        stage_transitions=transitions,
        async_chunk=True,
    )
    state = OrchestratorRequestState(
        request_id="r",
        sampling_params_list=[SamplingParams(max_tokens=1) for _ in pools],
        final_stage_id=receiver,
        final_output_stage_ids={receiver},
        prompt={"prompt_token_ids": [1, 2]},
    )
    orch.request_states["r"] = state
    pools[0].select_replica_id("r")
    assert await orch._prewarm_async_chunk_stages(
        "r", FakePromptRequest(request_id="r", prompt_token_ids=[1, 2], resumable=False), state
    )
    assert pools[bridge].get_bound_replica_id("r") == 1
    assert bridge not in state.stage_submit_ts
    assert not any(client.add_request_calls for client in clients[bridge])
    advertised = clients[receiver][0].add_request_calls[0][0].payload_sender_info
    await orch._forward_to_next_stage("r", 0, _build_request_output("r"), state)
    assert len(clients[bridge][1].add_request_calls) == 1
    assert not clients[bridge][0].add_request_calls
    assert advertised == pools[bridge].get_bound_client("r").get_payload_sender_info()
    await orch._cleanup_request_ids(["r"], abort=True)
    assert all(pool.get_bound_replica_id("r") is None for pool in pools)


@pytest.mark.asyncio
@pytest.mark.parametrize("transitions", [((0, 2),), ((0, 2), (2, 1))])
@pytest.mark.parametrize("raw_terminal", [False, True])
async def test_endpoint_completion_waits_for_other_selected_output(transitions, raw_terminal):
    orch, state, clients = _direct_orchestrator(transitions, async_chunk=True)
    endpoint = state.final_stage_id
    other = 0
    clients[other].final_output = True
    state.final_output_stage_ids = {other, endpoint}
    state.stage_submit_ts = {sid: 1.0 for sid in orch._request_stage_path(state)}
    state.finished_stage_ids.update(set(state.stage_submit_ts) - {other, endpoint})
    if raw_terminal:
        raw = OmniEngineCoreOutput(request_id="r", new_token_ids=[], finish_reason=FinishReason.STOP)
        await orch._apply_raw_terminal_stage_finish(endpoint, raw, state)
        await orch._finish_raw_terminal_requests(endpoint, 0, {"r"})
    else:
        await orch._route_output(endpoint, 0, _build_request_output("r"), state, None)
    assert orch.output_async_queue.empty()
    assert state.pending_final_output.stage_id == endpoint
    await orch._route_output(other, 0, _build_request_output("r"), state, None)
    intermediate = orch.output_async_queue.get_nowait()
    terminal = orch.output_async_queue.get_nowait()
    assert intermediate.stage_id == other and not intermediate.finished
    assert terminal.stage_id == endpoint and terminal.finished
    assert orch.output_async_queue.empty() and "r" not in orch.request_states


@pytest.mark.asyncio
@pytest.mark.parametrize("transitions", [None, ((0, 2), (2, 1))])
async def test_abort_outputs_follow_route_and_finish_each_request_once(transitions):
    orch, state, clients = _direct_orchestrator(transitions)
    for client in clients:
        client.final_output = True
    state.final_output_stage_ids = {1, 2}
    other = OrchestratorRequestState(
        request_id="other", final_stage_id=state.final_stage_id, final_output_stage_ids={1, 2}
    )
    orch.request_states["other"] = other
    for pool in orch.stage_pools:
        for rid in ("r", "other"):
            pool.select_replica_id(rid)
    outputs = await orch._cleanup_request_ids(["r", "other"], abort=True)
    for rid in ("r", "other"):
        selected = [msg for msg in outputs if msg.request_id == rid]
        assert [msg.stage_id for msg in selected] == [sid for sid in orch._stage_routing.stage_order if sid in {1, 2}]
        assert [msg.finished for msg in selected] == [False, True]
        assert selected[-1].stage_id == state.final_stage_id
    assert not orch.request_states


@pytest.mark.asyncio
async def test_empty_bridge_aborts_prewarmed_receiver_and_preserves_submission_metrics():
    orch, state, clients = _direct_orchestrator(((0, 2), (2, 1)), async_chunk=True)
    orch.stage_pools[2].stage_vllm_config.model_config.stage_connector_config = {"extra": {"role": "sender"}}
    endpoint_pool = orch.stage_pools[1]
    receiver = FakeStageClient(final_output=True)
    endpoint_pool.add_client("receiver-1", receiver, replica_id=1)
    endpoint_pool.mark_replica_unavailable(0)
    orch.stage_pools[0].select_replica_id("r")
    state.stage_submit_ts[0] = 1.0
    assert await orch._prewarm_async_chunk_stages(
        "r", FakePromptRequest(request_id="r", prompt_token_ids=[1], resumable=False), state
    )
    submit_ts = state.stage_submit_ts[1]
    await orch._forward_to_next_stage("r", 0, _build_request_output("r"), state)
    terminal = orch.output_async_queue.get_nowait()
    assert terminal.finished and terminal.stage_id == 1 and terminal.replica_id == 1
    assert terminal.stage_submit_ts == submit_ts
    assert set(state.stage_submit_ts) == {0, 1}
    assert receiver.abort_calls == [["r"]]
    assert not clients[2].add_request_calls
    assert "r" not in orch.request_states
    assert all(pool.get_bound_replica_id("r") is None for pool in orch.stage_pools)


@pytest.mark.asyncio
@pytest.mark.parametrize("transitions", [None, ((0, 2),), ((0, 2), (2, 1))])
@pytest.mark.parametrize("empty_inputs", [None, []])
async def test_empty_diffusion_inputs_fail_only_request(mocker, transitions, empty_inputs):
    orch, state, clients = _direct_orchestrator(transitions)
    target = state.final_stage_id
    source = orch._request_stage_path(state)[-2]
    clients[target].stage_type = "diffusion"
    clients[target].custom_process_input_func = mocker.Mock(return_value=empty_inputs)
    orch.stage_pools[source].select_replica_id("r")

    await orch._forward_to_next_stage("r", source, _build_request_output("r"), state)

    terminal = orch.output_async_queue.get_nowait()
    assert terminal.finished and terminal.stage_id == target
    assert terminal.engine_outputs.error is not None
    assert "r" not in orch.request_states
    assert not clients[target].add_request_calls
    assert all(pool.get_bound_replica_id("r") is None for pool in orch.stage_pools)
    assert orch.rpc_async_queue.empty()


@pytest.mark.asyncio
@pytest.mark.parametrize("transitions", [((0, 2),), ((0, 2), (2, 1))])
async def test_configured_route_runs_through_real_loop(configured_orchestrator_factory, transitions):
    order = (0, 2) if len(transitions) == 1 else (0, 2, 1)
    final = order[-1]
    clients = [
        FakeStageClient(final_output=(sid == final), next_inputs=[{"prompt_token_ids": [sid + 10]}]) for sid in range(3)
    ]
    processors = [FakeOutputProcessor(request_outputs=[_build_request_output("r")]) for _ in clients]
    harness = configured_orchestrator_factory(clients, output_processors=processors, stage_transitions=transitions)
    try:
        await _enqueue_add_request(
            harness,
            request_id="r",
            prompt=FakePromptRequest(request_id="r", prompt_token_ids=[1, 2], resumable=False),
            original_prompt={"prompt_token_ids": [1, 2]},
            sampling_params_list=[SamplingParams(max_tokens=1) for _ in clients],
            final_stage_id=final,
        )
        for sid in order:
            await _wait_for(lambda: len(clients[sid].add_request_calls) == 1)
            clients[sid].push_engine_core_outputs(_engine_core_outputs(f"raw-{sid}", float(sid + 1)))
        output = await _get_output_message(harness)
        assert output.stage_id == final and output.finished
        await _wait_for(lambda: "r" not in harness.orchestrator.request_states)
        for sid in set(range(3)) - set(order):
            assert not clients[sid].add_request_calls
        assert all(pool.get_bound_replica_id("r") is None for pool in harness.orchestrator.stage_pools)
    finally:
        await _shutdown_orchestrator(harness)


@pytest.mark.asyncio
async def test_async_prewarm_uses_actual_edge_and_sender(mocker):
    orch, state, clients = _direct_orchestrator(async_chunk=True)
    sender = mocker.patch.object(orch, "_build_payload_sender_info", return_value={"source": 0})
    emit = mocker.patch.object(orch, "_emit_tx_edge")
    assert await orch._prewarm_async_chunk_stages(
        "r", FakePromptRequest(request_id="r", prompt_token_ids=[1, 2], resumable=False), state
    )
    assert not clients[1].add_request_calls
    request = clients[2].add_request_calls[0][0]
    assert request.payload_sender_info == {"source": 0}
    sender.assert_called_once_with(0, request_id="r", req_state=state)
    assert emit.call_args.kwargs["from_stage"] == 0
    assert emit.call_args.kwargs["to_stage"] == 2
    assert set(state.stage_submit_ts) == {2}


@pytest.mark.asyncio
async def test_async_prewarm_traverses_route_order(mocker):
    orch, state, _ = _direct_orchestrator(((0, 2), (2, 1)), async_chunk=True)
    state.final_stage_id = 1
    state.final_output_stage_ids = {1}
    sender = mocker.patch.object(orch, "_build_payload_sender_info", return_value=None)
    assert await orch._prewarm_async_chunk_stages(
        "r", FakePromptRequest(request_id="r", prompt_token_ids=[1, 2], resumable=False), state
    )
    assert list(state.stage_submit_ts) == [2, 1]
    assert [call.args[0] for call in sender.call_args_list] == [0, 2]


@pytest.mark.asyncio
async def test_async_terminal_waits_for_submitted_route_only():
    orch, state, _ = _direct_orchestrator(async_chunk=True)
    state.stage_submit_ts = {0: 1.0, 2: 2.0}
    await orch._route_output(2, 0, _build_request_output("r"), state, None)
    assert orch.output_async_queue.empty()
    assert state.pending_final_output is not None
    await orch._route_output(0, 0, _build_request_output("r"), state, None)
    assert orch.output_async_queue.get_nowait().finished
    assert orch.output_async_queue.empty()
    assert "r" not in orch.request_states


@pytest.mark.asyncio
async def test_cancellation_during_prewarm_stops_later_route_edges(mocker):
    orch, state, clients = _direct_orchestrator(((0, 2), (2, 1)), async_chunk=True)
    state.final_stage_id = 1
    on_submit = mocker.patch.object(orch, "_on_stage_submitted")

    async def cancel_after_admission(request, **kwargs):
        await orch._cleanup_request_ids(["r"], abort=True)

    mocker.patch.object(clients[2], "add_request_async", side_effect=cancel_after_admission)
    assert not await orch._prewarm_async_chunk_stages(
        "r", FakePromptRequest(request_id="r", prompt_token_ids=[1, 2], resumable=False), state
    )
    assert not clients[1].add_request_calls
    assert all(pool.get_bound_replica_id("r") is None for pool in orch.stage_pools)
    on_submit.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("already_submitted", [False, True])
async def test_cfg_companion_release_uses_actual_successor(mocker, already_submitted):
    orch, state, _ = _direct_orchestrator()
    orch._cfg_tracker.register_companion("r", "unconditional", "r-cfg")
    orch._cfg_tracker.defer_parent("r", _build_request_output("r"), 0)
    if already_submitted:
        state.stage_submit_ts[2] = 1.0
    forward = mocker.patch.object(orch, "_forward_to_next_stage", new_callable=mocker.AsyncMock)
    await orch._handle_cfg_companion_ready("r-cfg")
    assert forward.await_count == (0 if already_submitted else 1)


@pytest.mark.asyncio
async def test_kv_ready_routes_once_to_configured_target(mocker):
    orch, state, _ = _direct_orchestrator()
    forward = mocker.patch.object(orch, "_forward_to_next_stage", new_callable=mocker.AsyncMock)
    raw = EngineCoreOutputs(
        outputs=[OmniEngineCoreOutput(request_id="r", new_token_ids=[], kv_transfer_params={"kv_ready": True})]
    )
    await orch._handle_kv_ready_raw_outputs(0, raw)
    forward.assert_awaited_once()
    state.stage_submit_ts[2] = 1.0
    await orch._handle_kv_ready_raw_outputs(0, raw)
    assert forward.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("final,outputs", [(1, [1]), (2, [1]), (2, [0])])
async def test_invalid_request_route_fails_before_submission(final, outputs):
    orch, _, clients = _direct_orchestrator()
    orch.request_states.clear()
    await orch._handle_add_request(_submission(final_stage_id=final, final_output_stage_ids=outputs))
    assert isinstance(orch.output_async_queue.get_nowait(), ErrorMessage)
    assert not orch.request_states
    assert all(not client.add_request_calls for client in clients)


@pytest.mark.asyncio
@pytest.mark.parametrize("outputs", [[2.0], [True], [[2]]])
async def test_bad_output_ids_fail_one_request_without_killing_dispatch(outputs):
    orch, _, clients = _direct_orchestrator()
    orch.request_states.clear()
    await orch._handle_add_request(_submission(final_output_stage_ids=outputs))
    error = orch.output_async_queue.get_nowait()
    assert isinstance(error, ErrorMessage) and not error.fatal
    assert error.status_code == 400
    assert all(not client.add_request_calls for client in clients)


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [0, 1, 2])
async def test_incomplete_sampling_array_fails_before_any_submission(count):
    orch, _, clients = _direct_orchestrator()
    orch.request_states.clear()
    await orch._handle_add_request(_submission(sampling_params_list=[SamplingParams() for _ in range(count)]))
    assert isinstance(orch.output_async_queue.get_nowait(), ErrorMessage)
    assert all(not client.add_request_calls for client in clients)


@pytest.mark.asyncio
async def test_endpoint_must_be_selected_to_prevent_premature_completion():
    orch, _, clients = _direct_orchestrator()
    clients[0].final_output = True
    orch.request_states.clear()
    await orch._handle_add_request(_submission(final_output_stage_ids=[0]))
    error = orch.output_async_queue.get_nowait()
    assert "final stage" in error.error
    assert all(not client.add_request_calls for client in clients)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "overrides",
    [
        {"final_stage_id": 0},
        {"final_output_stage_ids": [0, 2]},
        {"final_output_stage_ids": [2.0]},
        {"final_output_stage_ids": [True]},
        {"final_output_stage_ids": [[2]]},
        {"sampling_params_list": [SamplingParams()]},
    ],
)
async def test_streaming_update_preserves_admitted_route(overrides):
    orch, _, clients = _direct_orchestrator()
    await orch._handle_streaming_update(_submission(type="streaming_update", **overrides))
    assert isinstance(orch.output_async_queue.get_nowait(), ErrorMessage)
    assert "r" not in orch.request_states
    assert all(not client.add_request_calls for client in clients)


@pytest.mark.asyncio
async def test_streaming_update_can_reuse_sampling_params():
    orch, state, clients = _direct_orchestrator()
    params = state.sampling_params_list
    await orch._handle_streaming_update(_submission(type="streaming_update", sampling_params_list=[]))
    assert state.sampling_params_list is params
    assert state.final_stage_id == 2 and state.final_output_stage_ids == {2}
    assert len(clients[0].add_request_calls) == 1
    assert orch.output_async_queue.empty()


def test_frontend_selects_final_outputs_in_route_order(mocker):
    stages = [
        StagePipelineConfig(stage_id=0, model_stage="test", final_output=False),
        StagePipelineConfig(stage_id=1, model_stage="test", final_output=True, final_output_type="audio"),
        StagePipelineConfig(stage_id=2, model_stage="test", final_output=True, final_output_type="text"),
    ]
    base = object.__new__(OmniBase)
    base._stage_meta_list = stages
    base.output_modalities = ["text", "audio"]
    pipeline = PipelineConfig(
        model_type="test",
        stages=tuple(stages),
        stage_transitions=((0, 2), (2, 1)),
    )
    base.engine = mocker.Mock(spec=OmniEngineBase, pipeline_config=pipeline)
    assert base._compute_final_stage_id(["text", "audio"]) == 1
    assert base._compute_final_stage_id(["text"]) == 2
    assert base._compute_final_output_stage_ids(["text"]) == [2]
    assert set(base._compute_final_output_stage_ids(["text", "audio"])) == {1, 2}
    assert get_final_stage_id_for_e2e(["text", "audio"], ["text", "audio"], stages) == 2


@pytest.mark.parametrize("modalities", [["audio"], ["text", "audio"]])
@pytest.mark.parametrize("default_modalities", [["text", "audio"], ["text"]])
def test_frontend_rejects_outputs_only_available_on_inactive_stages(modalities, default_modalities):
    stages = [
        FakeStageClient(final_output=False),
        FakeStageClient(final_output=True, final_output_type="audio"),
        FakeStageClient(final_output=True, final_output_type="text"),
    ]
    with pytest.raises(ValueError, match="output modalities are unreachable"):
        get_final_stage_id_for_e2e(modalities, default_modalities, stages, stage_order=(0, 2))
    assert get_final_stage_id_for_e2e(None, ["text", "audio"], stages, stage_order=(0, 2)) == 2
    assert get_final_stage_id_for_e2e(modalities, ["text", "audio"], stages) == (1 if modalities == ["audio"] else 2)


@pytest.mark.asyncio
async def test_cancel_while_waiting_for_replica_stops_old_request_admission():
    orch, state, clients = _direct_orchestrator(async_chunk=True)
    pool = orch.stage_pools[2]
    pool.add_client("tcp://127.0.0.1:5555", clients[2], replica_id=0)
    polled = asyncio.Event()

    def no_replicas(_stage_id):
        polled.set()
        return SimpleNamespace(replicas=[])

    pool.attach_hub(SimpleNamespace(get_replicas_for_stage=no_replicas))
    pool.attach_load_balancer(SimpleNamespace(select=lambda task, replicas: 0))
    task = asyncio.create_task(
        orch._prewarm_async_chunk_stages(
            "r", FakePromptRequest(request_id="r", prompt_token_ids=[1], resumable=False), state
        )
    )
    await asyncio.wait_for(polled.wait(), timeout=1)
    await orch._cleanup_request_ids(["r"], abort=True)
    new_state = OrchestratorRequestState(request_id="r")
    orch.request_states["r"] = new_state
    pool.bind("r", "tcp://127.0.0.1:5555")
    assert not await asyncio.wait_for(task, timeout=1)
    assert orch.request_states["r"] is new_state
    assert pool.get_bound_replica_id("r") == 0
    assert not clients[2].add_request_calls
    await orch._cleanup_request_ids(["r"])


@pytest.mark.asyncio
async def test_lost_preselected_replica_cannot_be_reselected():
    orch, state, clients = _direct_orchestrator(async_chunk=True)
    pool = orch.stage_pools[2]
    pool.add_client("replica-1", FakeStageClient(final_output=True), replica_id=1)
    assert await pool.pick("r", request_state=state) == 0
    pool.evict_replica(0)
    assert not await orch._dispatch_or_fail_request(
        lambda: pool.submit_initial("r", state, SimpleNamespace()),
        req_id="r",
        stage_id=2,
        operation="bridge submission",
    )
    assert isinstance(orch.output_async_queue.get_nowait(), ErrorMessage)
    assert "r" not in orch.request_states
    assert not pool.clients[1].add_request_calls


@pytest.mark.asyncio
@pytest.mark.parametrize("async_chunk", [False, True])
async def test_reordered_route_starts_at_admitted_nonzero_entry(mocker, async_chunk):
    orch, _, clients = _direct_orchestrator(((0, 2), (2, 1)), async_chunk=async_chunk)
    orch.request_states.clear()
    processed = _processed_request("r", SamplingParams(max_tokens=1))
    build_entry = mocker.patch.object(orch, "_build_entry_stage_request", return_value=processed)
    await orch._handle_add_request(
        _submission(
            entry_stage_id=2,
            final_stage_id=1,
            final_output_stage_ids=[1],
            prompt={"prompt_token_ids": [1, 2]},
        )
    )
    state = orch.request_states["r"]
    assert orch._request_stage_path(state) == (2, 1)
    build_entry.assert_awaited_once()
    assert clients[2].add_request_calls[0][0] is processed
    assert not clients[0].add_request_calls
    assert bool(clients[1].add_request_calls) is async_chunk
    await orch._cleanup_request_ids(["r"], abort=True)


@pytest.mark.asyncio
async def test_closing_request_does_not_consume_its_abort_snapshot(mocker):
    orch, state, _ = _direct_orchestrator()
    state.closing = "aborting"
    orch.request_states["healthy"] = OrchestratorRequestState(request_id="healthy", final_stage_id=2)
    raw = EngineCoreOutputs(
        outputs=[OmniEngineCoreOutput(request_id=key, new_token_ids=[1]) for key in ("r", "healthy")]
    )
    process = mocker.patch.object(orch.stage_pools[0], "process_llm_raw_outputs", return_value=[])
    await orch._process_llm_stage_outputs(0, 0, raw, set())
    filtered = process.call_args.args[1]
    assert filtered is not raw
    assert [output.request_id for output in filtered.outputs] == ["healthy"]
    assert [output.request_id for output in raw.outputs] == ["r", "healthy"]


@pytest.mark.asyncio
@pytest.mark.parametrize("transitions", [None, ((0, 2), (2, 1))])
async def test_prewarm_reuses_replica_selected_by_waiting_bridge(monkeypatch, transitions):
    orch, state, clients = _direct_orchestrator(transitions, async_chunk=True)
    _, bridge, receiver = orch._request_stage_path(state)
    pool = orch.stage_pools[bridge]
    pool.stage_vllm_config.model_config.async_chunk = False
    replicas = [clients[bridge], FakeStageClient()]
    addresses = ["tcp://127.0.0.1:50101", "tcp://127.0.0.1:50102"]
    for replica_id, client in enumerate(replicas):
        pool.add_client(addresses[replica_id], client, replica_id=replica_id)
        monkeypatch.setattr(
            client,
            "get_payload_sender_info",
            lambda port=50101 + replica_id: {"host": "127.0.0.1", "zmq_port": port},
            raising=False,
        )
    orch.stage_pools[0].select_replica_id("r")
    hub = _FakeHub([])
    pool.attach_hub(hub)
    pool.attach_load_balancer(_RoundRobinLB())
    entered = {name: asyncio.Event() for name in ("bridge-submit", "prewarm")}
    release = {name: asyncio.Event() for name in entered}
    sleep = asyncio.sleep

    async def wait_for_capacity(delay):
        task = asyncio.current_task()
        assert task is not None
        name = task.get_name()
        if name in entered:
            entered[name].set()
            await release[name].wait()
        else:
            await sleep(delay)

    monkeypatch.setattr("vllm_omni.engine.stage_pool.asyncio.sleep", wait_for_capacity)
    state.stage_submit_ts[bridge] = 1.0
    submit = asyncio.create_task(
        pool.submit_initial("r", state, _processed_request("r", state.sampling_params_list[bridge])),
        name="bridge-submit",
    )
    prewarm = None
    try:
        await asyncio.wait_for(entered["bridge-submit"].wait(), 1)
        prewarm = asyncio.create_task(
            orch._prewarm_async_chunk_stages("r", FakePromptRequest("r", [1, 2], False), state), name="prewarm"
        )
        await asyncio.wait_for(entered["prewarm"].wait(), 1)
        hub._replicas = [replace(_replica(address), stage_id=bridge) for address in addresses]
        release["bridge-submit"].set()
        selected = await asyncio.wait_for(submit, 1)
        release["prewarm"].set()
        assert await asyncio.wait_for(prewarm, 1)
        assert pool.get_bound_replica_id("r") == selected
        advertised = clients[receiver].add_request_calls[0][0].payload_sender_info
        assert advertised == replicas[selected].get_payload_sender_info()
        await orch._cleanup_request_ids(["r"], abort=True)
        assert replicas[selected].abort_calls == [["r"]]
        assert not replicas[1 - selected].abort_calls
    finally:
        for task in (submit, prewarm):
            if task is not None:
                task.cancel()
        await asyncio.gather(*(task for task in (submit, prewarm) if task is not None), return_exceptions=True)


@pytest.mark.asyncio
async def test_prewarm_does_not_bind_bridge_only_stages():
    orch, state, _ = _direct_orchestrator(None, async_chunk=True)
    orch.stage_pools[0].select_replica_id("r")
    for pool in orch.stage_pools[1:]:
        pool.stage_vllm_config.model_config.async_chunk = False
    assert await orch._prewarm_async_chunk_stages("r", FakePromptRequest("r", [1, 2], False), state)
    assert all(pool.get_bound_replica_id("r") is None for pool in orch.stage_pools[1:])
    assert not state.stage_submit_ts


@pytest.mark.asyncio
@pytest.mark.parametrize("transitions", [None, ((0, 2),), ((0, 2), (2, 1))])
@pytest.mark.parametrize("fail_first", [False, True])
@pytest.mark.parametrize("finish_during_abort", [False, True])
async def test_abort_failure_drains_without_resuming_or_leaking_request(
    monkeypatch, mocker, transitions, fail_first, finish_during_abort
):
    orch, state, clients = _direct_orchestrator(transitions, async_chunk=True)
    orch._running_counter = FakeRunningCounter()
    orch._register_running_request(state)
    release_resources = mocker.spy(orch, "_release_stage_transfer_resources")
    path = orch._request_stage_path(state)
    failed_stage = path[0] if fail_first else path[-1]
    for sid in path:
        clients[sid].final_output = True
        processor = MultimodalOutputProcessor(tokenizer=None, log_stats=False)
        orch.stage_pools[sid]._output_processor = processor
        request = _processed_request(f"internal-r-{sid}", state.sampling_params_list[sid])
        request.external_req_id = "r"
        await orch.stage_pools[sid].submit_initial("r", state, request)
        state.stage_submit_ts[sid] = 1.0
    failed_pool = orch.stage_pools[failed_stage]
    processor = failed_pool.output_processor
    healthy = OrchestratorRequestState(
        request_id="healthy",
        entry_stage_id=failed_stage,
        final_stage_id=failed_stage,
        final_output_stage_ids={failed_stage},
        sampling_params_list=state.sampling_params_list,
        stage_submit_ts={failed_stage: 1.0},
    )
    orch.request_states["healthy"] = healthy
    orch._register_running_request(healthy)
    request = _processed_request("internal-healthy", state.sampling_params_list[failed_stage])
    request.external_req_id = "healthy"
    await failed_pool.submit_initial("healthy", healthy, request)

    async def terminal(request_id):
        terminals: set[str] = set()
        processed = await orch._process_llm_stage_outputs(
            failed_stage,
            0,
            EngineCoreOutputs(
                outputs=[
                    OmniEngineCoreOutput(
                        request_id=request_id,
                        new_token_ids=[],
                        finish_reason=FinishReason.STOP,
                    )
                ]
            ),
            terminals,
        )
        await orch._handle_processed_outputs(failed_stage, 0, processed)
        await orch._finish_raw_terminal_requests(failed_stage, 0, terminals)

    async def fail_abort(request_ids):
        if finish_during_abort:
            await terminal(f"internal-r-{failed_stage}")
            assert "r" in orch.request_states
        raise RuntimeError("abort transport failed")

    monkeypatch.setattr(clients[failed_stage], "abort_requests_async", fail_abort)
    await orch._handle_abort(AbortRequestMessage(request_ids=["r"], rpc_id="abort-r"))
    assert not orch.rpc_async_queue.get_nowait().success
    for sid in set(path) - {failed_stage}:
        assert clients[sid].abort_calls == [[f"internal-r-{sid}"]]
        assert orch.stage_pools[sid].get_bound_replica_id("r") is None
    if not finish_during_abort:
        assert state.closing
        assert orch._running_counter.value == 2
        release_resources.assert_not_called()
        with pytest.raises(RequestClosedError, match="r"):
            await failed_pool.submit_initial("r", state, request)
        await terminal(f"internal-r-{failed_stage}")
    assert "r" not in orch.request_states
    assert orch._running_counter.value == 1
    release_resources.assert_called_once_with(["r"])
    assert all(pool.get_bound_replica_id("r") is None for pool in orch.stage_pools)
    assert f"internal-r-{failed_stage}" not in processor.request_states
    assert "internal-healthy" in processor.request_states
    assert failed_pool.get_bound_replica_id("healthy") == 0
    error = orch.output_async_queue.get_nowait()
    assert isinstance(error, ErrorMessage) and error.request_id == "r" and not error.fatal
    assert orch.output_async_queue.empty()
    await terminal("internal-healthy")
    assert orch.output_async_queue.get_nowait().request_id == "healthy"
    assert not orch.request_states
    assert not processor.request_states
    assert orch._running_counter.value == 0


@pytest.mark.asyncio
async def test_failed_abort_waits_for_each_remaining_stage(monkeypatch):
    orch, state, clients = _direct_orchestrator()
    state.stage_submit_ts = {0: 1.0, 2: 1.0}
    for sid in (0, 2):
        processor = MultimodalOutputProcessor(tokenizer=None, log_stats=False)
        orch.stage_pools[sid]._output_processor = processor
        request = _processed_request("r", state.sampling_params_list[sid])
        request.external_req_id = "r"
        await orch.stage_pools[sid].submit_initial("r", state, request)

    async def fail_abort(request_ids):
        raise RuntimeError("abort failed")

    for sid in (0, 2):
        monkeypatch.setattr(clients[sid], "abort_requests_async", fail_abort)
    await orch._handle_abort(AbortRequestMessage(request_ids=["r"], rpc_id="abort-r"))
    assert not orch.rpc_async_queue.get_nowait().success
    for sid in (2, 0):
        terminals: set[str] = set()
        await orch._process_llm_stage_outputs(
            sid,
            0,
            EngineCoreOutputs(
                outputs=[
                    OmniEngineCoreOutput(
                        request_id="r",
                        new_token_ids=[],
                        finish_reason=FinishReason.STOP,
                    )
                ]
            ),
            terminals,
        )
        await orch._finish_raw_terminal_requests(sid, 0, terminals)
        assert ("r" in orch.request_states) == (sid == 2)
    assert all(not orch.stage_pools[sid].output_processor.request_states for sid in (0, 2))
    assert all(pool.get_bound_replica_id("r") is None for pool in orch.stage_pools)


@pytest.mark.asyncio
async def test_abort_failure_on_one_replica_does_not_skip_other_replicas(monkeypatch):
    orch, state, clients = _direct_orchestrator()
    pool = orch.stage_pools[2]
    pool._output_processor = MultimodalOutputProcessor(tokenizer=None, log_stats=False)
    pool.add_client("other", FakeStageClient(final_output=True), replica_id=1)
    for replica_id, request_id in enumerate(("r", "other")):
        request = _processed_request(request_id, state.sampling_params_list[2])
        request.external_req_id = request_id
        await pool.submit_initial(request_id, state, request, pinned_replica_id=replica_id)

    async def fail_abort(request_ids):
        raise RuntimeError("abort failed")

    monkeypatch.setattr(clients[2], "abort_requests_async", fail_abort)
    with pytest.raises(RuntimeError, match="abort failed"):
        await pool.abort_requests(["r", "other"])
    assert pool.get_bound_replica_id("r") == 0
    assert pool.get_bound_replica_id("other") is None
    assert set(pool.output_processor.request_states) == {"r"}
    assert pool.clients[1].abort_calls == [["other"]]


@pytest.mark.asyncio
async def test_draining_streaming_request_ignores_processed_segment_stop(monkeypatch):
    orch, state, clients = _direct_orchestrator(async_chunk=True)
    state.streaming.enabled = True
    state.stage_submit_ts[2] = 1.0
    orch.stage_pools[2].select_replica_id("r")

    async def fail_abort(request_ids):
        raise RuntimeError("abort failed")

    monkeypatch.setattr(clients[2], "abort_requests_async", fail_abort)
    await orch._handle_abort(AbortRequestMessage(request_ids=["r"], rpc_id="abort-r"))
    state.streaming.segments[2] = StreamingSegmentState(finished=True)
    await orch._route_output(2, 0, _build_request_output("r"), state, None)
    assert orch.request_states["r"] is state
    assert 2 not in state.finished_stage_ids
    state.streaming.segments[2] = StreamingSegmentState(finished=False)
    await orch._route_output(2, 0, _build_request_output("r"), state, None)
    assert "r" not in orch.request_states


@pytest.mark.asyncio
async def test_abort_failure_unblocks_frontend_output_stream(monkeypatch):
    orch, state, clients = _direct_orchestrator()
    orch.stage_pools[2].select_replica_id("r")
    state.stage_submit_ts[2] = 1.0
    frontend = object.__new__(AsyncOmniBase)
    frontend.request_states = {"r": ClientRequestState("r", queue=orch.output_async_queue, final_stage_id=2)}
    stream = frontend._process_orchestrator_results("r", OrchestratorAggregator(3, False, 1.0, 2), 2, {}, 1.0)
    waiting = asyncio.create_task(anext(stream))

    async def fail_abort(request_ids):
        raise RuntimeError("abort transport failed")

    abort = clients[2].abort_requests_async
    monkeypatch.setattr(clients[2], "abort_requests_async", fail_abort)
    try:
        await orch._handle_abort(AbortRequestMessage(request_ids=["r"], rpc_id="abort-r"))
        with pytest.raises(RuntimeError, match="abort transport failed"):
            await asyncio.wait_for(waiting, 1)
    finally:
        waiting.cancel()
        await asyncio.gather(waiting, return_exceptions=True)
        await stream.aclose()
        monkeypatch.setattr(clients[2], "abort_requests_async", abort)
        await orch._cleanup_request_ids(["r"], abort=True)


@pytest.mark.asyncio
async def test_failed_abort_drains_and_reuses_id_through_real_loops(configured_orchestrator_factory, monkeypatch):
    clients = [FakeStageClient(final_output=(sid == 2)) for sid in range(3)]
    pools = _build_stage_pools([[client] for client in clients])
    for pool in pools:
        pool._output_processor = MultimodalOutputProcessor(tokenizer=None, log_stats=False)
    abort = clients[2].abort_requests_async

    async def fail_abort(request_ids):
        raise RuntimeError("abort transport failed")

    monkeypatch.setattr(clients[2], "abort_requests_async", fail_abort)
    harness = configured_orchestrator_factory(clients, stage_pools=pools, stage_transitions=((0, 2),), async_chunk=True)
    orch = harness.orchestrator
    params = [SamplingParams(max_tokens=1) for _ in clients]

    async def submit():
        request = _processed_request("r", params[0])
        request.external_req_id = "r"
        await _enqueue_add_request(
            harness,
            request_id="r",
            prompt=request,
            original_prompt={"prompt_token_ids": [1, 2]},
            sampling_params_list=params,
            final_stage_id=2,
        )

    try:
        await submit()
        await _wait_for(lambda: len(clients[2].add_request_calls) == 1)
        old_state = orch.request_states["r"]
        harness.request_sync_q.put_nowait(AbortRequestMessage(request_ids=["r"], rpc_id="abort-r"))
        result = await _get_rpc_message(harness)
        assert isinstance(result, AbortResultMessage)
        assert result.rpc_id == "abort-r" and not result.success
        await _wait_for(lambda: not harness.output_sync_q.empty())
        error = harness.output_sync_q.get_nowait()
        assert isinstance(error, ErrorMessage) and not error.fatal and error.request_id == "r"
        assert orch.request_states["r"] is old_state and old_state.closing
        assert pools[0].get_bound_replica_id("r") is None
        assert pools[2].get_bound_replica_id("r") == 0
        clients[2].push_engine_core_outputs(
            EngineCoreOutputs(
                outputs=[OmniEngineCoreOutput(request_id="r", new_token_ids=[], finish_reason=FinishReason.STOP)]
            )
        )
        await _wait_for(lambda: "r" not in orch.request_states)
        assert all(pool.get_bound_replica_id("r") is None for pool in pools)
        assert all(not pool.output_processor.request_states for pool in pools)
        assert harness.output_sync_q.empty()

        monkeypatch.setattr(clients[2], "abort_requests_async", abort)
        await submit()
        await _wait_for(lambda: len(clients[2].add_request_calls) == 2)
        assert orch.request_states["r"] is not old_state
        assert not orch.request_states["r"].closing
        harness.request_sync_q.put_nowait(AbortRequestMessage(request_ids=["r"], rpc_id="abort-reused-r"))
        result = await _get_rpc_message(harness)
        assert isinstance(result, AbortResultMessage)
        assert result.rpc_id == "abort-reused-r" and result.success
        assert "r" not in orch.request_states
        assert not clients[1].add_request_calls
    finally:
        await _shutdown_orchestrator(harness)


@pytest.mark.asyncio
async def test_cancelled_bridge_validation_cannot_fail_reused_request(monkeypatch):
    orch, old_state, clients = _direct_orchestrator()
    clients[2].next_inputs = [{"prompt": "late input"}]
    processor = FakeInputProcessor()
    orch._stage_input_processors[2] = processor
    entered = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()

    def reject_input(**kwargs):
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(5), "Input validation was not released"
        raise ValueError("late validation failure")

    monkeypatch.setattr(processor, "process_inputs", reject_input)
    await orch.stage_pools[0].submit_initial("r", old_state, _processed_request("r", old_state.sampling_params_list[0]))
    forward = asyncio.create_task(orch._forward_to_next_stage("r", 0, _build_request_output("r"), old_state))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        await orch._handle_abort(AbortRequestMessage(request_ids=["r"], rpc_id="abort-old-r"))
        assert orch.rpc_async_queue.get_nowait().success
        await orch._handle_add_request(_submission())
        new_state = orch.request_states["r"]
        assert new_state is not old_state
        release.set()
        await asyncio.wait_for(forward, 2)
        assert orch.request_states["r"] is new_state
        assert not new_state.closing
        assert orch.stage_pools[0].get_bound_replica_id("r") == 0
        assert clients[0].abort_calls == [["r"]]
        assert orch.output_async_queue.empty()
    finally:
        release.set()
        await asyncio.gather(forward, return_exceptions=True)
        await orch._cleanup_request_ids(["r"], abort=True)
