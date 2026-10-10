# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real orchestrator and StagePool logic with lightweight engine clients (L1)."""

import asyncio
from types import SimpleNamespace

import pytest
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreOutputs, FinishReason

from tests.engine.test_orchestrator import (
    FakeOutputProcessor,
    FakePromptRequest,
    FakeStageClient,
    _build_request_output,
    _build_stage_pools,
    _engine_core_outputs,
    _enqueue_add_request,
    _get_output_message,
    _shutdown_orchestrator,
    _wait_for,
    orchestrator_factory,
)
from vllm_omni.config.stage_config import PipelineConfig, StagePipelineConfig
from vllm_omni.config.stage_routing import StageRouting
from vllm_omni.engine import OmniEngineCoreOutput
from vllm_omni.engine.errors import ResourceReleaseError
from vllm_omni.engine.messages import ErrorMessage, StageSubmissionMessage
from vllm_omni.engine.omni_engine_base import OmniEngineBase
from vllm_omni.engine.orchestrator import Orchestrator, OrchestratorRequestState
from vllm_omni.engine.stage_pool import RequestClosedError, StageUnavailableError
from vllm_omni.entrypoints.omni_base import OmniBase
from vllm_omni.entrypoints.utils import get_final_stage_id_for_e2e

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
configured_orchestrator_factory = orchestrator_factory


@pytest.mark.parametrize("selected", [set(), {0, 1}])
def test_completion_derives_selected_outputs_from_stage_termination(selected):
    state = OrchestratorRequestState(
        request_id="r", final_stage_id=1, final_output_stage_ids=selected, finished_stage_ids={2}
    )
    assert not state.final_outputs_finished
    state.finished_stage_ids.add(0)
    assert not state.final_outputs_finished
    state.finished_stage_ids.add(1)
    assert state.final_outputs_finished


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
    configs = [
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
async def test_cancel_while_waiting_for_replica_stops_old_request_admission(mocker):
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
async def test_duplicate_cleanup_waiter_does_not_cancel_abort_owner(mocker):
    orch, state, clients = _direct_orchestrator(async_chunk=True)
    orch.stage_pools[0].select_replica_id("r")
    started, finish = asyncio.Event(), asyncio.Event()

    async def abort(_ids):
        started.set()
        await finish.wait()

    abort_call = mocker.patch.object(clients[0], "abort_requests_async", side_effect=abort)
    owner = asyncio.create_task(orch._cleanup_request_ids(["r"], abort=True))
    await asyncio.wait_for(started.wait(), timeout=1)
    waiter = asyncio.create_task(orch._cleanup_request_ids(["r"], abort=True))
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert state.closing and not state.cleanup_future.done()
    with pytest.raises(RequestClosedError):
        await orch.stage_pools[0].submit_update("r", state, SimpleNamespace())
    await orch._handle_add_request(_submission())
    error = orch.output_async_queue.get_nowait()
    assert isinstance(error, ErrorMessage) and error.status_code == 409
    assert orch.request_states["r"] is state
    finish.set()
    await asyncio.wait_for(owner, timeout=1)
    assert abort_call.await_count == 1
    assert "r" not in orch.request_states


@pytest.mark.asyncio
@pytest.mark.parametrize("own_release_fails", [False, True])
async def test_automatic_cleanup_joins_failed_owner_and_reports_only_failed_requests(mocker, own_release_fails):
    orch, state, _ = _direct_orchestrator()
    other_state = OrchestratorRequestState(request_id="s")
    orch.request_states["s"] = other_state
    entered, second_entered, finish = asyncio.Event(), asyncio.Event(), asyncio.Event()
    failures = {"r", "s"} if own_release_fails else {"r"}

    async def release(ids):
        if ids == ["r"]:
            entered.set()
            await finish.wait()
        else:
            second_entered.set()
        if failures.intersection(ids):
            raise ResourceReleaseError("READ ownership remains active")

    mocker.patch.object(orch.stage_pools[0], "release_request_resources", side_effect=release)
    owner = asyncio.create_task(orch._cleanup_request_ids(["r"], abort=True))
    await asyncio.wait_for(entered.wait(), timeout=1)
    automatic = asyncio.create_task(orch._cleanup_request_ids(["r", "s"]))
    await asyncio.wait_for(second_entered.wait(), timeout=1)
    await asyncio.sleep(0)
    assert not automatic.done()
    finish.set()
    with pytest.raises(ResourceReleaseError):
        await asyncio.wait_for(owner, timeout=1)
    await asyncio.wait_for(automatic, timeout=1)
    errors = []
    while not orch.output_async_queue.empty():
        errors.append(orch.output_async_queue.get_nowait())
    assert {error.request_id for error in errors} == failures
    assert all(isinstance(error, ErrorMessage) and not error.fatal for error in errors)
    assert state.closing and orch.request_states["r"] is state
    assert ("s" in orch.request_states) is own_release_fails
    failures.clear()
    await orch._cleanup_request_ids(["r", "s"], abort=True)
    assert not orch.request_states


@pytest.mark.asyncio
@pytest.mark.parametrize("release_fails", [False, True])
async def test_parent_cleanup_joins_completed_companion_cleanup(mocker, release_fails):
    orch, state, _ = _direct_orchestrator()
    companion_id = "r__neg"
    companion = OrchestratorRequestState(request_id=companion_id)
    orch.request_states[companion_id] = companion
    orch._cfg_tracker.register_companion("r", "neg", companion_id)
    orch._cfg_tracker.set_companion_output(companion_id, {"conditioning": 1})
    orch._cfg_tracker.on_companion_completed(companion_id)
    entered, parent_entered, finish = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def release(ids):
        if ids == [companion_id]:
            entered.set()
            await finish.wait()
            if release_fails:
                raise ResourceReleaseError("READ ownership remains active")
        else:
            parent_entered.set()

    release_call = mocker.patch.object(orch.stage_pools[0], "release_request_resources", side_effect=release)
    owner = asyncio.create_task(orch._cleanup_request_ids([companion_id]))
    await asyncio.wait_for(entered.wait(), timeout=1)
    companion_future = companion.cleanup_future
    parent = asyncio.create_task(orch._cleanup_request_ids(["r"], abort=True))
    try:
        await asyncio.wait_for(parent_entered.wait(), timeout=1)
        assert companion.cleanup_future is companion_future
        assert orch.request_states["r"] is state and state.closing
        assert not parent.done()
    finally:
        finish.set()
        results = await asyncio.wait_for(asyncio.gather(owner, parent, return_exceptions=True), timeout=1)
    assert [call.args[0] for call in release_call.await_args_list] == [[companion_id], ["r"]]
    assert results[0] == []
    if release_fails:
        assert isinstance(results[1], ResourceReleaseError)
        assert orch.request_states["r"] is state and state.closing
        assert companion.closing and orch._cfg_tracker.has_companions("r")
        release_fails = False
        await orch._cleanup_request_ids(["r"], abort=True)
        assert set(release_call.await_args_list[-1].args[0]) == {"r", companion_id}
    else:
        assert results[1] == []
    assert not orch.request_states


@pytest.mark.asyncio
@pytest.mark.parametrize("retry_id", ["r", "r__neg"])
async def test_failed_cfg_cleanup_retry_releases_the_complete_owner_batch(mocker, retry_id):
    orch, state, _ = _direct_orchestrator()
    companion_id = "r__neg"
    companion = OrchestratorRequestState(request_id=companion_id)
    orch.request_states[companion_id] = companion
    orch._cfg_tracker.register_companion("r", "neg", companion_id)
    release_call = mocker.patch.object(
        orch.stage_pools[0],
        "release_request_resources",
        side_effect=ResourceReleaseError("READ ownership remains active"),
    )
    with pytest.raises(ResourceReleaseError):
        await orch._cleanup_request_ids(["r"], abort=True)
    assert state.closing and companion.closing
    assert state.cleanup_future is companion.cleanup_future
    assert orch._cfg_tracker.has_companions("r")
    release_call.side_effect = None
    await orch._cleanup_request_ids([retry_id], abort=True)
    assert not orch.request_states
    assert all(set(call.args[0]) == {"r", companion_id} for call in release_call.await_args_list)


@pytest.mark.asyncio
async def test_failed_abort_can_be_retried_without_reopening_admission(mocker):
    orch, state, clients = _direct_orchestrator()
    orch.stage_pools[0].select_replica_id("r")
    abort = mocker.patch.object(clients[0], "abort_requests_async", side_effect=[RuntimeError("RPC unavailable"), None])
    with pytest.raises(RuntimeError, match="RPC unavailable"):
        await orch._cleanup_request_ids(["r"], abort=True)
    assert state.closing and orch.request_states["r"] is state
    with pytest.raises(RequestClosedError):
        await orch.stage_pools[0].pick("r", request_state=state)
    await orch._cleanup_request_ids(["r"], abort=True)
    assert abort.await_count == 2
    assert "r" not in orch.request_states


@pytest.mark.asyncio
async def test_lost_preselected_replica_cannot_be_reselected(mocker):
    orch, state, clients = _direct_orchestrator(async_chunk=True)
    pool = orch.stage_pools[2]
    pool.add_client("replica-1", FakeStageClient(final_output=True), replica_id=1)
    assert await pool.pick("r", request_state=state) == 0
    pool.evict_replica(0)
    with pytest.raises(StageUnavailableError):
        await pool.submit_initial("r", state, SimpleNamespace())
    await orch._handle_dead_replica(2, 0, RuntimeError("replica exited"))
    assert isinstance(orch.output_async_queue.get_nowait(), ErrorMessage)
    assert "r" not in orch.request_states
    assert not pool.clients[1].add_request_calls


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
@pytest.mark.parametrize("empty_inputs", [None, []])
async def test_empty_diffusion_cleanup_failure_keeps_request_closed(mocker, empty_inputs):
    orch, state, clients = _direct_orchestrator()
    clients[2].stage_type = "diffusion"
    clients[2].custom_process_input_func = mocker.Mock(return_value=empty_inputs)
    mocker.patch.object(orch, "_release_stage_transfer_resources", side_effect=ResourceReleaseError("release failed"))

    await orch._forward_to_next_stage("r", 0, _build_request_output("r"), state)

    assert orch.output_async_queue.get_nowait().finished
    error = orch.output_async_queue.get_nowait()
    assert isinstance(error, ErrorMessage) and not error.fatal and error.request_id == "r"
    assert orch.request_states["r"] is state and state.closing
    assert not clients[2].add_request_calls
    assert orch.rpc_async_queue.empty()


@pytest.mark.asyncio
@pytest.mark.parametrize("event_driven", ["0", "1"])
@pytest.mark.parametrize("transitions", [((0, 2),), ((0, 2), (2, 1))])
async def test_configured_route_runs_through_real_loop(
    monkeypatch, configured_orchestrator_factory, event_driven, transitions
):
    monkeypatch.setenv("VLLM_OMNI_EVENT_DRIVEN_ORCH", event_driven)
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
    sender.assert_called_once_with(0, request_id="r")
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
async def test_cancelled_prewarm_releases_bindings_without_resurrection():
    orch, state, clients = _direct_orchestrator(async_chunk=True)
    await orch._prewarm_async_chunk_stages(
        "r", FakePromptRequest(request_id="r", prompt_token_ids=[1, 2], resumable=False), state
    )
    await orch._cleanup_request_ids(["r"], abort=True)
    assert "r" not in orch.request_states
    assert all(pool.get_bound_replica_id("r") is None for pool in orch.stage_pools)
    assert not await orch._prewarm_async_chunk_stages(
        "r", FakePromptRequest(request_id="r", prompt_token_ids=[1, 2], resumable=False), state
    )
    assert len(clients[2].add_request_calls) == 1


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
