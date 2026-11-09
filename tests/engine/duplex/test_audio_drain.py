# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared drain lifecycle with real runner/manager and synthetic stage outputs."""

import asyncio

import pytest
import torch
from vllm.v1.engine import EngineCoreRequest, EngineCoreRequestType, UtilityOutput
from vllm.v1.engine.core_client import AsyncMPClient, DPAsyncMPClient, DPLBAsyncMPClient, _process_utility_output
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder, UtilityResult

from tests.engine.duplex.test_session_runner import close_harness
from tests.engine.duplex.test_session_runner_personaplex import code2wav_output, frame, open_personaplex_harness
from tests.engine.test_orchestrator import FakeStageClient, _build_stage_pools
from vllm_omni.engine.duplex import commands
from vllm_omni.engine.duplex.contracts import AudioDrainTarget
from vllm_omni.engine.duplex.messages import DrainDuplexAudioMessage, DuplexControlResultMessage
from vllm_omni.engine.stage_engine_core_client import (
    DPLBStageEngineCoreClient,
    StageEngineCoreClient,
    StageEngineCoreClientBase,
)
from vllm_omni.entrypoints.duplex_omni import DuplexOmni, DuplexSessionHandle
from vllm_omni.model_executor.models.personaplex.duplex import stage0
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_declared_transport_keeps_cooperative_dp_mro():
    mro = DPLBStageEngineCoreClient.__mro__
    assert mro.index(StageEngineCoreClientBase) < mro.index(DPLBAsyncMPClient) < mro.index(AsyncMPClient)
    assert StageEngineCoreClient.__mro__.index(StageEngineCoreClientBase) < StageEngineCoreClient.__mro__.index(
        AsyncMPClient
    )


@pytest.fixture(autouse=True)
def _prefill(monkeypatch):
    monkeypatch.setattr(stage0, "personaplex_prefill_slots", lambda *_: 4)


async def _until(predicate):
    async def wait():
        while not predicate():
            await asyncio.sleep(0)

    await asyncio.wait_for(wait(), 2)


def _generated(h, sequence):
    h.deliver(
        OmniRequestOutput(request_id=h.stage0_request_id(), finished=False),
        stage_id=0,
        segment_finished=True,
        segment_token_ids=(11,),
        segment_output_metadata={
            "meta.duplex_epoch": torch.tensor([h.session.epoch]),
            "meta.duplex_generated_seq": torch.tensor([sequence]),
        },
    )


async def _open(mocker):
    h = await open_personaplex_harness()
    mocker.patch.object(h.runner.model, "_schedule_silence_continuation", new=mocker.AsyncMock(return_value=False))
    return h


@pytest.mark.asyncio
async def test_drain_waits_for_generation_flush_and_event_specific_send(mocker):
    h = await _open(mocker)
    flush = mocker.patch.object(h.port, "flush_audio_prefix", new=mocker.AsyncMock(return_value=3))
    try:
        for _ in range(3):
            await h.run(frame())
        drain = h.runner.begin_audio_drain(timeout=2)
        await _until(lambda: not h.runner._mailbox.qsize() and not h.runner.tasks.append_tasks)
        assert not drain.done()
        flush.assert_not_awaited()
        for sequence in (1, 2, 3):
            _generated(h, sequence)
        await _until(lambda: flush.await_count == 1)
        flush.assert_awaited_once_with(h.stage0_request_id(), epoch=0, sequence=3)
        h.deliver(code2wav_output(h.stage0_request_id(), samples=3840, text="hello"))
        await _until(lambda: h.session.audio_delivery.projected_samples == 3840)
        assert not drain.done()
        event = None
        while h.output_buffer.pending_events:
            candidate = await h.output_buffer.get()
            if candidate.type == "response.output_audio.delta":
                event = candidate
                break
        receipt = h.output_buffer.send_receipt(event)
        assert receipt is not None
        h.submit(commands.AudioSendCompleted(receipt=receipt))
        target = await drain
        assert (target.accepted_seq, target.expected_samples) == (3, 3840)
        assert h.session.audio_delivery.completed_samples == 3840
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_drain_freezes_before_later_append_while_prior_submit_is_blocked(mocker):
    h = await _open(mocker)
    flush = mocker.patch.object(h.port, "flush_audio_prefix", new=mocker.AsyncMock(return_value=1))
    h.port.submit_gate = asyncio.Event()
    try:
        h.submit(frame())
        await h.port.submit_started.wait()
        drain = h.runner.begin_audio_drain(timeout=2)
        h.submit(frame())
        h.submit(commands.Heartbeat())
        await _until(lambda: h.runner._mailbox.empty())
        assert not drain.done() and h.session.audio_delivery is None
        h.port.submit_gate.set()
        await _until(lambda: h.session.audio_delivery is not None and h.session.audio_delivery.accepted_seq == 2)
        _generated(h, 1)
        target = await drain
        assert (target.accepted_seq, target.expected_samples) == (1, 0)
        flush.assert_awaited_once_with(h.stage0_request_id(), epoch=0, sequence=1)
    finally:
        h.port.submit_gate.set()
        await close_harness(h)


@pytest.mark.asyncio
@pytest.mark.parametrize("action", [commands.CloseSession, commands.CancelResponse])
async def test_close_or_epoch_cancel_rejects_pending_drain(action, mocker):
    h = await _open(mocker)
    mocker.patch.object(h.port, "flush_audio_prefix", new=mocker.AsyncMock(return_value=2))
    try:
        await h.run(frame())
        await h.run(frame())
        drain = h.runner.begin_audio_drain(timeout=2)
        await _until(lambda: h.runner._mailbox.empty() and not h.runner.tasks.append_tasks)
        h.submit(action())
        with pytest.raises(RuntimeError, match="invalidated|closed"):
            await asyncio.wait_for(drain, 1)
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_timeout_bounds_work_and_allows_an_explicit_retry(mocker):
    h = await _open(mocker)
    mocker.patch.object(h.port, "flush_audio_prefix", new=mocker.AsyncMock(return_value=1))
    try:
        await h.run(frame())
        drain = h.runner.begin_audio_drain(timeout=0.02)
        with pytest.raises(RuntimeError, match="pending"):
            h.runner.begin_audio_drain(timeout=2)
        with pytest.raises(TimeoutError):
            await drain
        await _until(lambda: not h.runner._background_tasks)
        retry = h.runner.begin_audio_drain(timeout=1)
        _generated(h, 1)
        assert (await retry).expected_samples == 0
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_partial_input_does_not_require_generation_or_flush(mocker):
    h = await _open(mocker)
    flush = mocker.patch.object(h.port, "flush_audio_prefix", new=mocker.AsyncMock())
    try:
        await h.run(frame(960))
        target = await h.runner.begin_audio_drain(timeout=1)
        assert target.accepted_seq == target.expected_samples == 0
        assert h.port.submissions == []
        flush.assert_not_awaited()
    finally:
        await close_harness(h)


@pytest.mark.asyncio
@pytest.mark.parametrize("binding", [None, 0, 1])
@pytest.mark.parametrize("keyword", [False, True])
async def test_flush_only_uses_existing_live_bound_replica(binding, mocker, keyword):
    clients = [FakeStageClient(stage_type="llm"), FakeStageClient(stage_type="llm")]
    pool = _build_stage_pools([clients])[0]
    calls = [
        mocker.patch.object(client, "flush_codec_prefix_async", new=mocker.AsyncMock(return_value=4), create=True)
        for client in clients
    ]
    if binding is not None:
        pool._request_bindings["epoch-qualified"] = binding
    args, kwargs = ((), {"request_id": "epoch-qualified", "sequence": 4}) if keyword else (("epoch-qualified", 4), {})
    if binding is None:
        with pytest.raises(RuntimeError, match="bound"):
            await pool.flush_codec_prefix(*args, **kwargs)
        assert all(call.await_count == 0 for call in calls)
    else:
        assert await pool.flush_codec_prefix(*args, **kwargs) == 4
        calls[binding].assert_awaited_once_with("epoch-qualified", 4)
        assert calls[1 - binding].await_count == 0


@pytest.mark.asyncio
async def test_rebinding_during_flush_cannot_credit_the_old_replica(mocker):
    clients = [FakeStageClient(stage_type="llm"), FakeStageClient(stage_type="llm")]
    pool = _build_stage_pools([clients])[0]
    pool._request_bindings["epoch-qualified"] = 0

    async def rpc(*_):
        pool._request_bindings["epoch-qualified"] = 1
        return 4

    mocker.patch.object(clients[0], "flush_codec_prefix_async", new=rpc, create=True)

    with pytest.raises(RuntimeError, match="binding"):
        await pool.flush_codec_prefix("epoch-qualified", 4)


@pytest.mark.asyncio
async def test_control_dispatch_freezes_in_command_order_and_does_not_block_close(mocker):
    h = await _open(mocker)
    try:
        h.submit(frame())
        h.manager.dispatch(DrainDuplexAudioMessage(control_id="drain", session_id=h.session.session_id, timeout_s=2))
        h.submit(commands.Heartbeat())
        await _until(lambda: h.runner._mailbox.empty() and not h.runner.tasks.append_tasks)
        assert h.results.empty()
        h.submit(commands.CloseSession())
        result = await asyncio.wait_for(h.results.get(), 1)
        assert isinstance(result, DuplexControlResultMessage)
        assert result.control_id == "drain" and not result.ok
        assert result.audio_drain_target is None
        assert "closed" in result.error_message
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_handle_returns_correlated_target_without_a_wire_command(mocker):
    from tests.engine.test_duplex_omni_engine import _engine, _ok

    target = AudioDrainTarget("epoch-qualified", 1, 4, 5760)
    engine = _engine(_ok("drain_audio", audio_drain_target=target))
    from vllm_omni.config.stage_config import DuplexSessionRuntimeConfig

    engine.duplex_session_config = DuplexSessionRuntimeConfig()
    omni = DuplexOmni.__new__(DuplexOmni)
    omni.engine = engine
    handle = DuplexSessionHandle(omni, "sid")
    assert await handle.drain_audio(timeout=2) is target
    ((key, message, timeout),) = engine.rpc_client.calls
    assert key == ("duplex", message.control_id)
    assert isinstance(message, DrainDuplexAudioMessage)
    assert 0 < message.timeout_s == timeout <= 2
    assert engine.request_queue.sync_q.items == []


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan"), True])
async def test_nonfinite_or_nonpositive_timeout_is_rejected_before_enqueue(timeout, mocker):
    h = await _open(mocker)
    try:
        with pytest.raises(ValueError):
            h.runner.begin_audio_drain(timeout=timeout)
        assert h.runner._audio_drain_result is None
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_plugin_without_audio_drain_contract_rejects_before_enqueue(mocker):
    h = await _open(mocker)
    contract = mocker.patch.object(h.runner.plugin, "audio_drain_samples", return_value=None)
    try:
        await h.run(frame())
        pending = h.runner._mailbox.qsize()
        with pytest.raises(NotImplementedError, match="no acoustic drain contract"):
            h.runner.begin_audio_drain(timeout=1)
        contract.assert_called_once_with(0)
        assert h.runner._mailbox.qsize() == pending
        assert h.runner._audio_drain_result is None
        assert h.runner._audio_drain_barrier is None
        await h.run(frame())
        assert h.session.audio_delivery.accepted_seq == 2
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_cancelled_caller_releases_waiter_without_poisoning_append_tail(mocker):
    h = await _open(mocker)
    try:
        h.port.submit_gate = asyncio.Event()
        h.submit(frame())
        await h.port.submit_started.wait()
        drain = h.runner.begin_audio_drain(timeout=2)
        await _until(lambda: h.runner._mailbox.empty())
        drain.cancel()
        await _until(lambda: not h.runner._background_tasks)
        h.port.submit_gate.set()
        await _until(lambda: not h.runner.tasks.append_tasks)
        await h.run(frame())
        assert h.session.audio_delivery.accepted_seq == 2
    finally:
        h.port.submit_gate.set()
        await close_harness(h)


@pytest.mark.asyncio
@pytest.mark.parametrize("keyword", [False, True])
async def test_dp_flush_targets_only_the_request_engine_and_never_broadcasts(mocker, keyword):
    client = DPLBStageEngineCoreClient.__new__(DPLBStageEngineCoreClient)
    client.reqs_in_flight = {"epoch-qualified": b"engine-two"}
    targeted = mocker.patch.object(client, "_call_utility_async", new=mocker.AsyncMock(return_value=4))
    broadcast = mocker.patch.object(client, "call_utility_async", new=mocker.AsyncMock())
    if keyword:
        result = await client.flush_codec_prefix_async(request_id="epoch-qualified", sequence=4)
    else:
        result = await client.flush_codec_prefix_async("epoch-qualified", 4)
    assert result == 4
    targeted.assert_awaited_once_with("omni_flush_codec_prefix", "epoch-qualified", 4, engine=b"engine-two")
    broadcast.assert_not_awaited()


@pytest.mark.asyncio
async def test_dp_flush_uses_upstream_admission_and_utility_wire(mocker):
    """Run the vLLM DP route and utility helper; no processes or sockets.

    Bootstrap stops at the transport-owning base constructor. The real DPLB
    constructor initializes routing, and real admission populates the route.
    The utility helper, encoder and utility-result handler remain unpatched.
    """

    def bootstrap(
        client,
        vllm_config,
        executor_class,
        log_stats,
        client_addresses,
        client_count,
        client_index,
        renderer,
    ):
        client.core_engines = [b"engine-one", b"engine-two"]
        client.client_index = client_index
        client.current_wave = 0
        client.engines_running = True
        client.encoder = MsgpackEncoder()
        client.utility_results = {}

    mocker.patch.object(DPAsyncMPClient, "__init__", new=bootstrap)
    client = DPLBStageEngineCoreClient.__new__(DPLBStageEngineCoreClient)
    client.stage_id = 0
    DPLBAsyncMPClient.__init__(client, mocker.sentinel.config, mocker.sentinel.executor, False)
    mocker.patch.object(client, "_ensure_stats_update_task")
    mocker.patch.object(client, "_ensure_output_queue_task")
    broadcast = mocker.patch.object(client, "call_utility_async", new=mocker.AsyncMock())
    request_id = "session:epoch-3:stage-0"
    request = EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=[1],
        mm_features=None,
        sampling_params=None,
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=1,
        resumable=True,
        external_req_id=request_id,
    )
    admissions = []
    utilities = []

    async def send(message, engine):
        if message[0] == EngineCoreRequestType.ADD.value:
            admitted = MsgpackDecoder(EngineCoreRequest).decode(message[1:])
            admissions.append((engine, admitted.request_id, admitted.external_req_id, admitted.resumable))
            return
        assert message[0] == EngineCoreRequestType.UTILITY.value
        client_index, call_id, method, args = MsgpackDecoder().decode(message[1:])
        utilities.append((engine, client_index, method, args))
        _process_utility_output(UtilityOutput(call_id=call_id, result=UtilityResult(4)), client.utility_results)

    mocker.patch.object(client, "_send_input_message", new=send)
    assert not client.reqs_in_flight
    await client.add_request_async(request)
    assert client.reqs_in_flight == {request_id: b"engine-two"}
    # A resumable append keeps its admission route, even if a new rank is
    # requested. The flush must use that same recorded engine.
    request.data_parallel_rank = 0
    await client.add_request_async(request)
    assert await client.flush_codec_prefix_async(request_id, 4) == 4
    assert admissions == [(b"engine-two", request_id, request_id, True)] * 2
    assert utilities == [(b"engine-two", 0, "omni_flush_codec_prefix", [request_id, 4])]
    assert not client.utility_results
    broadcast.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", [False, True])
async def test_missing_or_rebound_dp_route_cannot_claim_flush(mocker, missing):
    client = DPLBStageEngineCoreClient.__new__(DPLBStageEngineCoreClient)
    client.reqs_in_flight = {} if missing else {"epoch-qualified": b"old"}

    async def targeted(*_, **__):
        client.reqs_in_flight["epoch-qualified"] = b"new"
        return 4

    rpc = mocker.patch.object(client, "_call_utility_async", new=mocker.AsyncMock(side_effect=targeted))
    with pytest.raises(RuntimeError, match="bound|binding"):
        await client.flush_codec_prefix_async("epoch-qualified", 4)
    assert rpc.await_count == int(not missing)


@pytest.mark.asyncio
@pytest.mark.parametrize("keyword", [False, True])
async def test_single_engine_client_uses_scheduler_utility_not_worker_collective_rpc(mocker, keyword):
    client = StageEngineCoreClient.__new__(StageEngineCoreClient)
    utility = mocker.patch.object(client, "call_utility_async", new=mocker.AsyncMock(return_value=4))
    collective = mocker.patch.object(client, "collective_rpc_async", new=mocker.AsyncMock())
    if keyword:
        result = await client.flush_codec_prefix_async(request_id="epoch-qualified", sequence=4)
    else:
        result = await client.flush_codec_prefix_async("epoch-qualified", 4)
    assert result == 4
    utility.assert_awaited_once_with("omni_flush_codec_prefix", "epoch-qualified", 4)
    collective.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("command", [commands.CancelResponse(response_id="unrelated"), commands.BargeIn()])
async def test_unrelated_cancel_or_unsupported_barge_in_does_not_cancel_drain(command, mocker):
    h = await _open(mocker)
    mocker.patch.object(h.port, "flush_audio_prefix", new=mocker.AsyncMock(return_value=1))
    try:
        await h.run(frame())
        drain = h.runner.begin_audio_drain(timeout=2)
        h.submit(command)
        await _until(lambda: h.runner._mailbox.empty() and not h.runner.tasks.append_tasks)
        assert not drain.done()
        _generated(h, 1)
        assert (await drain).expected_samples == 0
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_expired_prefix_barrier_is_bounded_while_submit_remains_blocked(mocker):
    h = await _open(mocker)
    try:
        h.port.submit_gate = asyncio.Event()
        h.submit(frame())
        await h.port.submit_started.wait()
        drain = h.runner.begin_audio_drain(timeout=0.02)
        with pytest.raises(TimeoutError):
            await drain
        for _ in range(3):
            with pytest.raises(RuntimeError, match="pending"):
                h.runner.begin_audio_drain(timeout=1)
        assert len(h.runner.tasks.append_tasks) == 2  # submit and one barrier
        h.port.submit_gate.set()
        await _until(lambda: not h.runner.tasks.append_tasks)
        assert h.session.audio_delivery.accepted_seq == 1
    finally:
        h.port.submit_gate.set()
        await close_harness(h)


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", [None, True, 0, 3])
async def test_invalid_codec_reply_cannot_claim_acoustic_drain(reply, mocker):
    h = await _open(mocker)
    mocker.patch.object(h.port, "flush_audio_prefix", new=mocker.AsyncMock(return_value=reply))
    try:
        await h.run(frame())
        drain = h.runner.begin_audio_drain(timeout=2)
        _generated(h, 1)
        with pytest.raises(RuntimeError, match="not flushed"):
            await drain
        assert h.session.audio_delivery.completed_samples == 0
    finally:
        await close_harness(h)
