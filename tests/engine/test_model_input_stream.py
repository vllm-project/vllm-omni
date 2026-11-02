# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Ordered input append, wire transport, and request ownership contracts."""

import asyncio
from queue import Queue
from types import SimpleNamespace

import msgspec
import pytest

from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
from vllm_omni.engine.messages import TextInputUpdate
from vllm_omni.engine.orchestrator import Orchestrator, OrchestratorRequestState
from vllm_omni.engine.stage_engine_core_client import StageEngineCoreClient
from vllm_omni.engine.stage_engine_core_proc import StageEngineCoreProc
from vllm_omni.engine.stage_pool import StagePool, StageUnavailableError
from vllm_omni.errors import OmniClientError

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.asyncio
async def test_append_uses_the_sender_side_of_the_cross_thread_queue():
    engine = AsyncOmniEngine.__new__(AsyncOmniEngine)
    engine.request_queue = SimpleNamespace(sync_q=Queue())
    tokens = [32]
    await engine.append_text_async("request", tokens)
    tokens.clear()
    await engine.append_text_async("request", [], final=True)

    assert [engine.request_queue.sync_q.get(), engine.request_queue.sync_q.get()] == [
        TextInputUpdate(request_id="request", token_ids=[32]),
        TextInputUpdate(request_id="request", token_ids=[], final=True),
    ]


@pytest.mark.asyncio
async def test_append_uses_only_the_existing_stage_replica(mocker):
    clients = [SimpleNamespace(stage_type="llm", append_text_async=mocker.AsyncMock()) for _ in range(2)]
    pool = StagePool(0, clients)
    pool.bind_request_to_replica("request", 1)
    update = TextInputUpdate(request_id="request", token_ids=[32])
    await pool.append_text(update)
    clients[0].append_text_async.assert_not_awaited()
    clients[1].append_text_async.assert_awaited_once_with(update)
    with pytest.raises(StageUnavailableError):
        await pool.append_text(TextInputUpdate(request_id="missing", token_ids=[32]))


@pytest.mark.asyncio
async def test_append_utility_errors_are_request_scoped(mocker):
    client = StageEngineCoreClient.__new__(StageEngineCoreClient)
    client.call_utility_async = mocker.AsyncMock(return_value={"error": "input already closed"})
    with pytest.raises(OmniClientError, match="closed"):
        await client.append_text_async(TextInputUpdate(request_id="request", token_ids=[32]))
    client.call_utility_async.assert_awaited_once()


def test_actual_utility_argument_decoding_restores_the_input_envelope(mocker):
    core = StageEngineCoreProc.__new__(StageEngineCoreProc)
    append = mocker.Mock()
    core.scheduler = SimpleNamespace(append_text=append)
    update = TextInputUpdate(request_id="request", token_ids=[32])
    wire_args = msgspec.msgpack.decode(msgspec.msgpack.encode([update]))
    args = core._convert_msgspec_args(core.omni_append_text, wire_args)
    assert isinstance(args[0], dict)
    assert core.omni_append_text(*args) == {"accepted": True}
    append.assert_called_once_with(update)


@pytest.mark.asyncio
@pytest.mark.parametrize("entry_stage_id", [0, 1])
async def test_orchestrator_append_targets_the_entry_stage_and_reports_unknown_requests(mocker, entry_stage_id):
    orchestrator = Orchestrator.__new__(Orchestrator)
    state = OrchestratorRequestState(
        request_id="request", sampling_params_list=[], entry_stage_id=entry_stage_id, final_stage_id=entry_stage_id + 1
    )
    orchestrator.request_states = {"request": state}
    orchestrator.stage_pools = [SimpleNamespace(append_text=mocker.AsyncMock()) for _ in range(entry_stage_id + 2)]
    orchestrator.output_async_queue = asyncio.Queue()
    update = TextInputUpdate(request_id="request", token_ids=[32])
    assert await orchestrator._dispatch_message(update)
    orchestrator.stage_pools[entry_stage_id].append_text.assert_awaited_once_with(update)
    for stage_id, pool in enumerate(orchestrator.stage_pools):
        if stage_id != entry_stage_id:
            pool.append_text.assert_not_awaited()
    assert orchestrator.request_states == {"request": state}
    missing = TextInputUpdate(request_id="missing", token_ids=[32])
    await orchestrator._dispatch_message(missing)
    error = orchestrator.output_async_queue.get_nowait()
    assert error.request_id == "missing" and error.status_code == 400
    assert orchestrator.request_states == {"request": state}
