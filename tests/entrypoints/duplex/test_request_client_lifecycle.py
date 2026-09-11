# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
import time

import pytest

from vllm_omni.engine.duplex.contracts import duplex_resource_request_id
from vllm_omni.engine.duplex.control_client import DuplexControlRequestError
from vllm_omni.engine.duplex.messages import DuplexFence
from vllm_omni.entrypoints.duplex_request_client import DuplexRequestClient, DuplexRequestOutputPort

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _client(engine):
    return DuplexRequestClient(
        engine,
        DuplexRequestOutputPort(
            request_states={},
            num_stages=1,
            log_stats=False,
            start_output_handler=lambda: None,
            process_single_result=lambda *_: None,
        ),
    )


def _result(fence, role="stage0"):
    return {
        "stage_results": [
            {
                "result": {
                    "data_plane_append": True,
                    "request_id": duplex_resource_request_id(fence, role),
                    "response_stage_id": 0,
                }
            }
        ]
    }


async def _append(client, fence, *, timeout=1.0, collect_outputs=False):
    return await client.append(
        fence.session_id,
        mode="append_tokens",
        payload={},
        operation_id="op",
        final=False,
        expected_epoch=fence.epoch,
        fence=fence,
        timeout=timeout,
        collect_outputs=collect_outputs,
    )


@pytest.mark.asyncio
async def test_append_timeout_includes_waiting_for_session_lock(mocker):
    fence = DuplexFence("deadline-queue")
    entered, release = asyncio.Event(), asyncio.Event()

    async def engine_append(*_args, **_kwargs):
        entered.set()
        await release.wait()
        return _result(fence)

    engine = mocker.Mock(append_duplex_input_async=mocker.AsyncMock(side_effect=engine_append))
    client = _client(engine)
    first = asyncio.create_task(_append(client, fence, timeout=None))
    second = None
    try:
        await asyncio.wait_for(entered.wait(), 1)
        second = asyncio.create_task(_append(client, fence, timeout=0.01))
        done, _ = await asyncio.wait((second,), timeout=0.2)
        assert second in done, "append ignored its deadline while queued behind another append"
        with pytest.raises(TimeoutError):
            await second
        assert engine.append_duplex_input_async.await_count == 1
    finally:
        release.set()
        for task in (first, second):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(*(task for task in (first, second) if task is not None), return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", ["close", "cancel"])
@pytest.mark.parametrize("reply", ["success", "rebuild_failure"])
async def test_late_append_reply_cannot_restore_retired_fence_state(mocker, terminal, reply):
    fence = DuplexFence("retired-append")
    entered, release = asyncio.Event(), asyncio.Event()

    async def engine_append(*_args, **_kwargs):
        entered.set()
        await release.wait()
        if reply == "rebuild_failure":
            raise DuplexControlRequestError({"error": {"code": "kv_recovery_failed"}})
        return _result(fence)

    engine = mocker.Mock(
        append_duplex_input_async=mocker.AsyncMock(side_effect=engine_append),
        close_duplex_session_async=mocker.AsyncMock(return_value={}),
        signal_duplex_turn_async=mocker.AsyncMock(return_value={}),
    )
    client = _client(engine)
    pending = asyncio.create_task(_append(client, fence))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        if terminal == "close":
            await client.close(fence.session_id, reason="test", fence=fence, timeout=1)
        else:
            await client.signal(
                fence.session_id,
                event="input.cancel",
                fence=fence,
                next_fence=None,
                session_config=None,
                runtime_config=None,
                timeout=1,
            )
        release.set()
        await asyncio.gather(pending, return_exceptions=True)
        assert not client._resource_generations, "late success resurrected generation state after close/cancel"
        assert not client._append_locks
        assert not client.output_port.request_states
    finally:
        release.set()
        if not pending.done():
            pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)


@pytest.mark.asyncio
async def test_append_output_collection_uses_remaining_deadline(mocker):
    from vllm_omni.entrypoints import duplex_request_client as module

    fence = DuplexFence("one-budget")
    clock = [100.0]
    mocker.patch.object(module, "time", mocker.Mock(time=time.time, monotonic=lambda: clock[0]))

    async def engine_append(*_args, **_kwargs):
        clock[0] += 0.6
        return _result(fence)

    client = _client(mocker.Mock(append_duplex_input_async=mocker.AsyncMock(side_effect=engine_append)))
    collector = mocker.patch.object(client, "collect_outputs", new=mocker.AsyncMock(return_value=[]))
    await _append(client, fence, timeout=1.0, collect_outputs=True)
    assert collector.call_args.kwargs["timeout"] == pytest.approx(0.4)


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", ["close", "cancel"])
async def test_retired_route_wakes_unbounded_collectors_and_queued_append(mocker, terminal):
    fence = DuplexFence("retire-readers")
    engine = mocker.Mock(
        append_duplex_input_async=mocker.AsyncMock(return_value=_result(fence)),
        close_duplex_session_async=mocker.AsyncMock(return_value={}),
        signal_duplex_turn_async=mocker.AsyncMock(return_value={}),
    )
    client = _client(engine)
    collecting = asyncio.Event()
    original_collect = client.collect_outputs

    async def collect(*args, **kwargs):
        collecting.set()
        return await original_collect(*args, **kwargs)

    mocker.patch.object(client, "collect_outputs", side_effect=collect)
    first = asyncio.create_task(_append(client, fence, timeout=None, collect_outputs=True))
    tasks = [first]
    try:
        await asyncio.wait_for(collecting.wait(), 1)
        request_id = duplex_resource_request_id(fence, "stage0")
        tasks.append(
            asyncio.create_task(
                client.collect_registered_outputs(
                    request_id,
                    response_stage_id=0,
                    timeout=None,
                )
            )
        )
        tasks.append(asyncio.create_task(_append(client, fence, timeout=None)))
        await asyncio.sleep(0)
        if terminal == "close":
            await client.close(fence.session_id, reason="test", fence=fence, timeout=1)
        else:
            await client.signal(
                fence.session_id,
                event="input.cancel",
                fence=fence,
                next_fence=None,
                session_config=None,
                runtime_config=None,
                timeout=1,
            )
        done, pending = await asyncio.wait(tasks, timeout=0.2)
        assert not pending, "close/cancel left an infinite output or append-lock wait"
        assert isinstance(first.exception(), RuntimeError)
        assert tasks[1].result() == []
        assert isinstance(tasks[2].exception(), RuntimeError)
        assert engine.append_duplex_input_async.await_count == 1
        assert not client._resource_generations and not client._append_locks
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["rollover", "rebuild_failure"])
async def test_removed_generation_wakes_its_unbounded_collector(mocker, outcome):
    fence = DuplexFence("generation-readers")
    reply = (
        _result(fence, "stage0g1")
        if outcome == "rollover"
        else DuplexControlRequestError({"error": {"code": "kv_recovery_failed"}})
    )
    engine = mocker.Mock(append_duplex_input_async=mocker.AsyncMock(side_effect=[_result(fence), reply]))
    client = _client(engine)
    await _append(client, fence)
    request_id = duplex_resource_request_id(fence, "stage0")
    reader = asyncio.create_task(client.collect_registered_outputs(request_id, response_stage_id=0, timeout=None))
    try:
        await asyncio.sleep(0)
        if outcome == "rollover":
            await _append(client, fence)
        else:
            with pytest.raises(DuplexControlRequestError):
                await _append(client, fence)
        done, _ = await asyncio.wait((reader,), timeout=0.2)
        assert reader in done, "removing the old KV generation stranded its output reader"
        assert reader.result() == []
        if outcome == "rollover":
            next_id = duplex_resource_request_id(fence, "stage0g1")
            assert client.output_port.request_states[next_id] not in client._retired_output_routes
    finally:
        if not reader.done():
            reader.cancel()
        await asyncio.gather(reader, return_exceptions=True)
