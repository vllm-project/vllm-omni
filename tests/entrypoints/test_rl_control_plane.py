# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from types import SimpleNamespace

import pytest

from vllm_omni.entrypoints.async_omni import AsyncOmni

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_omni(mocker):
    omni = object.__new__(AsyncOmni)
    omni.engine = mocker.MagicMock()
    omni.engine.stage_clients = [SimpleNamespace(), SimpleNamespace()]
    omni.request_states = {}
    omni._pause_cond = asyncio.Condition()
    omni._pause_lock = asyncio.Lock()
    omni._paused = False
    omni._pause_barrier_complete = False
    omni._weight_update_lock = asyncio.Lock()
    omni._weight_update_active = False
    omni._weight_update_stage_ids = None
    omni._weight_update_component = None
    omni._sleeping_stage_ids = set()
    omni._sleeping_tags = set()
    return omni


@pytest.mark.asyncio
async def test_collective_rpc_with_acks_preserves_stage_ids(mocker):
    omni = _make_omni(mocker)
    omni.engine.collective_rpc_result_async = mocker.AsyncMock(
        return_value=SimpleNamespace(
            stage_ids=[1, 1],
            results=[{"rank": 0}, {"rank": 1}],
        )
    )

    acks = await omni.collective_rpc_with_acks("update_weights", stage_ids=[1])

    assert [ack["stage_id"] for ack in acks] == [1, 1]
    assert all(ack["success"] for ack in acks)


@pytest.mark.asyncio
async def test_collective_rpc_with_acks_fails_closed(mocker):
    omni = _make_omni(mocker)
    omni.engine.collective_rpc_result_async = mocker.AsyncMock(
        return_value=SimpleNamespace(
            stage_ids=[0, 1],
            results=[True, {"supported": False, "error": "missing method"}],
        )
    )

    with pytest.raises(RuntimeError, match="stage 1"):
        await omni.collective_rpc_with_acks("reset_mm_cache")


@pytest.mark.asyncio
async def test_pause_abort_establishes_barrier_before_cache_reset(mocker):
    omni = _make_omni(mocker)
    events = []

    async def abort_all():
        events.append("abort")
        return 1

    async def reset_prefix_cache(**kwargs):
        events.append("prefix")
        return True

    async def reset_mm_cache(**kwargs):
        events.append("mm")

    async def reset_encoder_cache(**kwargs):
        events.append("encoder")

    omni.abort_all = abort_all
    omni.reset_prefix_cache = reset_prefix_cache
    omni.reset_mm_cache = reset_mm_cache
    omni.reset_encoder_cache = reset_encoder_cache

    await omni.pause_generation(mode="abort")

    assert await omni.is_paused() is True
    assert events == ["abort", "prefix", "mm", "encoder"]


@pytest.mark.asyncio
async def test_weight_update_transaction_targets_initialized_stages(mocker):
    omni = _make_omni(mocker)
    omni.collective_rpc_with_acks = mocker.AsyncMock(return_value=[{"stage_id": 1, "success": True, "result": None}])

    await omni.init_weight_transfer_engine(
        {"init_info": {"backend": "safetensors", "stage_ids": [1], "component": "transformer"}}
    )
    omni._paused = True
    omni._pause_barrier_complete = True
    await omni.start_weight_update()
    await omni.update_weights({"update_info": {"path": "/tmp/policy.safetensors"}})
    await omni.finish_weight_update()

    calls = omni.collective_rpc_with_acks.await_args_list
    assert calls[0].kwargs["stage_ids"] == [1]
    assert calls[1].kwargs["stage_ids"] == [1]
    assert calls[2].kwargs["kwargs"]["update_info"]["component"] == "transformer"
    assert calls[3].kwargs["stage_ids"] == [1]
    assert omni._weight_update_active is False


@pytest.mark.asyncio
async def test_weight_update_requires_pause_barrier_and_fixed_target(mocker):
    omni = _make_omni(mocker)
    omni.collective_rpc_with_acks = mocker.AsyncMock(return_value=[])
    await omni.init_weight_transfer_engine({"init_info": {"stage_ids": [1], "component": "transformer"}})

    with pytest.raises(RuntimeError, match="pause must complete"):
        await omni.start_weight_update()

    omni._paused = True
    omni._pause_barrier_complete = True
    await omni.start_weight_update()
    with pytest.raises(ValueError, match="stage_ids"):
        await omni.update_weights({"update_info": {"stage_ids": [0], "component": "transformer"}})
    with pytest.raises(ValueError, match="component"):
        await omni.update_weights({"update_info": {"stage_ids": [1], "component": "vae"}})

    await omni.finish_weight_update()
