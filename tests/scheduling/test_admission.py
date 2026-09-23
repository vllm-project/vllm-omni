# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
from types import SimpleNamespace

import pytest

from vllm_omni.scheduling.config import TailAwareSchedulingConfig
from vllm_omni.scheduling.controller import TailAwareController, TailAwareQueueFullError

pytestmark = [pytest.mark.cpu, pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.asyncio]


def _controller(replica_ids=(7,), limit=8):
    settings = TailAwareSchedulingConfig(enabled=True, max_pending_requests=limit, hardware_profile="910B2")
    return TailAwareController(list(replica_ids), settings)


def _acquire(policy, request_id):
    params = SimpleNamespace(width=512, height=512, num_inference_steps=20)
    return policy.acquire(request_id, params, model_class_name="QwenImagePipeline")


async def test_fifo_bounded_single_slot_replicas():
    policy = _controller((7, 3), limit=2)
    tasks = []
    try:
        assert (await _acquire(policy, "first")).replica_id == 3
        assert (await _acquire(policy, "second")).replica_id == 7
        third = asyncio.create_task(_acquire(policy, "third"))
        fourth = asyncio.create_task(_acquire(policy, "fourth"))
        tasks.extend((third, fourth))
        await asyncio.sleep(0)
        with pytest.raises(TailAwareQueueFullError):
            await _acquire(policy, "overflow")
        await asyncio.sleep(0)
        assert not third.done() and not fourth.done()
        policy.complete("second")
        assert (await asyncio.wait_for(third, 1)).replica_id == 7
        policy.complete("second")  # A late completion cannot release third's slot.
        await asyncio.sleep(0)
        assert not fourth.done()
        policy.complete("first")
        assert (await asyncio.wait_for(fourth, 1)).replica_id == 3
    finally:
        policy.close()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.parametrize("pending", [True, False], ids=["pending", "bound-before-resume"])
async def test_cancellation_releases_capacity(pending):
    policy = _controller()
    tasks = []
    try:
        if pending:
            await _acquire(policy, "occupied")
        raced = asyncio.create_task(_acquire(policy, "raced"))
        tasks.append(raced)
        await asyncio.sleep(0)  # acquire has queued the drain callback.
        if pending:
            raced.cancel()
            policy.complete("occupied")
        else:
            # The callback runs after binding but before acquire resumes.
            asyncio.get_running_loop().call_soon(policy.cancel, "raced")
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(raced, 1)
        policy.cancel("raced")
        policy.complete("raced")
        successor = asyncio.create_task(_acquire(policy, "successor"))
        tasks.append(successor)
        assert (await asyncio.wait_for(successor, 1)).replica_id == 7
    finally:
        policy.close()
        await asyncio.gather(*tasks, return_exceptions=True)


async def test_replica_failure_keeps_remaining_replica_usable():
    policy = _controller((3, 7))
    tasks = []
    try:
        await _acquire(policy, "on3")
        await _acquire(policy, "on7")
        waiting = asyncio.create_task(_acquire(policy, "waiting"))
        tasks.append(waiting)
        await asyncio.sleep(0)
        assert policy.remove_replica(3) == ("on3",)
        assert policy.remove_replica(3) == ()
        policy.complete("on7")
        assert (await asyncio.wait_for(waiting, 1)).replica_id == 7
        stranded = asyncio.create_task(_acquire(policy, "stranded"))
        tasks.append(stranded)
        await asyncio.sleep(0)
        assert policy.remove_replica(7) == ("waiting",)
        with pytest.raises(RuntimeError, match="no diffusion replicas"):
            await asyncio.wait_for(stranded, 1)
    finally:
        policy.close()
        await asyncio.gather(*tasks, return_exceptions=True)
