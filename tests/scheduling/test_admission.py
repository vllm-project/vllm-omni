# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
from types import SimpleNamespace

import pytest

from tests.scheduling.helpers import controller_context, drain
from vllm_omni.scheduling.controller import TailAwareQueueFullError

pytestmark = [pytest.mark.cpu, pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.asyncio]


def _acquire(policy, request_id):
    params = SimpleNamespace(width=512, height=512, num_inference_steps=20)
    return policy.acquire(request_id, params, model_class_name="QwenImagePipeline")


async def test_fifo_bounded_single_slot_replicas():
    async with controller_context([7, 3], max_pending_requests=2) as (policy, tasks):
        assert (await _acquire(policy, "first")).replica_id == 3
        assert (await _acquire(policy, "second")).replica_id == 7
        third = asyncio.create_task(_acquire(policy, "third"))
        fourth = asyncio.create_task(_acquire(policy, "fourth"))
        tasks.update(third=third, fourth=fourth)
        await drain()
        with pytest.raises(TailAwareQueueFullError):
            await _acquire(policy, "overflow")
        await drain()
        assert not third.done() and not fourth.done()
        policy.complete("second")
        assert (await asyncio.wait_for(third, 1)).replica_id == 7
        policy.complete("second")  # A late completion cannot release third's slot.
        await drain()
        assert not fourth.done()
        policy.complete("first")
        assert (await asyncio.wait_for(fourth, 1)).replica_id == 3


@pytest.mark.parametrize("pending", [True, False], ids=["pending", "bound-before-resume"])
async def test_cancellation_releases_capacity(pending):
    async with controller_context([7], max_pending_requests=8) as (policy, tasks):
        if pending:
            await _acquire(policy, "occupied")
        raced = asyncio.create_task(_acquire(policy, "raced"))
        tasks["raced"] = raced
        await asyncio.sleep(0)  # Keep this single turn: cancellation races with the drain callback.
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
        tasks["successor"] = successor
        assert (await asyncio.wait_for(successor, 1)).replica_id == 7


async def test_replica_failure_keeps_remaining_replica_usable():
    async with controller_context([3, 7], max_pending_requests=8) as (policy, tasks):
        await _acquire(policy, "on3")
        await _acquire(policy, "on7")
        waiting = asyncio.create_task(_acquire(policy, "waiting"))
        tasks["waiting"] = waiting
        await drain()
        assert policy.remove_replica(3) == ("on3",)
        assert policy.remove_replica(3) == ()
        policy.complete("on7")
        assert (await asyncio.wait_for(waiting, 1)).replica_id == 7
        stranded = asyncio.create_task(_acquire(policy, "stranded"))
        tasks["stranded"] = stranded
        await drain()
        assert policy.remove_replica(7) == ("waiting",)
        with pytest.raises(RuntimeError, match="no diffusion replicas"):
            await asyncio.wait_for(stranded, 1)
