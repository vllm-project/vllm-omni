# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
from types import SimpleNamespace

import pytest

from vllm_omni.scheduling import controller as controller_module
from vllm_omni.scheduling.config import TailAwareSchedulingConfig
from vllm_omni.scheduling.controller import TailAwareController

pytestmark = [pytest.mark.cpu, pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.asyncio]


def params(size=512, steps=20):
    return SimpleNamespace(width=size, height=size, num_inference_steps=steps, num_frames=1, num_outputs_per_prompt=1)


def controller(replicas=(0,), **settings):
    config = TailAwareSchedulingConfig(enabled=True, hardware_profile="910B2", **settings)
    return TailAwareController(list(replicas), config)


def admit(policy, request_id, sampling):
    return asyncio.create_task(policy.acquire(request_id, sampling, "QwenImagePipeline"))


async def flush():
    for _ in range(3):
        await asyncio.sleep(0)


async def test_calibrated_service_estimates_drive_actual_dispatch_order(monkeypatch):
    monkeypatch.setattr(controller_module.time, "perf_counter", lambda: 100.0)
    policy, waiting = controller(), {}
    try:
        await policy.acquire("running", params(), "QwenImagePipeline")
        for request_id, sampling in (("short", params()), ("medium", params(1024, 25)), ("long", params(1536, 35))):
            waiting[request_id] = admit(policy, request_id, sampling)
        await flush()
        assert not any(task.done() for task in waiting.values())
        # Equal wait: the real 910B2 estimates are 8.60, 14.22 and 43.22 seconds.
        previous = "running"
        for expected in ("long", "medium", "short"):
            policy.complete(previous)
            await flush()
            assert [name for name, task in waiting.items() if task.done()] == [expected]
            assert (await waiting.pop(expected)).replica_id == 0
            previous = expected
    finally:
        policy.close()
        await asyncio.gather(*waiting.values(), return_exceptions=True)


@pytest.mark.parametrize("band,expected", [((3, 3), "newer"), ((2, 2), "older")])
async def test_waiting_time_and_queue_depth_beta_change_next_dispatch(monkeypatch, band, expected):
    now = [100.0]
    monkeypatch.setattr(controller_module.time, "perf_counter", lambda: now[0])
    policy = controller(band_risk_beta=0.2, band_min_pending=band[0], band_max_pending=band[1])
    waiting = {}
    try:
        await policy.acquire("running", params(), "QwenImagePipeline")
        waiting["older"] = admit(policy, "older", params())
        await flush()
        now[0] += 10.0
        waiting["newer"] = admit(policy, "newer", params(1536, 35))
        await flush()
        assert not any(task.done() for task in waiting.values())
        policy.complete("running")
        await flush()
        # At depth two, beta=.2 favors the older short request; beta=.85
        # outside the configured band favors the newer, longer request.
        assert [name for name, task in waiting.items() if task.done()] == [expected]
        assert (await waiting[expected]).replica_id == 0
    finally:
        policy.close()
        await asyncio.gather(*waiting.values(), return_exceptions=True)


@pytest.mark.parametrize("success,expected_replica", [(True, 0), (False, 1)])
async def test_completion_feedback_changes_replica_choice_without_learning_failures(
    monkeypatch, success, expected_replica
):
    now = [0.0]
    monkeypatch.setattr(controller_module.time, "perf_counter", lambda: now[0])
    policy = controller(replicas=(0, 1))
    try:
        assert (await policy.acquire("slow", params(), "QwenImagePipeline")).replica_id == 0
        assert (await policy.acquire("fast", params(), "QwenImagePipeline")).replica_id == 1
        now[0] = 10.0
        policy.complete("fast")
        now[0] = 20.0
        policy.complete("slow")
        # Both slots are idle: successful latency feedback now beats ID order.
        assert (await policy.acquire("probe", params(), "QwenImagePipeline")).replica_id == 1
        now[0] = 220.0
        policy.complete("probe", success=success)
        # A slow success raises replica 1's EMA above replica 0; failure must
        # release capacity without teaching the scheduler that failed latency.
        assert (await policy.acquire("next", params(), "QwenImagePipeline")).replica_id == expected_replica
    finally:
        policy.close()
