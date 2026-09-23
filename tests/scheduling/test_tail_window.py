# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Tail admission and rolling-window decisions through the real controller."""

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from vllm_omni.scheduling import controller as controller_module
from vllm_omni.scheduling.config import TailAwareSchedulingConfig
from vllm_omni.scheduling.controller import TailAwareController

pytestmark = [pytest.mark.cpu, pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.asyncio]


async def _flush():
    # Register admissions, drain the queue, then resume admitted coroutines.
    for _ in range(3):
        await asyncio.sleep(0)


def _dispatched(pending):
    return {name: task.result().replica_id for name, task in pending.items() if task.done()}


@asynccontextmanager
async def _policy(**options):
    policy = TailAwareController([0, 1], TailAwareSchedulingConfig(enabled=True, hardware_profile="910B2", **options))
    pending = {}
    try:
        yield policy, pending
    finally:
        policy.close()
        await asyncio.gather(*pending.values(), return_exceptions=True)


@asynccontextmanager
async def _queued_window(monkeypatch, **options):
    now = [0.0]
    monkeypatch.setattr(controller_module.time, "perf_counter", lambda: now[0])
    # Known service times isolate planning; classification and dispatch remain real.
    monkeypatch.setattr(controller_module, "estimate_service_time_s", lambda service, *args: service)
    settings = dict(
        quota_every=2,
        quota_amount=0,
        threshold_ratio=1.0,
        beam_min_pending=2,
        beam_max_pending=8,
        beam_horizon=2,
        beam_width=4,
        beam_branch_width=4,
        beam_risk_slack_s=1000.0,
    )
    async with _policy(**(settings | options)) as (policy, pending):
        assert (await policy.acquire("blocker-0", 1.0, "QwenImagePipeline")).replica_id == 0
        assert (await policy.acquire("blocker-1", 150.0, "QwenImagePipeline")).replica_id == 1
        for arrival, name, service in (
            (0.0, "old-short", 10.0),
            (0.0, "long", 100.0),
            (10.0, "later-short", 10.0),
            (40.0, "medium", 20.0),
        ):
            now[0] = arrival
            pending[name] = asyncio.create_task(policy.acquire(name, service, "QwenImagePipeline"))
            await _flush()
        now[0] = 50.0
        assert not _dispatched(pending)
        assert policy.active_count == 2 and policy.pending_count == 4
        yield policy, now, pending


async def test_real_estimates_classify_gate_and_pack_tails(monkeypatch):
    monkeypatch.setattr(controller_module.time, "perf_counter", lambda: 0.0)
    short, medium, long = [
        SimpleNamespace(width=size, height=size, num_inference_steps=steps, num_frames=1, num_outputs_per_prompt=1)
        for size, steps in ((512, 20), (1024, 25), (1536, 35))
    ]
    async with _policy(quota_every=2) as (policy, pending):
        classified = []
        for index, sampling in enumerate((short, short, long, medium, long, long, long)):
            name = f"warmup-{index}"
            classified.append((await policy.acquire(name, sampling, "QwenImagePipeline")).is_tail)
            policy.complete(name)
        # Short fails the ratio, medium fails the maximum threshold, and no
        # quota remains for the final long despite its qualifying duration.
        assert classified == [False, False, True, False, True, True, False]
        for replica in (0, 1):
            assert (await policy.acquire(f"blocker-{replica}", short, "QwenImagePipeline")).replica_id == replica
        for name, sampling in (("tail-0", long), ("tail-1", long), ("tail-2", long), ("normal", short)):
            pending[name] = asyncio.create_task(policy.acquire(name, sampling, "QwenImagePipeline"))
        await _flush()
        assert not _dispatched(pending)
        policy.complete("blocker-0")
        policy.complete("blocker-1")
        await _flush()
        # Tail reservations pack onto 0, preserving 1 for Normal; only the
        # newest Tail may backfill after the central Normal queue drains.
        assert _dispatched(pending) == {"normal": 1, "tail-2": 0}
        assert not pending["normal"].result().is_tail and pending["tail-2"].result().is_tail
        policy.complete("normal")
        assert (await pending["tail-1"]).replica_id == 1
        assert pending["tail-1"].result().is_tail and not pending["tail-0"].done()
        policy.complete("tail-2")
        assert (await pending["tail-0"]).replica_id == 0
        assert pending["tail-0"].result().is_tail


@pytest.mark.parametrize("tail,expected", [(False, "old-short"), (True, "later-short")])
async def test_window_changes_immediate_risk_choice(monkeypatch, tail, expected):
    async with _queued_window(monkeypatch, quota_amount=int(tail)) as (policy, _now, pending):
        policy.complete("blocker-0")
        await _flush()
        # Immediate risk chooses long: 50 + .85*100 = 135. Window planning
        # considers the other occupied replica and whether its request is Tail.
        assert _dispatched(pending) == {expected: 0}
        assert not pending[expected].result().is_tail
        assert policy.active_count == 2 and policy.pending_count == 3


@pytest.mark.parametrize("new_arrival,expected", [(False, "long"), (True, "later-short")])
async def test_arrivals_replan_after_completion_without_preemption(monkeypatch, new_arrival, expected):
    async with _queued_window(monkeypatch) as (policy, now, pending):
        policy.complete("blocker-0")
        await _flush()
        assert _dispatched(pending) == {"old-short": 0}
        now[0] = 55.0
        if new_arrival:
            pending["new"] = asyncio.create_task(policy.acquire("new", 200.0, "QwenImagePipeline"))
        await _flush()
        assert _dispatched(pending) == {"old-short": 0}
        now[0] = 60.0
        policy.complete("old-short")
        await _flush()
        assert _dispatched(pending) == {"old-short": 0, expected: 0}
        now[0] = 1000.0  # Both running estimates have expired, but neither completed.
        pending["late"] = asyncio.create_task(policy.acquire("late", 1.0, "QwenImagePipeline"))
        await _flush()
        assert _dispatched(pending) == {"old-short": 0, expected: 0}
        assert policy.active_count == 2


@pytest.mark.parametrize("window", [(5, 8), (2, 3)])
async def test_outside_window_falls_back_to_risk_order(monkeypatch, window):
    async with _queued_window(monkeypatch, beam_min_pending=window[0], beam_max_pending=window[1]) as (
        policy,
        _now,
        pending,
    ):
        policy.complete("blocker-0")
        await _flush()
        # The same four requests pick a short request inside the window.
        assert _dispatched(pending) == {"long": 0}
        assert not pending["long"].result().is_tail
        assert policy.active_count == 2 and policy.pending_count == 3
