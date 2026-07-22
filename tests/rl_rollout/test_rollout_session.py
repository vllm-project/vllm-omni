# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for RolloutSessionStore (RFC #3747, P0)."""

import asyncio

import pytest

from vllm_omni.entrypoints.openai.rollout_session import (
    RolloutSession,
    RolloutSessionClosedError,
    RolloutSessionNotFoundError,
    RolloutSessionStepError,
    RolloutSessionStore,
)


@pytest.fixture
def store() -> RolloutSessionStore:
    return RolloutSessionStore()


@pytest.mark.asyncio
async def test_create_returns_session(store):
    s = await store.create("s1", model="dreamzero", mode="world_model_env")
    assert isinstance(s, RolloutSession)
    assert s.session_id == "s1"
    assert s.committed_step_id == -1
    assert s.context_length == 0
    assert not s.closed


@pytest.mark.asyncio
async def test_get_existing(store):
    await store.create("s1", model="dreamzero", mode="world_model_env")
    s = await store.get("s1")
    assert s.session_id == "s1"


@pytest.mark.asyncio
async def test_get_missing_raises(store):
    with pytest.raises(RolloutSessionNotFoundError):
        await store.get("does-not-exist")


@pytest.mark.asyncio
async def test_get_closed_raises(store):
    await store.create("s1", model="dreamzero", mode="world_model_env")
    await store.close("s1")
    with pytest.raises(RolloutSessionClosedError):
        await store.get("s1")


@pytest.mark.asyncio
async def test_close_missing_raises(store):
    with pytest.raises(RolloutSessionNotFoundError):
        await store.close("does-not-exist")


@pytest.mark.asyncio
async def test_close_closed_raises(store):
    await store.create("s1", model="dreamzero", mode="world_model_env")
    await store.close("s1")
    with pytest.raises(RolloutSessionClosedError):
        await store.close("s1")


@pytest.mark.asyncio
async def test_advance_updates_committed_step_id(store):
    await store.create("s1", model="dreamzero", mode="world_model_env")
    await store.advance("s1", step_id=0)
    s = await store.get("s1")
    assert s.committed_step_id == 0
    assert s.context_length == 1

    await store.advance("s1", step_id=1)
    s = await store.get("s1")
    assert s.committed_step_id == 1
    assert s.context_length == 2


@pytest.mark.asyncio
async def test_advance_rejects_duplicate_step_id(store):
    await store.create("s1", model="dreamzero", mode="world_model_env")
    await store.advance("s1", step_id=0)
    with pytest.raises(RolloutSessionStepError):
        await store.advance("s1", step_id=0)

    s = await store.get("s1")
    assert s.committed_step_id == 0
    assert s.context_length == 1


@pytest.mark.asyncio
async def test_advance_rejects_out_of_order_step_id(store):
    await store.create("s1", model="dreamzero", mode="world_model_env")
    with pytest.raises(RolloutSessionStepError):
        await store.advance("s1", step_id=2)

    s = await store.get("s1")
    assert s.committed_step_id == -1
    assert s.context_length == 0


@pytest.mark.asyncio
async def test_reset_clears_state(store):
    await store.create("s1", model="dreamzero", mode="world_model_env")
    await store.advance("s1", step_id=3)
    await store.reset("s1")
    s = await store.get("s1")
    assert s.committed_step_id == -1
    assert s.context_length == 0


@pytest.mark.asyncio
async def test_reset_closed_raises(store):
    await store.create("s1", model="dreamzero", mode="world_model_env")
    await store.close("s1")
    with pytest.raises(RolloutSessionClosedError):
        await store.reset("s1")


@pytest.mark.asyncio
async def test_failed_step_does_not_advance(store):
    """committed_step_id must not change when a step raises (RFC section 6.5 invariant)."""
    await store.create("s1", model="dreamzero", mode="world_model_env")
    await store.advance("s1", step_id=0)

    # Simulate a failed step: advance is NOT called
    s = await store.get("s1")
    assert s.committed_step_id == 0  # unchanged


@pytest.mark.asyncio
async def test_concurrent_steps_are_serialised(store):
    """Per-session lock must prevent concurrent context mutations."""
    await store.create("s1", model="dreamzero", mode="world_model_env")
    s = await store.get("s1")

    results: list[int] = []

    async def slow_advance(step_id: int, delay: float) -> None:
        await asyncio.sleep(delay)
        async with s.lock:
            await asyncio.sleep(0.01)
            await store.advance("s1", step_id=step_id)
            results.append(step_id)

    await asyncio.gather(slow_advance(0, 0.0), slow_advance(1, 0.01))
    # Both steps ran under the per-session lock and committed in order.
    assert sorted(results) == [0, 1]
