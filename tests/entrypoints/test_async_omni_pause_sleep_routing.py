"""Unit tests for AR EngineCore vs diffusion worker pause/sleep routing."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, call

import pytest

from vllm_omni.diffusion.data import CuMemTag, OmniACK
from vllm_omni.entrypoints.async_omni import AsyncOmni

pytestmark = [pytest.mark.core_model]


def _make_omni(*, stage_types: list[str]) -> AsyncOmni:
    omni = object.__new__(AsyncOmni)
    omni._pause_cond = asyncio.Condition()
    omni._paused = False
    omni._admitting = 0
    omni._hold_admission_until_resume = False
    omni._sleeping_tags = set()
    omni._stage_sleeping_tags = {}
    omni._level2_sleeping = False
    omni.event_resolver = SimpleNamespace(watch_task=lambda *a, **k: None, resolve=AsyncMock())
    omni._final_output_handler = lambda: None
    omni.reset_mm_cache = AsyncMock()

    stage_clients = [SimpleNamespace(stage_type=stage_type) for stage_type in stage_types]
    omni.engine = SimpleNamespace(
        stage_clients=stage_clients,
        stage_vllm_configs=[None] * len(stage_types),
        collective_rpc_async=AsyncMock(return_value=[True]),
    )
    omni.collective_rpc = AsyncMock(return_value=[True])
    return omni


@pytest.mark.cpu
def test_split_stage_ids_by_type():
    omni = _make_omni(stage_types=["llm", "diffusion", "llm"])
    ar_ids, diff_ids = omni._split_stage_ids_by_type()
    assert ar_ids == [0, 2]
    assert diff_ids == [1]


@pytest.mark.cpu
def test_split_stage_ids_by_type_rejects_out_of_range():
    omni = _make_omni(stage_types=["llm", "diffusion"])
    with pytest.raises(ValueError, match=r"Invalid stage_ids \[2\].*0\.\.1"):
        omni._split_stage_ids_by_type([0, 2])
    with pytest.raises(ValueError, match=r"Invalid stage_ids \[-1\]"):
        omni._split_stage_ids_by_type([-1])


@pytest.mark.cpu
def test_pause_generation_routes_ar_via_collective_rpc():
    async def run() -> None:
        omni = _make_omni(stage_types=["llm", "diffusion"])
        omni.reset_prefix_cache = AsyncMock(return_value=True)
        omni.reset_mm_cache = AsyncMock()
        omni.reset_encoder_cache = AsyncMock()

        await omni.pause_generation(mode="abort", clear_cache=True)

        assert omni._paused is True
        assert omni._hold_admission_until_resume is True
        omni.collective_rpc.assert_awaited_once_with(
            method="pause_scheduler",
            args=(),
            kwargs={"mode": "abort", "clear_cache": True},
            stage_ids=[0],
        )

    asyncio.run(run())


@pytest.mark.cpu
def test_pause_generation_still_rpcs_when_already_paused():
    async def run() -> None:
        omni = _make_omni(stage_types=["llm", "diffusion"])
        omni.reset_prefix_cache = AsyncMock(return_value=True)
        omni.reset_mm_cache = AsyncMock()
        omni.reset_encoder_cache = AsyncMock()
        omni._paused = True

        await omni.pause_generation(mode="abort", clear_cache=True, stage_ids=[0])

        omni.collective_rpc.assert_awaited_once_with(
            method="pause_scheduler",
            args=(),
            kwargs={"mode": "abort", "clear_cache": True},
            stage_ids=[0],
        )
        omni.reset_prefix_cache.assert_awaited_once()
        omni.reset_mm_cache.assert_awaited_once()
        omni.reset_encoder_cache.assert_awaited_once()

    asyncio.run(run())


@pytest.mark.cpu
def test_resume_generation_resumes_ar_then_clears_frontend_pause():
    async def run() -> None:
        omni = _make_omni(stage_types=["llm", "diffusion"])
        omni._paused = True

        await omni.resume_generation()

        omni.collective_rpc.assert_awaited_once_with(
            method="resume_scheduler",
            args=(),
            kwargs=None,
            stage_ids=[0],
        )
        assert omni._paused is False
        assert omni._hold_admission_until_resume is False

    asyncio.run(run())


@pytest.mark.cpu
def test_sleep_routes_ar_via_collective_rpc_and_diffusion_to_worker_rpc():
    async def run() -> None:
        omni = _make_omni(stage_types=["llm", "diffusion"])
        diffusion_ack = OmniACK(task_id="diff", status="SUCCESS", stage_id=1, rank=0)
        omni._sleep_diffusion = AsyncMock(return_value=[diffusion_ack])

        acks = await omni.sleep(stage_ids=[0, 1], level=1, mode="abort")

        omni.collective_rpc.assert_awaited_once_with(
            method="sleep",
            args=(1, "abort"),
            kwargs=None,
            stage_ids=[0],
        )
        omni._sleep_diffusion.assert_awaited_once_with([1], 1)
        assert {ack.stage_id for ack in acks} == {0, 1}
        assert any(ack.metadata.get("path") == "engine_core" for ack in acks if ack.stage_id == 0)
        assert CuMemTag.WEIGHTS.value in omni._sleeping_tags
        assert CuMemTag.KV_CACHE.value in omni._sleeping_tags
        # Sleep gates frontend admission for the trainer resume contract.
        assert omni._paused is True
        assert omni._hold_admission_until_resume is True

    asyncio.run(run())


@pytest.mark.cpu
def test_sleep_blocks_admission_before_engine_core_rpc():
    """Problem 3: _paused must be set before awaiting sleep RPC."""

    async def run() -> None:
        omni = _make_omni(stage_types=["llm"])
        paused_at_rpc: list[bool] = []

        async def rpc_side_effect(**kwargs):
            if kwargs.get("method") == "sleep":
                paused_at_rpc.append(omni._paused)
            return [True]

        omni.collective_rpc = AsyncMock(side_effect=rpc_side_effect)

        assert omni._paused is False
        await omni.sleep(level=1, mode="abort")

        assert paused_at_rpc == [True]
        assert omni._paused is True
        assert omni._hold_admission_until_resume is True
        # Sleep must not also call pause_scheduler (EngineCore.sleep pauses).
        assert all(c.kwargs.get("method") != "pause_scheduler" for c in omni.collective_rpc.await_args_list)

    asyncio.run(run())


@pytest.mark.cpu
def test_sleep_waits_for_in_flight_generate_admission():
    """Sleep must not offload while generate() is still in add_request."""

    async def run() -> None:
        omni = _make_omni(stage_types=["llm"])
        omni._admitting = 1
        rpc_started = asyncio.Event()

        async def rpc_side_effect(**kwargs):
            if kwargs.get("method") == "sleep":
                rpc_started.set()
            return [True]

        omni.collective_rpc = AsyncMock(side_effect=rpc_side_effect)
        sleep_task = asyncio.create_task(omni.sleep(level=1, mode="abort"))
        await asyncio.sleep(0.05)
        assert not sleep_task.done()
        assert not rpc_started.is_set()
        await omni._release_generate_admission()
        await asyncio.wait_for(sleep_task, timeout=1)
        assert rpc_started.is_set()
        assert omni._admitting == 0

    asyncio.run(run())


@pytest.mark.cpu
def test_wake_up_routes_ar_via_collective_rpc_and_diffusion_to_worker_rpc():
    async def run() -> None:
        omni = _make_omni(stage_types=["llm", "diffusion"])
        omni._paused = True
        omni._hold_admission_until_resume = True
        omni._sleeping_tags = {CuMemTag.WEIGHTS.value, CuMemTag.KV_CACHE.value}
        diffusion_ack = OmniACK(task_id="diff", status="SUCCESS", stage_id=1, rank=0)
        omni._wake_diffusion = AsyncMock(return_value=[diffusion_ack])

        acks = await omni.wake_up(stage_ids=[0, 1])

        omni.collective_rpc.assert_awaited_once()
        wake_kwargs = omni.collective_rpc.await_args.kwargs
        assert wake_kwargs["method"] == "wake_up"
        assert wake_kwargs["stage_ids"] == [0]
        assert set(wake_kwargs["kwargs"]["tags"]) == {
            CuMemTag.WEIGHTS.value,
            CuMemTag.KV_CACHE.value,
        }
        omni._wake_diffusion.assert_awaited_once()
        assert {ack.stage_id for ack in acks} == {0, 1}
        assert not omni._sleeping_tags
        # Mixed/AR wake restores memory but does not resume frontend admission.
        assert omni._paused is True
        assert omni._hold_admission_until_resume is True

    asyncio.run(run())


@pytest.mark.cpu
def test_wake_up_does_not_resume_frontend_admission():
    async def run() -> None:
        omni = _make_omni(stage_types=["llm"])
        await omni.sleep(level=1, mode="abort")
        assert omni._paused is True
        assert omni._hold_admission_until_resume is True

        await omni.wake_up()

        assert omni._paused is True
        assert omni._hold_admission_until_resume is True
        assert not omni._sleeping_tags
        # Explicit resume is required after AR sleep/wake.
        await omni.resume_generation()
        assert omni._paused is False
        assert omni._hold_admission_until_resume is False

    asyncio.run(run())


@pytest.mark.cpu
def test_sleep_level1_wake_without_tags_clears_all_sleeping_tags():
    """sleep(level=1) tracks WEIGHTS+KV; untagged wake_up must clear both."""

    async def run() -> None:
        omni = _make_omni(stage_types=["llm"])

        await omni.sleep(level=1, mode="abort")
        assert CuMemTag.WEIGHTS.value in omni._sleeping_tags
        assert CuMemTag.KV_CACHE.value in omni._sleeping_tags

        await omni.wake_up()
        assert not omni._sleeping_tags
        assert omni._paused is True
        assert omni._hold_admission_until_resume is True

    asyncio.run(run())


@pytest.mark.cpu
def test_sleep_diffusion_only_skips_engine_core_collective_rpc():
    async def run() -> None:
        omni = _make_omni(stage_types=["diffusion"])
        omni._sleep_diffusion = AsyncMock(return_value=[OmniACK(task_id="d", status="SUCCESS", stage_id=0, rank=0)])

        await omni.sleep(level=1)

        # AR EngineCore sleep path must not run for diffusion-only.
        assert call(method="sleep", args=(1, "abort"), kwargs=None, stage_ids=[0]) not in (
            omni.collective_rpc.await_args_list
        )
        omni._sleep_diffusion.assert_awaited_once_with([0], 1)
        assert omni._paused is True
        assert omni._hold_admission_until_resume is False

    asyncio.run(run())


@pytest.mark.cpu
def test_streaming_wait_for_first_chunk_does_not_block_sleep():
    """Waiting for the next client chunk must not hold an admission slot."""

    async def run() -> None:
        omni = _make_omni(stage_types=["llm"])
        rpc_started = asyncio.Event()
        first_chunk = asyncio.Event()

        async def rpc_side_effect(**kwargs):
            if kwargs.get("method") == "sleep":
                rpc_started.set()
            return [True]

        omni.collective_rpc = AsyncMock(side_effect=rpc_side_effect)

        async def wait_then_submit() -> None:
            await first_chunk.wait()
            await omni._submit_with_admission(asyncio.sleep(0))

        wait_task = asyncio.create_task(wait_then_submit())
        sleep_task = asyncio.create_task(omni.sleep(level=1, mode="abort"))
        await asyncio.sleep(0.05)
        assert not wait_task.done()
        assert omni._admitting == 0
        await asyncio.wait_for(sleep_task, timeout=1)
        assert rpc_started.is_set()

        wait_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await wait_task
        assert omni._admitting == 0

    asyncio.run(run())


@pytest.mark.cpu
def test_streaming_in_flight_add_blocks_sleep():
    """Sleep must wait while a streaming ADD/update holds an admission slot."""

    async def run() -> None:
        omni = _make_omni(stage_types=["llm"])
        add_started = asyncio.Event()
        add_release = asyncio.Event()
        rpc_started = asyncio.Event()

        async def rpc_side_effect(**kwargs):
            if kwargs.get("method") == "sleep":
                rpc_started.set()
            return [True]

        omni.collective_rpc = AsyncMock(side_effect=rpc_side_effect)

        async def slow_add() -> None:
            add_started.set()
            await add_release.wait()

        add_task = asyncio.create_task(omni._submit_with_admission(slow_add()))
        await add_started.wait()
        sleep_task = asyncio.create_task(omni.sleep(level=1, mode="abort"))
        await asyncio.sleep(0.05)
        assert not sleep_task.done()
        assert not rpc_started.is_set()
        assert omni._admitting == 1

        add_release.set()
        await asyncio.wait_for(add_task, timeout=1)
        await asyncio.wait_for(sleep_task, timeout=1)
        assert rpc_started.is_set()
        assert omni._admitting == 0

    asyncio.run(run())


@pytest.mark.cpu
def test_wake_up_restores_admission_for_diffusion_only():
    """Pure diffusion must keep sleep → wake → generate (no resume)."""

    async def run() -> None:
        omni = _make_omni(stage_types=["diffusion"])
        omni._sleep_diffusion = AsyncMock(return_value=[OmniACK(task_id="d", status="SUCCESS", stage_id=0, rank=0)])
        omni._wake_diffusion = AsyncMock(return_value=[OmniACK(task_id="w", status="SUCCESS", stage_id=0, rank=0)])

        await omni.sleep(level=1)
        assert omni._paused is True

        async def wait_for_admission() -> None:
            async with omni._pause_cond:
                await omni._pause_cond.wait_for(lambda: not omni._paused)

        waiter = asyncio.create_task(wait_for_admission())
        await omni.wake_up()
        await asyncio.wait_for(waiter, timeout=1.0)
        assert omni._paused is False
        assert omni._hold_admission_until_resume is False
        assert not omni._sleeping_tags

    asyncio.run(run())


@pytest.mark.cpu
def test_pause_then_sleep_wake_keeps_admission_paused_for_diffusion():
    """Explicit pause_generation still requires resume after diffusion wake."""

    async def run() -> None:
        omni = _make_omni(stage_types=["diffusion"])
        omni.reset_prefix_cache = AsyncMock(return_value=True)
        omni.reset_mm_cache = AsyncMock()
        omni.reset_encoder_cache = AsyncMock()
        omni._sleep_diffusion = AsyncMock(return_value=[OmniACK(task_id="d", status="SUCCESS", stage_id=0, rank=0)])
        omni._wake_diffusion = AsyncMock(return_value=[OmniACK(task_id="w", status="SUCCESS", stage_id=0, rank=0)])

        await omni.pause_generation()
        await omni.sleep(level=1)
        await omni.wake_up()

        assert omni._paused is True
        assert omni._hold_admission_until_resume is True
        await omni.resume_generation()
        assert omni._paused is False

    asyncio.run(run())


@pytest.mark.cpu
def test_partial_wake_does_not_skip_remaining_sleeping_stage():
    """sleep(stage_ids=[0]) then wake([0]) must not skip a later wake([1])."""

    async def run() -> None:
        omni = _make_omni(stage_types=["llm", "llm"])
        await omni.sleep(stage_ids=[0, 1], level=1, mode="abort")
        assert omni._stage_sleeping_tags.keys() == {0, 1}

        await omni.wake_up(stage_ids=[0])
        assert 0 not in omni._stage_sleeping_tags
        assert 1 in omni._stage_sleeping_tags
        assert CuMemTag.WEIGHTS.value in omni._sleeping_tags

        omni.collective_rpc.reset_mock()
        acks = await omni.wake_up(stage_ids=[1])
        assert acks
        omni.collective_rpc.assert_awaited_once()
        wake_kwargs = omni.collective_rpc.await_args.kwargs
        assert wake_kwargs["method"] == "wake_up"
        assert wake_kwargs["stage_ids"] == [1]
        assert not omni._sleeping_tags
        assert not omni._stage_sleeping_tags

    asyncio.run(run())
