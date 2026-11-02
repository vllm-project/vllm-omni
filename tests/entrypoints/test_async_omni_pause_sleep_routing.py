# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Unit tests for AR EngineCore vs diffusion worker pause/sleep routing."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call

import pytest
from vllm.renderers import BaseRenderer
from vllm.v1.engine.input_processor import InputProcessor

from vllm_omni.diffusion.data import CuMemTag, OmniACK
from vllm_omni.engine.messages import ErrorMessage
from vllm_omni.entrypoints.async_omni import CACHE_RESET_TIMEOUT_S, AsyncOmni

pytestmark = [pytest.mark.core_model]


def _make_omni(*, stage_types: list[str]) -> AsyncOmni:
    omni = object.__new__(AsyncOmni)
    omni._pause_cond = asyncio.Condition()
    omni._paused = False
    omni._admitting = 0
    omni._hold_admission_until_resume = False
    omni._paused_stage_ids = set()
    omni._sleeping_tags = set()
    omni._stage_sleeping_tags = {}
    omni._level2_sleeping = False
    omni.event_resolver = SimpleNamespace(watch_task=lambda *a, **k: None, resolve=AsyncMock())
    omni._final_output_handler = lambda: None
    omni._clear_frontend_mm_cache = AsyncMock()

    stage_configs = [SimpleNamespace(stage_type=stage_type) for stage_type in stage_types]
    omni.engine = SimpleNamespace(
        stage_configs=stage_configs,
        stage_vllm_configs=[None] * len(stage_types),
        collective_rpc_async=AsyncMock(return_value=[True]),
    )
    omni.collective_rpc = AsyncMock(return_value=[True])
    return omni


@pytest.mark.cpu
def test_reset_mm_cache_routes_through_renderer(mocker):
    async def run() -> None:
        omni = _make_omni(stage_types=["llm"])
        del omni._clear_frontend_mm_cache
        renderer = mocker.create_autospec(BaseRenderer, instance=True)
        input_processor = mocker.create_autospec(InputProcessor, instance=True)
        input_processor.renderer = renderer
        omni.input_processor = input_processor

        await omni.reset_mm_cache()

        renderer.clear_mm_cache_async.assert_awaited_once_with()

    asyncio.run(run())


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
        omni.reset_prefix_cache.assert_not_awaited()
        omni.reset_mm_cache.assert_not_awaited()
        omni.reset_encoder_cache.assert_not_awaited()
        omni._clear_frontend_mm_cache.assert_awaited_once_with()

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
def test_reset_prefix_cache_forwards_to_ar_stages():
    async def run() -> None:
        omni = _make_omni(stage_types=["llm", "diffusion"])

        result = await omni.reset_prefix_cache(reset_running_requests=True, reset_connector=True)

        assert result is True
        omni.collective_rpc.assert_awaited_once_with(
            method="reset_prefix_cache",
            args=(True, True),
            kwargs=None,
            stage_ids=[0],
            timeout=CACHE_RESET_TIMEOUT_S,
        )

    asyncio.run(run())


@pytest.mark.cpu
def test_reset_prefix_cache_diffusion_only_skips_rpc():
    async def run() -> None:
        omni = _make_omni(stage_types=["diffusion"])

        result = await omni.reset_prefix_cache()

        assert result is True
        omni.collective_rpc.assert_not_awaited()

    asyncio.run(run())


@pytest.mark.cpu
def test_reset_encoder_cache_forwards_to_ar_stages():
    async def run() -> None:
        omni = _make_omni(stage_types=["llm", "diffusion"])

        await omni.reset_encoder_cache()

        omni.collective_rpc.assert_awaited_once_with(
            method="reset_encoder_cache",
            args=(),
            kwargs=None,
            stage_ids=[0],
            timeout=CACHE_RESET_TIMEOUT_S,
        )

    asyncio.run(run())


@pytest.mark.cpu
def test_release_kv_memory_requires_pause_and_wakes_only_ar_kv_allocations():
    async def run() -> None:
        omni = _make_omni(stage_types=["llm", "diffusion", "llm"])
        with pytest.raises(RuntimeError, match="completed pause"):
            await omni.release_kv_cache_memory()
        omni.collective_rpc.assert_not_awaited()

        omni._paused = True
        await omni.release_kv_cache_memory()
        omni._clear_frontend_mm_cache.assert_awaited_once_with()
        omni.collective_rpc.assert_awaited_once_with(
            method="release_kv_cache_memory",
            args=(),
            kwargs=None,
            stage_ids=[0, 2],
            timeout=CACHE_RESET_TIMEOUT_S,
        )
        assert omni._stage_sleeping_tags == {0: {"kv_cache"}, 2: {"kv_cache"}}
        assert not omni._level2_sleeping

        omni.collective_rpc.reset_mock()
        await omni.wake_up(stage_ids=[0, 2], tags=["kv_cache"])
        omni.collective_rpc.assert_awaited_once_with(
            method="wake_up", args=(), kwargs={"tags": ["kv_cache"]}, stage_ids=[0, 2]
        )
        assert not omni._sleeping_tags
        assert omni._paused

    asyncio.run(run())


@pytest.mark.cpu
def test_release_kv_memory_skips_diffusion_and_reports_engine_failures():
    async def run() -> None:
        diffusion = _make_omni(stage_types=["diffusion"])
        await diffusion.release_kv_cache_memory()
        diffusion.collective_rpc.assert_not_awaited()
        assert not diffusion._paused

        omni = _make_omni(stage_types=["llm"])
        omni._paused = True
        omni.collective_rpc.return_value = [{"supported": False, "error": "requires all executor memory resident"}]
        with pytest.raises(RuntimeError, match="all executor memory resident"):
            await omni.release_kv_cache_memory()
        assert not omni._sleeping_tags

    asyncio.run(run())


@pytest.mark.cpu
def test_reset_mm_cache_clears_frontend_then_forwards_to_ar_stages():
    async def run() -> None:
        omni = _make_omni(stage_types=["llm", "diffusion"])
        calls = []
        renderer = SimpleNamespace(clear_mm_cache_async=AsyncMock(side_effect=lambda: calls.append("frontend")))
        omni.input_processor = SimpleNamespace(renderer=renderer)
        del omni._clear_frontend_mm_cache

        async def rpc(**kwargs):
            calls.append("engine")
            return [None]

        omni.collective_rpc.side_effect = rpc
        await omni.reset_mm_cache()

        assert calls == ["frontend", "engine"]
        omni.collective_rpc.assert_awaited_once_with(
            method="reset_mm_cache",
            args=(),
            kwargs=None,
            stage_ids=[0],
            timeout=CACHE_RESET_TIMEOUT_S,
        )

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
def test_sleep_rejects_invalid_stage_ids_without_blocking_admission():
    """Invalid stage_ids must raise before _paused is set.

    Raising afterwards leaves no sleeping tags behind, so wake_up() early-returns
    "already warm" and nothing ever clears _paused: generate() hangs forever.
    """

    async def run() -> None:
        omni = _make_omni(stage_types=["llm", "diffusion"])

        with pytest.raises(ValueError, match=r"Invalid stage_ids \[99\]"):
            await omni.sleep(stage_ids=[99], level=1, mode="abort")

        assert omni._paused is False
        assert omni._hold_admission_until_resume is False
        omni._clear_frontend_mm_cache.assert_not_awaited()
        omni.collective_rpc.assert_not_awaited()

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
    """Waiting for the next client chunk must not hold an admission slot, and
    a chunk submit arriving after sleep() parks at the gate slot-free."""

    async def run() -> None:
        omni = _make_omni(stage_types=["llm"])
        omni._pending_submits = {}
        rpc_started = asyncio.Event()
        first_chunk = asyncio.Event()

        async def rpc_side_effect(**kwargs):
            if kwargs.get("method") == "sleep":
                rpc_started.set()
            return [True]

        omni.collective_rpc = AsyncMock(side_effect=rpc_side_effect)

        async def wait_then_submit() -> None:
            await first_chunk.wait()
            await omni._submit_tracked(
                "stream-1",
                lambda: asyncio.sleep(0),
                hold_admission=True,
                on_revoke=lambda: asyncio.sleep(0),
            )

        wait_task = asyncio.create_task(wait_then_submit())
        sleep_task = asyncio.create_task(omni.sleep(level=1, mode="abort"))
        await asyncio.sleep(0.05)
        assert not wait_task.done()
        assert omni._admitting == 0
        await asyncio.wait_for(sleep_task, timeout=1)
        assert rpc_started.is_set()

        first_chunk.set()
        await asyncio.sleep(0.05)
        # Parked at the (now closed) admission gate: still no slot held.
        assert not wait_task.done()
        assert omni._admitting == 0

        wait_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await wait_task
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


# ───────────────────── diffusion pause_generation(mode="keep") ─────────────────────


def _rpc_methods(omni: AsyncOmni) -> list[tuple[str, list[int] | None]]:
    return [(call.kwargs["method"], call.kwargs.get("stage_ids")) for call in omni.collective_rpc.await_args_list]


@pytest.mark.cpu
def test_diffusion_keep_routes_to_backend():
    async def run() -> None:
        omni = _make_omni(stage_types=["diffusion"])

        await omni.pause_generation(mode="keep", clear_cache=False)

        omni.collective_rpc.assert_awaited_once_with(
            method="pause_scheduler",
            args=(),
            kwargs={"mode": "keep"},
            stage_ids=[0],
        )
        assert await omni.is_paused() is True
        assert omni._hold_admission_until_resume is True

    asyncio.run(run())


@pytest.mark.cpu
@pytest.mark.parametrize(
    "pause_kwargs",
    [
        {},
        {"mode": "abort"},
        {"mode": "wait"},
        {"mode": "keep", "wait_for_inflight_requests": True},
    ],
)
def test_diffusion_non_keep_modes_pause_frontend_only(pause_kwargs):
    async def run() -> None:
        omni = _make_omni(stage_types=["diffusion"])
        omni.reset_prefix_cache = AsyncMock(return_value=True)
        omni.reset_mm_cache = AsyncMock()
        omni.reset_encoder_cache = AsyncMock()

        await omni.pause_generation(**pause_kwargs)
        assert await omni.is_paused() is True
        omni.collective_rpc.assert_not_awaited()

        await omni.resume_generation()
        omni.collective_rpc.assert_not_awaited()
        assert await omni.is_paused() is False

    asyncio.run(run())


@pytest.mark.cpu
def test_mixed_engine_keep_sends_engine_core_and_diffusion_pauses():
    async def run() -> None:
        omni = _make_omni(stage_types=["llm", "diffusion"])
        omni.reset_prefix_cache = AsyncMock(return_value=True)
        omni.reset_mm_cache = AsyncMock()
        omni.reset_encoder_cache = AsyncMock()

        await omni.pause_generation(mode="keep", clear_cache=True)

        assert omni.collective_rpc.await_args_list[0].kwargs == {
            "method": "pause_scheduler",
            "args": (),
            "kwargs": {"mode": "keep", "clear_cache": True},
            "stage_ids": [0],
        }
        assert omni.collective_rpc.await_args_list[1].kwargs == {
            "method": "pause_scheduler",
            "args": (),
            "kwargs": {"mode": "keep"},
            "stage_ids": [1],
        }
        # Only P0 needs clearing here: pause_scheduler already resets P1.
        omni.reset_prefix_cache.assert_not_awaited()
        omni._clear_frontend_mm_cache.assert_awaited_once_with()

        await omni.resume_generation()
        assert _rpc_methods(omni)[2:] == [("resume_scheduler", [0]), ("resume_scheduler", [1])]

    asyncio.run(run())


@pytest.mark.cpu
def test_keep_targets_only_requested_diffusion_stages():
    async def run() -> None:
        omni = _make_omni(stage_types=["diffusion", "diffusion"])

        await omni.pause_generation(mode="keep", clear_cache=False, stage_ids=[1])
        assert _rpc_methods(omni) == [("pause_scheduler", [1])]

        await omni.resume_generation(stage_ids=[0])
        assert _rpc_methods(omni) == [("pause_scheduler", [1])]
        # Stage 1's scheduler is still closed: admission must stay closed too,
        # otherwise new requests queue on a stage that will not schedule them.
        assert await omni.is_paused() is True

        await omni.resume_generation(stage_ids=[1])
        assert _rpc_methods(omni) == [("pause_scheduler", [1]), ("resume_scheduler", [1])]
        assert await omni.is_paused() is False

        await omni.resume_generation()
        assert len(_rpc_methods(omni)) == 2

    asyncio.run(run())


@pytest.mark.cpu
def test_keep_then_sleep_wake_resume_keeps_admission_paused_until_resume():
    async def run() -> None:
        omni = _make_omni(stage_types=["diffusion"])
        omni.reset_prefix_cache = AsyncMock(return_value=True)
        omni.reset_encoder_cache = AsyncMock()
        omni.collective_rpc = AsyncMock(
            side_effect=lambda **kw: (
                [OmniACK(task_id="t", status="SUCCESS", stage_id=0, rank=0)]
                if kw["method"] in {"handle_sleep_task", "handle_wake_task"}
                else [None]
            )
        )

        await omni.pause_generation(mode="keep", clear_cache=False)
        await omni.sleep(level=1)
        await omni.wake_up()
        assert await omni.is_paused() is True

        await omni.resume_generation()
        assert await omni.is_paused() is False
        assert [m for m, _ in _rpc_methods(omni)] == [
            "pause_scheduler",
            "handle_sleep_task",
            "handle_wake_task",
            "resume_scheduler",
        ]

    asyncio.run(run())


@pytest.mark.cpu
@pytest.mark.parametrize(
    "failure",
    [
        AsyncMock(side_effect=NotImplementedError("request-level execution only")),
        AsyncMock(return_value=[{"supported": False, "error": "TimeoutError: outputs did not drain"}]),
    ],
)
def test_keep_failure_surfaces_and_leaves_explicit_resume_possible(failure):
    async def run() -> None:
        omni = _make_omni(stage_types=["diffusion"])
        omni.collective_rpc = failure

        with pytest.raises((NotImplementedError, RuntimeError)):
            await omni.pause_generation(mode="keep", clear_cache=False)

        assert await omni.is_paused() is True
        assert omni._paused_stage_ids == {0}

        omni.collective_rpc = AsyncMock(return_value=[None])
        await omni.resume_generation()
        omni.collective_rpc.assert_awaited_once_with(
            method="resume_scheduler",
            args=(),
            kwargs=None,
            stage_ids=[0],
        )
        assert await omni.is_paused() is False
        assert omni._paused_stage_ids == set()

    asyncio.run(run())


@pytest.mark.cpu
def test_resume_failure_keeps_admission_paused_and_target_recorded():
    async def run() -> None:
        omni = _make_omni(stage_types=["diffusion"])
        await omni.pause_generation(mode="keep", clear_cache=False)

        omni.collective_rpc = AsyncMock(return_value=[{"error": "replica gone"}])
        with pytest.raises(RuntimeError, match="resume_scheduler failed"):
            await omni.resume_generation()

        assert await omni.is_paused() is True
        assert omni._paused_stage_ids == {0}

    asyncio.run(run())


@pytest.mark.cpu
@pytest.mark.parametrize("first, second", [([1], [0]), ([0], [1])])
def test_mixed_engine_partial_resume_keeps_admission_paused(first, second):
    async def run() -> None:
        omni = _make_omni(stage_types=["llm", "diffusion"])
        await omni.pause_generation(mode="keep", clear_cache=False)

        await omni.resume_generation(stage_ids=first)
        assert await omni.is_paused() is True

        await omni.resume_generation(stage_ids=second)
        assert await omni.is_paused() is False
        assert _rpc_methods(omni)[2:] == [("resume_scheduler", first), ("resume_scheduler", second)]

    asyncio.run(run())


@pytest.mark.cpu
def test_two_ar_stages_partial_resume_keeps_admission_paused():
    async def run() -> None:
        omni = _make_omni(stage_types=["llm", "llm"])
        omni.reset_prefix_cache = AsyncMock(return_value=True)
        omni.reset_mm_cache = AsyncMock()
        omni.reset_encoder_cache = AsyncMock()
        await omni.pause_generation(mode="abort")

        await omni.resume_generation(stage_ids=[0])
        assert await omni.is_paused() is True

        await omni.resume_generation(stage_ids=[1])
        assert await omni.is_paused() is False

    asyncio.run(run())


@pytest.mark.cpu
@pytest.mark.parametrize("method", ["reset_prefix_cache", "reset_encoder_cache", "reset_mm_cache"])
@pytest.mark.parametrize("stage_ids, targets", [(None, [0, 2]), ([2], [2]), ([1], []), ([], [])])
def test_cache_resets_respect_stage_selection(method, stage_ids, targets):
    async def run():
        omni = _make_omni(stage_types=["llm", "diffusion", "llm"])
        await getattr(omni, method)(stage_ids=stage_ids, timeout=2.0)
        if targets:
            assert omni.collective_rpc.await_args.kwargs["stage_ids"] == targets
            assert omni.collective_rpc.await_args.kwargs["timeout"] == 2.0
        else:
            omni.collective_rpc.assert_not_awaited()
        assert omni._clear_frontend_mm_cache.await_count == int(method == "reset_mm_cache" and 0 in targets)

    asyncio.run(run())


@pytest.mark.cpu
@pytest.mark.parametrize("method", ["reset_prefix_cache", "reset_encoder_cache", "reset_mm_cache"])
@pytest.mark.parametrize("failure", [{"todo": "unsupported"}, {"supported": False}, {"error": "reset failed"}])
def test_cache_resets_reject_failure_from_any_replica(method, failure):
    async def run():
        omni = _make_omni(stage_types=["llm"])
        # Exercise the actual collective_rpc too: a single stage can have
        # multiple replicas, so result indexes are not stage IDs.
        del omni.collective_rpc
        omni.engine.collective_rpc_async.return_value = [True, failure]
        with pytest.raises(RuntimeError, match=method):
            await getattr(omni, method)()

    asyncio.run(run())


@pytest.mark.cpu
def test_prefix_reset_returns_false_if_any_replica_refuses():
    async def run():
        omni = _make_omni(stage_types=["llm"])
        omni.collective_rpc.return_value = [True, [True, False]]
        assert await omni.reset_prefix_cache() is False

    asyncio.run(run())


@pytest.mark.cpu
@pytest.mark.parametrize("operation", ["pause_generation", "sleep"])
@pytest.mark.parametrize("stage_ids", [[0], [2]])
def test_pause_and_sleep_do_not_reset_unselected_stages_or_reset_twice(operation, stage_ids):
    async def run():
        omni = _make_omni(stage_types=["llm", "diffusion", "llm"])
        kwargs = {"clear_cache": True} if operation == "pause_generation" else {}
        await getattr(omni, operation)(stage_ids=stage_ids, **kwargs)
        assert omni.collective_rpc.await_count == 1
        rpc = omni.collective_rpc.await_args.kwargs
        assert rpc["method"] == ("pause_scheduler" if operation == "pause_generation" else "sleep")
        assert rpc["stage_ids"] == stage_ids
        assert omni._clear_frontend_mm_cache.await_count == int(0 in stage_ids)

    asyncio.run(run())


_SLEEP_TAGS = {CuMemTag.WEIGHTS.value, CuMemTag.KV_CACHE.value}

# In-process stages return OmniACK, subprocess stages return its dict form,
# and StagePool returns an error dict when the RPC itself failed.
_FAILED_DIFFUSION_RESULTS = [
    pytest.param(OmniACK(task_id="t", status="ERROR", error_msg="out of memory"), id="ack"),
    pytest.param({"task_id": "t", "status": "ERROR", "error_msg": "out of memory"}, id="ack-dict"),
    pytest.param({"supported": False, "error": "out of memory"}, id="rpc-error"),
]


@pytest.mark.cpu
@pytest.mark.parametrize("result", _FAILED_DIFFUSION_RESULTS)
def test_sleep_raises_when_diffusion_stage_fails(result):
    async def run() -> None:
        omni = _make_omni(stage_types=["diffusion"])
        omni.collective_rpc = AsyncMock(return_value=[result])

        with pytest.raises(RuntimeError, match="handle_sleep_task failed: out of memory"):
            await omni.sleep(level=1)

        # The stage may be partly asleep, so it stays on record for wake_up.
        assert omni._sleeping_tags == _SLEEP_TAGS

    asyncio.run(run())


@pytest.mark.cpu
@pytest.mark.parametrize("result", _FAILED_DIFFUSION_RESULTS)
def test_wake_up_raises_when_diffusion_stage_fails(result):
    async def run() -> None:
        omni = _make_omni(stage_types=["diffusion"])
        await omni.sleep(level=1)
        omni.collective_rpc = AsyncMock(return_value=[result])

        with pytest.raises(RuntimeError, match="handle_wake_task failed: out of memory"):
            await omni.wake_up()

        assert omni._sleeping_tags == _SLEEP_TAGS
        assert omni._paused is True

    asyncio.run(run())


@pytest.mark.cpu
def test_sleep_records_all_stages_when_diffusion_stage_fails():
    async def run() -> None:
        omni = _make_omni(stage_types=["llm", "diffusion"])
        omni._sleep_diffusion = AsyncMock(side_effect=RuntimeError("handle_sleep_task failed"))

        with pytest.raises(RuntimeError):
            await omni.sleep(level=1)

        assert omni._stage_sleeping_tags == {0: _SLEEP_TAGS, 1: _SLEEP_TAGS}

    asyncio.run(run())


@pytest.mark.cpu
def test_wake_up_clears_ar_stage_when_diffusion_stage_fails():
    async def run() -> None:
        omni = _make_omni(stage_types=["llm", "diffusion"])
        await omni.sleep(level=1)
        omni._wake_diffusion = AsyncMock(side_effect=RuntimeError("handle_wake_task failed"))

        with pytest.raises(RuntimeError):
            await omni.wake_up()

        assert omni._stage_sleeping_tags == {1: _SLEEP_TAGS}

    asyncio.run(run())


@pytest.mark.cpu
def test_sleep_level2_is_recorded_when_diffusion_stage_fails():
    async def run() -> None:
        omni = _make_omni(stage_types=["llm", "diffusion"])
        omni._sleep_diffusion = AsyncMock(side_effect=RuntimeError("handle_sleep_task failed"))

        with pytest.raises(RuntimeError):
            await omni.sleep(level=2)

        with pytest.raises(NotImplementedError):
            await omni.wake_up(stage_ids=[0])

    asyncio.run(run())


@pytest.mark.cpu
def test_wake_up_reaches_diffusion_stage_after_failed_sleep():
    async def run() -> None:
        omni = _make_omni(stage_types=["diffusion"])
        omni.collective_rpc = AsyncMock(
            return_value=[
                [
                    OmniACK(task_id="t", status="SUCCESS", stage_id=0, rank=0),
                    OmniACK(task_id="t", status="ERROR", error_msg="out of memory"),
                ]
            ]
        )
        with pytest.raises(RuntimeError, match="out of memory"):
            await omni.sleep(level=1)

        omni.collective_rpc = AsyncMock(return_value=[OmniACK(task_id="t", status="SUCCESS", stage_id=0, rank=0)])
        await omni.wake_up()

        omni.collective_rpc.assert_awaited_once()
        assert not omni._sleeping_tags
        assert omni._paused is False

    asyncio.run(run())


@pytest.mark.cpu
def test_wake_up_clears_diffusion_stages_that_woke_before_a_failure():
    async def run() -> None:
        omni = _make_omni(stage_types=["diffusion", "diffusion"])
        await omni.sleep(level=1)
        omni.collective_rpc = AsyncMock(
            side_effect=[
                [OmniACK(task_id="t", status="SUCCESS", stage_id=0, rank=0)],
                [OmniACK(task_id="t", status="ERROR", error_msg="out of memory")],
            ]
        )

        with pytest.raises(RuntimeError, match="out of memory"):
            await omni.wake_up()

        assert omni._stage_sleeping_tags == {1: _SLEEP_TAGS}

    asyncio.run(run())


@pytest.mark.cpu
def test_wake_up_settles_once_for_several_diffusion_stages(mocker):
    async def run() -> None:
        omni = _make_omni(stage_types=["diffusion", "diffusion"])
        await omni.sleep(level=1)
        settle = mocker.patch("vllm_omni.entrypoints.async_omni.asyncio.sleep", new=AsyncMock())

        await omni.wake_up()

        assert omni.collective_rpc.await_count == 3
        settle.assert_awaited_once_with(0.1)

    asyncio.run(run())


@pytest.mark.cpu
@pytest.mark.parametrize("result", [{"supported": False}, {"todo": "not supported yet"}], ids=["unsupported", "todo"])
def test_sleep_raises_on_rpc_result_without_error(result):
    async def run() -> None:
        omni = _make_omni(stage_types=["diffusion"])
        omni.collective_rpc = AsyncMock(return_value=[result])

        with pytest.raises(RuntimeError, match="handle_sleep_task failed"):
            await omni.sleep(level=1)

    asyncio.run(run())


@pytest.mark.cpu
def test_rpc_failure_result_is_not_resolved_as_ack():
    async def run() -> None:
        omni = _make_omni(stage_types=["diffusion"])
        omni.event_resolver.resolve = AsyncMock()
        omni.collective_rpc = AsyncMock(return_value=[{"supported": False, "error": "out of memory"}])

        with pytest.raises(RuntimeError, match="out of memory"):
            await omni.sleep(level=1)

        omni.event_resolver.resolve.assert_not_awaited()

    asyncio.run(run())


@pytest.mark.cpu
def test_sleep_raises_with_reason_of_rpc_error_envelope():
    async def run() -> None:
        omni = _make_omni(stage_types=["diffusion"])
        omni.collective_rpc = AsyncMock(return_value=[{"error": True, "reason": "worker crashed"}])

        with pytest.raises(RuntimeError, match="handle_sleep_task failed: worker crashed"):
            await omni.sleep(level=1)

    asyncio.run(run())


@pytest.mark.cpu
def test_failed_level2_sleep_can_still_be_woken():
    async def run() -> None:
        omni = _make_omni(stage_types=["diffusion"])
        omni.collective_rpc = AsyncMock(return_value=[OmniACK(task_id="t", status="ERROR", error_msg="out of memory")])
        with pytest.raises(RuntimeError, match="out of memory"):
            await omni.sleep(level=2)

        omni.collective_rpc = AsyncMock(return_value=[OmniACK(task_id="t", status="SUCCESS", stage_id=0, rank=0)])
        await omni.wake_up()

        omni.collective_rpc.assert_awaited_once()
        assert omni._paused is False

    asyncio.run(run())


@pytest.mark.cpu
def test_partial_level2_sleep_still_blocks_wake_up():
    async def run() -> None:
        omni = _make_omni(stage_types=["diffusion"])
        omni.collective_rpc = AsyncMock(
            return_value=[
                [
                    OmniACK(task_id="t", status="SUCCESS", stage_id=0, rank=0),
                    OmniACK(task_id="t", status="ERROR", error_msg="out of memory"),
                ]
            ]
        )
        with pytest.raises(RuntimeError, match="out of memory"):
            await omni.sleep(level=2)

        with pytest.raises(NotImplementedError):
            await omni.wake_up()

    asyncio.run(run())


@pytest.mark.cpu
def test_is_sleeping_for_given_stages():
    async def run() -> None:
        omni = _make_omni(stage_types=["llm", "diffusion"])
        await omni.sleep(stage_ids=[1], level=1)

        assert await omni.is_sleeping(stage_ids=[1])
        assert not await omni.is_sleeping(stage_ids=[0])

    asyncio.run(run())


@pytest.mark.cpu
def test_pause_abort_cancels_pending_submits():
    """pause_generation(mode="abort") cancels generate() submits stuck on a
    full request queue, so their admission slots release (#8678)."""

    async def run() -> None:
        omni = _make_omni(stage_types=["llm"])
        omni._pending_submits = {}

        stuck = asyncio.create_task(asyncio.Event().wait())
        omni._pending_submits["req-1"] = stuck

        await omni.pause_generation(mode="abort", clear_cache=False)

        with pytest.raises(asyncio.CancelledError):
            await stuck

    asyncio.run(run())


@pytest.mark.cpu
def test_sleep_releases_pending_submit_admission():
    """sleep() must not wait forever behind a submit suspended on a full
    queue: the abort-mode cancel releases the slot and sleep proceeds."""

    async def run() -> None:
        omni = _make_omni(stage_types=["llm"])
        omni._pending_submits = {}
        omni._admitting = 1

        async def suspended_submit() -> None:
            await asyncio.Event().wait()

        task = asyncio.create_task(suspended_submit())

        async def release_on_cancel() -> None:
            try:
                await task
            except asyncio.CancelledError:
                await omni._release_generate_admission()
                raise

        waiter = asyncio.create_task(release_on_cancel())
        omni._pending_submits["req-1"] = task

        await asyncio.wait_for(omni.sleep(level=1, mode="abort"), timeout=2)
        assert omni._admitting == 0
        with pytest.raises(asyncio.CancelledError):
            await waiter

    asyncio.run(run())


@pytest.mark.cpu
def test_add_request_async_suspends_on_full_queue_without_wedging(mocker):
    """A full request queue suspends the submit coroutine (cancellable), not
    the event loop thread (#8678)."""

    import janus

    from vllm_omni.engine.async_omni_engine import AsyncOmniEngine

    async def run() -> None:
        engine = object.__new__(AsyncOmniEngine)
        engine.request_queue = janus.Queue(maxsize=1)
        engine.request_queue.sync_q.put_nowait(object())
        engine.prompt_expand_func = None
        mocker.patch.object(
            engine,
            "_build_add_request_message",
            return_value=SimpleNamespace(request_artifact_dirs=None, original_prompt=None),
        )

        submit = asyncio.create_task(engine.add_request_async(request_id="req-1", prompt={"prompt": "hi"}))
        # If this loop turn runs at all, the event loop is not wedged.
        await asyncio.sleep(0.05)
        assert not submit.done()

        submit.cancel()
        with pytest.raises(asyncio.CancelledError):
            await submit
        # A cancelled put enqueues nothing.
        assert engine.request_queue.sync_q.qsize() == 1

    asyncio.run(run())


@pytest.mark.cpu
def test_add_request_async_survives_saturated_queue_on_orchestrator_loop(mocker):
    """Deterministic #8678 regression at the production loop boundary.

    The janus queue is bound to a dedicated orchestrator loop — what the real
    orchestrator thread does by awaiting get() on it — and janus >= 2 raises
    RuntimeError when any other loop touches async_q, even with capacity free.
    With the queue saturated and never drained, a frontend submit must
    suspend (frontend loop stays responsive), a cancelled submit must admit
    nothing, and an unsaturated submit must complete. The deadline is
    enforced by a watchdog outside the frontend loop: a blocking put would
    freeze every in-loop timer, so no asyncio.wait_for guard can fire.
    """

    import faulthandler
    import threading

    import janus

    from vllm_omni.engine.async_omni_engine import AsyncOmniEngine

    # Built with no running loop, like OmniEngineBase does on the main thread.
    queue = janus.Queue(maxsize=2)
    orch_loop = asyncio.new_event_loop()
    orch_thread = threading.Thread(target=orch_loop.run_forever, daemon=True, name="orchestrator")
    orch_thread.start()

    faulthandler.dump_traceback_later(30, exit=True)
    try:

        async def run() -> None:
            engine = object.__new__(AsyncOmniEngine)
            engine.request_queue = queue
            engine.orchestrator_loop = orch_loop
            engine.prompt_expand_func = None
            mocker.patch.object(
                engine,
                "_build_add_request_message",
                return_value=SimpleNamespace(request_artifact_dirs=None, original_prompt=None),
            )

            # Bind async_q to the orchestrator loop the way the production
            # request handler's await get() does.
            await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(queue.async_q.join(), orch_loop))
            # Saturate the queue without draining it.
            queue.sync_q.put_nowait(object())
            queue.sync_q.put_nowait(object())

            submit = asyncio.create_task(engine.add_request_async(request_id="req-sat", prompt={"prompt": "hi"}))
            for _ in range(10):
                # The frontend loop must keep servicing timers while the
                # submit is parked on the full queue.
                await asyncio.sleep(0.01)
            assert not submit.done()

            submit.cancel()
            with pytest.raises(asyncio.CancelledError):
                await submit
            assert queue.sync_q.qsize() == 2, "cancelled submit must not admit its item"

            # Drain from the orchestrator loop: only the seeded items come
            # out, proving the cancelled submission never crossed.
            async def get_two():
                return [await queue.async_q.get(), await queue.async_q.get()]

            drained = await asyncio.wait_for(
                asyncio.wrap_future(asyncio.run_coroutine_threadsafe(get_two(), orch_loop)), timeout=5
            )
            assert len(drained) == 2

            # With room available a cross-loop submit completes instead of
            # raising janus's "bound to a different event loop" RuntimeError.
            await asyncio.wait_for(
                asyncio.create_task(engine.add_request_async(request_id="req-ok", prompt={"prompt": "hi"})),
                timeout=5,
            )
            assert queue.sync_q.qsize() == 1

        asyncio.run(run())
    finally:
        faulthandler.cancel_dump_traceback_later()
        queue.close()
        orch_loop.call_soon_threadsafe(orch_loop.stop)
        orch_thread.join(timeout=5)
        orch_loop.close()


@pytest.mark.cpu
def test_abort_revokes_queue_parked_submit():
    """abort(request_id) must cancel and settle a submit parked on a full
    request queue, so the request cannot be admitted after abort returns
    (#8678)."""

    async def run() -> None:
        omni = _make_omni(stage_types=["llm"])
        omni._pending_submits = {}
        omni.request_states = {
            "ext-1#uuid": SimpleNamespace(request_id="ext-1#uuid", external_request_id="ext-1"),
        }
        parked = asyncio.create_task(asyncio.Event().wait())
        omni._pending_submits["ext-1#uuid"] = parked
        parked_cancelled_at_delegation: list[bool] = []

        async def engine_abort(internal_ids, timeout=None):
            parked_cancelled_at_delegation.append(parked.cancelled())
            return []

        omni._abort = engine_abort

        await omni.abort("ext-1")

        assert parked.cancelled(), "parked submit survived abort()"
        assert parked_cancelled_at_delegation == [True], (
            "abort() delegated to the Orchestrator before revoking the parked submit"
        )
        assert omni._pending_submits == {}

    asyncio.run(run())


@pytest.mark.cpu
def test_revoked_submit_finishes_generate_as_abort():
    """pause/sleep/abort revoking a parked submit must end generate() with a
    finished abort output — not a CancelledError the caller never asked for
    (the abort-retry RL client gathers over rollout batches, and a cancelled
    member kills the whole gather)."""

    async def run() -> None:
        omni = _make_omni(stage_types=["llm"])
        omni._pending_submits = {}
        omni._paused = False
        omni._hold_admission_until_resume = False
        omni.request_states = {}
        omni.engine.num_stages = 1
        omni.log_stats = False
        omni._record_request_failure_once = lambda *args, **kwargs: None
        omni._maybe_expand_sampling_params = lambda spl: spl
        omni.resolve_sampling_params_list = lambda spl, **_: spl
        omni._compute_final_stage_id = lambda output_modalities: 0
        omni._compute_final_output_stage_ids = lambda output_modalities: [0]
        omni._get_pd_separation_pair = lambda: None
        omni._publish_request_gauges = lambda total: None
        omni._log_summary_and_cleanup = lambda request_id, reason="stage_error": None

        submit_parked = asyncio.Event()

        async def parked_submit(*args, **kwargs):
            await submit_parked.wait()

        omni.engine.add_request_async = parked_submit

        async def drain_until_finished(request_id, *args, **kwargs):
            state = omni.request_states[request_id]
            while True:
                message = await state.queue.get()
                assert message.finished
                yield message
                return  # a finished message ends the real results loop

        omni._process_orchestrator_results = drain_until_finished

        outputs: list = []

        async def consume() -> None:
            async for output in omni.generate("hi", sampling_params_list=[SimpleNamespace(max_tokens=1)]):
                outputs.append(output)

        task = asyncio.create_task(consume())
        await asyncio.sleep(0.05)
        assert not task.done()
        assert len(omni._pending_submits) == 1

        # pause_generation(mode="abort")/sleep()/abort() revoke the parked submit.
        for pending in list(omni._pending_submits.values()):
            pending.cancel()
        await asyncio.wait_for(task, timeout=2)

        assert outputs, "revoked submit produced no terminal output"
        assert outputs[-1].finished
        assert outputs[-1].engine_outputs.outputs[0].finish_reason == "abort"
        assert not task.cancelled(), "generate() ended as cancelled, not aborted"

    asyncio.run(run())


@pytest.mark.cpu
def test_sleep_aborts_tracked_streaming_submit():
    """A streaming ADD tracked via _submit_tracked holds an admission slot;
    sleep(mode="abort") must cancel it and pass the ``_admitting == 0``
    barrier without leaking a CancelledError into the pump (streaming
    submits were invisible to the cancel set before this fix)."""

    async def run() -> None:
        omni = _make_omni(stage_types=["llm"])
        omni._pending_submits = {}
        omni._admitting = 0
        revoked = asyncio.Event()

        async def parked_add() -> None:
            await asyncio.Event().wait()

        async def on_revoke() -> None:
            revoked.set()
            await asyncio.sleep(0)

        async def pump() -> None:
            await omni._submit_tracked("stream-1", parked_add, hold_admission=True, on_revoke=on_revoke)

        pump_task = asyncio.create_task(pump())
        await asyncio.sleep(0.05)
        assert omni._admitting == 1, "streaming ADD did not take its admission slot"

        await asyncio.wait_for(omni.sleep(level=1, mode="abort"), timeout=2)
        await asyncio.wait_for(pump_task, timeout=1)  # completes instead of raising

        assert revoked.is_set(), "streaming submit was cancelled without the revoke conversion"
        assert omni._admitting == 0

    asyncio.run(run())


@pytest.mark.cpu
def test_pump_stops_after_revoked_first_chunk(mocker):
    """After a revoke closes the request, the streaming pump must stop: no
    further chunk submits and no final empty update for an already-aborted
    request (the synthetic abort output already ended it)."""

    async def run() -> None:
        omni = _make_omni(stage_types=["llm"])
        omni._pending_submits = {}
        omni._paused = False
        omni._admitting = 0
        mocker.patch.object(AsyncOmni, "model_config", new=SimpleNamespace())
        omni._record_request_failure_once = lambda *args, **kwargs: None
        omni._validate_streaming_input_sampling_params = lambda params: None
        mocker.patch(
            "vllm_omni.entrypoints.async_omni.extract_prompt_components",
            return_value=("text", None, None),
        )

        state = SimpleNamespace(queue=asyncio.Queue())
        omni.request_states = {"req-1": state}

        submits: list[str] = []
        first_submit_parked = asyncio.Event()
        gate = asyncio.Event()

        async def parked_add(**kwargs) -> None:
            submits.append("add")
            first_submit_parked.set()
            await gate.wait()

        async def chunk_update(**kwargs) -> None:
            submits.append("update")

        omni.engine.add_request_async = parked_add
        omni.engine.add_streaming_update_async = chunk_update

        async def input_stream():
            yield SimpleNamespace(sampling_params=None, prompt="chunk-1")
            yield SimpleNamespace(sampling_params=None, prompt="chunk-2")

        stage0_params = SimpleNamespace(skip_clone=True)
        first_chunk_submitted = asyncio.get_running_loop().create_future()

        asyncio.create_task(
            omni._add_streaming_input_request(
                request_id="req-1",
                input_stream=input_stream(),
                sampling_params_list=[stage0_params],
                final_stage_id=0,
                final_output_stage_ids=[0],
                arrival_time=0.0,
                first_chunk_submitted=first_chunk_submitted,
            )
        )
        await first_submit_parked.wait()
        assert len(omni._pending_submits) == 1

        # pause_generation(mode="abort")/sleep() revokes the parked first ADD.
        for pending in list(omni._pending_submits.values()):
            pending.cancel()
        # The wrapper returns as soon as the pump task is created; the pump
        # itself is registered on the request state.
        await asyncio.wait_for(state.input_stream_task, timeout=2)

        assert submits == ["add"], f"pump submitted after the revoke: {submits}"
        assert first_chunk_submitted.done(), "generate() would stay parked on the first chunk"
        assert not state.queue.empty(), "revoked request got no terminal abort output"

    asyncio.run(run())


@pytest.mark.cpu
def test_revoked_final_update_is_not_an_input_error(mocker):
    """A revoke of the final empty update must not be reported as an input error.

    That submit runs inside handle_inputs's finally, after the request was
    already closed with a synthetic abort. _SubmitRevokedError subclasses
    Exception, so the inner handler has to ignore it explicitly.
    """

    async def run() -> None:
        omni = _make_omni(stage_types=["llm"])
        omni._pending_submits = {}
        omni._paused = False
        omni._admitting = 0
        mocker.patch.object(AsyncOmni, "model_config", new=SimpleNamespace())
        omni._record_request_failure_once = lambda *args, **kwargs: None
        omni._validate_streaming_input_sampling_params = lambda params: None
        mocker.patch(
            "vllm_omni.entrypoints.async_omni.extract_prompt_components",
            return_value=("text", None, None),
        )

        state = SimpleNamespace(queue=asyncio.Queue())
        omni.request_states = {"req-1": state}
        final_parked = asyncio.Event()
        gate = asyncio.Event()

        async def parked_add(**kwargs) -> None:
            final_parked.set()
            await gate.wait()

        omni.engine.add_request_async = parked_add
        omni.engine.add_streaming_update_async = AsyncMock()

        async def input_stream():
            return
            yield  # pragma: no cover

        stage0_params = SimpleNamespace(skip_clone=True)
        first_chunk_submitted = asyncio.get_running_loop().create_future()
        asyncio.create_task(
            omni._add_streaming_input_request(
                request_id="req-1",
                input_stream=input_stream(),
                sampling_params_list=[stage0_params],
                final_stage_id=0,
                final_output_stage_ids=[0],
                arrival_time=0.0,
                first_chunk_submitted=first_chunk_submitted,
            )
        )
        await asyncio.wait_for(final_parked.wait(), timeout=2)
        for pending in list(omni._pending_submits.values()):
            pending.cancel()
        await asyncio.wait_for(state.input_stream_task, timeout=2)

        queued = []
        while not state.queue.empty():
            queued.append(state.queue.get_nowait())
        assert queued, "revoked final update produced no terminal output"
        assert not any(isinstance(message, ErrorMessage) for message in queued)

    asyncio.run(run())


@pytest.mark.cpu
def test_tracked_non_streaming_submit_runs_under_callers_slot():
    """Non-streaming generate() holds the outer admission slot across
    add_request: _submit_tracked must not stack a second one, and must not
    re-wait on _paused — sleep()'s ``_admitting == 0`` barrier already waits
    behind the caller's slot, so a second wait would deadlock
    ``sleep(mode="wait")`` against it."""

    async def run() -> None:
        omni = _make_omni(stage_types=["llm"])
        omni._pending_submits = {}
        omni._admitting = 1  # generate()'s outer slot
        omni._paused = True
        body_started = asyncio.Event()

        async def add() -> None:
            body_started.set()

        async def generate_like() -> None:
            await omni._submit_tracked("req-1", add, hold_admission=False)

        task = asyncio.create_task(generate_like())
        await asyncio.wait_for(body_started.wait(), timeout=1)
        assert omni._admitting == 1, "tracked non-streaming submit stacked a second admission slot"
        await asyncio.wait_for(task, timeout=1)
        assert omni._admitting == 1  # the slot still held is the caller's

    asyncio.run(run())


@pytest.mark.cpu
def test_sleep_wait_survives_non_streaming_submit_scheduling_window():
    """sleep(mode="wait") landing after a non-streaming submit is scheduled
    but before it runs must not deadlock: the child runs under the caller's
    outer slot and sleep's ``_admitting == 0`` barrier passes once the ADD
    settles — wait mode has no cancel to break the cycle."""

    async def run() -> None:
        omni = _make_omni(stage_types=["llm"])
        omni._pending_submits = {}
        omni._admitting = 1  # generate() passed the outer gate; slot held
        submit_started = asyncio.Event()
        release_submit = asyncio.Event()

        async def add() -> None:
            submit_started.set()
            await release_submit.wait()

        async def generate_like() -> None:
            await omni._submit_tracked("req-1", add, hold_admission=False)
            await omni._release_generate_admission()

        generate_task = asyncio.create_task(generate_like())
        await asyncio.sleep(0)  # submit registered, its first step not run
        assert list(omni._pending_submits) == ["req-1"]

        # Close the gate inside the scheduling window, like a control call
        # slipping in between create_task and the submit's first step.
        omni._paused = True
        sleep_task = asyncio.create_task(omni.sleep(level=1, mode="wait"))
        await asyncio.sleep(0.05)
        assert omni._paused is True
        assert submit_started.is_set(), "submit re-waited on _paused behind sleep's own barrier"
        release_submit.set()

        await asyncio.wait_for(generate_task, timeout=2)
        await asyncio.wait_for(sleep_task, timeout=2)  # hangs if the submit double-waits
        assert omni._admitting == 0

    asyncio.run(run())


@pytest.mark.cpu
def test_sleep_releases_gate_parked_streaming_submit():
    """A streaming submit parked at the per-submit gate when sleep() runs must
    be revoked and unwound, so sleep's ``_admitting == 0`` barrier cannot hang
    behind a paused engine (non-streaming submits no longer re-gate: their
    caller holds the outer slot)."""

    async def run() -> None:
        omni = _make_omni(stage_types=["llm"])
        omni._pending_submits = {}
        omni._paused = True  # the gate closed before this submit registered
        revoked = asyncio.Event()

        async def add() -> None:
            raise AssertionError("streaming submit body must not run while paused")

        async def on_revoke() -> None:
            revoked.set()
            await asyncio.sleep(0)

        async def pump() -> None:
            await omni._submit_tracked("stream-1", add, hold_admission=True, on_revoke=on_revoke)

        task = asyncio.create_task(pump())
        await asyncio.sleep(0.05)
        assert not task.done()
        assert omni._admitting == 0  # parked before the gate: no slot held

        await asyncio.wait_for(omni.sleep(level=1, mode="abort"), timeout=2)
        # The parked submit holds no slot, so sleep's barrier passes at once
        # and the cancelled child unwinds after sleep releases the condition.
        await asyncio.wait_for(task, timeout=1)
        assert revoked.is_set()
        assert not task.cancelled(), "pump ended as cancelled instead of revoked"

    asyncio.run(run())


@pytest.mark.cpu
def test_cancelled_companion_put_revokes_admitted_parent(mocker):
    """Cancelling between parent and companion puts must revoke the parent:
    it would otherwise sit admitted with scheduler and KV capacity for a CFG
    group that can never complete."""

    import faulthandler
    import threading

    import janus

    from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
    from vllm_omni.engine.messages import AbortRequestMessage

    queue = janus.Queue(maxsize=2)
    orch_loop = asyncio.new_event_loop()
    orch_thread = threading.Thread(target=orch_loop.run_forever, daemon=True, name="orchestrator")
    orch_thread.start()

    faulthandler.dump_traceback_later(30, exit=True)
    try:

        async def run() -> None:
            engine = object.__new__(AsyncOmniEngine)
            engine.request_queue = queue
            engine.orchestrator_loop = orch_loop
            clear_mm_cache = AsyncMock()
            engine.input_processor = SimpleNamespace(renderer=SimpleNamespace(clear_mm_cache_async=clear_mm_cache))

            await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(queue.async_q.join(), orch_loop))
            queue.sync_q.put_nowait(object())  # leave room for exactly the parent

            parent = SimpleNamespace(request_id="parent-1", request_artifact_dirs=None)
            companion = object()
            outcome: dict[str, bool] = {"parent_enqueued": False}
            group = asyncio.create_task(engine._put_request_queue_group(parent, [companion], outcome))
            await asyncio.sleep(0.1)
            assert not group.done(), "companion put should be parked on the full queue"
            assert queue.sync_q.qsize() == 2, "parent should be enqueued ahead of the companion"

            group.cancel()
            with pytest.raises(asyncio.CancelledError):
                await group
            # The parent landed: its media is in the sender cache legitimately,
            # so the frontend cache reset must be skipped (only the never-
            # admitted case clears it).
            assert outcome["parent_enqueued"] is True
            clear_mm_cache.assert_not_awaited()

            # Free the room the detached revoke is waiting for, then give the
            # owning loop time to land the AbortRequestMessage behind the ADD.
            queue.sync_q.get_nowait()
            queue.sync_q.get_nowait()
            for _ in range(100):
                if queue.sync_q.qsize():
                    break
                await asyncio.sleep(0.02)
            assert queue.sync_q.qsize() == 1, "orphan-parent revoke never landed"
            revoked = queue.sync_q.get_nowait()
            assert isinstance(revoked, AbortRequestMessage)
            assert revoked.request_ids == ["parent-1"]

        asyncio.run(run())
    finally:
        faulthandler.cancel_dump_traceback_later()
        queue.close()
        orch_loop.call_soon_threadsafe(orch_loop.stop)
        orch_thread.join(timeout=5)
        orch_loop.close()


@pytest.mark.cpu
def test_cancelled_submit_with_landed_parent_keeps_mm_cache(mocker):
    """Cancellation must settle from the owning-loop task's unwind, not from
    cf.cancel(): the proxy future completes at cancel time while the group's
    finally has not run, so a stale parent_enqueued=False resets the global
    sender cache for an already-enqueued parent and returns before the queue
    operation has settled."""

    import faulthandler
    import threading

    import janus

    from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
    from vllm_omni.engine.messages import AbortRequestMessage

    queue = janus.Queue(maxsize=2)
    orch_loop = asyncio.new_event_loop()
    orch_thread = threading.Thread(target=orch_loop.run_forever, daemon=True, name="orchestrator")
    orch_thread.start()
    freeze_entered = threading.Event()
    freeze_release = threading.Event()

    faulthandler.dump_traceback_later(30, exit=True)
    try:

        async def run() -> None:
            engine = object.__new__(AsyncOmniEngine)
            engine.request_queue = queue
            engine.orchestrator_loop = orch_loop
            engine.prompt_expand_func = object()
            clear_mm_cache = AsyncMock()
            engine.input_processor = SimpleNamespace(renderer=SimpleNamespace(clear_mm_cache_async=clear_mm_cache))
            parent = SimpleNamespace(
                request_id="parent-1",
                request_artifact_dirs=None,
                original_prompt=None,
                sampling_params_list=[object()],
            )
            mocker.patch.object(engine, "_build_add_request_message", return_value=parent)
            mocker.patch.object(engine, "_build_cfg_companions", return_value=[object()])

            await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(queue.async_q.join(), orch_loop))
            queue.sync_q.put_nowait(object())  # leave room for exactly the parent

            submit = asyncio.create_task(
                engine.add_request_async(request_id="parent-1", prompt={"prompt": "hi"}, final_stage_id=1)
            )
            await asyncio.sleep(0.1)
            assert not submit.done()
            assert queue.sync_q.qsize() == 2, "parent should be enqueued ahead of its parked companion"

            # Freeze the orchestrator loop before cancelling: its unwind can
            # no longer race the frontend's read of `outcome`.
            orch_loop.call_soon_threadsafe(freeze_entered.set)
            while not freeze_entered.is_set():
                await asyncio.sleep(0.01)
            orch_loop.call_soon_threadsafe(lambda: freeze_release.wait())

            submit.cancel()
            freeze_release.set()
            with pytest.raises(asyncio.CancelledError):
                await submit
            # The parent landed, so its media legitimately lives in the sender
            # cache: the global reset must be skipped even though the
            # companion never made it.
            clear_mm_cache.assert_not_awaited()

            # The half-enqueued parent is revoked behind the ADD.
            queue.sync_q.get_nowait()
            queue.sync_q.get_nowait()
            for _ in range(100):
                if queue.sync_q.qsize():
                    break
                await asyncio.sleep(0.02)
            revoked = queue.sync_q.get_nowait()
            assert isinstance(revoked, AbortRequestMessage)
            assert revoked.request_ids == ["parent-1"]

        asyncio.run(run())
    finally:
        faulthandler.cancel_dump_traceback_later()
        freeze_release.set()
        queue.close()
        orch_loop.call_soon_threadsafe(orch_loop.stop)
        orch_thread.join(timeout=5)
        orch_loop.close()


@pytest.mark.cpu
def test_revoked_submit_resets_frontend_mm_cache(mocker):
    """A submit revoked on a full queue must clear the frontend sender cache:
    process_inputs already recorded the request's media, EngineCore never got
    the request, and a later same-media request would fail with "Multi-modal
    processor cache miss" (HTTP 503)."""

    import janus

    from vllm_omni.engine.async_omni_engine import AsyncOmniEngine

    async def run() -> None:
        engine = object.__new__(AsyncOmniEngine)
        engine.request_queue = janus.Queue(maxsize=1)
        engine.request_queue.sync_q.put_nowait(object())
        engine.prompt_expand_func = None
        engine.orchestrator_loop = None  # exercise the same-loop path
        clear_mm_cache = AsyncMock()
        engine.input_processor = SimpleNamespace(renderer=SimpleNamespace(clear_mm_cache_async=clear_mm_cache))
        mocker.patch.object(
            engine,
            "_build_add_request_message",
            return_value=SimpleNamespace(request_artifact_dirs=None, original_prompt=None),
        )

        submit = asyncio.create_task(engine.add_request_async(request_id="req-mm", prompt={"prompt": "hi"}))
        await asyncio.sleep(0.05)
        assert not submit.done()

        submit.cancel()
        with pytest.raises(asyncio.CancelledError):
            await submit
        clear_mm_cache.assert_awaited_once()
        assert engine.request_queue.sync_q.qsize() == 1

    asyncio.run(run())


@pytest.mark.cpu
def test_revoked_submit_keeps_shm_sender_cache(mocker):
    """The shm sender cache hands EngineCore shared-memory addresses, so a
    revoked submit's entries stay readable; a full clear there would reset
    the ring buffer under requests that are still in flight."""

    import janus
    from vllm.multimodal.cache import ShmObjectStoreSenderCache

    from vllm_omni.engine.async_omni_engine import AsyncOmniEngine

    async def run() -> None:
        engine = object.__new__(AsyncOmniEngine)
        engine.request_queue = janus.Queue(maxsize=1)
        engine.request_queue.sync_q.put_nowait(object())
        engine.prompt_expand_func = None
        engine.orchestrator_loop = None
        clear_mm_cache = AsyncMock()
        engine.input_processor = SimpleNamespace(
            renderer=SimpleNamespace(
                mm_processor_cache=MagicMock(spec=ShmObjectStoreSenderCache),
                clear_mm_cache_async=clear_mm_cache,
            )
        )
        mocker.patch.object(
            engine,
            "_build_add_request_message",
            return_value=SimpleNamespace(request_artifact_dirs=None, original_prompt=None),
        )

        submit = asyncio.create_task(engine.add_request_async(request_id="req-mm", prompt={"prompt": "hi"}))
        await asyncio.sleep(0.05)
        assert not submit.done()

        submit.cancel()
        with pytest.raises(asyncio.CancelledError):
            await submit
        clear_mm_cache.assert_not_awaited()

    asyncio.run(run())


@pytest.mark.cpu
def test_prepare_add_request_drops_client_artifact_dirs_before_any_raise(mocker):
    """The server-reserved artifact-dirs key must be popped before anything
    can raise: the sampling-params ValueError fires before the transform's
    own sanitization, and the failure handler rmtrees whatever the key
    holds (CodeQL py/path-injection)."""

    from vllm_omni.data_entry_keys import REQUEST_ARTIFACT_DIRS_KEY
    from vllm_omni.engine.async_omni_engine import AsyncOmniEngine

    engine = object.__new__(AsyncOmniEngine)
    engine.prompt_expand_func = None
    prompt = {"prompt": "hi", REQUEST_ARTIFACT_DIRS_KEY: ["/etc", "/important-data"]}
    mocker.patch.object(engine, "_build_add_request_message", side_effect=ValueError("nope"))

    with pytest.raises(ValueError, match="nope"):
        engine._prepare_add_request(request_id="req-1", prompt=prompt, sampling_params_list=[object()])

    assert REQUEST_ARTIFACT_DIRS_KEY not in prompt, "client-supplied artifact dirs survived a prepare failure"


@pytest.mark.cpu
def test_build_failure_after_transform_cleans_extracted_dirs(mocker):
    """A failure after the transform attached artifact dirs must clean them at
    the source — without any cleanup path re-reading the client prompt
    (CodeQL py/path-injection)."""

    from vllm_omni.data_entry_keys import REQUEST_ARTIFACT_DIRS_KEY
    from vllm_omni.engine.async_omni_engine import AsyncOmniEngine

    engine = object.__new__(AsyncOmniEngine)
    engine.default_sampling_params_list = [SimpleNamespace(max_tokens=1)]
    engine.stage_metadata = [SimpleNamespace(stage_type="llm")]
    engine.supported_tasks = ("generate",)
    engine.prompt_transform_func = lambda prompt, params: {
        **prompt,
        REQUEST_ARTIFACT_DIRS_KEY: ["/tmp/omni-test-artifact"],
    }
    engine.input_processor = SimpleNamespace(process_inputs=lambda **kwargs: object())
    mocker.patch.object(engine, "_scope_stage0_multimodal_cache_to_replica", return_value=(None, {"prompt": "hi"}))
    rmtree = mocker.patch("vllm_omni.engine.async_omni_engine.shutil.rmtree")
    mocker.patch("vllm_omni.engine.async_omni_engine.upgrade_to_omni_request", side_effect=RuntimeError("boom"))

    prompt = {"prompt": "hi", REQUEST_ARTIFACT_DIRS_KEY: ["/client/controlled/path"]}
    with pytest.raises(RuntimeError, match="boom"):
        engine._prepare_add_request(
            request_id="req-1", prompt=prompt, sampling_params_list=[SimpleNamespace(max_tokens=1)]
        )

    # The transform's own dirs were cleaned at the source, and only they were:
    rmtree.assert_called_once_with("/tmp/omni-test-artifact", ignore_errors=True)
    assert rmtree.call_count == 1
    # The key now holds the transform's list - the client-supplied value is gone.
    assert prompt[REQUEST_ARTIFACT_DIRS_KEY] == ["/tmp/omni-test-artifact"]


@pytest.mark.cpu
def test_generate_waits_for_full_wake_instead_of_erroring():
    """After a partial wake (kv_cache still asleep), generate() parks at the
    admission gate instead of raising "partially or fully asleep" and forcing
    the abort-retry client into error-retry churn (#8678)."""

    async def run() -> None:
        omni = _make_omni(stage_types=["llm"])
        omni._pending_submits = {}
        # wake_up(tags=["weights"]) + resume_generation() already ran:
        # admission is open but the engine is still partially asleep.
        omni._paused = False
        omni._hold_admission_until_resume = False
        omni._stage_sleeping_tags = {0: {"weights", "kv_cache"}}
        omni._sleeping_tags = {"weights", "kv_cache"}
        add_request = omni.engine.add_request_async = AsyncMock()

        omni.request_states = {}
        omni.engine.num_stages = 1
        omni.log_stats = False
        omni._maybe_expand_sampling_params = lambda spl: spl
        omni.resolve_sampling_params_list = lambda spl, **_: spl
        omni._compute_final_stage_id = lambda output_modalities: 0
        omni._compute_final_output_stage_ids = lambda output_modalities: [0]
        omni._get_pd_separation_pair = lambda: None
        omni._publish_request_gauges = lambda total: None
        omni._log_summary_and_cleanup = lambda request_id, reason="stage_error": None

        async def no_outputs(*args, **kwargs):
            return
            yield  # pragma: no cover

        omni._process_orchestrator_results = no_outputs

        async def consume() -> None:
            async for _ in omni.generate("hi", sampling_params_list=[SimpleNamespace(max_tokens=1)]):
                pass

        task = asyncio.create_task(consume())
        await asyncio.sleep(0.05)
        assert not task.done()
        assert omni._admitting == 0
        add_request.assert_not_awaited()

        # Partial wake leaves kv_cache asleep: still parked, not erroring.
        await omni.wake_up(tags=["weights"])
        await asyncio.sleep(0.05)
        assert not task.done()
        add_request.assert_not_awaited()

        # Full wake clears the last tag and releases the gate.
        await omni.wake_up(tags=["kv_cache"])
        await asyncio.wait_for(task, timeout=2)
        add_request.assert_awaited_once()
        assert omni._admitting == 0

    asyncio.run(run())


@pytest.mark.cpu
def test_tracked_streaming_submit_waits_for_full_wake():
    """The streaming pump's per-submit admission gate waits out a partial
    wake inside the tracked task instead of submitting into a
    partially-asleep engine (#8678) — the gated body must not start early."""

    async def run() -> None:
        omni = _make_omni(stage_types=["llm"])
        omni._pending_submits = {}
        omni._sleeping_tags = {"kv_cache"}

        started = asyncio.Event()

        async def payload() -> None:
            started.set()

        async def pump() -> None:
            await omni._submit_tracked("stream-1", payload, hold_admission=True)

        task = asyncio.create_task(pump())
        await asyncio.sleep(0.05)
        # Parked before the gate: payload not started, no admission slot held.
        assert not started.is_set()
        assert omni._admitting == 0

        omni._sleeping_tags.clear()
        async with omni._pause_cond:
            omni._pause_cond.notify_all()

        await asyncio.wait_for(task, timeout=2)
        assert started.is_set()
        assert omni._admitting == 0

    asyncio.run(run())
