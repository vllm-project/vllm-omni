"""Unit tests for StagePool.collective_rpc EngineCore control dispatch."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from vllm_omni.engine.stage_pool import StagePool

pytestmark = [pytest.mark.core_model]


def _make_pool(*, stage_type: str = "llm", **client_methods: AsyncMock) -> tuple[StagePool, SimpleNamespace]:
    client = SimpleNamespace(stage_type=stage_type, **client_methods)
    if "collective_rpc_async" not in client_methods:
        client.collective_rpc_async = AsyncMock(return_value={"via": "collective"})
    pool = StagePool(0, [client])  # type: ignore[arg-type]
    return pool, client


@pytest.mark.cpu
def test_collective_rpc_normalizes_none_args_on_control_helper():
    async def run() -> None:
        pause = AsyncMock(return_value="paused")
        pool, client = _make_pool(pause_scheduler_async=pause)

        result = await pool.collective_rpc(0, "pause_scheduler", args=None, kwargs={"mode": "abort"})

        assert result == "paused"
        pause.assert_awaited_once_with(mode="abort")
        client.collective_rpc_async.assert_not_awaited()

    asyncio.run(run())


@pytest.mark.cpu
def test_collective_rpc_unrelated_async_helper_uses_collective_path():
    async def run() -> None:
        other = AsyncMock(return_value="should-not-run")
        pool, client = _make_pool(reset_prefix_cache_async=other)

        result = await pool.collective_rpc(0, "reset_prefix_cache", timeout=1.5, args=("x",), kwargs={"k": 1})

        assert result == {"via": "collective"}
        other.assert_not_awaited()
        client.collective_rpc_async.assert_awaited_once_with(
            method="reset_prefix_cache",
            timeout=1.5,
            args=("x",),
            kwargs={"k": 1},
        )

    asyncio.run(run())


@pytest.mark.cpu
def test_collective_rpc_control_helper_honors_timeout():
    async def run() -> None:
        async def slow_sleep(*_args, **_kwargs):
            await asyncio.sleep(1.0)
            return "slept"

        pool, _client = _make_pool(sleep_async=slow_sleep)
        result = await pool.collective_rpc(0, "sleep", timeout=0.01, args=(1,))
        assert result != "slept"
        assert result["supported"] is False

    asyncio.run(run())
