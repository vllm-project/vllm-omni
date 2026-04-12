"""Guard tests for AsyncOmniEngine.do_log_stats edge cases.

These are pure-Python tests that bypass __init__ and only exercise the
no-op branches of do_log_stats, so no stage cores / threads are needed.
"""

import asyncio

import pytest

from vllm_omni.engine.async_omni_engine import AsyncOmniEngine

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_bare_engine() -> AsyncOmniEngine:
    # Bypass __init__ so we don't spin up stage cores; we only need the
    # attributes do_log_stats touches.
    return AsyncOmniEngine.__new__(AsyncOmniEngine)


@pytest.mark.asyncio
async def test_do_log_stats_noop_when_manager_missing():
    engine = _make_bare_engine()
    engine.logger_manager = None
    engine.orchestrator_loop = None
    await engine.do_log_stats()  # should silently return


@pytest.mark.asyncio
async def test_do_log_stats_noop_when_loop_missing():
    engine = _make_bare_engine()

    class _Manager:
        def log(self) -> None:  # pragma: no cover - must not be called
            raise AssertionError("log() should not be called without a loop")

    engine.logger_manager = _Manager()
    engine.orchestrator_loop = None
    await engine.do_log_stats()


@pytest.mark.asyncio
async def test_do_log_stats_noop_when_loop_not_running():
    engine = _make_bare_engine()

    class _Manager:
        def log(self) -> None:  # pragma: no cover - must not be called
            raise AssertionError("log() should not be called on a stopped loop")

    dead_loop = asyncio.new_event_loop()
    dead_loop.close()

    engine.logger_manager = _Manager()
    engine.orchestrator_loop = dead_loop
    await engine.do_log_stats()
