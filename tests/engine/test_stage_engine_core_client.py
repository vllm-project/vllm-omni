# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for StageEngineCoreClient.check_health()."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from vllm.v1.engine.exceptions import EngineDeadError

from vllm_omni.engine.stage_engine_core_client import StageEngineCoreClient

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_client(*, engine_dead=False):
    client = object.__new__(StageEngineCoreClient)
    client.stage_id = 0
    client.resources = SimpleNamespace(engine_dead=engine_dead)
    return client


def test_check_health_passes_when_alive():
    client = _make_client(engine_dead=False)
    client.check_health()  # no exception


def test_check_health_raises_when_resources_engine_dead():
    client = _make_client(engine_dead=True)
    with pytest.raises(EngineDeadError, match="engine core is dead"):
        client.check_health()


@pytest.mark.asyncio
async def test_collective_rpc_routes_cache_control_to_engine_core(mocker):
    client = _make_client()
    client.reset_prefix_cache_async = mocker.AsyncMock(return_value=True)
    client.reset_mm_cache_async = mocker.AsyncMock()
    client.reset_encoder_cache_async = mocker.AsyncMock()

    result = await client.collective_rpc_async(
        "reset_prefix_cache",
        kwargs={"reset_running_requests": True, "reset_connector": True},
    )
    assert result is True
    client.reset_prefix_cache_async.assert_awaited_once_with(True, True)

    assert await client.collective_rpc_async("reset_mm_cache") is True
    client.reset_mm_cache_async.assert_awaited_once()
    assert await client.collective_rpc_async("reset_encoder_cache") is True
    client.reset_encoder_cache_async.assert_awaited_once()
