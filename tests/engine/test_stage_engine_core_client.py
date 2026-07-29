# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for StageEngineCoreClient.check_health()."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from vllm.v1.engine.core_client import AsyncMPClient
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
async def test_reset_prefix_cache_uses_engine_core_utility():
    client = _make_client()
    with patch.object(AsyncMPClient, "reset_prefix_cache_async", new=AsyncMock(return_value=True)) as reset:
        result = await client.reset_prefix_cache_async(
            reset_running_requests=True,
            reset_connector=True,
        )

    assert result is True
    reset.assert_awaited_once_with(
        reset_running_requests=True,
        reset_connector=True,
    )
