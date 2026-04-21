# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for StageEngineCoreClient.check_health().

Uses object.__new__ to construct a minimal client — check_health only
touches self.resources, self.stage_id, and self._proc.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from vllm.v1.engine.exceptions import EngineDeadError

from vllm_omni.engine.stage_engine_core_client import StageEngineCoreClient

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_client(*, engine_dead=False, proc_alive=True):
    client = object.__new__(StageEngineCoreClient)
    client.stage_id = 0
    client.resources = SimpleNamespace(engine_dead=engine_dead)
    client._proc = MagicMock(is_alive=MagicMock(return_value=proc_alive), exitcode=1)
    return client


def test_check_health_passes_when_alive():
    client = _make_client(engine_dead=False, proc_alive=True)
    client.check_health()  # no exception


def test_check_health_raises_when_resources_engine_dead():
    client = _make_client(engine_dead=True, proc_alive=True)
    with pytest.raises(EngineDeadError, match="engine core is dead"):
        client.check_health()


def test_check_health_raises_when_proc_not_alive():
    client = _make_client(engine_dead=False, proc_alive=False)
    with pytest.raises(EngineDeadError, match="not alive"):
        client.check_health()
    # Verify it set resources.engine_dead as a side effect
    assert client.resources.engine_dead is True
