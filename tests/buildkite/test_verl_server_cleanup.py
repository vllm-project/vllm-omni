# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
from types import SimpleNamespace
from typing import Any, cast

import pytest

from tests.e2e.features.helpers.verl_omni_server import vLLMOmniHttpServerLocal

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_shutdown_closes_engine_before_clearing_actor_reference():
    calls = []
    engine = SimpleNamespace(shutdown=lambda **kwargs: calls.append(kwargs))
    server = vLLMOmniHttpServerLocal.__new__(vLLMOmniHttpServerLocal)
    server.engine = cast(Any, engine)

    asyncio.run(vLLMOmniHttpServerLocal.shutdown(server))

    assert calls == [{"timeout": 30}]
    assert server.engine is None
    asyncio.run(vLLMOmniHttpServerLocal.shutdown(server))
    assert len(calls) == 1
