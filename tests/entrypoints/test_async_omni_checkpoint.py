# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Newer vLLM checkpoint methods must not make ordinary Omni serving abstract."""

import inspect

import pytest

from vllm_omni.entrypoints.async_omni import AsyncOmni

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_async_omni_is_concrete():
    # This catches new abstract EngineClient methods before any model loads.
    assert not inspect.isabstract(AsyncOmni), AsyncOmni.__abstractmethods__


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["checkpoint_prepare", "checkpoint_restore", "checkpoint_state"])
async def test_checkpoint_operations_fail_before_accessing_engine(method):
    # No initialized engine: rejection must not drain, pause or mutate stages.
    omni = object.__new__(AsyncOmni)
    with pytest.raises(NotImplementedError, match="does not support process checkpointing"):
        await getattr(omni, method)()
