# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio

import pytest

from vllm_omni.entrypoints.async_omni import AsyncOmni

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def make_omni():
    omni = AsyncOmni.__new__(AsyncOmni)
    omni._pause_cond = asyncio.Condition()
    omni._admitting = 0
    omni._paused = False
    omni._sleeping_tags = set()
    omni.request_states = {}
    return omni


@pytest.mark.asyncio
async def test_idle_sleep_snapshot_is_idle_only_when_all_work_is_clear():
    omni = make_omni()

    assert await omni.is_idle_for_sleep()

    for attribute, busy_value in (
        ("_admitting", 1),
        ("_paused", True),
        ("_sleeping_tags", {"weights"}),
        ("request_states", {"request": object()}),
    ):
        original = getattr(omni, attribute)
        setattr(omni, attribute, busy_value)
        assert not await omni.is_idle_for_sleep()
        setattr(omni, attribute, original)
