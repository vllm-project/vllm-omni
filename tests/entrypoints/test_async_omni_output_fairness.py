# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
from itertools import count

import pytest

from vllm_omni.engine.messages import ErrorMessage
from vllm_omni.entrypoints import async_omni_base

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["subclass", "ack", "request_error", "ignored"])
async def test_every_message_route_obeys_the_time_slice(mocker, monkeypatch, route):
    processed = []

    async def resolve(msg):
        processed.append(msg)

    def route_message(msg):
        if route != "ack":
            processed.append(msg)
        return route == "subclass"

    message = {"type": "ack", "ack": mocker.Mock(task_id="ack")} if route == "ack" else 0
    if route == "request_error":
        message = ErrorMessage(request_id="gone", error="failed")
    messages = [message] * 32

    async def get_outputs_async(**kwargs):
        if not processed:
            return messages
        await asyncio.Future()

    frontend = async_omni_base.AsyncOmniBase.__new__(async_omni_base.AsyncOmniBase)
    frontend.final_output_task = None
    frontend.engine = mocker.Mock(get_outputs_async=get_outputs_async)
    frontend._route_engine_message = route_message
    frontend._handle_output_message = lambda msg: (True, None, None, None)
    frontend.event_resolver = mocker.Mock(resolve=resolve)
    frontend.request_states = {}
    clock = count(step=0.0006)
    monkeypatch.setattr(async_omni_base, "time", mocker.Mock(monotonic=lambda: next(clock)))
    async_omni_base.AsyncOmniBase._final_output_handler(frontend)
    task = frontend.final_output_task
    try:
        await asyncio.sleep(0)
        assert 0 < len(processed) < len(messages)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
