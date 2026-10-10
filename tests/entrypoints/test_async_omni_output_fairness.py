# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio

import pytest

from vllm_omni.engine.messages import ErrorMessage
from vllm_omni.entrypoints import async_omni_base

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["subclass", "ack", "request_error", "ignored"])
async def test_every_message_route_obeys_the_time_slice(mocker, monkeypatch, route):
    processed = []
    clock = [0.0]

    async def resolve(msg):
        processed.append(msg)
        clock[0] += 0.002

    def route_message(msg):
        if route != "ack":
            processed.append(msg)
            clock[0] += 0.002
        return route == "subclass"

    message_routes: dict[str, list[object]] = {
        "subclass": list(range(32)),
        "ack": [{"type": "ack", "ack": mocker.Mock(task_id="ack")} for _ in range(32)],
        "request_error": [ErrorMessage(request_id="gone", error="failed") for _ in range(32)],
        "ignored": list(range(32)),
    }
    messages = message_routes[route]

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
    monkeypatch.setattr(async_omni_base, "time", mocker.Mock(monotonic=lambda: clock[0]))
    async_omni_base.AsyncOmniBase._final_output_handler(frontend)
    task = getattr(frontend, "final_output_task")
    assert isinstance(task, asyncio.Task)
    try:
        await asyncio.sleep(0)
        assert 0 < len(processed) < len(messages)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
