# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Real ASGI route, protocol, handle, runner and plugin; simulated model outputs.

The in-process control adapter replaces engine IPC, and RecordingStagePort
replaces model execution. These tests do not prove GPU inference or TCP delivery.
"""

import asyncio
import base64
import json
from typing import Any
from uuid import uuid4

import numpy as np
import pytest
from fastapi import FastAPI
from vllm.outputs import CompletionOutput

from tests.engine.duplex.test_session_manager import FakeClock
from tests.engine.duplex.test_session_runner_nemotron_voicechat import open_nemotron_harness
from tests.engine.duplex.test_tool_call_ledger import FakeTimerScheduler
from vllm_omni.config.stage_config import DuplexSessionRuntimeConfig
from vllm_omni.engine.duplex.contracts import DuplexOutputContext, DuplexRequestIdentity, duplex_resource_request_id
from vllm_omni.engine.duplex.messages import (
    CloseDuplexSessionMessage,
    DuplexSessionCommandMessage,
    DuplexSessionError,
    OpenDuplexSessionMessage,
)
from vllm_omni.entrypoints.duplex.serving import OmniDuplexSessionHandler
from vllm_omni.entrypoints.duplex_omni import DuplexOmni, DuplexSessionHandle
from vllm_omni.entrypoints.openai.api_server import realtime_websocket
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class InProcessControl:
    """Only the IPC seam: commands and admission run on the actual manager."""

    def __init__(self, harness):
        self.manager = harness.manager
        self.results = harness.results
        self.duplex_session_config = self.manager.runtime_config
        self.engine = self
        self.handle: DuplexSessionHandle | None = None

    async def open_session(self, payload):
        session_id = f"duplex-{uuid4().hex}"
        handle = DuplexSessionHandle(self, session_id)
        config = DuplexOmni._resolve_session_config(payload, model="nemotron")
        await self.manager.handle(
            OpenDuplexSessionMessage(
                control_id="transport-open",
                session_id=session_id,
                session_config=config,
                output_buffer=handle._outbox,
            )
        )
        result = await self.results.get()
        if not result.ok:
            raise DuplexSessionError(result.error_message, code=result.error_code, session_id=session_id)
        handle._adopt(result)
        self.handle = handle
        return handle

    async def submit_command_async(self, session_id, command):
        self.manager.dispatch(DuplexSessionCommandMessage(session_id=session_id, command=command))

    async def close_session(self, session_id, *, reason="client_close", timeout=None):
        await self.manager.handle(
            CloseDuplexSessionMessage(control_id="transport-close", session_id=session_id, reason=reason)
        )
        result = await self.results.get()
        assert result.ok, result
        assert self.handle is not None
        self.handle._mark_closed(reason)


class ASGIConnection:
    def __init__(self, app):
        self.app = app
        self.inbound: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.outbound: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.events: list[dict[str, Any]] = []
        self.inbound.put_nowait({"type": "websocket.connect"})

    async def run(self):
        await self.app(
            {
                "type": "websocket",
                "asgi": {"version": "3.0", "spec_version": "2.3"},
                "scheme": "ws",
                "path": "/v1/realtime",
                "raw_path": b"/v1/realtime",
                "query_string": b"duplex=1",
                "root_path": "",
                "headers": [],
                "client": ("127.0.0.1", 12345),
                "server": ("127.0.0.1", 8000),
                "subprotocols": [],
            },
            self.inbound.get,
            self.outbound.put,
        )

    def send(self, payload):
        self.inbound.put_nowait({"type": "websocket.receive", "text": json.dumps(payload)})

    async def expect(self, event_type):
        async def receive_expected():
            while True:
                message = await self.outbound.get()
                if message["type"] == "websocket.accept":
                    continue
                assert message["type"] == "websocket.send", message
                event = json.loads(message["text"])
                self.events.append(event)
                assert event["type"] != "error" or event_type == "error", event
                if event["type"] == event_type:
                    return event

        return await asyncio.wait_for(receive_expected(), timeout=3)

    async def append_frame(self, *, samples=1280):
        self.send(
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(np.ones(samples, dtype="<f4").tobytes()).decode(),
                "format": "pcm_f32le",
                "sample_rate_hz": 16000,
            }
        )
        # session.update waits for the preceding append tail, giving the test a
        # protocol barrier without changing clocks, epochs or model state.
        self.send({"type": "session.update", "session": {}})
        await self.expect("session.updated")


def push_model_output(runner, *, token_id=12, metadata=None, stage_id=0, audio=False):
    session = runner.session
    request_id = duplex_resource_request_id(session.fence, "stage0")
    output = OmniRequestOutput(
        request_id=request_id,
        finished=False,
        outputs=[CompletionOutput(index=0, text="", token_ids=[token_id], cumulative_logprob=None, logprobs=None)],
        _multimodal_output=({"model_outputs": [np.ones(16000, dtype=np.float32)], "sr": [16000]} if audio else {}),
    )
    runner.on_stage_output(
        stage_id,
        output,
        None,
        request_id=request_id,
        context=DuplexOutputContext(
            identity=DuplexRequestIdentity(session_id=session.session_id, fence=session.fence),
            final_stage_id=2,
            segment_finished=True,
            segment_token_ids=(token_id,),
            segment_output_metadata=metadata or {},
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("interruption", ["timeout", "cancel"])
async def test_tool_recovery_through_realtime_asgi_route(monkeypatch, interruption):
    clock = FakeClock()
    async with open_nemotron_harness(
        monkeypatch, clock=clock, runtime_config=DuplexSessionRuntimeConfig(max_sessions=2)
    ) as h:
        control = InProcessControl(h)
        app = FastAPI()
        app.add_api_websocket_route("/v1/realtime", realtime_websocket)
        app.state.openai_serving_duplex = OmniDuplexSessionHandler(duplex_omni=control)
        wire = ASGIConnection(app)
        connection = asyncio.create_task(wire.run())
        try:
            wire.send(
                {
                    "type": "session.update",
                    "session": {
                        "model": "nemotron",
                        "modalities": ["text"],
                        "extra_body": {"auto_response": True, "tool_result_timeout_s": 5},
                    },
                }
            )
            await wire.expect("session.created")
            assert control.handle is not None
            runner = control.manager.runners[control.handle.session_id]
            timers = FakeTimerScheduler()
            runner.tool_calls.set_scheduler(timers)
            await wire.append_frame()
            for token in (20, 7, 21):
                push_model_output(runner, metadata={"nvc_function_token": [token]})
            call = await wire.expect("response.function_call_arguments.done")
            call_id = call["call_id"]
            count = len(h.port.submissions)
            await wire.append_frame()
            assert len(h.port.submissions) == count + 1
            if interruption == "timeout":
                clock.advance(5)
                timers.fire_all()
                error = await wire.expect("error")
                assert error["error"]["code"] == "tool_result_timeout"
            else:
                wire.send({"type": "input.cancel"})
                wire.send({"type": "session.update", "session": {}})
                await wire.expect("session.updated")
            before_late = len(h.port.submissions)
            event_start = len(wire.events)
            for replay in range(2):
                event_id = f"late-result-{replay}"
                wire.send(
                    {
                        "type": "conversation.item.create",
                        "event_id": event_id,
                        "item": {"type": "function_call_output", "call_id": call_id, "output": "20"},
                    }
                )
                error = await wire.expect("error")
                assert error["error"]["code"] == "invalid_function_call_output"
                assert error["error"]["event_id"] == event_id
            assert len(h.port.submissions) == before_late
            assert not any(
                event["type"] in {"response.created", "response.audio.delta", "response.output_audio.delta"}
                for event in wire.events[event_start:]
            )
            assert not runner.session.runtime_config.get("nvc_function_response_batches")
            await wire.append_frame()
            push_model_output(runner, token_id=7)
            created = await wire.expect("response.created")
            push_model_output(runner, token_id=1)
            push_model_output(runner, stage_id=2, audio=True)
            done = await wire.expect("response.done")
            assert done["response"]["id"] == created["response"]["id"]
            assert done["response"]["status"] == "completed"
            assert not any(
                event["type"] == "conversation.item.created"
                and event.get("item", {}).get("type") == "function_call_output"
                for event in wire.events
            )
        finally:
            if control.handle is not None:
                await control.close_session(control.handle.session_id)
            wire.inbound.put_nowait({"type": "websocket.disconnect", "code": 1000})
            await asyncio.wait_for(connection, timeout=3)
