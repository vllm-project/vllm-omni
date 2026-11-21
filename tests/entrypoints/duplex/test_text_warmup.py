# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Text startup probes use deployment defaults and finish without audio."""

import json
from dataclasses import dataclass, field

import httpx
import pytest
import websockets

from tests.entrypoints.duplex.test_multi_frontend_warmup import _app, _Handle, _Omni
from vllm_omni.engine.duplex.config import DuplexCapabilities
from vllm_omni.entrypoints.duplex import warmup
from vllm_omni.protocol.duplex.events import ResponseDone

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("response", [None, {}, {"status": "cancelled"}, {"status": "completed", "output": []}])
def test_text_warmup_needs_a_completed_response_with_text(response):
    assert not warmup._completed_text_response(response)


@pytest.mark.asyncio
async def test_local_text_warmup_seeds_primes_and_releases_session(monkeypatch):
    omni = _Omni(2, 2, 0)
    omni.duplex_capabilities = DuplexCapabilities(
        supported_output_modalities=frozenset({"text"}),
        default_output_modalities=("text",),
        supports_session_resume=False,
        text_turn_priming_units=3,
    )

    async def append_text(handle, audio, **kwargs):
        handle.appends.append((audio, kwargs))
        if len(handle.appends) == 3:
            handle.events_queue.put_nowait(
                ResponseDone(
                    response={
                        "status": "completed",
                        "output": [{"content": [{"type": "output_text", "text": "ready"}]}],
                    }
                )
            )

    monkeypatch.setattr(_Handle, "append_audio", append_text)
    await warmup._warmup_local_duplex(omni, omni.engine.plugin, "silent_frames", 1)
    assert len(omni.handles) == 1
    assert omni.handles[0].closed
    assert len(omni.handles[0].appends) == 3
    assert omni.configs[0]["modalities"] == ["text"]
    assert omni.configs[0]["initial_user_text"]
    assert omni.placement.snapshot()["sessions"] == {}


@dataclass
class _Args:
    port: int = 8099
    model: str = "test-model"


@dataclass
class _Socket:
    received: list[dict[str, object]]
    sent: list[dict[str, object]] = field(default_factory=list)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def send(self, value):
        self.sent.append(json.loads(value))

    async def recv(self):
        return json.dumps(self.received.pop(0))


@pytest.mark.asyncio
async def test_websocket_text_warmup_seeds_before_open_and_closes_normally(monkeypatch):
    omni = _Omni(1, 1, 0)
    omni.duplex_capabilities = DuplexCapabilities(
        supported_output_modalities=frozenset({"text"}),
        default_output_modalities=("text",),
        text_turn_priming_units=2,
    )
    app = _app(omni, 1)
    socket = _Socket(
        [
            {"type": "session.created"},
            {"type": "response.output_text.delta", "delta": "ready"},
            {"type": "response.output_text.done", "text": "ready"},
            {
                "type": "response.done",
                "response": {
                    "status": "completed",
                    "output": [{"content": [{"type": "output_text", "text": "ready"}]}],
                },
            },
            {"type": "session.closed"},
        ]
    )
    client_class = httpx.AsyncClient
    monkeypatch.setattr(websockets, "connect", lambda *args, **kwargs: socket)
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda: client_class(transport=httpx.MockTransport(lambda request: httpx.Response(200)))
    )
    await warmup._warmup_duplex_realtime(app, _Args(), 1)
    session = socket.sent[0]["session"]
    assert isinstance(session, dict)
    assert session["modalities"] == ["text"]
    assert session["initial_user_text"]
    assert len([event for event in socket.sent if event["type"] == "input_audio_buffer.append"]) == 2
    assert socket.sent[-1]["type"] == "session.close"
    assert not socket.received
    assert app.state.duplex_warmup_done.is_set()
