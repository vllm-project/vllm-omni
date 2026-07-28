from __future__ import annotations

import asyncio
import base64
import json
import struct
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from starlette.websockets import WebSocketDisconnect

from vllm_omni.config.stage_config import DuplexSessionRuntimeConfig
from vllm_omni.experimental.fullduplex.engine.duplex_control_client import DuplexControlRequestError
from vllm_omni.experimental.fullduplex.engine.duplex_runtime import duplex_resource_request_id
from vllm_omni.experimental.fullduplex.engine.lease import DuplexLeaseActivity
from vllm_omni.experimental.fullduplex.engine.messages import DuplexFence, DuplexSessionLifecycleMessage
from vllm_omni.experimental.fullduplex.minicpmo45 import (
    MiniCPMO45NativeDuplexServingAdapter,
    MiniCPMO45PcmAppendBuffer,
)
from vllm_omni.experimental.fullduplex.minicpmo45.data_plane import (
    MiniCPMO45DataPlaneContext,
    MiniCPMO45DataPlaneSession,
)
from vllm_omni.experimental.fullduplex.minicpmo45.runtime import (
    MiniCPMO45DuplexRuntimeExtension,
)
from vllm_omni.experimental.fullduplex.minicpmo45.session import (
    MiniCPMO45ServingSessionState,
)
from vllm_omni.experimental.fullduplex.openai.protocol import (
    DuplexCapabilities,
    DuplexOverlapPolicy,
    DuplexPlaybackCommitPolicy,
    DuplexSession,
    DuplexSessionConfig,
    ResponseCreateOptions,
)
from vllm_omni.experimental.fullduplex.openai.realtime_session import NativeRealtimeSessionProtocol
from vllm_omni.experimental.fullduplex.openai.runtime_adapter import ServingRuntimeConfigError
from vllm_omni.experimental.fullduplex.openai.serving import (
    OmniDuplexSessionHandler,
    should_enable_duplex_endpoint,
)
from vllm_omni.experimental.fullduplex.openai.websocket import DuplexWebSocketActor
from vllm_omni.experimental.fullduplex.output import attach_duplex_output_decision
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _ModelConfig:
    model = "test-model"


class FakeEngineClient:
    output_modalities = ["text", "audio"]

    def __init__(
        self,
        *,
        fail_open: bool = False,
        fail_signal: bool = False,
        fail_touch: bool = False,
        fail_signal_events: set[str] | None = None,
        fail_close: bool = False,
        fail_abort: bool = False,
        control_result: dict[str, object] | None = None,
        open_result: dict[str, object] | None = None,
        append_result: dict[str, object] | None = None,
        append_results: list[dict[str, object]] | None = None,
        collect_outputs: list[list[object]] | None = None,
        collect_delay_s: float = 0.0,
        signal_result: dict[str, object] | None = None,
        close_result: dict[str, object] | None = None,
    ) -> None:
        self.fail_open = fail_open
        self.fail_signal = fail_signal
        self.fail_touch = fail_touch
        self.fail_signal_events = set(fail_signal_events or ())
        self.fail_close = fail_close
        self.fail_abort = fail_abort
        self.control_result = control_result
        self.open_result = open_result
        self.append_result = append_result
        self.append_results = list(append_results or [])
        self.collect_outputs = list(collect_outputs or [])
        self.collect_delay_s = collect_delay_s
        self.signal_result = signal_result
        self.close_result = close_result
        self.opened: list[str] = []
        self.opened_fences: list[DuplexFence | None] = []
        self.appended: list[tuple[str, str, object, bool]] = []
        self.append_operation_ids: list[str | None] = []
        self.appended_fences: list[DuplexFence | None] = []
        self.opened_configs: list[dict[str, object]] = []
        self.opened_runtime_configs: list[dict[str, object]] = []
        self.signals: list[tuple[str, str]] = []
        self.signal_fences: list[DuplexFence | None] = []
        self.signal_next_fences: list[DuplexFence | None] = []
        self.signal_session_configs: list[dict[str, object] | None] = []
        self.signal_runtime_configs: list[dict[str, object] | None] = []
        self.closed: list[tuple[str, str]] = []
        self.touched: list[tuple[str, DuplexLeaseActivity]] = []
        self.touch_fences: list[DuplexFence] = []
        self.resumed: list[tuple[str, int]] = []
        self.resume_fences: list[DuplexFence] = []
        self.aborted: list[str] = []
        self.abort_batches: list[list[str]] = []
        self.internal_abort_batches: list[list[str]] = []
        self.collected: list[tuple[str, int | None]] = []
        self.duplex_lifecycle_events: asyncio.Queue[DuplexSessionLifecycleMessage] = asyncio.Queue()

    async def open_duplex_session_async(
        self,
        session_id: str,
        *,
        session_mode: str = "duplex",
        capabilities: dict[str, object] | None = None,
        session_config: dict[str, object] | None = None,
        runtime_config: dict[str, object] | None = None,
        timeout: float | None = None,
        fence: DuplexFence | None = None,
    ) -> None:
        del timeout
        if self.fail_open:
            raise RuntimeError("open failed")
        self.opened.append(session_id)
        self.opened_fences.append(fence)
        self.opened_configs.append(dict(session_config or {}))
        self.opened_runtime_configs.append(dict(runtime_config or {}))
        return self.open_result if self.open_result is not None else self.control_result

    async def append_duplex_input_async(
        self,
        session_id: str,
        *,
        mode: str,
        payload: object,
        operation_id: str | None = None,
        final: bool = False,
        timeout: float | None = None,
        collect_outputs: bool = True,
        fence: DuplexFence | None = None,
    ) -> None:
        del timeout, collect_outputs
        self.appended.append((session_id, mode, payload, final))
        self.append_operation_ids.append(operation_id)
        self.appended_fences.append(fence)
        if self.append_results:
            return self.append_results.pop(0)
        return self.append_result if self.append_result is not None else self.control_result

    async def collect_duplex_data_plane_outputs_async(
        self,
        request_id: str,
        *,
        response_stage_id: int | None = None,
        timeout: float | None = None,
    ) -> list[object]:
        del timeout
        self.collected.append((request_id, response_stage_id))
        if self.collect_delay_s > 0:
            await asyncio.sleep(self.collect_delay_s)
        if not self.collect_outputs:
            return []
        return self.collect_outputs.pop(0)

    async def signal_duplex_turn_async(
        self,
        session_id: str,
        *,
        event: str,
        timeout: float | None = None,
        fence: DuplexFence | None = None,
        next_fence: DuplexFence | None = None,
        session_config: dict[str, object] | None = None,
        runtime_config: dict[str, object] | None = None,
    ) -> None:
        del timeout
        if self.fail_signal or event in self.fail_signal_events:
            raise RuntimeError("signal failed")
        self.signals.append((session_id, event))
        self.signal_fences.append(fence)
        self.signal_next_fences.append(next_fence)
        self.signal_session_configs.append(session_config)
        self.signal_runtime_configs.append(runtime_config)
        return self.signal_result if self.signal_result is not None else self.control_result

    async def close_duplex_session_async(
        self,
        session_id: str,
        *,
        reason: str = "client_close",
        timeout: float | None = None,
        fence: DuplexFence | None = None,
    ) -> None:
        del timeout
        if self.fail_close:
            raise RuntimeError("close failed")
        self.closed.append((session_id, reason))
        return self.close_result if self.close_result is not None else self.control_result

    async def touch_duplex_session_async(
        self,
        session_id: str,
        *,
        fence: DuplexFence,
        activity: DuplexLeaseActivity,
        timeout: float | None = None,
    ) -> dict[str, object]:
        del timeout
        if self.fail_touch:
            raise RuntimeError("touch failed")
        self.touched.append((session_id, activity))
        self.touch_fences.append(fence)
        return {
            "operation": "touch",
            "session_id": session_id,
            "ok": True,
            "stage_results": [
                {
                    "stage_id": -1,
                    "replica_id": -1,
                    "result": {"supported": True, "lease_generation": 0},
                }
            ],
        }

    async def resume_duplex_session_async(
        self,
        session_id: str,
        *,
        fence: DuplexFence,
        expected_lease_generation: int,
        timeout: float | None = None,
    ) -> dict[str, object]:
        del timeout
        self.resumed.append((session_id, expected_lease_generation))
        self.resume_fences.append(fence)
        return {
            "operation": "resume",
            "session_id": session_id,
            "ok": True,
            "stage_results": [
                {
                    "stage_id": -1,
                    "replica_id": -1,
                    "result": {
                        "supported": True,
                        "lease_generation": expected_lease_generation + 1,
                    },
                }
            ],
        }

    async def abort(self, request_ids: list[str]) -> None:
        if self.fail_abort:
            raise RuntimeError("abort failed")
        assert isinstance(request_ids, list)
        self.abort_batches.append(list(request_ids))
        self.aborted.extend(request_ids)

    async def _abort_internal_requests(self, request_ids: list[str]) -> None:
        if self.fail_abort:
            raise RuntimeError("abort failed")
        assert isinstance(request_ids, list)
        self.internal_abort_batches.append(list(request_ids))
        self.aborted.extend(request_ids)


class FakeChatService:
    duplex_serving_adapter_path = (
        "vllm_omni.experimental.fullduplex.minicpmo45.serving_adapter.MiniCPMO45ServingRuntimeAdapter"
    )

    def __init__(self, engine_client: FakeEngineClient) -> None:
        self.engine_client = engine_client
        self.model_config = _ModelConfig()
        self.seen_request_ids: list[str] = []

    async def create_chat_completion(self, request, raw_request=None):
        self.seen_request_ids.append(request.request_id)

        async def _gen():
            await asyncio.sleep(999)
            yield "data: [DONE]\n\n"

        return _gen()

    def create_audio(self, audio_obj):
        return SimpleNamespace(audio_data=f"wav-{int(audio_obj.audio_tensor.shape[0])}")


def _test_data_plane() -> MiniCPMO45DataPlaneSession:
    def encode_audio(
        audio: object,
        sample_rate_hz: int,
        response_format: str,
        speed: float | None,
    ) -> str | None:
        del sample_rate_hz, response_format, speed
        if audio is None:
            return None
        if hasattr(audio, "numel"):
            return f"wav-{int(audio.numel())}"
        return f"wav-{int(np.asarray(audio).size)}"

    return MiniCPMO45DataPlaneSession(encode_audio)


def _data_plane_context(session: DuplexSession | None = None) -> MiniCPMO45DataPlaneContext:
    if session is None:
        return MiniCPMO45DataPlaneContext()
    return MiniCPMO45DataPlaneContext(
        epoch=session.epoch,
        turn_id=session.turn_id,
        active_response_turn_id=session.active_response_turn_id,
        active_response_id=session.active_response_id,
        auto_responds=bool(session.config.extra_body.get("auto_response", False)),
        response_format=session.config.response_format,
        speed=session.config.speed,
        modalities=tuple(session.config.modalities),
    )


def _project_data_plane(
    data_plane: MiniCPMO45DataPlaneSession,
    result: object,
    *,
    session: DuplexSession | None = None,
) -> list[dict[str, object]]:
    return list(data_plane.project(result, context=_data_plane_context(session)))


class TimedWebSocket:
    def __init__(self, *, on_send=None, receive_timeout_s: float = 1.0):
        self._q: asyncio.Queue[str | BaseException] = asyncio.Queue()
        self.sent: list[dict[str, Any]] = []
        self.accepted = False
        self._on_send = on_send
        self._receive_timeout_s = receive_timeout_s
        self.query_params: dict[str, str] = {}

    async def accept(self):
        self.accepted = True

    async def receive_text(self) -> str:
        try:
            item = await asyncio.wait_for(self._q.get(), timeout=self._receive_timeout_s)
        except asyncio.TimeoutError as exc:
            raise WebSocketDisconnect(code=1000) from exc
        if isinstance(item, BaseException):
            raise item
        return item

    async def send_json(self, data: dict[str, Any]):
        self.sent.append(data)
        if self._on_send is not None:
            self._on_send(self, data)

    def put(self, payload: dict[str, Any]) -> None:
        self._q.put_nowait(json.dumps(payload))

    async def close(self, code: int = 1000, reason: str | None = None) -> None:
        del reason
        self._q.put_nowait(WebSocketDisconnect(code=code))

    def sent_types(self) -> list[str]:
        return [m.get("type", "") for m in self.sent]


def test_native_realtime_protocol_emits_speak_once_per_response():
    protocol = NativeRealtimeSessionProtocol({})
    response_id = "resp-speak-once"

    projected = protocol.encode_outbound_event(
        {
            "type": "response.created",
            "response_id": response_id,
            "modalities": ["audio", "text"],
        }
    )
    projected.extend(
        protocol.encode_outbound_event(
            {
                "type": "response.speak",
                "response_id": response_id,
                "text": "hello",
                "model_speak": True,
            }
        )
    )
    projected.extend(
        protocol.encode_outbound_event(
            {
                "type": "response.output_audio.delta",
                "response_id": response_id,
                "audio": base64.b64encode(b"\x00\x00").decode("ascii"),
                "format": "pcm16",
                "sample_rate_hz": 24000,
                "model_speak": True,
            }
        )
    )

    assert [event["type"] for event in projected].count("response.speak") == 1
    speak = next(event for event in projected if event["type"] == "response.speak")
    assert "text" not in speak
    assert speak["metadata"] == {"model_speak": True}


def test_native_realtime_protocol_speak_does_not_expose_internal_event():
    protocol = NativeRealtimeSessionProtocol({})

    projected = protocol.encode_outbound_event(
        {
            "type": "response.speak",
            "response_id": "resp-curated-speak",
            "session_id": "sid-curated-speak",
            "epoch": 3,
            "text": "transcript belongs on the delta channel",
            "model_speak": True,
            "data_plane_request_id": "internal-request-id",
            "uses_model_runner_scheduler": True,
        }
    )

    assert len(projected) == 1
    projected[0].pop("event_id")
    assert projected == [
        {
            "type": "response.speak",
            "response_id": "resp-curated-speak",
            "item_id": "item_resp-curated-speak",
            "output_index": 0,
            "content_index": 0,
            "metadata": {
                "session_id": "sid-curated-speak",
                "epoch": 3,
                "model_speak": True,
            },
        }
    ]


@pytest.mark.asyncio
async def test_native_realtime_protocol_drains_internal_conversation_item_control_event():
    ws = TimedWebSocket()
    protocol = NativeRealtimeSessionProtocol(ws)  # type: ignore[arg-type]
    protocol.bind_sender(ws.send_json)

    ws.put({"type": "session.update", "model": "test-model", "session_id": "rt-delete"})
    session_create = json.loads(await protocol.receive_internal_event_text(ws))
    assert session_create["type"] == "session.create"

    ws.put(
        {
            "type": "conversation.item.create",
            "item": {
                "id": "item-a",
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hello"}],
            },
        }
    )
    assert json.loads(await protocol.receive_internal_event_text(ws)) == {
        "type": "turn.signal",
        "event": "conversation.item.create",
        "payload": {
            "item": {
                "id": "item-a",
                "object": "realtime.item",
                "type": "message",
                "status": "completed",
                "role": "user",
                "content": [{"type": "input_text", "text": "hello"}],
            }
        },
    }

    ws.put({"type": "conversation.item.delete", "item_id": "item-a"})
    translated = json.loads(await protocol.receive_internal_event_text(ws))

    assert translated == {
        "type": "turn.signal",
        "event": "conversation.item.delete",
        "payload": {"item_id": "item-a"},
    }


@pytest.mark.asyncio
async def test_native_realtime_protocol_conversation_item_create_commits_user_text():
    ws = TimedWebSocket()
    protocol = NativeRealtimeSessionProtocol(ws)  # type: ignore[arg-type]
    protocol.bind_sender(ws.send_json)

    ws.put({"type": "session.update", "model": "test-model", "session_id": "rt-item-create"})
    session_create = json.loads(await protocol.receive_internal_event_text(ws))
    assert session_create["type"] == "session.create"

    ws.put(
        {
            "type": "conversation.item.create",
            "item": {
                "id": "item-user-text",
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hello"}],
            },
        }
    )

    translated = json.loads(await protocol.receive_internal_event_text(ws))

    assert translated == {
        "type": "turn.signal",
        "event": "conversation.item.create",
        "payload": {
            "item": {
                "id": "item-user-text",
                "object": "realtime.item",
                "type": "message",
                "status": "completed",
                "role": "user",
                "content": [{"type": "input_text", "text": "hello"}],
            }
        },
    }


@pytest.mark.asyncio
async def test_native_realtime_protocol_audio_commit_requires_non_empty_buffer():
    ws = TimedWebSocket()
    protocol = NativeRealtimeSessionProtocol(ws)  # type: ignore[arg-type]
    protocol.bind_sender(ws.send_json)

    ws.put({"type": "session.update", "model": "test-model", "session_id": "rt-audio-commit"})
    session_create = json.loads(await protocol.receive_internal_event_text(ws))
    assert session_create["type"] == "session.create"

    ws.put({"type": "input_audio_buffer.commit", "final": True})
    with pytest.raises(WebSocketDisconnect):
        await protocol.receive_internal_event_text(ws)

    assert ws.sent[-1]["type"] == "error"
    assert ws.sent[-1]["error"]["code"] == "input_audio_buffer_empty"


@pytest.mark.asyncio
async def test_native_realtime_protocol_audio_commit_does_not_auto_create_response():
    ws = TimedWebSocket()
    protocol = NativeRealtimeSessionProtocol(ws)  # type: ignore[arg-type]
    protocol.bind_sender(ws.send_json)

    ws.put({"type": "session.update", "model": "test-model", "session_id": "rt-audio-commit-full"})
    session_create = json.loads(await protocol.receive_internal_event_text(ws))
    assert session_create["type"] == "session.create"

    pcm = struct.pack("<2h", 1024, -1024)
    ws.put({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode("ascii"), "format": "pcm16"})
    append_event = json.loads(await protocol.receive_internal_event_text(ws))
    assert append_event["type"] == "input_audio_buffer.append"

    ws.put({"type": "input_audio_buffer.commit", "final": True})
    commit_event = json.loads(await protocol.receive_internal_event_text(ws))

    assert commit_event["type"] == "input_audio_buffer.commit"
    assert commit_event["response_create"] is False


@pytest.mark.asyncio
async def test_native_realtime_protocol_rejects_invalid_sample_rate_without_closing():
    ws = TimedWebSocket()
    protocol = NativeRealtimeSessionProtocol(ws)  # type: ignore[arg-type]
    protocol.bind_sender(ws.send_json)

    ws.put({"type": "session.update", "model": "test-model", "session_id": "rt-invalid-rate"})
    session_create = json.loads(await protocol.receive_internal_event_text(ws))
    assert session_create["type"] == "session.create"

    ws.put(
        {
            "type": "input_audio_buffer.append",
            "audio": _pcm16_b64(1),
            "format": "pcm16",
            "sample_rate_hz": 192_001,
        }
    )
    ws.put({"type": "input_audio_buffer.clear"})
    translated = json.loads(await protocol.receive_internal_event_text(ws))

    assert translated == {"type": "input_audio_buffer.clear", "reason": "input_audio_buffer.clear"}
    error = next(event for event in ws.sent if event.get("type") == "error")
    assert error["error"]["code"] == "bad_event"
    assert error["error"]["param"] == "sample_rate_hz"


@pytest.mark.asyncio
async def test_native_realtime_protocol_audio_clear_is_not_barge_in_cancel():
    ws = TimedWebSocket()
    protocol = NativeRealtimeSessionProtocol(ws)  # type: ignore[arg-type]
    protocol.bind_sender(ws.send_json)

    ws.put({"type": "session.update", "model": "test-model", "session_id": "rt-audio-clear"})
    session_create = json.loads(await protocol.receive_internal_event_text(ws))
    assert session_create["type"] == "session.create"

    ws.put({"type": "input_audio_buffer.clear"})
    clear_event = json.loads(await protocol.receive_internal_event_text(ws))

    assert clear_event == {"type": "input_audio_buffer.clear", "reason": "input_audio_buffer.clear"}


@pytest.mark.asyncio
async def test_native_realtime_protocol_preserves_input_turn_policy_hints():
    ws = TimedWebSocket()
    protocol = NativeRealtimeSessionProtocol(ws)  # type: ignore[arg-type]
    protocol.bind_sender(ws.send_json)

    translated = await protocol._to_duplex_event(
        {
            "type": "input_audio_buffer.append",
            "audio": "AAAA",
            "format": "pcm_f32le",
            "duration_ms": 240,
            "vad": {"is_speech": True, "speech_probability": 0.9},
            "overlap_action": "listen",
        }
    )

    assert translated is not None
    assert translated["duration_ms"] == 240
    assert translated["vad"] == {"is_speech": True, "speech_probability": 0.9}
    assert translated["overlap_action"] == "listen"


@pytest.mark.asyncio
async def test_native_realtime_protocol_rejects_unimplemented_server_vad():
    ws = TimedWebSocket()
    protocol = NativeRealtimeSessionProtocol(ws)  # type: ignore[arg-type]
    protocol.bind_sender(ws.send_json)

    translated = await protocol._to_duplex_event(
        {
            "type": "session.update",
            "model": "test-model",
            "session_id": "rt-turn-detection",
            "turn_detection": {
                "type": "server_vad",
                "interrupt_response": False,
                "silence_duration_ms": 900,
                "threshold": 0.4,
            },
        }
    )

    assert translated is None
    error = ws.sent[-1]
    assert error["type"] == "error"
    assert error["error"]["code"] == "unsupported_turn_detection"
    assert "server_vad" in error["error"]["message"]


@pytest.mark.asyncio
async def test_native_realtime_protocol_accepts_disabled_turn_detection():
    ws = TimedWebSocket()
    protocol = NativeRealtimeSessionProtocol(ws)  # type: ignore[arg-type]
    protocol.bind_sender(ws.send_json)

    ws.put(
        {
            "type": "session.update",
            "model": "test-model",
            "session_id": "rt-no-turn-detection",
            "turn_detection": None,
        }
    )
    translated = json.loads(await protocol.receive_internal_event_text(ws))

    assert translated["type"] == "session.create"
    assert translated["session"]["extra_body"]["realtime_session_payload"]["turn_detection"] is None


@pytest.mark.asyncio
async def test_native_realtime_protocol_rejects_nested_vad_even_when_top_level_is_disabled():
    ws = TimedWebSocket()
    protocol = NativeRealtimeSessionProtocol(ws)  # type: ignore[arg-type]
    protocol.bind_sender(ws.send_json)

    translated = await protocol._to_duplex_event(
        {
            "type": "session.update",
            "model": "test-model",
            "turn_detection": None,
            "audio": {
                "input": {
                    "turn_detection": {
                        "type": "server_vad",
                    }
                }
            },
        }
    )

    assert translated is None
    assert ws.sent[-1]["error"]["code"] == "unsupported_turn_detection"


def test_native_duplex_handler_has_no_fixed_session_admission_cap():
    handler = OmniDuplexSessionHandler(chat_service=FakeChatService(FakeEngineClient()))
    client_config = DuplexSessionConfig(
        extra_body={"max_duplex_sessions": 99, "duplex_max_sessions": 99},
    )

    assert not hasattr(handler, "_max_native_duplex_sessions")
    assert not hasattr(handler, "_max_duplex_sessions")
    assert client_config.extra_body["max_duplex_sessions"] == 99


def test_native_response_options_ignore_private_runtime_config():
    session = DuplexSession(
        session_id="sid-response-private-config",
        config=DuplexSessionConfig(),
        capabilities=DuplexCapabilities.minicpmo45_native(),
    )

    handler = OmniDuplexSessionHandler(chat_service=FakeChatService(FakeEngineClient()))
    handler._apply_response_create_options(
        session,
        {
            "extra_body": {
                "duplex_stage_max_tokens": {"0": 999},
                "client_trace_id": "trace-1",
            }
        },
    )
    session.begin_response()

    assert "duplex_stage_max_tokens" not in session.config.extra_body
    assert "client_trace_id" not in session.config.extra_body
    assert session.response_config.extra_body["client_trace_id"] == "trace-1"


@pytest.mark.parametrize(
    "payload",
    [
        {"instructions": "not wired"},
        {"voice": "not-wired"},
        {"temperature": 0.2},
        {"max_output_tokens": 8},
        {"tools": [{"type": "function", "name": "lookup"}]},
    ],
)
def test_native_response_create_rejects_unwired_generation_options(payload):
    session = DuplexSession(
        session_id="sid-response-unwired-options",
        config=DuplexSessionConfig(),
        capabilities=DuplexCapabilities.minicpmo45_native(),
    )

    handler = OmniDuplexSessionHandler(chat_service=FakeChatService(FakeEngineClient()))
    error = handler._apply_response_create_options(session, payload)

    assert error == "unsupported_native_response_options"
    session.begin_response()
    assert session.response_config == session.config


def test_chat_fallback_excludes_realtime_response_metadata_from_model_request():
    handler = OmniDuplexSessionHandler(chat_service=FakeChatService(FakeEngineClient()))
    session = DuplexSession(
        session_id="sid-response-protocol-metadata",
        config=DuplexSessionConfig(),
    )
    session.reserve_response_options(
        ResponseCreateOptions(
            extra_body={
                "user": "trace-1",
                "realtime_response_conversation": "none",
                "realtime_response_metadata": {"source": "test"},
                "realtime_response_prompt": {"id": "prompt-1"},
                "realtime_response_tools": [
                    {
                        "type": "function",
                        "function": {"name": "lookup", "parameters": {}},
                    }
                ],
                "realtime_response_tool_choice": "auto",
            }
        )
    )
    session.begin_response()

    request = handler._build_chat_request(session, "request-1")
    created = handler._response_created_payload(session, session.active_response_id or "response-1", epoch=0)
    dumped = request.model_dump()

    assert dumped["user"] == "trace-1"
    assert dumped["tools"][0]["function"]["name"] == "lookup"
    assert dumped["tool_choice"] == "auto"
    assert not any(key.startswith("realtime_response_") for key in dumped)
    assert created["conversation"] == "none"
    assert created["metadata"] == {"source": "test"}
    assert created["prompt"] == {"id": "prompt-1"}
    assert session.response_config.extra_body["realtime_response_conversation"] == "none"


@pytest.mark.asyncio
async def test_native_response_create_surfaces_unwired_generation_options_error():
    handler = OmniDuplexSessionHandler(
        chat_service=FakeChatService(FakeEngineClient()),
        config_timeout_s=0.1,
        idle_timeout_s=1,
    )
    ws = TimedWebSocket()
    ws.put(_native_session_create("sid-response-unwired-options-event"))
    ws.put(
        {
            "type": "response.create",
            "response": {"instructions": "not wired"},
        }
    )
    ws.put({"type": "session.close"})

    await handler.handle_session(ws)

    error = next(message for message in ws.sent if message.get("type") == "error")
    assert error["code"] == "unsupported_native_response_options"
    assert "response_create_without_input" not in {message.get("code") for message in ws.sent}


@pytest.mark.asyncio
async def test_realtime_session_update_preserves_tools_and_metadata():
    engine = FakeEngineClient()
    chat_service = FakeChatService(engine)
    handler = OmniDuplexSessionHandler(chat_service=chat_service, config_timeout_s=0.1, idle_timeout_s=1)
    ws = TimedWebSocket()
    ws.put(_session_create("sid-rt-update-fields"))
    ws.put(
        {
            "type": "turn.signal",
            "event": "session.update",
            "payload": {
                "tools": [{"type": "function", "name": "lookup"}],
                "tool_choice": "auto",
                "metadata": {"demo": "yes"},
            },
        }
    )
    ws.put({"type": "session.close"})

    await handler.handle_session(ws)

    updated = next(m for m in ws.sent if m.get("type") == "session.updated")
    session = updated["session"]
    assert session["tools"] == [{"type": "function", "name": "lookup"}]
    assert session["tool_choice"] == "auto"
    assert session["metadata"] == {"demo": "yes"}
    assert "turn_detection" not in session
    runtime_config = engine.signal_session_configs[-1]
    assert runtime_config is not None
    assert runtime_config["extra_body"]["realtime_tools"] == [{"type": "function", "name": "lookup"}]
    assert runtime_config["extra_body"]["realtime_metadata"] == {"demo": "yes"}


@pytest.mark.asyncio
async def test_native_realtime_protocol_non_speech_append_does_not_emit_speech_started():
    ws = TimedWebSocket()
    protocol = NativeRealtimeSessionProtocol(ws)  # type: ignore[arg-type]
    protocol.bind_sender(ws.send_json)

    translated = await protocol._to_duplex_event(
        {
            "type": "input_audio_buffer.append",
            "audio": "AAAA",
            "format": "pcm_f32le",
            "vad": {"is_speech": False},
        }
    )

    assert translated is not None
    assert translated["vad"] == {"is_speech": False}
    assert not any(event.get("type") == "input_audio_buffer.speech_started" for event in ws.sent)


@pytest.mark.asyncio
async def test_native_session_update_rebuilds_server_runtime_policy():
    engine = FakeEngineClient()
    handler = OmniDuplexSessionHandler(
        chat_service=FakeChatService(engine),
        config_timeout_s=0.1,
        idle_timeout_s=1,
    )
    ws = TimedWebSocket()
    ws.put(_native_session_create("sid-runtime-update"))
    ws.put(
        {
            "type": "turn.signal",
            "event": "session.update",
            "payload": {"temperature": 0.2, "max_output_tokens": 7},
        }
    )
    ws.put({"type": "session.close"})

    await handler.handle_session(ws)

    assert "session.updated" in ws.sent_types()
    public_config = engine.signal_session_configs[-1]
    runtime_config = engine.signal_runtime_configs[-1]
    assert public_config is not None
    assert runtime_config is not None
    assert "duplex_stage_sampling_params" not in public_config["extra_body"]
    assert runtime_config["duplex_stage_max_tokens"]["0"] == 7
    assert runtime_config["duplex_stage_sampling_params"]["0"]["temperature"] == 0.2
    assert runtime_config["duplex_stage_sampling_params"]["0"]["top_k"] == 20
    assert runtime_config["duplex_stage_sampling_params"]["0"]["top_p"] == 0.8


@pytest.mark.asyncio
async def test_native_session_update_rejects_client_runtime_config():
    engine = FakeEngineClient()
    handler = OmniDuplexSessionHandler(
        chat_service=FakeChatService(engine),
        config_timeout_s=0.1,
        idle_timeout_s=1,
    )
    ws = TimedWebSocket()
    ws.put(_native_session_create("sid-runtime-update-reject"))
    ws.put(
        {
            "type": "turn.signal",
            "event": "session.update",
            "payload": {
                "extra_body": {"duplex_stage_max_tokens": {"0": 999}},
            },
        }
    )
    ws.put({"type": "session.close"})

    await handler.handle_session(ws)

    error = next(message for message in ws.sent if message.get("type") == "error")
    assert error["code"] == "invalid_duplex_runtime_config"
    assert ("sid-runtime-update-reject", "session.update") not in engine.signals


@pytest.mark.asyncio
async def test_minicpmo_native_session_update_requires_ref_audio_before_enabling_audio():
    engine = FakeEngineClient()
    handler = OmniDuplexSessionHandler(
        chat_service=FakeChatService(engine),
        config_timeout_s=0.1,
        idle_timeout_s=1,
    )
    ws = TimedWebSocket()
    ws.put(_native_session_create("sid-runtime-update-ref-required", modalities=["text"]))
    ws.put(
        {
            "type": "turn.signal",
            "event": "session.update",
            "payload": {"modalities": ["text", "audio"]},
        }
    )
    ws.put({"type": "session.close"})

    await handler.handle_session(ws)

    error = next(message for message in ws.sent if message.get("type") == "error")
    assert error["code"] == "ref_audio_required"
    assert "session.updated" not in ws.sent_types()
    assert ("sid-runtime-update-ref-required", "session.update") not in engine.signals


def test_native_realtime_protocol_audio_delta_preserves_sample_rate_hz():
    ws = TimedWebSocket()
    protocol = NativeRealtimeSessionProtocol(ws)  # type: ignore[arg-type]

    payloads = protocol._from_duplex_event(
        {
            "type": "response.output_audio.delta",
            "response_id": "resp-a",
            "audio": "AAAA",
            "format": "pcm",
            "sample_rate_hz": 24000,
        }
    )

    audio_events = [
        payload for payload in payloads if payload["type"] in {"response.output_audio.delta", "response.audio.delta"}
    ]
    assert {payload["type"] for payload in audio_events} == {"response.audio.delta"}
    assert {payload["format"] for payload in audio_events} == {"pcm16"}
    assert {payload["sample_rate_hz"] for payload in audio_events} == {24000}


def test_native_realtime_protocol_ignores_removed_legacy_event_switches():
    ws = TimedWebSocket()
    ws.query_params = {
        "legacy_audio_events": "true",
        "vllm_omni_legacy_events": "true",
        "output_audio_events": "true",
        "vllm_omni_output_audio_events": "true",
    }
    protocol = NativeRealtimeSessionProtocol(ws)  # type: ignore[arg-type]

    payloads = protocol._from_duplex_event(
        {
            "type": "response.output_audio.delta",
            "response_id": "resp-canonical",
            "audio": "AAAA",
            "text": "hello",
            "format": "pcm",
            "sample_rate_hz": 24000,
            "end_of_turn": True,
        }
    )

    event_types = {payload["type"] for payload in payloads}
    assert "response.audio.delta" in event_types
    assert "response.audio.done" in event_types
    assert "response.audio_transcript.delta" in event_types
    assert "response.audio_transcript.done" in event_types
    assert "response.output_audio.delta" not in event_types
    assert "response.output_audio.done" not in event_types
    assert "response.output_audio_transcript.delta" not in event_types
    assert "response.output_audio_transcript.done" not in event_types
    assert "response.output_item.created" not in event_types
    assert "response.text.delta" not in event_types


def test_native_realtime_protocol_emits_terminal_audio_transcript_events():
    ws = TimedWebSocket()
    protocol = NativeRealtimeSessionProtocol(ws)  # type: ignore[arg-type]

    payloads = protocol._from_duplex_event(
        {
            "type": "response.output_audio.delta",
            "response_id": "resp-a",
            "audio": "AAAA",
            "text": "hello",
            "format": "pcm",
            "sample_rate_hz": 24000,
            "end_of_turn": True,
        }
    )

    by_type = {payload["type"]: payload for payload in payloads}
    assert "response.audio.done" in by_type
    assert "response.audio_transcript.delta" in by_type
    assert "response.audio_transcript.done" in by_type
    assert by_type["response.content_part.done"]["part"]["transcript"] == "hello"
    assert by_type["response.output_item.done"]["item"]["object"] == "realtime.item"
    assert by_type["response.done"]["response"]["output"][0]["object"] == "realtime.item"


def test_native_realtime_protocol_terminal_transcript_equals_joined_deltas():
    protocol = NativeRealtimeSessionProtocol(TimedWebSocket())  # type: ignore[arg-type]
    response_id = "resp-transcript-contract"
    events: list[dict[str, object]] = []

    for text, end_of_turn in (("It's Canberra.", False), (" Next question.", True)):
        events.extend(
            protocol._from_duplex_event(
                {
                    "type": "response.output_audio.delta",
                    "response_id": response_id,
                    "audio": "AAAA",
                    "text": text,
                    "format": "pcm",
                    "sample_rate_hz": 24000,
                    "end_of_turn": end_of_turn,
                }
            )
        )

    transcript = "".join(str(event["delta"]) for event in events if event["type"] == "response.audio_transcript.delta")
    done = [str(event["transcript"]) for event in events if event["type"] == "response.audio_transcript.done"]

    assert transcript == "It's Canberra. Next question."
    assert done == [transcript]


def test_native_realtime_protocol_updates_in_progress_item_for_audio_truncate():
    ws = TimedWebSocket()
    protocol = NativeRealtimeSessionProtocol(ws)  # type: ignore[arg-type]

    protocol._from_duplex_event({"type": "response.created", "response_id": "resp-truncate"})
    protocol._from_duplex_event(
        {
            "type": "response.output_audio.delta",
            "response_id": "resp-truncate",
            "audio": "AAAA",
            "text": "hello",
            "format": "pcm",
            "sample_rate_hz": 24000,
            "audio_duration_ms": 100,
            "audio_text_marks": [{"text_chars": 5, "audio_end_ms": 100}],
        }
    )

    item = protocol._conversation_items["item_resp-truncate"]
    assert item["content"][0]["type"] == "output_audio"
    assert item["content"][0]["transcript"] == "hello"
    assert item["content"][0]["audio_duration_ms"] == 100

    translated = asyncio.run(
        protocol._to_duplex_event(
            {
                "type": "conversation.item.truncate",
                "item_id": "item_resp-truncate",
                "content_index": 0,
                "audio_end_ms": 50,
            }
        )
    )

    assert translated is None
    assert protocol._conversation_items["item_resp-truncate"]["content"][0]["transcript"] == "he"


@pytest.mark.asyncio
async def test_native_realtime_protocol_rejects_truncate_for_user_item():
    ws = TimedWebSocket()
    protocol = NativeRealtimeSessionProtocol(ws)  # type: ignore[arg-type]
    protocol.bind_sender(ws.send_json)
    protocol._conversation_items["item-user"] = {
        "id": "item-user",
        "object": "realtime.item",
        "type": "message",
        "role": "user",
        "status": "completed",
        "content": [{"type": "input_audio", "transcript": "hello"}],
    }

    translated = await protocol._to_duplex_event(
        {
            "type": "conversation.item.truncate",
            "item_id": "item-user",
            "content_index": 0,
            "audio_end_ms": 10,
        }
    )

    assert translated is None
    error = next(event for event in ws.sent if event.get("type") == "error")
    assert error["error"]["code"] == "bad_event"


def test_native_realtime_protocol_response_done_emits_audio_done_before_terminal_events():
    ws = TimedWebSocket()
    protocol = NativeRealtimeSessionProtocol(ws)  # type: ignore[arg-type]
    protocol._from_duplex_event({"type": "response.created", "response_id": "resp-done"})
    protocol._from_duplex_event(
        {
            "type": "response.output_audio.delta",
            "response_id": "resp-done",
            "audio": "AAAA",
            "format": "pcm",
            "sample_rate_hz": 24000,
        }
    )

    payloads = protocol._from_duplex_event({"type": "response.done", "response_id": "resp-done"})
    event_types = [payload["type"] for payload in payloads]

    assert event_types.index("response.audio.done") < event_types.index("response.content_part.done")
    assert event_types.index("response.content_part.done") < event_types.index("response.output_item.done")
    assert event_types.index("conversation.item.done") < event_types.index("response.done")
    assert event_types.index("response.output_item.done") < event_types.index("response.done")


def test_native_realtime_protocol_audio_cancelled_uses_active_response_id():
    ws = TimedWebSocket()
    protocol = NativeRealtimeSessionProtocol(ws)  # type: ignore[arg-type]
    protocol._from_duplex_event({"type": "response.created", "response_id": "resp-cancel"})

    payloads = protocol._from_duplex_event(
        {
            "type": "audio.cancelled",
            "reason": "barge_in",
            "committed_ms": 0,
        }
    )

    done = [payload for payload in payloads if payload.get("type") == "response.done"]
    assert len(done) == 1
    assert done[0]["response_id"] == "resp-cancel"
    assert done[0]["response"]["id"] == "resp-cancel"


def test_native_realtime_protocol_audio_cancelled_does_not_reopen_completed_response():
    ws = TimedWebSocket()
    protocol = NativeRealtimeSessionProtocol(ws)  # type: ignore[arg-type]
    protocol._from_duplex_event({"type": "response.created", "response_id": "resp-complete"})
    protocol._from_duplex_event(
        {
            "type": "response.output_audio.delta",
            "response_id": "resp-complete",
            "audio": "AAAA",
            "format": "pcm",
            "sample_rate_hz": 24000,
        }
    )
    protocol._from_duplex_event({"type": "response.done", "response_id": "resp-complete"})

    payloads = protocol._from_duplex_event(
        {
            "type": "audio.cancelled",
            "reason": "session_close",
            "committed_ms": 999999,
        }
    )

    assert [payload.get("type") for payload in payloads] == []


def test_native_audio_text_marks_are_normalized_to_session_cumulative_offsets():
    marks = OmniDuplexSessionHandler._normalize_native_audio_text_marks(
        [{"text_chars": 3, "audio_end_ms": 200}],
        audio_offset_ms=800,
        text_offset_chars=5,
    )

    assert marks == [{"text_chars": 8, "audio_end_ms": 1000}]


def test_duplex_session_playback_commit_uses_multi_delta_audio_text_marks():
    session = DuplexSession(
        session_id="sid-marks",
        config=DuplexSessionConfig(playback_commit_policy=DuplexPlaybackCommitPolicy.ACK_ONLY.value),
    )
    session.begin_response()
    session.append_assistant_text("hello ")
    session.mark_audio_sent(1000, text_chars=6)
    session.append_assistant_text("world")
    session.mark_audio_sent(2000, text_chars=11)
    session.acknowledge_playback(played_ms=1500, committed_ms=1500)

    committed = session.end_response(commit_text=True)

    assert committed == {"role": "assistant", "content": "hello wo"}
    assert session.history == (committed,)


@pytest.mark.asyncio
async def test_duplex_session_actor_preserves_wire_order_before_control():
    ws = TimedWebSocket()
    actor = DuplexWebSocketActor(ws)

    await actor.enqueue_event({"type": "input_audio_buffer.append", "audio": "AAAA"})
    await actor.enqueue_event({"type": "response.cancel"})

    first = await actor.next_event()
    second = await actor.next_event()

    assert first["type"] == "input_audio_buffer.append"
    assert second["type"] == "response.cancel"


def _pcm_f32_b64(samples: int, *, value: float = 0.05) -> str:
    return base64.b64encode(struct.pack(f"<{samples}f", *([value] * samples))).decode("ascii")


def _pcm16_b64(samples: int = 1, *, value: int = 1000) -> str:
    return base64.b64encode(struct.pack(f"<{samples}h", *([value] * samples))).decode("ascii")


def _session_create(session_id: str = "duplex-test") -> dict[str, Any]:
    return {
        "type": "session.create",
        "session_id": session_id,
        "session": {
            "model": "test-model",
            "modalities": ["text", "audio"],
            "idle_timeout_s": 1,
        },
    }


def _native_session_create(
    session_id: str = "duplex-native",
    *,
    modalities: list[str] | None = None,
) -> dict[str, Any]:
    event = _session_create(session_id)
    event["session"]["model"] = "openbmb/MiniCPM-o-4_5"
    event["session"]["modalities"] = list(modalities or ["text"])
    event["session"]["instructions"] = "You are a concise assistant."
    event["session"]["extra_body"] = {"minicpmo45_native_duplex": True}
    return event


def _native_realtime_session_update(
    session_id: str,
    *,
    modalities: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "type": "session.update",
        "session": {
            "session_id": session_id,
            "model": "openbmb/MiniCPM-o-4_5",
            "modalities": list(modalities or ["text"]),
            "instructions": "You are a concise assistant.",
            "idle_timeout_s": 1,
            "extra_body": {"minicpmo45_native_duplex": True},
        },
    }


def test_native_stage0_request_id_matches_engine_after_same_session_id_reopen():
    handler = OmniDuplexSessionHandler(chat_service=FakeChatService(FakeEngineClient()))
    first = handler._registry.create(session_id="sid-reused")
    handler._registry.close(first.session_id)
    reopened = handler._registry.create(session_id="sid-reused")
    expected = duplex_resource_request_id(
        DuplexFence(
            reopened.session_id,
            epoch=reopened.epoch,
            incarnation=reopened.incarnation,
        ),
        "stage0",
    )

    assert reopened.incarnation == 1
    assert handler._native_stage0_request_id(reopened, reopened.epoch) == expected


def _native_audio_payload(
    *,
    samples: int = 16000,
    value: float = 0.05,
    is_speech: bool | None = True,
    force_listen: bool | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": "audio",
        "audio": _pcm_f32_b64(samples, value=value),
        "format": "pcm_f32le",
        "sample_rate_hz": 16000,
    }
    if is_speech is not None:
        payload["is_speech"] = is_speech
    if force_listen is not None:
        payload["force_listen"] = force_listen
    return payload


def _auto_response_context(
    session_id: str,
    *,
    playback_active: bool = False,
) -> tuple[OmniDuplexSessionHandler, DuplexSession]:
    handler = OmniDuplexSessionHandler(
        chat_service=FakeChatService(FakeEngineClient()),
        config_timeout_s=0.1,
        idle_timeout_s=1,
    )
    session = DuplexSession(
        session_id=session_id,
        config=DuplexSessionConfig(extra_body={"auto_response": True}),
    )
    if playback_active:
        session.config.playback_commit_policy = DuplexPlaybackCommitPolicy.ACK_ONLY.value
        session.begin_response()
        session.mark_audio_sent(duration_ms=1000)
    return handler, session


def _install_direct_silence_scheduler(
    handler: OmniDuplexSessionHandler,
    session: DuplexSession,
) -> None:
    native = handler._minicpmo_session_state(session)

    async def _schedule(payload: object, **kwargs: Any) -> bool:
        if handler._native_silence_continuation_is_stale(
            session,
            request_id=kwargs["request_id"],
            response_id=kwargs["response_id"],
            response_owned=kwargs["response_owned"],
            expected_epoch=kwargs["expected_epoch"],
            expected_incarnation=kwargs["expected_incarnation"],
            expected_model_turn_id=kwargs["expected_model_turn_id"],
        ):
            return False
        append_ok, _ = await handler._append_runtime_input(
            session,
            payload,
            final=False,
            send_json=kwargs["send_json"],
            mode="append_audio_chunk",
            expected_epoch=kwargs["expected_epoch"],
        )
        return append_ok

    native.silence_continuation_scheduler = _schedule


def test_minicpmo_pcm_append_buffer_flush_preserves_accumulated_speech_marker():
    buffer = MiniCPMO45PcmAppendBuffer()
    payload = _native_audio_payload(samples=8000)

    assert buffer.append(payload, chunk_period_ms=1000) is None
    flushed = buffer.flush(chunk_period_ms=1000)

    assert flushed is not None
    assert flushed["is_speech"] is True


def test_minicpmo_pcm_append_buffer_drops_serving_new_user_turn_marker():
    buffer = MiniCPMO45PcmAppendBuffer()
    first = _native_audio_payload(samples=8000)
    first.update(
        new_user_turn=True,
        new_user_turn_prefix_variant="clean_response_done",
        force_speak=True,
    )
    second = _native_audio_payload(samples=8000)

    assert buffer.append(first, chunk_period_ms=1000) is None
    emitted = buffer.append(second, chunk_period_ms=1000)

    assert emitted is not None
    assert emitted["is_speech"] is True
    assert "new_user_turn" not in emitted
    assert "new_user_turn_prefix_variant" not in emitted
    assert "force_speak" not in emitted


def test_minicpmo_merge_native_audio_payloads_preserves_speech_marker():
    first = _native_audio_payload(samples=8000)
    second = _native_audio_payload(samples=8000, value=0.0, is_speech=False)

    merged = OmniDuplexSessionHandler._merge_native_audio_payloads(first, second)

    assert merged["is_speech"] is True


def test_minicpmo_merge_drops_serving_new_user_turn_marker():
    first = _native_audio_payload(samples=8000)
    first.update(
        new_user_turn=True,
        new_user_turn_prefix_variant="clean_response_done",
        force_speak=True,
    )
    second = _native_audio_payload(samples=8000, is_speech=False)

    merged = OmniDuplexSessionHandler._merge_native_audio_payloads(first, second)

    assert merged["is_speech"] is True
    assert "new_user_turn" not in merged
    assert "new_user_turn_prefix_variant" not in merged
    assert "force_speak" not in merged


@pytest.mark.asyncio
async def test_minicpmo_clear_continuation_does_not_cancel_pending_silence_task():
    async def _pending() -> bool:
        await asyncio.sleep(3600)
        return True

    native = MiniCPMO45ServingSessionState()
    task = asyncio.create_task(_pending())
    native.continuation_owner_id = "owner"
    native.continuation_units = 1
    native.pending_silence_task = task
    native.pending_silence_owner_id = "owner"

    native.clear_continuation()

    assert native.continuation_owner_id is None
    assert native.continuation_units == 0
    assert native.pending_silence_task is None
    assert native.pending_silence_owner_id is None
    assert not task.cancelled()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_auto_response_playback_overlap_keeps_model_owned_listen_speak_decision():
    handler, session = _auto_response_context("sid-auto-silent-overlap", playback_active=True)
    session.mark_audio_sent(duration_ms=1000)
    silent_payload = _native_audio_payload(value=0.0, is_speech=None, force_listen=False)
    speech_payload = _native_audio_payload(is_speech=None, force_listen=False)

    assert handler._should_force_listen_for_auto_response_overlap(session, {}, silent_payload) is False
    assert handler._should_force_listen_for_auto_response_overlap(session, {}, speech_payload) is False
    assert (
        handler._should_force_listen_for_auto_response_overlap(
            session,
            {"force_listen": True},
            speech_payload,
        )
        is True
    )
    assert (
        handler._should_force_listen_for_auto_response_overlap(
            session,
            {"force_barge_in": True},
            speech_payload,
        )
        is False
    )

    session.acknowledge_playback(played_ms=1000, committed_ms=1000)
    assert handler._should_force_listen_for_auto_response_overlap(session, {}, speech_payload) is False


def test_auto_response_overlap_silence_advances_model_unit_and_preserves_realtime_input():
    handler, session = _auto_response_context(
        "sid-auto-silent-wire-buffer",
        playback_active=True,
    )
    decision = handler._overlap_decision(
        session,
        {"is_speech": False},
        _native_audio_payload(samples=3200, is_speech=False),
    )

    assert decision["action"] == "listen"
    assert decision["buffer_audio"] is True
    assert decision["defer_runtime_append"] is False
    assert decision["force_listen"] is False
    assert decision["preserve_realtime_input"] is True


def test_auto_response_playback_overlap_admits_model_units_and_tracks_speech():
    handler, session = _auto_response_context("sid-auto-deferred-overlap", playback_active=True)
    payload = _native_audio_payload(samples=8640)

    decision = handler._overlap_decision(
        session,
        {"duration_ms": 540, "is_speech": True},
        payload,
    )

    assert decision["action"] == "listen"
    assert decision["defer_runtime_append"] is False
    assert decision["force_listen"] is False
    assert decision["buffer_audio"] is True
    assert session.overlap_speech_ms == 540


@pytest.mark.asyncio
async def test_auto_response_overlap_exact_unit_commit_does_not_block_or_replay_next_unit():
    class ModelUnitEngine(FakeEngineClient):
        def __init__(self) -> None:
            super().__init__()
            self.append_started = asyncio.Event()
            self.active_response_ids_at_append: list[str | None] = []
            self.response_done_seen_at_append = False
            self.session: DuplexSession | None = None
            self.websocket: TimedWebSocket | None = None

        async def append_duplex_input_async(self, session_id: str, **kwargs):
            assert self.session is not None
            assert self.websocket is not None
            self.active_response_ids_at_append.append(self.session.active_response_id)
            self.response_done_seen_at_append = "response.done" in self.websocket.sent_types()
            kwargs.pop("expected_epoch", None)
            result = await super().append_duplex_input_async(session_id, **kwargs)
            self.append_started.set()
            return result

    engine = ModelUnitEngine()
    handler = OmniDuplexSessionHandler(
        chat_service=FakeChatService(engine),
        config_timeout_s=0.1,
        idle_timeout_s=1,
    )
    ws = TimedWebSocket()
    engine.websocket = ws
    create = _native_session_create("sid-auto-overlap-model-unit")
    create["session"]["extra_body"]["auto_response"] = True
    ws.put(create)
    handler_task = asyncio.create_task(handler.handle_session(ws))

    try:
        for _ in range(100):
            session = handler._registry.get("sid-auto-overlap-model-unit")
            if session is not None and "session.created" in ws.sent_types():
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("native duplex session did not open")

        engine.session = session
        session.config.playback_commit_policy = DuplexPlaybackCommitPolicy.ACK_ONLY.value
        response_id = session.begin_response()
        session.mark_audio_sent(duration_ms=2000)
        request_id = handler._native_stage0_request_id(session, session.epoch)
        session.bind_request(request_id)
        expected_fence = DuplexFence(
            session.session_id,
            epoch=session.epoch,
            turn_id=session.turn_id,
            incarnation=session.incarnation,
        )

        chunk = {
            "type": "input_audio_buffer.append",
            "audio": _pcm_f32_b64(3200, value=0.05),
            "format": "pcm_f32le",
            "sample_rate_hz": 16000,
            "duration_ms": 200,
            "is_speech": True,
        }
        for _ in range(4):
            ws.put(chunk)
        await asyncio.sleep(0.05)
        assert engine.appended == []

        ws.put(chunk)
        await asyncio.wait_for(engine.append_started.wait(), timeout=0.5)

        ws.put({"type": "input_audio_buffer.commit", "final": True})
        for _ in range(100):
            if "input.committed" in ws.sent_types():
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("exact model unit input did not commit")

        engine.append_started.clear()
        for _ in range(5):
            ws.put(chunk)
        await asyncio.wait_for(engine.append_started.wait(), timeout=0.5)
        await asyncio.sleep(0.05)
        assert "error" not in ws.sent_types()
    finally:
        ws.put({"type": "session.close"})
        await asyncio.wait_for(handler_task, timeout=2)

    assert len(engine.appended) == 2
    for appended_session_id, mode, payload, final in engine.appended:
        assert appended_session_id == session.session_id
        assert mode == "append_audio_chunk"
        assert final is False
        assert isinstance(payload, dict)
        assert np.frombuffer(base64.b64decode(payload["audio"]), dtype="<f4").shape == (16000,)
        assert payload.get("force_listen") is not True
    assert engine.appended_fences == [expected_fence, expected_fence]
    assert engine.active_response_ids_at_append == [response_id, response_id]
    assert engine.response_done_seen_at_append is False
    assert not any(event in {"barge_in", "input.cancel"} for _, event in engine.signals)
    assert ws.sent_types().count("response.speak") <= 1


@pytest.mark.asyncio
async def test_stale_terminal_event_does_not_reset_current_session_overlap():
    handler, session = _auto_response_context("sid-stale-terminal")
    session.epoch = 2
    session.accumulate_overlap_speech(540)
    actor = DuplexWebSocketActor(TimedWebSocket(), current_epoch=lambda: session.epoch)
    native = MiniCPMO45ServingSessionState()

    accepted, deferred = await handler._apply_outbound_session_event(
        {"type": "response.done", "epoch": 1},
        session=session,
        actor=actor,
        native=native,
        realtime_protocol=None,
    )

    assert accepted is False
    assert deferred is None
    assert session.overlap_speech_ms == 540


@pytest.mark.asyncio
async def test_current_terminal_event_resets_session_owned_overlap():
    handler, session = _auto_response_context("sid-current-terminal")
    session.accumulate_overlap_speech(320)
    actor = DuplexWebSocketActor(TimedWebSocket(), current_epoch=lambda: session.epoch)
    native = MiniCPMO45ServingSessionState()

    accepted, deferred = await handler._apply_outbound_session_event(
        {"type": "response.done", "epoch": session.epoch},
        session=session,
        actor=actor,
        native=native,
        realtime_protocol=None,
    )

    assert accepted is True
    assert deferred is None
    assert session.overlap_speech_ms == 0


@pytest.mark.asyncio
async def test_response_listen_terminal_does_not_drive_serving_generation_state():
    handler, session = _auto_response_context("sid-listen-terminal")
    actor = DuplexWebSocketActor(TimedWebSocket(), current_epoch=lambda: session.epoch)
    native = MiniCPMO45ServingSessionState()

    accepted, deferred = await handler._apply_outbound_session_event(
        {"type": "response.listen", "epoch": session.epoch},
        session=session,
        actor=actor,
        native=native,
        realtime_protocol=None,
    )

    assert accepted is True
    assert deferred is None
    assert session.turn_id == 0


@pytest.mark.asyncio
async def test_stale_audio_cancelled_does_not_reset_current_session_overlap():
    handler, session = _auto_response_context("sid-stale-audio-cancelled")
    session.epoch = 3
    session.accumulate_overlap_speech(640)
    actor = DuplexWebSocketActor(TimedWebSocket(), current_epoch=lambda: session.epoch)
    native = MiniCPMO45ServingSessionState()

    accepted, deferred = await handler._apply_outbound_session_event(
        {"type": "audio.cancelled", "epoch": 2},
        session=session,
        actor=actor,
        native=native,
        realtime_protocol=None,
    )

    assert accepted is False
    assert deferred is None
    assert session.overlap_speech_ms == 640


def test_auto_response_post_response_silence_remains_model_owned_input():
    handler, session = _auto_response_context("sid-auto-waiting-silence", playback_active=True)
    silent_payload = _native_audio_payload(value=0.0, is_speech=False, force_listen=False)
    speech_payload = _native_audio_payload(force_listen=False)

    assert handler._assistant_playback_active(session) is True
    assert handler._should_force_listen_for_auto_response_overlap(session, {}, silent_payload) is False
    assert handler._should_force_listen_for_auto_response_overlap(session, {}, speech_payload) is False


@pytest.mark.asyncio
async def test_auto_response_terminal_advances_open_realtime_input_without_commit():
    handler, session = _auto_response_context("sid-auto-realtime-open-input", playback_active=True)
    session.capabilities = DuplexCapabilities.minicpmo45_native()
    response_id = session.active_response_id
    assert response_id is not None

    native = handler._minicpmo_session_state(session)
    native.input_since_commit = True
    native.speech_since_commit = True
    session.accumulate_overlap_speech(800)
    assert (
        native.audio_buffer.append(
            _native_audio_payload(samples=8000),
            chunk_period_ms=session.capabilities.chunk_period_ms or 1000,
        )
        is None
    )
    session.end_response(commit_text=False, preserve_request=True)

    allowed, promoted = await handler._apply_outbound_session_event(
        {
            "type": "response.done",
            "response_id": response_id,
            "epoch": session.epoch,
            "status": "completed",
        },
        session=session,
        actor=DuplexWebSocketActor(TimedWebSocket()),
        native=native,
        realtime_protocol=NativeRealtimeSessionProtocol({}),
    )

    assert allowed is True
    assert promoted is None
    assert native.audio_buffer.has_pending()
    assert native.committed_audio_payload is None
    assert native.deferred_response_create is False
    assert native.input_since_commit is True
    assert native.speech_since_commit is True
    next_speech = _native_audio_payload(samples=8000)
    assert "new_user_turn" not in next_speech


@pytest.mark.asyncio
async def test_auto_response_terminal_allows_next_model_unit_while_playback_drains():
    handler, session = _auto_response_context(
        "sid-auto-realtime-open-before-overlap",
        playback_active=True,
    )
    session.capabilities = DuplexCapabilities.minicpmo45_native()
    response_id = session.active_response_id
    assert response_id is not None

    native = handler._minicpmo_session_state(session)
    native.input_since_commit = True
    native.speech_since_commit = True
    session.end_response(commit_text=False, preserve_request=True)

    allowed, promoted = await handler._apply_outbound_session_event(
        {
            "type": "response.done",
            "response_id": response_id,
            "epoch": session.epoch,
            "status": "completed",
        },
        session=session,
        actor=DuplexWebSocketActor(TimedWebSocket()),
        native=native,
        realtime_protocol=NativeRealtimeSessionProtocol({}),
    )

    assert allowed is True
    assert promoted is None
    assert native.input_since_commit is True
    assert handler._assistant_playback_active(session) is True

    next_speech = _native_audio_payload(samples=16_000)
    assert "new_user_turn" not in next_speech


@pytest.mark.asyncio
async def test_realtime_committed_overlap_promotes_after_response_done():
    handler, session = _auto_response_context("sid-auto-realtime-committed-input", playback_active=True)
    session.capabilities = DuplexCapabilities.minicpmo45_native()
    response_id = session.active_response_id
    assert response_id is not None

    native = handler._minicpmo_session_state(session)
    native.committed_audio_payload = _native_audio_payload()
    native.deferred_response_create = True
    session.accumulate_overlap_speech(1000)
    session.end_response(commit_text=False, preserve_request=True)

    allowed, promoted = await handler._apply_outbound_session_event(
        {
            "type": "response.done",
            "response_id": response_id,
            "epoch": session.epoch,
            "status": "completed",
        },
        session=session,
        actor=DuplexWebSocketActor(TimedWebSocket()),
        native=native,
        realtime_protocol=NativeRealtimeSessionProtocol({}),
    )

    assert allowed is True
    assert promoted is not None
    assert promoted["force_listen"] is False
    assert native.committed_audio_payload is promoted
    assert native.deferred_response_create is False
    assert "new_user_turn" not in promoted


@pytest.mark.asyncio
async def test_auto_response_committed_overlap_does_not_precreate_empty_response():
    session_id = "sid-auto-overlap-no-empty-response"
    request_id = f"duplex-{session_id}-e0-stage0"

    def _stage_output(samples: int, *, turn_end: bool = False):
        return SimpleNamespace(
            request_id=request_id,
            finished=False,
            outputs=[
                SimpleNamespace(
                    text="hello",
                    multimodal_output={
                        "audio": np.zeros(samples, dtype=np.float32),
                        "sr": 24000,
                        "meta": {
                            "turn_end": turn_end,
                            "duplex_turn_id": 0,
                            "duplex_epoch": 0,
                        },
                    },
                )
            ],
        )

    first_append = {
        "operation": "append",
        "session_id": session_id,
        "ok": True,
        "unsupported_count": 0,
        "error_count": 0,
        "stage_results": [
            {
                "stage_id": 0,
                "replica_id": 0,
                "result": {
                    "supported": True,
                    "implementation_level": "model_native_duplex",
                    "data_plane_append": True,
                    "request_id": request_id,
                    "response_stage_id": 1,
                },
            }
        ],
        "data_plane_outputs": [_stage_output(2400)],
    }
    promoted_append = {
        "operation": "append",
        "session_id": session_id,
        "ok": True,
        "unsupported_count": 0,
        "error_count": 0,
        "stage_results": [],
        "data_plane_outputs": [],
    }
    engine = FakeEngineClient(
        append_results=[first_append, promoted_append],
        collect_outputs=[[_stage_output(4800, turn_end=True)]],
        collect_delay_s=0.1,
    )
    handler = OmniDuplexSessionHandler(
        chat_service=FakeChatService(engine),
        config_timeout_s=0.1,
        idle_timeout_s=2,
    )

    overlap_sent = False

    def commit_overlap_after_first_response(ws: TimedWebSocket, data: dict[str, Any]) -> None:
        nonlocal overlap_sent
        if data.get("type") != "response.audio.delta" or overlap_sent:
            return
        overlap_sent = True
        chunk = {
            "type": "input_audio_buffer.append",
            "audio": _pcm_f32_b64(3200, value=0.05),
            "format": "pcm_f32le",
            "sample_rate_hz": 16000,
            "duration_ms": 200,
            "is_speech": True,
        }
        for _ in range(4):
            ws.put(chunk)
        ws.put({"type": "input_audio_buffer.commit", "final": True})

    ws = TimedWebSocket(on_send=commit_overlap_after_first_response, receive_timeout_s=3)
    create = _native_realtime_session_update(session_id)
    create["session"]["extra_body"]["auto_response"] = True
    ws.put(create)
    ws.put(
        {
            "type": "input_audio_buffer.append",
            "audio": _pcm_f32_b64(16000, value=0.05),
            "format": "pcm_f32le",
            "sample_rate_hz": 16000,
            "duration_ms": 1000,
            "is_speech": True,
        }
    )
    ws.put({"type": "input_audio_buffer.commit", "final": True})

    handler_task = asyncio.create_task(handler.handle_realtime_session(ws))
    try:
        for _ in range(300):
            if len(engine.appended) >= 2:
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail(
                f"committed overlap was not promoted after response.done: "
                f"appends={len(engine.appended)}, events={ws.sent_types()}"
            )

        assert overlap_sent is True
        assert len(engine.appended) == 2
        assert engine.appended[1][3] is True
        assert ws.sent_types().count("response.created") == 1
    finally:
        ws.put({"type": "session.close"})
        await asyncio.wait_for(handler_task, timeout=3)

    assert ws.sent_types().count("response.done") >= 1


def test_auto_response_barge_in_does_not_mark_serving_new_user_turn_payload():
    handler, session = _auto_response_context("sid-auto-barge-new-turn", playback_active=True)
    payload = _native_audio_payload()

    assert handler._should_force_listen_for_auto_response_overlap(session, {"force_barge_in": True}, payload) is False
    assert "new_user_turn" not in payload
    assert "new_user_turn_prefix_variant" not in payload
    assert session.turn_id == 0
    assert payload.get("force_speak") is not True


def test_native_realtime_barge_in_payload_does_not_create_serving_turn_boundary():
    payload: dict[str, object] = {
        "type": "audio",
        "audio": _pcm_f32_b64(16000, value=0.05),
        "format": "pcm_f32le",
        "sample_rate_hz": 16000,
        "is_speech": True,
    }
    session = DuplexSession(session_id="sid-realtime-barge", config=DuplexSessionConfig())
    session.turn_id = 2

    assert "new_user_turn" not in payload
    assert "new_user_turn_prefix_variant" not in payload
    assert session.turn_id == 2


def test_auto_response_force_barge_in_has_no_waiting_turn_variant():
    handler, session = _auto_response_context("sid-auto-barge-keeps-variant")
    payload = _native_audio_payload()

    assert (
        handler._should_force_listen_for_auto_response_overlap(
            session,
            {"force_barge_in": True},
            payload,
        )
        is False
    )


def test_duplex_endpoint_requires_explicit_session_mode_duplex():
    assert should_enable_duplex_endpoint(None) is False
    assert should_enable_duplex_endpoint([]) is False
    assert should_enable_duplex_endpoint([SimpleNamespace(session_mode="turn")]) is False
    assert should_enable_duplex_endpoint([{"session_mode": "turn"}]) is False
    assert (
        should_enable_duplex_endpoint(
            [
                SimpleNamespace(session_mode="turn"),
                SimpleNamespace(session_mode="duplex"),
            ]
        )
        is True
    )
    assert should_enable_duplex_endpoint([{"session_mode": "duplex"}]) is True


def test_duplex_endpoint_supports_top_level_session_mode(tmp_path):
    config_path = tmp_path / "stage_config.yaml"
    config_path.write_text(
        """
session_mode: duplex
stage_args:
  - stage_id: 0
    engine_args: {}
""",
        encoding="utf-8",
    )

    assert should_enable_duplex_endpoint([], config_path=str(config_path)) is True


def test_duplex_handler_splits_data_plane_audio_list_into_deltas():
    import torch

    data_plane = _test_data_plane()
    output = SimpleNamespace(
        finished=True,
        outputs=[
            SimpleNamespace(
                text="hello",
                multimodal_output={
                    "audio": [
                        torch.zeros(10, dtype=torch.float32),
                        torch.zeros(20, dtype=torch.float32),
                    ],
                    "sr": 24000,
                },
            )
        ],
    )

    native_results = list(data_plane.project_output(output))

    assert [result["audio_data"] for result in native_results] == ["wav-10", "wav-20"]
    assert [result["text"] for result in native_results] == ["hello", ""]
    assert [result["end_of_turn"] for result in native_results] == [False, True]


def test_duplex_listen_latent_does_not_poison_cumulative_audio_offset():
    import torch

    data_plane = _test_data_plane()
    request_id = "duplex-duplex-sess-stage0"

    # A model-listen decision wraps the segment with a latent tensor that is
    # NOT reply audio; it must not advance the cumulative audio offset.
    listen_output = SimpleNamespace(
        request_id=request_id,
        finished=True,
        outputs=[],
        multimodal_output={
            "duplex_native_decision": "listen",
            "model_listen": True,
            "latent": torch.zeros(331776, dtype=torch.float32),
            "meta": {"sr": 24000},
        },
    )
    listen_results = list(data_plane.project_output(listen_output))
    assert [result.get("is_listen") for result in listen_results] == [True]

    # The first speak unit carries cumulative stage-1 audio far smaller than
    # the listen latent; it must still be delivered from sample 0.
    speak_output = SimpleNamespace(
        request_id=request_id,
        finished=False,
        outputs=[
            SimpleNamespace(
                text=" It was a very",
                multimodal_output={},
            )
        ],
        multimodal_output={
            "audio": torch.zeros(32768, dtype=torch.float32),
            "sr": 24000,
        },
    )
    speak_results = list(data_plane.project_output(speak_output))
    assert [result.get("audio_data") for result in speak_results] == ["wav-32768"]
    assert speak_results[0]["text"] == " It was a very"


def test_direct_listen_decision_survives_inner_completion_metadata():
    inner_output = SimpleNamespace(
        outputs=[
            SimpleNamespace(
                multimodal_output={"special_token_ids": {"listen_token_id": 151705}},
            )
        ]
    )
    decision = MiniCPMO45DuplexRuntimeExtension().decide_output(
        stage_id=0,
        final_stage_id=1,
        segment_finished=True,
        segment_token_ids=(151705,),
        segment_output_metadata={"special_token_ids": {"listen_token_id": 151705}},
        output=inner_output,
    )
    assert decision is not None
    output = attach_duplex_output_decision(
        OmniRequestOutput(
            request_id="duplex-direct-listen",
            finished=True,
            stage_id=0,
            final_output_type=decision.final_output_type,
            request_output=inner_output,
            metrics={
                "stage_metrics": {
                    "0": {
                        "vllm_ttft_ms": 125.0,
                        "vllm_tpot_ms": 18.0,
                        "vllm_itl_ms": 17.5,
                        "vllm_itls_ms": [17.0, 18.0],
                    }
                }
            },
        ),
        decision,
    )

    results = list(_test_data_plane().project_output(output))

    assert len(results) == 1
    assert results[0]["is_listen"] is True
    assert results[0]["model_listen"] is True
    assert results[0]["listen_source"] == "model_listen"
    assert results[0]["stage_metrics"] == output.metrics["stage_metrics"]


def test_duplex_segment_text_is_attached_once_across_streaming_batches():
    import torch

    class _ChatService:
        model_config = _ModelConfig()

        def create_audio(self, audio_obj):
            return SimpleNamespace(audio_data=f"wav-{int(audio_obj.audio_tensor.shape[0])}")

    data_plane = _test_data_plane()
    request_id = "duplex-duplex-sess-stage0"
    session = DuplexSession(
        session_id="sid-auto-respond",
        config=DuplexSessionConfig(extra_body={"auto_response": True}),
    )

    def _speak(total_samples: int, text: str, *, finished: bool):
        output = SimpleNamespace(
            request_id=request_id,
            finished=finished,
            outputs=[SimpleNamespace(text=text, multimodal_output={})],
            multimodal_output={
                "audio": torch.zeros(total_samples, dtype=torch.float32),
                "sr": 24000,
            },
        )
        return _project_data_plane(data_plane, {"data_plane_outputs": [output]}, session=session)

    def _audio(results):
        return [r for r in results if r.get("is_listen") is False]

    def _listen(results):
        return [r for r in results if r.get("is_listen") is True]

    # One talker segment streams several cumulative-audio batches, each
    # carrying the SAME segment text; only the first may attach it.
    assert [r["text"] for r in _audio(_speak(100, " movie called", finished=False))] == [" movie called"]
    assert [r["text"] for r in _audio(_speak(220, " movie called", finished=False))] == [""]
    finished_results = _speak(300, " movie called", finished=True)
    assert [r["text"] for r in _audio(finished_results)] == [""]
    assert _listen(finished_results) == []

    # Continuation units re-run the talker with the SAME handed text past a
    # finished boundary (every engine segment ends finished=True); the text
    # must stay suppressed or each continuation duplicates the transcript.
    assert [r["text"] for r in _audio(_speak(400, " movie called", finished=False))] == [""]
    assert [r["text"] for r in _audio(_speak(450, " movie called", finished=True))] == [""]

    # A genuinely new segment text is attached in full.
    assert [r["text"] for r in _audio(_speak(500, " Titanic", finished=True))] == [" Titanic"]

    # Text growing within a segment is delivered as suffix deltas.
    assert [r["text"] for r in _audio(_speak(600, " by", finished=False))] == [" by"]
    assert [r["text"] for r in _audio(_speak(700, " by James", finished=True))] == [" James"]

    # A segment whose finished batch slices to an EMPTY audio delta (all
    # samples already delivered) must not block the next segment's text.
    assert [r["text"] for r in _audio(_speak(800, " James Cameron. The", finished=False))] == [" James Cameron. The"]
    assert _speak(800, " James Cameron. The", finished=True) == []
    assert [r["text"] for r in _audio(_speak(900, " 997.", finished=False))] == [" 997."]


def test_duplex_data_plane_output_prefers_audio_segment_text_metadata():
    data_plane = _test_data_plane()
    request_id = "duplex-sid-native-segment-text-e0-stage0"
    segment_text = "你好，有什么想聊的吗？"
    cumulative_text = segment_text * 2
    output = SimpleNamespace(
        request_id=request_id,
        finished=False,
        outputs=[SimpleNamespace(text=cumulative_text, multimodal_output={})],
        multimodal_output={
            "audio": np.zeros(24000, dtype=np.float32),
            "sr": 24000,
            "meta.llm_output_text_utf8": np.frombuffer(segment_text.encode("utf-8"), dtype=np.uint8),
        },
    )

    results = list(data_plane.project_output(output))

    assert len(results) == 1
    assert results[0]["audio_data"] == "wav-24000"
    assert results[0]["text"] == segment_text

    next_output_without_segment_metadata = SimpleNamespace(
        request_id=request_id,
        finished=False,
        outputs=[SimpleNamespace(text=segment_text * 3, multimodal_output={})],
        multimodal_output={
            "audio": np.zeros(30720, dtype=np.float32),
            "sr": 24000,
        },
    )
    next_results = list(data_plane.project_output(next_output_without_segment_metadata))

    assert len(next_results) == 1
    assert next_results[0]["audio_data"] == "wav-6720"
    assert next_results[0]["text"] == ""


@pytest.mark.parametrize(
    ("request_id", "second_samples", "expected_second_audio"),
    [
        ("duplex-sid-model-turn-audio-e0-stage0", 36000, "wav-12000"),
        ("duplex-sid-model-turn-audio-restart-e0-stage0", 12000, "wav-12000"),
    ],
    ids=["cumulative-across-model-turns", "upstream-cumulative-audio-restarts"],
)
def test_duplex_audio_cursor_across_model_turns(
    request_id: str,
    second_samples: int,
    expected_second_audio: str,
):
    data_plane = _test_data_plane()

    def output(turn_id: int, samples: int):
        return SimpleNamespace(
            request_id=request_id,
            finished=False,
            outputs=[SimpleNamespace(text="", multimodal_output={})],
            multimodal_output={
                "audio": np.zeros(samples, dtype=np.float32),
                "sr": 24000,
                "meta.duplex_turn_id": np.array([turn_id], dtype=np.int32),
            },
        )

    first = list(data_plane.project_output(output(0, 24000)))
    second = list(data_plane.project_output(output(1, second_samples)))

    assert first[0]["audio_data"] == "wav-24000"
    assert second[0]["audio_data"] == expected_second_audio


def test_duplex_data_plane_accepts_model_turn_output_across_client_commits():
    data_plane = _test_data_plane()
    session = DuplexSession(
        session_id="sid-auto-stale-turn",
        config=DuplexSessionConfig(extra_body={"auto_response": True}),
    )
    session.commit_native_audio_input()
    session.commit_native_audio_input()
    session.turn_id = 1
    session.epoch = 0
    request_id = "duplex-sid-auto-stale-turn-stage0"
    output = SimpleNamespace(
        request_id=request_id,
        finished=False,
        outputs=[SimpleNamespace(text="old tail", multimodal_output={})],
        multimodal_output={
            "audio": np.zeros(24000, dtype=np.float32),
            "sr": 24000,
            "meta.duplex_turn_id": np.array([1], dtype=np.int32),
            "meta.llm_output_text_utf8": np.frombuffer("旧尾巴".encode(), dtype=np.uint8),
        },
    )

    results = list(data_plane.project_output(output, context=_data_plane_context(session)))

    assert len(results) == 1
    assert results[0]["audio_data"] == "wav-24000"
    assert results[0]["model_turn_id"] == 1


def test_duplex_data_plane_accepts_active_response_turn_identity():
    data_plane = _test_data_plane()
    session = DuplexSession(
        session_id="sid-active-response-turn",
        config=DuplexSessionConfig(extra_body={"auto_response": True}),
    )
    session.turn_id = 1
    session.epoch = 0
    session.begin_response(turn_id=0)
    output = SimpleNamespace(
        request_id="duplex-sid-active-response-turn-stage0",
        finished=True,
        outputs=[SimpleNamespace(text="", multimodal_output={})],
        multimodal_output={
            "audio": np.zeros(10, dtype=np.float32),
            "sr": 24000,
            "meta.duplex_turn_id": np.array([0], dtype=np.int32),
            "meta.duplex_epoch": np.array([0], dtype=np.int32),
        },
    )

    results = _project_data_plane(data_plane, {"data_plane_outputs": [output]}, session=session)

    assert len(results) == 1


def test_duplex_data_plane_listen_preserves_model_turn_identity():
    data_plane = _test_data_plane()
    session = DuplexSession(
        session_id="sid-listen-turn",
        config=DuplexSessionConfig(extra_body={"auto_response": True}),
    )
    output = SimpleNamespace(
        request_id="duplex-sid-listen-turn-stage0",
        finished=True,
        outputs=[SimpleNamespace(text="", token_ids=[], multimodal_output={})],
        multimodal_output={
            "duplex_native_decision": "listen",
            "meta.duplex_turn_id": np.array([2], dtype=np.int32),
            "meta.duplex_epoch": np.array([0], dtype=np.int32),
        },
    )

    results = _project_data_plane(data_plane, {"data_plane_outputs": [output]}, session=session)

    assert len(results) == 1
    assert results[0]["is_listen"] is True
    assert results[0]["model_turn_id"] == 2


def test_duplex_data_plane_does_not_drop_future_model_turn_while_response_active():
    data_plane = _test_data_plane()
    session = DuplexSession(
        session_id="sid-future-response-turn",
        config=DuplexSessionConfig(extra_body={"auto_response": True}),
    )
    session.turn_id = 0
    session.epoch = 0
    session.begin_response(turn_id=0)
    output = SimpleNamespace(
        request_id="duplex-sid-future-response-turn-stage0",
        finished=False,
        outputs=[SimpleNamespace(text="", multimodal_output={})],
        multimodal_output={
            "audio": np.zeros(10, dtype=np.float32),
            "sr": 24000,
            "meta.duplex_turn_id": np.array([1], dtype=np.int32),
            "meta.duplex_epoch": np.array([0], dtype=np.int32),
        },
    )

    results = _project_data_plane(data_plane, {"data_plane_outputs": [output]}, session=session)

    assert len(results) == 1
    assert results[0]["model_turn_id"] == 1
    assert results[0]["audio_data"] == "wav-10"


def _duplex_tts_output(
    *,
    request_id: str,
    samples: int,
    finished: bool,
    text: str = "hello",
    tts_is_last_chunk: bool = False,
    turn_end: bool = False,
    turn_id: int = 0,
    token_ids: list[int] | None = None,
):
    return SimpleNamespace(
        request_id=request_id,
        finished=finished,
        outputs=[SimpleNamespace(text=text, token_ids=list(token_ids or []), multimodal_output={})],
        multimodal_output={
            "audio": np.zeros(samples, dtype=np.float32),
            "sr": 24000,
            "meta.tts_is_last_chunk": np.array([int(tts_is_last_chunk)], dtype=np.int32),
            "meta.turn_end": np.array([int(turn_end)], dtype=np.int32),
            "meta.duplex_turn_id": np.array([turn_id], dtype=np.int32),
            "meta.duplex_epoch": np.array([0], dtype=np.int32),
        },
    )


@pytest.mark.parametrize(
    (
        "finished",
        "tts_is_last_chunk",
        "turn_end",
        "consume_audio_first",
        "expected_end_of_turn",
        "expected_audio",
    ),
    [
        (False, True, False, False, False, "wav-24000"),
        (True, True, False, True, False, ""),
        (True, True, True, True, True, ""),
    ],
    ids=[
        "last-audio-batch-does-not-end-response",
        "scheduler-eos-only-ends-tts-segment",
        "explicit-turn-end-metadata-ends-response",
    ],
)
def test_duplex_auto_response_tts_boundary(
    finished: bool,
    tts_is_last_chunk: bool,
    turn_end: bool,
    consume_audio_first: bool,
    expected_end_of_turn: bool,
    expected_audio: str,
):
    data_plane = _test_data_plane()
    request_id = "duplex-sid-tts-boundary-e0-stage0"
    session = DuplexSession(
        session_id="sid-tts-boundary",
        config=DuplexSessionConfig(extra_body={"auto_response": True}),
    )
    session.bind_response_turn(0)
    if consume_audio_first:
        data_plane.slice_cumulative_audio(request_id, np.zeros(24000, dtype=np.float32))
    output = _duplex_tts_output(
        request_id=request_id,
        samples=24000,
        finished=finished,
        tts_is_last_chunk=tts_is_last_chunk,
        turn_end=turn_end,
        token_ids=[151645] if finished else None,
    )

    results = _project_data_plane(data_plane, {"data_plane_outputs": [output]}, session=session)

    assert len(results) == 1
    boundary = results[0]
    assert boundary["stage_role"] == "tts"
    assert boundary["end_of_turn"] is expected_end_of_turn
    assert boundary["abort_data_plane_request"] is True
    assert boundary["audio_data"] == expected_audio


def test_duplex_auto_response_tts_scheduler_eos_fallback():
    data_plane = _test_data_plane()
    request_id = "duplex-sid-tts-eos-no-profile-e0-stage0"
    session = DuplexSession(
        session_id="sid-tts-eos-no-profile",
        config=DuplexSessionConfig(extra_body={"auto_response": True}),
    )
    session.bind_response_turn(0)
    data_plane.slice_cumulative_audio(request_id, np.zeros(24000, dtype=np.float32))
    output = _duplex_tts_output(
        request_id=request_id,
        samples=24000,
        finished=True,
        tts_is_last_chunk=False,
        token_ids=[151645],
    )

    results = _project_data_plane(data_plane, {"data_plane_outputs": [output]}, session=session)

    assert len(results) == 1
    assert results[0]["stage_role"] == "tts"
    assert results[0]["abort_data_plane_request"] is True


def test_duplex_turn_end_is_not_swallowed_by_finished_segment_fallback():
    data_plane = _test_data_plane()
    request_id = "duplex-sid-turn-end-after-segment-e0-stage0"
    session = DuplexSession(
        session_id="sid-turn-end-after-segment",
        config=DuplexSessionConfig(extra_body={"auto_response": True}),
    )
    session.begin_response(turn_id=0)
    session.bind_request(request_id)
    _project_data_plane(
        data_plane,
        {
            "data_plane_outputs": [
                _duplex_tts_output(
                    request_id=request_id,
                    samples=24000,
                    finished=False,
                    tts_is_last_chunk=True,
                )
            ]
        },
        session=session,
    )
    output = _duplex_tts_output(
        request_id=request_id,
        samples=24000,
        finished=True,
        text="",
        tts_is_last_chunk=False,
        turn_end=True,
        token_ids=[151645],
    )

    results = _project_data_plane(data_plane, {"data_plane_outputs": [output]}, session=session)

    assert len(results) == 1
    assert results[0]["stage_role"] == "tts"
    assert results[0]["end_of_turn"] is True

    duplicate = _project_data_plane(data_plane, {"data_plane_outputs": [output]}, session=session)

    assert not any(result.get("end_of_turn") is True for result in duplicate)


def test_duplex_auto_response_discards_terminal_only_audio_before_response_creation():
    data_plane = _test_data_plane()
    request_id = "duplex-sid-terminal-only-audio-e0-stage0"
    session = DuplexSession(
        session_id="sid-terminal-only-audio",
        config=DuplexSessionConfig(extra_body={"auto_response": True}),
    )
    session.bind_request(request_id)
    first_audio = _duplex_tts_output(
        request_id=request_id,
        samples=3840,
        finished=False,
        text="",
    )

    assert _project_data_plane(data_plane, {"data_plane_outputs": [first_audio]}, session=session) == []

    terminal = _duplex_tts_output(
        request_id=request_id,
        samples=3840,
        finished=True,
        text="",
        tts_is_last_chunk=True,
        turn_end=True,
        token_ids=[151645],
    )
    results = _project_data_plane(data_plane, {"data_plane_outputs": [terminal]}, session=session)

    assert len(results) == 1
    assert results[0]["audio_data"] == ""
    assert results[0]["end_of_turn"] is True
    assert session.active_request_id == request_id
    assert not data_plane.is_terminal(request_id)


def test_duplex_auto_response_releases_buffered_audio_when_transcript_arrives():
    data_plane = _test_data_plane()
    request_id = "duplex-sid-delayed-transcript-e0-stage0"
    session = DuplexSession(
        session_id="sid-delayed-transcript",
        config=DuplexSessionConfig(extra_body={"auto_response": True}),
    )
    audio_before_text = _duplex_tts_output(
        request_id=request_id,
        samples=100,
        finished=False,
        text="",
    )

    assert _project_data_plane(data_plane, {"data_plane_outputs": [audio_before_text]}, session=session) == []

    audio_with_text = _duplex_tts_output(
        request_id=request_id,
        samples=200,
        finished=False,
        text="hello",
    )
    results = _project_data_plane(data_plane, {"data_plane_outputs": [audio_with_text]}, session=session)

    assert [result["audio_data"] for result in results] == ["wav-100", "wav-100"]
    assert [result["text"] for result in results] == ["", "hello"]


def test_duplex_auto_response_releases_buffered_audio_on_text_only_terminal():
    data_plane = _test_data_plane()
    request_id = "duplex-sid-delayed-terminal-text-e0-stage0"
    session = DuplexSession(
        session_id="sid-delayed-terminal-text",
        config=DuplexSessionConfig(extra_body={"auto_response": True}),
    )
    audio_before_text = _duplex_tts_output(
        request_id=request_id,
        samples=100,
        finished=False,
        text="",
        turn_id=1,
    )

    assert _project_data_plane(data_plane, {"data_plane_outputs": [audio_before_text]}, session=session) == []

    textless_turn_terminal = _duplex_tts_output(
        request_id=request_id,
        samples=100,
        finished=True,
        text="",
        tts_is_last_chunk=True,
        turn_end=True,
        turn_id=1,
        token_ids=[151645],
    )
    terminal_results = list(
        data_plane.project(
            {"data_plane_outputs": [textless_turn_terminal]},
            context=_data_plane_context(session),
        )
    )

    assert len(terminal_results) == 1
    assert terminal_results[0]["audio_data"] == ""
    assert terminal_results[0]["end_of_turn"] is True

    text_only_terminal = SimpleNamespace(
        request_id=request_id,
        finished=True,
        outputs=[SimpleNamespace(text="hello", token_ids=[151645], multimodal_output={})],
        multimodal_output={
            "sr": 24000,
            "meta.tts_is_last_chunk": np.array([1], dtype=np.int32),
            "meta.turn_end": np.array([1], dtype=np.int32),
            "meta.duplex_turn_id": np.array([2], dtype=np.int32),
            "meta.duplex_epoch": np.array([0], dtype=np.int32),
        },
    )
    results = _project_data_plane(data_plane, {"data_plane_outputs": [text_only_terminal]}, session=session)

    assert len(results) == 1
    assert results[0]["audio_data"] == "wav-100"
    assert results[0]["text"] == "hello"
    assert results[0]["end_of_turn"] is True
    assert results[0]["abort_data_plane_request"] is True
    assert not data_plane.has_pending_audio(request_id)


@pytest.mark.asyncio
async def test_native_append_propagates_current_turn_fence_to_engine():
    engine = FakeEngineClient()
    handler = OmniDuplexSessionHandler(chat_service=FakeChatService(engine))
    session = DuplexSession(
        session_id="sid-fenced-append",
        config=DuplexSessionConfig(extra_body={"auto_response": True}),
    )
    session.capabilities = DuplexCapabilities.minicpmo45_native()
    session.turn_id = 2
    session.begin_response(turn_id=2)
    ws = TimedWebSocket()

    await handler._append_runtime_input(
        session,
        {"duplex_turn_id": 2, "audio": "", "format": "pcm_f32le"},
        final=True,
        send_json=ws.send_json,
        mode="append_audio_chunk",
        expected_epoch=0,
    )

    assert engine.appended_fences == [DuplexFence("sid-fenced-append", epoch=0, turn_id=2)]


@pytest.mark.asyncio
async def test_minicpmo_auto_response_tts_segment_boundary_appends_silence_unit():
    request_id = "duplex-sid-segment-boundary-e0-stage0"
    engine = FakeEngineClient()
    handler = OmniDuplexSessionHandler(
        chat_service=FakeChatService(engine),
        config_timeout_s=0.1,
        idle_timeout_s=1,
    )
    session = DuplexSession(
        session_id="sid-segment-boundary",
        config=DuplexSessionConfig(extra_body={"auto_response": True}),
    )
    session.capabilities = DuplexCapabilities.minicpmo45_native()
    session.begin_response(turn_id=0)
    session.turn_id = 1
    session.bind_request(request_id)
    session.mark_audio_sent(100)
    _install_direct_silence_scheduler(handler, session)
    ws = TimedWebSocket()

    await handler._send_one_native_duplex_event(
        ws.send_json,
        {
            "supported": True,
            "stage_role": "tts",
            "is_listen": False,
            "data_plane_request_id": request_id,
            "text": "",
            "audio_data": "",
            "audio_format": "pcm16",
            "end_of_turn": False,
            "abort_data_plane_request": True,
            "uses_model_runner_scheduler": True,
            "runner_kv_backed": True,
            "runtime_impl": "scheduler_data_plane",
        },
        session=session,
        expected_epoch=session.epoch,
    )
    await asyncio.sleep(0.01)

    assert len(engine.appended) == 1
    _, mode, payload, final = engine.appended[0]
    assert mode == "append_audio_chunk"
    assert payload["format"] == "pcm_f32le"
    assert payload["duplex_turn_id"] == 0
    assert "force_speak" not in payload
    assert final is False
    assert session.active_response_id is not None
    assert "response.done" not in ws.sent_types()


@pytest.mark.asyncio
async def test_minicpmo_auto_response_pre_response_tts_boundary_continues_model_turn():
    request_id = "duplex-sid-pre-response-boundary-e0-stage0"
    engine = FakeEngineClient()
    handler = OmniDuplexSessionHandler(chat_service=FakeChatService(engine))
    session = DuplexSession(
        session_id="sid-pre-response-boundary",
        config=DuplexSessionConfig(extra_body={"auto_response": True}),
    )
    session.capabilities = DuplexCapabilities.minicpmo45_native()
    session.bind_request(request_id)
    _install_direct_silence_scheduler(handler, session)
    ws = TimedWebSocket()

    close_reason, emitted = await handler._send_one_native_duplex_event(
        ws.send_json,
        {
            "supported": True,
            "stage_role": "tts",
            "is_listen": False,
            "data_plane_request_id": request_id,
            "model_turn_id": session.turn_id,
            "text": "",
            "audio_data": "",
            "audio_format": "pcm16",
            "end_of_turn": False,
            "abort_data_plane_request": True,
            "uses_model_runner_scheduler": True,
            "runner_kv_backed": True,
            "runtime_impl": "scheduler_data_plane",
        },
        session=session,
        expected_epoch=session.epoch,
    )
    await asyncio.sleep(0.01)

    assert close_reason is None
    assert emitted is False
    assert session.active_response_id is None
    assert ws.sent == []
    assert len(engine.appended) == 1
    _, mode, payload, final = engine.appended[0]
    assert mode == "append_audio_chunk"
    assert payload["duplex_turn_id"] == session.turn_id
    assert final is False


@pytest.mark.asyncio
async def test_minicpmo_pre_response_continuation_drops_after_model_turn_ends():
    request_id = "duplex-sid-stale-pre-response-boundary-e0-stage0"
    engine = FakeEngineClient()
    handler = OmniDuplexSessionHandler(chat_service=FakeChatService(engine))
    session = DuplexSession(
        session_id="sid-stale-pre-response-boundary",
        config=DuplexSessionConfig(extra_body={"auto_response": True}),
    )
    session.capabilities = DuplexCapabilities.minicpmo45_native()
    session.bind_request(request_id)
    model_turn_id = session.turn_id

    native = handler._minicpmo_session_state(session)

    async def _stale_before_append(payload: object, **kwargs: Any) -> bool:
        session.complete_model_turn(model_turn_id)
        if handler._native_silence_continuation_is_stale(
            session,
            request_id=kwargs["request_id"],
            response_id=kwargs["response_id"],
            response_owned=kwargs["response_owned"],
            expected_epoch=kwargs["expected_epoch"],
            expected_incarnation=kwargs["expected_incarnation"],
            expected_model_turn_id=kwargs["expected_model_turn_id"],
        ):
            return False
        append_ok, _ = await handler._append_runtime_input(
            session,
            payload,
            final=False,
            send_json=kwargs["send_json"],
            mode="append_audio_chunk",
            expected_epoch=kwargs["expected_epoch"],
        )
        return append_ok

    native.silence_continuation_scheduler = _stale_before_append

    await handler._maybe_continue_native_response(
        TimedWebSocket().send_json,
        session=session,
        expected_epoch=session.epoch,
        expected_model_turn_id=model_turn_id,
    )
    await asyncio.sleep(0.01)

    assert engine.appended == []


@pytest.mark.asyncio
async def test_minicpmo_auto_response_continuation_has_no_semantic_unit_cap():
    request_id = "duplex-sid-long-response-e0-stage0"
    engine = FakeEngineClient()
    handler = OmniDuplexSessionHandler(chat_service=FakeChatService(engine))
    session = DuplexSession(
        session_id="sid-long-response",
        config=DuplexSessionConfig(extra_body={"auto_response": True}),
    )
    session.capabilities = DuplexCapabilities.minicpmo45_native()
    session.begin_response(turn_id=0)
    session.turn_id = 1
    session.bind_request(request_id)
    session.mark_audio_sent(100)
    _install_direct_silence_scheduler(handler, session)
    ws = TimedWebSocket()

    for _ in range(9):
        await handler._maybe_continue_native_response(
            ws.send_json,
            session=session,
            expected_epoch=session.epoch,
        )
    await asyncio.sleep(0.02)

    assert len(engine.appended) == 9
    assert all(payload["duplex_turn_id"] == 0 for _, _, payload, _ in engine.appended)
    assert all("force_speak" not in payload for _, _, payload, _ in engine.appended)


@pytest.mark.asyncio
async def test_minicpmo_auto_response_pre_speak_listen_continues_same_response():
    request_id = "duplex-sid-pre-speak-listen-e0-stage0"
    engine = FakeEngineClient()
    handler = OmniDuplexSessionHandler(chat_service=FakeChatService(engine))
    session = DuplexSession(
        session_id="sid-pre-speak-listen",
        config=DuplexSessionConfig(extra_body={"auto_response": True}),
    )
    session.capabilities = DuplexCapabilities.minicpmo45_native()
    response_id = session.begin_response(turn_id=0)
    session.turn_id = 1
    session.bind_request(request_id)
    _install_direct_silence_scheduler(handler, session)
    ws = TimedWebSocket()

    await handler._send_one_native_duplex_event(
        ws.send_json,
        {
            "supported": True,
            "stage_role": "llm",
            "is_listen": True,
            "model_listen": True,
            "data_plane_request_id": request_id,
            "end_of_turn": False,
            "uses_model_runner_scheduler": True,
            "runner_kv_backed": True,
            "runtime_impl": "scheduler_data_plane",
        },
        session=session,
        expected_epoch=session.epoch,
    )
    await asyncio.sleep(0.01)

    assert len(engine.appended) == 1
    assert session.active_response_id == response_id
    assert session.active_request_id == request_id
    assert "response.listen" not in ws.sent_types()
    assert "response.done" not in ws.sent_types()


@pytest.mark.asyncio
async def test_minicpmo_auto_response_listen_before_response_keeps_resumable_request():
    request_id = "duplex-sid-listen-before-response-e0-stage0"
    handler = OmniDuplexSessionHandler(chat_service=FakeChatService(FakeEngineClient()))
    session = DuplexSession(
        session_id="sid-listen-before-response",
        config=DuplexSessionConfig(extra_body={"auto_response": True}),
    )
    session.capabilities = DuplexCapabilities.minicpmo45_native()
    session.bind_request(request_id)
    ws = TimedWebSocket()

    close_reason, emitted = await handler._send_one_native_duplex_event(
        ws.send_json,
        {
            "supported": True,
            "stage_role": "llm",
            "is_listen": True,
            "model_listen": True,
            "data_plane_request_id": request_id,
            "end_of_turn": False,
            "uses_model_runner_scheduler": True,
            "runner_kv_backed": True,
            "runtime_impl": "scheduler_data_plane",
        },
        session=session,
        expected_epoch=session.epoch,
    )

    assert close_reason is None
    assert emitted is True
    assert session.active_request_id == request_id
    assert not handler._minicpmo_data_plane.is_terminal(request_id)
    assert ws.sent_types().count("response.listen") == 1


@pytest.mark.asyncio
async def test_minicpmo_auto_response_listen_without_response_does_not_defer_commit():
    session_id = "sid-listen-before-response-commit"
    request_id = duplex_resource_request_id(DuplexFence(session_id), "stage0")
    inner_output = SimpleNamespace(
        outputs=[
            SimpleNamespace(
                multimodal_output={"special_token_ids": {"listen_token_id": 151705}},
            )
        ]
    )
    decision = MiniCPMO45DuplexRuntimeExtension().decide_output(
        stage_id=0,
        final_stage_id=1,
        segment_finished=True,
        segment_token_ids=(151705,),
        segment_output_metadata={"special_token_ids": {"listen_token_id": 151705}},
        output=inner_output,
    )
    assert decision is not None
    listen_output = attach_duplex_output_decision(
        OmniRequestOutput(
            request_id=request_id,
            finished=True,
            stage_id=0,
            final_output_type=decision.final_output_type,
            request_output=inner_output,
        ),
        decision,
    )
    append_result = {
        "operation": "append",
        "session_id": session_id,
        "ok": True,
        "unsupported_count": 0,
        "error_count": 0,
        "stage_results": [
            {
                "stage_id": 0,
                "replica_id": 0,
                "result": {
                    "supported": True,
                    "implementation_level": "model_native_duplex",
                    "data_plane_append": True,
                    "request_id": request_id,
                    "response_stage_id": 1,
                },
            }
        ],
        "data_plane_outputs": [listen_output],
    }
    engine = FakeEngineClient(append_result=append_result)
    handler = OmniDuplexSessionHandler(
        chat_service=FakeChatService(engine),
        config_timeout_s=0.1,
        idle_timeout_s=1,
    )
    listen_count = 0

    def append_residual_after_listen(ws: TimedWebSocket, data: dict[str, Any]) -> None:
        nonlocal listen_count
        if data.get("type") == "response.listen":
            listen_count += 1
            if listen_count == 1:
                ws.put(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": _pcm_f32_b64(3_200),
                        "format": "pcm_f32le",
                        "sample_rate_hz": 16_000,
                        "duration_ms": 200,
                        "is_speech": True,
                    }
                )
                ws.put({"type": "input_audio_buffer.commit", "final": True})
        elif data.get("type") == "input_audio_buffer.committed":
            asyncio.get_running_loop().call_later(0.05, ws.put, {"type": "session.close"})

    ws = TimedWebSocket(on_send=append_residual_after_listen, receive_timeout_s=2)
    create = _native_realtime_session_update(session_id)
    create["session"]["extra_body"]["auto_response"] = True
    ws.put(create)
    ws.put(
        {
            "type": "input_audio_buffer.append",
            "audio": _pcm_f32_b64(16_000),
            "format": "pcm_f32le",
            "sample_rate_hz": 16_000,
            "duration_ms": 1_000,
            "is_speech": True,
        }
    )

    await handler.handle_realtime_session(ws)

    committed = [event for event in ws.sent if event.get("type") == "input_audio_buffer.committed"]
    assert len(engine.appended) == 2
    assert engine.appended[1][3] is True
    assert len(committed) == 1
    assert committed[0].get("event", {}).get("overlap_deferred") is not True
    assert committed[0].get("event", {}).get("response_create_deferred") is not True
    assert "response.created" not in ws.sent_types()


@pytest.mark.asyncio
async def test_minicpmo_pcm_f32_residual_commit_releases_pending_input_bytes():
    session_id = "sid-pcm-residual-byte-accounting"
    control_result = {
        "operation": "append",
        "session_id": session_id,
        "ok": True,
        "unsupported_count": 0,
        "error_count": 0,
        "stage_results": [],
    }
    engine = FakeEngineClient(control_result=control_result)
    handler = OmniDuplexSessionHandler(
        chat_service=FakeChatService(engine),
        config_timeout_s=0.1,
        idle_timeout_s=1,
    )
    ws = TimedWebSocket(receive_timeout_s=2)
    create = _native_realtime_session_update(session_id)
    create["session"]["extra_body"]["auto_response"] = True
    ws.put(create)
    handler_task = asyncio.create_task(handler.handle_realtime_session(ws))

    try:
        for _ in range(100):
            session = handler._registry.get(session_id)
            if session is not None and "session.created" in ws.sent_types():
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("native duplex session did not open")

        pending_bytes_after_turn: list[int] = []
        for turn_index in range(3):
            ws.put(
                {
                    "type": "input_audio_buffer.append",
                    "audio": _pcm_f32_b64(19_200),
                    "format": "pcm_f32le",
                    "sample_rate_hz": 16_000,
                    "duration_ms": 1_200,
                    "is_speech": True,
                }
            )
            ws.put({"type": "input_audio_buffer.commit", "final": True})
            expected_appends = (turn_index + 1) * 2
            for _ in range(100):
                if len(engine.appended) >= expected_appends:
                    await asyncio.sleep(0)
                    break
                await asyncio.sleep(0.01)
            else:
                pytest.fail(f"turn {turn_index} residual append did not complete")
            pending_bytes_after_turn.append(session.pending_input_bytes)
    finally:
        ws.put({"type": "session.close"})
        await asyncio.wait_for(handler_task, timeout=2)

    assert pending_bytes_after_turn == [0, 0, 0]
    assert not any(
        event.get("type") == "error"
        and isinstance(event.get("error"), dict)
        and event["error"].get("code") == "input_backpressure"
        for event in ws.sent
    )


@pytest.mark.asyncio
async def test_minicpmo_auto_response_post_speak_listen_continues_same_response():
    request_id = "duplex-sid-post-speak-listen-e0-stage0"
    engine = FakeEngineClient()
    handler = OmniDuplexSessionHandler(chat_service=FakeChatService(engine))
    session = DuplexSession(
        session_id="sid-post-speak-listen",
        config=DuplexSessionConfig(extra_body={"auto_response": True}),
    )
    session.capabilities = DuplexCapabilities.minicpmo45_native()
    response_id = session.begin_response(turn_id=0)
    session.turn_id = 1
    session.bind_request(request_id)
    session.mark_audio_sent(100)
    _install_direct_silence_scheduler(handler, session)
    ws = TimedWebSocket()

    await handler._send_one_native_duplex_event(
        ws.send_json,
        {
            "supported": True,
            "stage_role": "llm",
            "is_listen": True,
            "model_listen": True,
            "data_plane_request_id": request_id,
            "end_of_turn": False,
            "uses_model_runner_scheduler": True,
            "runner_kv_backed": True,
            "runtime_impl": "scheduler_data_plane",
        },
        session=session,
        expected_epoch=session.epoch,
    )
    await asyncio.sleep(0.01)

    assert len(engine.appended) == 1
    assert session.active_response_id == response_id
    assert session.active_request_id == request_id
    assert "response.listen" not in ws.sent_types()
    assert "response.done" not in ws.sent_types()


@pytest.mark.asyncio
async def test_minicpmo_auto_response_turn_end_preserves_resumable_request():
    request_id = "duplex-sid-turn-end-preserve-request-e0-stage0"
    handler = OmniDuplexSessionHandler(chat_service=FakeChatService(FakeEngineClient()))
    session = DuplexSession(
        session_id="sid-turn-end-preserve-request",
        config=DuplexSessionConfig(extra_body={"auto_response": True}),
    )
    session.capabilities = DuplexCapabilities.minicpmo45_native()
    session.begin_response(turn_id=0)
    session.bind_request(request_id)
    ws = TimedWebSocket()

    await handler._send_one_native_duplex_event(
        ws.send_json,
        {
            "supported": True,
            "stage_role": "tts",
            "is_listen": False,
            "data_plane_request_id": request_id,
            "text": "done",
            "audio_data": "audio",
            "audio_format": "pcm16",
            "audio_duration_ms": 100,
            "end_of_turn": True,
            "model_turn_id": 0,
            "uses_model_runner_scheduler": True,
            "runner_kv_backed": True,
            "runtime_impl": "scheduler_data_plane",
        },
        session=session,
        expected_epoch=session.epoch,
    )

    assert session.active_response_id is None
    assert session.active_request_id == request_id
    assert "response.done" in ws.sent_types()


@pytest.mark.asyncio
async def test_minicpmo_auto_response_empty_turn_end_emits_model_listen():
    request_id = "duplex-sid-empty-turn-end-e0-stage0"
    handler = OmniDuplexSessionHandler(chat_service=FakeChatService(FakeEngineClient()))
    session = DuplexSession(
        session_id="sid-empty-turn-end",
        config=DuplexSessionConfig(extra_body={"auto_response": True}),
    )
    session.capabilities = DuplexCapabilities.minicpmo45_native()
    session.bind_request(request_id)
    ws = TimedWebSocket()

    close_reason, emitted = await handler._send_one_native_duplex_event(
        ws.send_json,
        {
            "supported": True,
            "stage_role": "tts",
            "is_listen": False,
            "data_plane_request_id": request_id,
            "text": "",
            "audio_data": "",
            "audio_format": "pcm16",
            "end_of_turn": True,
            "abort_data_plane_request": True,
            "model_turn_id": 0,
            "uses_model_runner_scheduler": True,
            "runner_kv_backed": True,
            "runtime_impl": "scheduler_data_plane",
        },
        session=session,
        expected_epoch=session.epoch,
    )

    assert close_reason is None
    assert emitted is True
    assert session.turn_id == 1
    assert session.active_request_id == request_id
    assert session.active_response_id is None
    assert ws.sent_types() == ["response.listen"]
    assert ws.sent[0]["model_listen"] is True
    assert ws.sent[0]["reason"] == "model_turn_completed_without_output"


@pytest.mark.asyncio
async def test_minicpmo_auto_response_drops_late_audio_from_completed_model_turn():
    request_id = "duplex-sid-late-completed-turn-e0-stage0"
    handler = OmniDuplexSessionHandler(chat_service=FakeChatService(FakeEngineClient()))
    session = DuplexSession(
        session_id="sid-late-completed-turn",
        config=DuplexSessionConfig(extra_body={"auto_response": True}),
    )
    session.capabilities = DuplexCapabilities.minicpmo45_native()
    session.bind_request(request_id)
    session.complete_model_turn(0)
    ws = TimedWebSocket()

    close_reason, emitted = await handler._send_one_native_duplex_event(
        ws.send_json,
        {
            "supported": True,
            "stage_role": "tts",
            "is_listen": False,
            "data_plane_request_id": request_id,
            "text": "late duplicate",
            "audio_data": "late-audio",
            "audio_format": "pcm16",
            "audio_duration_ms": 100,
            "end_of_turn": True,
            "model_turn_id": 0,
            "uses_model_runner_scheduler": True,
            "runner_kv_backed": True,
            "runtime_impl": "scheduler_data_plane",
        },
        session=session,
        expected_epoch=session.epoch,
    )

    assert close_reason is None
    assert emitted is False
    assert session.turn_id == 1
    assert session.active_response_id is None
    assert "response.created" not in ws.sent_types()
    assert "response.done" not in ws.sent_types()


def test_duplex_data_plane_drops_stale_epoch_audio_after_barge_in():
    class _ChatService:
        model_config = _ModelConfig()

        def create_audio(self, audio_obj):
            return SimpleNamespace(audio_data=f"wav-{int(audio_obj.audio_tensor.shape[0])}")

    data_plane = _test_data_plane()
    session = DuplexSession(
        session_id="sid-auto-stale-epoch",
        config=DuplexSessionConfig(extra_body={"auto_response": True}),
    )
    session.turn_id = 1
    session.epoch = 2
    output = SimpleNamespace(
        request_id="duplex-sid-auto-stale-epoch-stage0",
        finished=False,
        outputs=[SimpleNamespace(text="old epoch", multimodal_output={})],
        multimodal_output={
            "audio": np.zeros(24000, dtype=np.float32),
            "sr": 24000,
            "meta.duplex_epoch": np.array([1], dtype=np.int32),
            "meta.duplex_turn_id": np.array([1], dtype=np.int32),
            "meta.llm_output_text_utf8": np.frombuffer("旧轮".encode(), dtype=np.uint8),
        },
    )

    assert list(data_plane.project_output(output, context=_data_plane_context(session))) == []


def test_duplex_data_plane_text_delta_preserves_repeated_suffix_growth():
    data_plane = _test_data_plane()
    request_id = "duplex-sid-native-repeated-word-e0-stage0"

    assert data_plane.segment_text_delta(request_id, "好的") == "好的"
    assert data_plane.segment_text_delta(request_id, "好的好的") == "好的"


def test_duplex_data_plane_text_delta_appends_distinct_non_prefix_segments():
    data_plane = _test_data_plane()
    request_id = "duplex-sid-native-rewrite-e0-stage0"

    deltas = [
        data_plane.segment_text_delta(request_id, "It's Canberra."),
        data_plane.segment_text_delta(request_id, " Next question."),
    ]

    assert deltas == ["It's Canberra.", " Next question."]
    assert "".join(deltas) == "It's Canberra. Next question."


def test_duplex_data_plane_close_session_removes_encoded_request_state():
    data_plane = _test_data_plane()
    target_request_id = duplex_resource_request_id(
        DuplexFence("sid-to-close", incarnation=1, epoch=2),
        "stage0",
    )
    other_request_id = duplex_resource_request_id(
        DuplexFence("sid-to-keep", incarnation=1, epoch=2),
        "stage0",
    )
    data_plane.begin_request(target_request_id)
    data_plane.begin_request(other_request_id)

    data_plane.close_session("sid-to-close")

    assert data_plane.has_request(target_request_id) is False
    assert data_plane.has_request(other_request_id) is True


def test_duplex_auto_response_segment_complete_keeps_data_plane_request_open():
    import numpy as np

    class _ChatService:
        model_config = _ModelConfig()

        def create_audio(self, audio_obj):
            return SimpleNamespace(audio_data=f"wav-{int(audio_obj.audio_tensor.shape[0])}")

    data_plane = _test_data_plane()
    request_id = "duplex-duplex-sess-stage0"
    session = DuplexSession(
        session_id="sid-auto-respond",
        config=DuplexSessionConfig(extra_body={"auto_response": True}),
    )
    session.begin_response()
    session.mark_audio_sent(1280)
    data_plane.slice_cumulative_audio(request_id, np.zeros(30720, dtype=np.float32))

    output = SimpleNamespace(
        request_id=request_id,
        finished=True,
        outputs=[SimpleNamespace(text="hello", multimodal_output={})],
        multimodal_output={
            "audio": np.zeros(30720, dtype=np.float32),
            "sr": 24000,
        },
    )

    native_results = _project_data_plane(data_plane, {"data_plane_outputs": [output]}, session=session)

    assert len(native_results) == 1
    assert native_results[0]["is_listen"] is True
    assert native_results[0]["reason"] == "auto_response_segment_complete"
    assert native_results[0]["data_plane_request_id"] == request_id
    assert "abort_data_plane_request" not in native_results[0]


def test_duplex_auto_response_prior_playback_does_not_abort_before_current_response():
    data_plane = _test_data_plane()
    request_id = "duplex-duplex-sess-stage0"
    session = DuplexSession(
        session_id="sid-auto-respond-next-turn",
        config=DuplexSessionConfig(extra_body={"auto_response": True}),
    )
    session.mark_audio_sent(1280)
    session.acknowledge_playback(played_ms=1280, committed_ms=1280)

    output = SimpleNamespace(
        request_id=request_id,
        finished=True,
        outputs=[SimpleNamespace(text="", multimodal_output={})],
        multimodal_output={},
    )

    native_results = _project_data_plane(data_plane, {"data_plane_outputs": [output]}, session=session)

    assert native_results == []


def test_duplex_auto_response_text_only_flush_does_not_consume_audio_transcript():
    import numpy as np

    data_plane = _test_data_plane()
    request_id = "duplex-duplex-sess-stage0"
    session = DuplexSession(
        session_id="sid-auto-respond-delayed-audio",
        config=DuplexSessionConfig(extra_body={"auto_response": True}),
    )
    text = "你好呀，有什莫问的吗？"

    text_only_output = SimpleNamespace(
        request_id=request_id,
        finished=False,
        outputs=[SimpleNamespace(text=text, multimodal_output={})],
        multimodal_output={"sr": 24000},
    )

    assert _project_data_plane(data_plane, {"data_plane_outputs": [text_only_output]}, session=session) == []

    audio_output = SimpleNamespace(
        request_id=request_id,
        finished=False,
        outputs=[SimpleNamespace(text=text, multimodal_output={})],
        multimodal_output={
            "audio": np.zeros(24000, dtype=np.float32),
            "sr": 24000,
        },
    )

    native_results = _project_data_plane(data_plane, {"data_plane_outputs": [audio_output]}, session=session)

    assert len(native_results) == 1
    assert native_results[0]["audio_data"].startswith("wav-")
    assert native_results[0]["text"] == text


@pytest.mark.asyncio
async def test_duplex_auto_response_empty_terminal_does_not_create_empty_response():
    handler = OmniDuplexSessionHandler(chat_service=FakeChatService(FakeEngineClient()))
    request_id = "duplex-sid-native-empty-terminal-e0-stage0"
    session = DuplexSession(
        session_id="sid-native-empty-terminal",
        config=DuplexSessionConfig(extra_body={"auto_response": True}),
    )
    session.mark_audio_sent(1280)
    sent: list[dict[str, Any]] = []

    async def send_json(payload: dict[str, Any]) -> None:
        sent.append(payload)

    output = SimpleNamespace(
        request_id=request_id,
        finished=False,
        outputs=[SimpleNamespace(text="hello", multimodal_output={})],
        multimodal_output={
            "sr": 24000,
            "meta": {"turn_end": True, "duplex_turn_id": 0},
        },
    )

    native_results = list(
        handler._minicpmo_data_plane.project(
            {"data_plane_outputs": [output]},
            context=_data_plane_context(session),
        )
    )
    assert len(native_results) == 1
    assert native_results[0]["end_of_turn"] is True
    assert native_results[0]["model_turn_id"] == 0

    for native_result in native_results:
        await handler._send_one_native_duplex_event(send_json, native_result, session=session)

    assert "response.created" not in [m.get("type") for m in sent]
    assert "response.done" not in [m.get("type") for m in sent]
    assert session.turn_id == 1
    assert not handler._minicpmo_data_plane.is_terminal(request_id)


@pytest.mark.asyncio
async def test_minicpmo_native_duplex_session_close_cleans_auto_response_data_plane_state():
    session_id = "sid-native-cleanup"
    request_id = f"duplex-{session_id}-e0-stage0"
    engine = FakeEngineClient()
    chat_service = FakeChatService(engine)
    handler = OmniDuplexSessionHandler(chat_service=chat_service, config_timeout_s=0.1, idle_timeout_s=1)
    handler._minicpmo_data_plane.slice_cumulative_audio(request_id, np.zeros(24000, dtype=np.float32))
    handler._minicpmo_data_plane.segment_text_delta(request_id, "hello")
    handler._minicpmo_data_plane.mark_terminal(request_id)
    native = MiniCPMO45ServingSessionState()
    handler._minicpmo_sessions[session_id] = native

    event = _native_session_create(session_id)
    event["session"]["extra_body"]["auto_response"] = True
    ws = TimedWebSocket()
    ws.put(event)
    ws.put({"type": "session.close"})

    await handler.handle_session(ws)

    assert not handler._minicpmo_data_plane.has_request(request_id)
    assert session_id not in handler._minicpmo_sessions


@pytest.mark.asyncio
async def test_duplex_chat_audio_stream_uses_output_audio_delta_event():
    engine = FakeEngineClient()
    chat_service = FakeChatService(engine)
    handler = OmniDuplexSessionHandler(chat_service=chat_service, config_timeout_s=0.1, idle_timeout_s=1)
    session = DuplexSession(session_id="sid-chat-audio", config=DuplexSessionConfig())
    response_id = session.begin_response()
    sent: list[dict[str, Any]] = []

    async def send_json(data: dict[str, Any]) -> None:
        sent.append(data)

    await handler._emit_chat_payload(
        session,
        {
            "modality": "audio",
            "choices": [
                {
                    "delta": {
                        "content": "AAAA",
                    }
                }
            ],
        },
        session.epoch,
        response_id,
        send_json,
    )

    assert sent == [
        {
            "type": "response.output_audio.delta",
            "session_id": "sid-chat-audio",
            "response_id": response_id,
            "epoch": 0,
            "audio": "AAAA",
            "format": "wav",
        }
    ]


@pytest.mark.asyncio
async def test_duplex_chat_rejects_unknown_output_modality():
    handler = OmniDuplexSessionHandler(
        chat_service=FakeChatService(FakeEngineClient()),
        config_timeout_s=0.1,
        idle_timeout_s=1,
    )
    session = DuplexSession(session_id="sid-chat-unknown", config=DuplexSessionConfig())
    response_id = session.begin_response()
    sent: list[dict[str, Any]] = []

    async def send_json(data: dict[str, Any]) -> None:
        sent.append(data)

    await handler._emit_chat_payload(
        session,
        {
            "modality": "video",
            "choices": [{"delta": {"content": "not-text"}}],
        },
        session.epoch,
        response_id,
        send_json,
    )

    assert sent == [
        {
            "type": "error",
            "session_id": session.session_id,
            "response_id": response_id,
            "epoch": session.epoch,
            "code": "unsupported_response_modality",
            "error": "Unsupported chat response modality: video",
        }
    ]
    assert session.assistant_text_buffer == ()


@pytest.mark.asyncio
async def test_minicpmo_model_name_does_not_auto_enable_experimental_native_duplex():
    event = _session_create("sid-minicpmo-default-protocol")
    event["session"]["model"] = "openbmb/MiniCPM-o-4_5"
    engine = FakeEngineClient()
    chat_service = FakeChatService(engine)
    handler = OmniDuplexSessionHandler(chat_service=chat_service, config_timeout_s=0.1, idle_timeout_s=1)
    ws = TimedWebSocket()
    ws.put(event)
    ws.put({"type": "session.close"})

    await handler.handle_session(ws)

    created = next(m for m in ws.sent if m.get("type") == "session.created")
    capabilities = created["session"]["capabilities"]
    assert capabilities["implementation_level"] == "serving_session_adapter"
    assert capabilities["supports_input_append"] is False


@pytest.mark.asyncio
async def test_duplex_handler_aborts_current_chat_request_id_on_barge_in():
    engine = FakeEngineClient()
    chat_service = FakeChatService(engine)
    handler = OmniDuplexSessionHandler(chat_service=chat_service, config_timeout_s=0.1, idle_timeout_s=1)

    def on_send(ws: TimedWebSocket, data: dict[str, Any]) -> None:
        if data.get("type") == "response.created":
            ws.put({"type": "input.cancel", "reason": "test_barge_in"})

    ws = TimedWebSocket(on_send=on_send)
    ws.put(_session_create("sid-a"))
    ws.put({"type": "input.text.append", "text": "hello"})
    ws.put({"type": "input.commit"})

    await handler.handle_session(ws)

    assert ws.accepted
    assert "response.created" in ws.sent_types()
    assert "audio.cancelled" in ws.sent_types()
    assert chat_service.seen_request_ids == ["duplex-sid-a-0-1"]
    assert engine.aborted == ["chatcmpl-duplex-sid-a-0-1"]
    assert engine.internal_abort_batches == [["chatcmpl-duplex-sid-a-0-1"]]


@pytest.mark.asyncio
async def test_duplex_handler_runtime_open_failure_is_reported_to_client():
    engine = FakeEngineClient(fail_open=True)
    chat_service = FakeChatService(engine)
    handler = OmniDuplexSessionHandler(chat_service=chat_service, config_timeout_s=0.1, idle_timeout_s=1)
    ws = TimedWebSocket()
    ws.put(_session_create("sid-open-fail"))
    ws.put({"type": "session.close"})

    await handler.handle_session(ws)

    assert ws.sent_types() == ["error"]
    assert ws.sent[0]["code"] == "runtime_open_failed"
    assert engine.opened == []


@pytest.mark.asyncio
async def test_duplex_handler_preserves_typed_admission_error():
    class AdmissionRejectingEngine(FakeEngineClient):
        async def open_duplex_session_async(self, session_id: str, **kwargs):
            del kwargs
            raise DuplexControlRequestError(
                {
                    "operation": "open",
                    "session_id": session_id,
                    "error": {
                        "code": "resource_exhausted",
                        "message": "duplex session capacity exhausted",
                        "retryable": True,
                    },
                    "accepted_fence": None,
                    "lease_generation": None,
                }
            )

    handler = OmniDuplexSessionHandler(
        chat_service=FakeChatService(AdmissionRejectingEngine()),
        config_timeout_s=0.1,
        idle_timeout_s=1,
    )
    ws = TimedWebSocket()
    ws.put(_session_create("sid-admission-rejected"))

    await handler.handle_session(ws)

    assert ws.sent_types() == ["error"]
    assert ws.sent[0]["code"] == "resource_exhausted"
    assert ws.sent[0]["retryable"] is True


@pytest.mark.asyncio
async def test_duplex_cancel_reports_playback_committed_cursor():
    engine = FakeEngineClient()
    chat_service = FakeChatService(engine)
    handler = OmniDuplexSessionHandler(chat_service=chat_service, config_timeout_s=0.1, idle_timeout_s=1)

    def on_send(ws: TimedWebSocket, data: dict[str, Any]) -> None:
        if data.get("type") == "response.created":
            ws.put({"type": "playback.ack", "played_ms": 1200, "committed_ms": 1000})
            ws.put({"type": "input.cancel", "reason": "test_barge_in"})

    ws = TimedWebSocket(on_send=on_send)
    ws.put(_session_create("sid-playback"))
    ws.put({"type": "input.text.append", "text": "hello"})
    ws.put({"type": "input.commit"})

    await handler.handle_session(ws)

    ack = next(m for m in ws.sent if m.get("type") == "playback.acknowledged")
    cancelled = next(m for m in ws.sent if m.get("type") == "audio.cancelled")
    assert ack["playback"]["played_ms"] == 1200
    assert ack["playback"]["committed_ms"] == 1000
    assert cancelled["committed_ms"] == 1000
    assert cancelled["epoch"] == 1


@pytest.mark.asyncio
async def test_playback_ack_response_id_selects_matching_pending_history_item():
    handler = OmniDuplexSessionHandler(
        chat_service=FakeChatService(FakeEngineClient()),
        config_timeout_s=0.1,
        idle_timeout_s=1,
    )
    session = DuplexSession(
        session_id="sid-playback-response-id",
        config=DuplexSessionConfig(),
    )
    first_item_id = "item_resp-first"
    second_item_id = "item_resp-second"
    session.stage_pending_history_item(
        first_item_id,
        {"role": "assistant", "content": "first response"},
    )
    session.stage_pending_history_item(
        second_item_id,
        {"role": "assistant", "content": "second response"},
    )
    ws = TimedWebSocket()

    await handler._handle_playback_ack(
        session,
        {
            "type": "playback.ack",
            "response_id": "resp-second",
            "played_ms": 1000,
            "committed_ms": 1000,
        },
        ws.send_json,
    )

    ack = next(message for message in ws.sent if message.get("type") == "playback.acknowledged")
    assert ack["item_id"] == second_item_id
    assert ack["history_committed"] is True
    assert first_item_id in session.pending_history_item_ids
    assert second_item_id not in session.pending_history_item_ids


@pytest.mark.asyncio
async def test_late_playback_ack_does_not_advance_new_response_cursor():
    handler = OmniDuplexSessionHandler(
        chat_service=FakeChatService(FakeEngineClient()),
        config_timeout_s=0.1,
        idle_timeout_s=1,
    )
    session = DuplexSession(
        session_id="sid-late-playback-ack",
        config=DuplexSessionConfig(),
    )
    first_response_id = session.begin_response()
    session.append_assistant_text("first response")
    session.mark_audio_sent(11480, text_chars=len("first response"))
    session.end_response(commit_text=False, preserve_request=True)
    session.register_history_item(f"item_{first_response_id}", None)

    second_response_id = session.begin_response()
    session.mark_audio_sent(2200, text_chars=len("second response"))
    ws = TimedWebSocket()

    await handler._handle_playback_ack(
        session,
        {
            "type": "playback.ack",
            "response_id": first_response_id,
            "item_id": f"item_{first_response_id}",
            "played_ms": 11480,
            "committed_ms": 11480,
        },
        ws.send_json,
    )

    assert session.active_response_id == second_response_id
    assert session.playback.as_dict() == {
        "generated_ms": 2200,
        "sent_ms": 2200,
        "played_ms": 0,
        "committed_ms": 0,
    }
    ack = next(message for message in ws.sent if message.get("type") == "playback.acknowledged")
    assert ack["playback"] == {
        "generated_ms": 11480,
        "sent_ms": 11480,
        "played_ms": 11480,
        "committed_ms": 11480,
    }
    assert first_response_id not in session.response_playbacks
    assert session.response_playbacks[second_response_id] == session.playback


@pytest.mark.asyncio
async def test_late_playback_ack_truncates_history_with_matching_response_cursor():
    handler = OmniDuplexSessionHandler(
        chat_service=FakeChatService(FakeEngineClient()),
        config_timeout_s=0.1,
        idle_timeout_s=1,
    )
    session = DuplexSession(
        session_id="sid-late-playback-history",
        config=DuplexSessionConfig(),
    )
    first_response_id = session.begin_response()
    session.append_assistant_text("abcdefghij")
    session.mark_audio_sent(1000)
    session.end_response(commit_text=False, preserve_request=True)
    session.register_history_item(f"item_{first_response_id}", None)

    session.begin_response()
    session.mark_audio_sent(10000)
    ws = TimedWebSocket()

    await handler._handle_playback_ack(
        session,
        {
            "type": "playback.ack",
            "response_id": first_response_id,
            "item_id": f"item_{first_response_id}",
            "played_ms": 500,
            "committed_ms": 500,
        },
        ws.send_json,
    )

    assert session.history[-1] == {"role": "assistant", "content": "abcde"}
    assert first_response_id in session.response_playbacks


@pytest.mark.asyncio
async def test_cancel_active_response_commits_only_played_assistant_history():
    engine = FakeEngineClient()
    chat_service = FakeChatService(engine)
    handler = OmniDuplexSessionHandler(chat_service=chat_service, config_timeout_s=0.1, idle_timeout_s=1)
    ws = TimedWebSocket()
    session = DuplexSession(
        session_id="sid-playback-partial",
        config=DuplexSessionConfig(playback_commit_policy=DuplexPlaybackCommitPolicy.COMMIT_ALL_ON_DONE.value),
    )
    response_id = session.begin_response()
    session.append_assistant_text("hello world")
    session.mark_audio_sent(1000, text_chars=len("hello world"))
    session.acknowledge_playback(played_ms=500, committed_ms=500)

    cancelled = await handler._cancel_active_response(session, None, ws.send_json, reason="barge_in")

    assert cancelled is True
    assert session.epoch == 1
    assert session.history == ({"role": "assistant", "content": "hello"},)
    assert session.history_item_ids[f"item_{response_id}"] == session.history[0]
    event = next(m for m in ws.sent if m.get("type") == "audio.cancelled")
    assert event["committed_ms"] == 500


@pytest.mark.asyncio
async def test_cancel_active_response_keeps_truncated_history_in_serving():
    engine = FakeEngineClient()
    chat_service = FakeChatService(engine)
    handler = OmniDuplexSessionHandler(chat_service=chat_service, config_timeout_s=0.1, idle_timeout_s=1)
    ws = TimedWebSocket()
    session = DuplexSession(
        session_id="sid-playback-runtime-rebuild",
        config=DuplexSessionConfig(playback_commit_policy=DuplexPlaybackCommitPolicy.COMMIT_ALL_ON_DONE.value),
    )
    response_id = session.begin_response()
    session.append_assistant_text("hello world")
    session.mark_audio_sent(1000, text_chars=len("hello world"))
    session.acknowledge_playback(played_ms=500, committed_ms=500)

    cancelled = await handler._cancel_active_response(
        session,
        None,
        ws.send_json,
        reason="barge_in",
    )

    assert cancelled is True
    assert session.history == ({"role": "assistant", "content": "hello"},)
    assert session.history_item_ids[f"item_{response_id}"] == session.history[0]
    assert engine.signals == []
    assert "error" not in ws.sent_types()


@pytest.mark.asyncio
async def test_cancel_active_native_data_plane_request_aborts_stage_request_id():
    engine = FakeEngineClient()
    chat_service = FakeChatService(engine)
    handler = OmniDuplexSessionHandler(chat_service=chat_service, config_timeout_s=0.1, idle_timeout_s=1)
    ws = TimedWebSocket()
    session = DuplexSession(session_id="sid-native-abort", config=DuplexSessionConfig())
    session.bind_request("duplex-sid-native-abort-e0-stage0-s1")

    cancelled = await handler._cancel_active_response(session, None, ws.send_json, reason="barge_in")

    assert cancelled is True
    assert engine.aborted == ["duplex-sid-native-abort-e0-stage0-s1"]
    assert engine.internal_abort_batches == [["duplex-sid-native-abort-e0-stage0-s1"]]
    assert "audio.cancelled" in ws.sent_types()


@pytest.mark.asyncio
async def test_duplex_handler_explicit_close_closes_runtime_once_with_client_reason():
    engine = FakeEngineClient()
    chat_service = FakeChatService(engine)
    handler = OmniDuplexSessionHandler(chat_service=chat_service, config_timeout_s=0.1, idle_timeout_s=1)
    ws = TimedWebSocket()
    ws.put(_session_create("sid-close"))
    ws.put({"type": "session.close"})

    await handler.handle_session(ws)

    assert ws.sent_types() == ["session.created", "session.closed"]
    assert engine.closed == [("sid-close", "session_close")]


@pytest.mark.asyncio
async def test_realtime_disconnect_detaches_resumable_session_without_runtime_close():
    engine = FakeEngineClient()
    handler = OmniDuplexSessionHandler(
        chat_service=FakeChatService(engine),
        config_timeout_s=0.1,
        idle_timeout_s=1,
    )
    ws = TimedWebSocket(receive_timeout_s=0.01)
    ws.put(_native_realtime_session_update("sid-resumable-disconnect"))

    await handler.handle_realtime_session(ws)

    created = next(message for message in ws.sent if message.get("type") == "session.created")
    assert created["session"]["id"] == "sid-resumable-disconnect"
    assert isinstance(created["resume_token"], str)
    assert created["resume_token"]
    assert created["attachment_generation"] == 1
    assert engine.closed == []
    assert engine.touched == [
        ("sid-resumable-disconnect", DuplexLeaseActivity.DETACH),
    ]
    assert handler._registry.get("sid-resumable-disconnect") is not None
    assert "sid-resumable-disconnect" in handler._minicpmo_sessions


@pytest.mark.asyncio
async def test_realtime_initial_resume_token_delivery_failure_closes_unrecoverable_session():
    class FailingCreatedWebSocket(TimedWebSocket):
        async def send_json(self, data: dict[str, Any]):
            if data.get("type") == "session.created":
                raise WebSocketDisconnect(code=1006)
            await super().send_json(data)

    engine = FakeEngineClient()
    handler = OmniDuplexSessionHandler(
        chat_service=FakeChatService(engine),
        config_timeout_s=0.1,
        idle_timeout_s=1,
    )
    ws = FailingCreatedWebSocket(receive_timeout_s=0.01)
    ws.put(_native_realtime_session_update("sid-created-delivery-failure"))

    await handler.handle_realtime_session(ws)

    assert engine.closed == [("sid-created-delivery-failure", "disconnect")]
    assert handler._registry.get("sid-created-delivery-failure") is None
    assert "sid-created-delivery-failure" not in handler._minicpmo_sessions


@pytest.mark.asyncio
async def test_realtime_send_after_created_disconnect_preserves_resumable_session():
    class ClosedAfterCreatedWebSocket(TimedWebSocket):
        def __init__(self) -> None:
            super().__init__(receive_timeout_s=0.01)
            self.transport_closed = False

        async def send_json(self, data: dict[str, Any]):
            if self.transport_closed:
                raise RuntimeError(
                    "Unexpected ASGI message 'websocket.send', "
                    "after sending 'websocket.close' or response already completed."
                )
            await super().send_json(data)
            if data.get("type") == "session.created":
                self.transport_closed = True

    engine = FakeEngineClient()
    handler = OmniDuplexSessionHandler(
        chat_service=FakeChatService(engine),
        config_timeout_s=0.1,
        idle_timeout_s=1,
    )
    ws = ClosedAfterCreatedWebSocket()
    ws.put(_native_realtime_session_update("sid-created-then-disconnected"))

    await handler.handle_realtime_session(ws)

    assert engine.closed == []
    assert engine.touched == [
        ("sid-created-then-disconnected", DuplexLeaseActivity.DETACH),
    ]
    assert handler._registry.get("sid-created-then-disconnected") is not None
    assert "sid-created-then-disconnected" in handler._minicpmo_sessions


@pytest.mark.asyncio
async def test_realtime_resume_rotates_token_replays_and_preserves_runtime_identity():
    engine = FakeEngineClient()
    handler = OmniDuplexSessionHandler(
        chat_service=FakeChatService(engine),
        config_timeout_s=0.1,
        idle_timeout_s=1,
    )
    first = TimedWebSocket(receive_timeout_s=0.01)
    first.put(_native_realtime_session_update("sid-realtime-resume"))
    first.put({"type": "session.heartbeat"})

    await handler.handle_realtime_session(first)

    created = next(message for message in first.sent if message.get("type") == "session.created")
    heartbeat = next(message for message in first.sent if message.get("type") == "session.heartbeat_ack")
    token = created["resume_token"]
    incarnation = created["incarnation"]
    second = TimedWebSocket(receive_timeout_s=0.1)
    second.query_params = {
        "model": "openbmb/MiniCPM-o-4_5",
        "minicpmo45_native_duplex": "1",
        "resume": "1",
    }
    second.put(
        {
            "type": "session.resume",
            "session_id": "sid-realtime-resume",
            "incarnation": incarnation,
            "resume_token": token,
            "last_received_server_event_seq": heartbeat["server_event_seq"] - 1,
        }
    )
    second.put({"type": "session.close"})

    await handler.handle_realtime_session(second)

    resumed = next(message for message in second.sent if message.get("type") == "session.resumed")
    replayed = [
        message
        for message in second.sent
        if message.get("type") == "session.heartbeat_ack"
        and message.get("server_event_seq") == heartbeat["server_event_seq"]
    ]
    assert resumed["session_id"] == "sid-realtime-resume"
    assert resumed["incarnation"] == incarnation
    assert resumed["attachment_generation"] == 2
    assert resumed["resume_token"] != token
    assert replayed == [heartbeat]
    assert engine.opened == ["sid-realtime-resume"]
    assert engine.resumed == [("sid-realtime-resume", 0)]
    assert engine.resume_fences[0] == engine.opened_fences[0]
    assert engine.closed == [("sid-realtime-resume", "session_close")]


@pytest.mark.asyncio
async def test_realtime_resume_preserves_append_tail_order_across_connections():
    class SlowFirstAppendEngine(FakeEngineClient):
        def __init__(self) -> None:
            super().__init__()
            self.first_append_started = asyncio.Event()
            self.second_append_started = asyncio.Event()
            self.release_first_append = asyncio.Event()
            self.append_start_order: list[int] = []

        async def append_duplex_input_async(self, session_id: str, **kwargs):
            call_index = len(self.append_start_order) + 1
            self.append_start_order.append(call_index)
            if call_index == 1:
                self.first_append_started.set()
                await self.release_first_append.wait()
            else:
                self.second_append_started.set()
            kwargs.pop("expected_epoch", None)
            return await super().append_duplex_input_async(session_id, **kwargs)

    engine = SlowFirstAppendEngine()
    handler = OmniDuplexSessionHandler(
        chat_service=FakeChatService(engine),
        config_timeout_s=0.1,
        idle_timeout_s=1,
    )
    first = TimedWebSocket(receive_timeout_s=1.0)
    session_update = _native_realtime_session_update("sid-resume-append-order")
    session_update["session"]["extra_body"]["auto_response"] = True
    first.put(session_update)
    first_task = asyncio.create_task(handler.handle_realtime_session(first))
    for _ in range(100):
        created = next((message for message in first.sent if message.get("type") == "session.created"), None)
        if created is not None:
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("resumable session did not open")
    session = handler._registry.get("sid-resume-append-order")
    assert session is not None
    native_state = handler._minicpmo_session_state(session)
    first_scheduler = native_state.silence_continuation_scheduler
    assert first_scheduler is not None
    first.put(
        {
            "type": "input_audio_buffer.append",
            "audio": _pcm16_b64(16000),
            "sample_rate_hz": 16000,
            "is_speech": True,
        }
    )
    await asyncio.wait_for(engine.first_append_started.wait(), timeout=1)
    await first.close(code=1006)
    await asyncio.wait_for(first_task, timeout=1)

    second = TimedWebSocket(receive_timeout_s=0.5)
    second.query_params = {
        "model": "openbmb/MiniCPM-o-4_5",
        "minicpmo45_native_duplex": "1",
        "resume": "1",
    }
    second.put(
        {
            "type": "session.resume",
            "session_id": "sid-resume-append-order",
            "incarnation": created["incarnation"],
            "resume_token": created["resume_token"],
            "last_received_server_event_seq": created.get("server_event_seq", 0),
        }
    )
    second_task = asyncio.create_task(handler.handle_realtime_session(second))
    for _ in range(100):
        if "session.resumed" in second.sent_types():
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("session did not resume")
    assert native_state.silence_continuation_scheduler is not first_scheduler
    second.put(
        {
            "type": "input_audio_buffer.append",
            "audio": _pcm16_b64(16000),
            "sample_rate_hz": 16000,
            "is_speech": True,
        }
    )
    try:
        await asyncio.sleep(0.05)
        assert not engine.second_append_started.is_set()
    finally:
        engine.release_first_append.set()

    await asyncio.wait_for(engine.second_append_started.wait(), timeout=1)
    second.put({"type": "session.close"})
    await asyncio.wait_for(second_task, timeout=2)
    assert engine.append_start_order == [1, 2]


@pytest.mark.asyncio
async def test_realtime_takeover_ignores_late_detach_from_replaced_attachment():
    engine = FakeEngineClient()
    handler = OmniDuplexSessionHandler(
        chat_service=FakeChatService(engine),
        config_timeout_s=0.1,
        idle_timeout_s=1,
    )
    first = TimedWebSocket(receive_timeout_s=1.0)
    first.put(_native_realtime_session_update("sid-takeover"))
    first_task = asyncio.create_task(handler.handle_realtime_session(first))
    for _ in range(100):
        created = next((message for message in first.sent if message.get("type") == "session.created"), None)
        if created is not None:
            break
        await asyncio.sleep(0.001)
    else:
        raise AssertionError("first attachment did not create the session")

    second = TimedWebSocket(receive_timeout_s=0.03)
    second.put(
        {
            "type": "session.resume",
            "session_id": "sid-takeover",
            "incarnation": created["incarnation"],
            "resume_token": created["resume_token"],
            "last_received_server_event_seq": 0,
        }
    )
    await handler.handle_realtime_session(second)
    await first_task

    detach_touches = [item for item in engine.touched if item[1] is DuplexLeaseActivity.DETACH]
    assert detach_touches == [("sid-takeover", DuplexLeaseActivity.DETACH)]
    assert "session.replaced" in first.sent_types()
    assert engine.resumed == [("sid-takeover", 0)]
    assert engine.closed == []


@pytest.mark.asyncio
async def test_realtime_takeover_does_not_clear_new_attachment_pending_turns():
    class DelayedReplacementCloseWebSocket(TimedWebSocket):
        def __init__(self) -> None:
            super().__init__(receive_timeout_s=1.0)
            self.replacement_close_started = asyncio.Event()
            self.release_replacement_close = asyncio.Event()

        async def close(self, code: int = 1000, reason: str | None = None) -> None:
            if not self.replacement_close_started.is_set():
                self.replacement_close_started.set()
                await self.release_replacement_close.wait()
            await super().close(code=code, reason=reason)

    engine = FakeEngineClient()
    handler = OmniDuplexSessionHandler(
        chat_service=FakeChatService(engine),
        config_timeout_s=0.1,
        idle_timeout_s=1,
    )
    first = DelayedReplacementCloseWebSocket()
    first.put(_native_realtime_session_update("sid-takeover-pending-turn"))
    first_task = asyncio.create_task(handler.handle_realtime_session(first))
    for _ in range(100):
        created = next((message for message in first.sent if message.get("type") == "session.created"), None)
        if created is not None:
            break
        await asyncio.sleep(0.001)
    else:
        pytest.fail("first attachment did not create the session")

    second = TimedWebSocket(receive_timeout_s=1.0)
    second.put(
        {
            "type": "session.resume",
            "session_id": "sid-takeover-pending-turn",
            "incarnation": created["incarnation"],
            "resume_token": created["resume_token"],
            "last_received_server_event_seq": 0,
        }
    )
    second_task = asyncio.create_task(handler.handle_realtime_session(second))
    await asyncio.wait_for(first.replacement_close_started.wait(), timeout=1)

    session = handler._registry.get("sid-takeover-pending-turn")
    assert session is not None
    assert session.reserve_pending_turn(limit=2)
    first.release_replacement_close.set()
    await asyncio.wait_for(first_task, timeout=1)

    assert session.pending_input_turns == 1
    session.release_pending_turn()
    second.put({"type": "session.close"})
    await asyncio.wait_for(second_task, timeout=2)


@pytest.mark.asyncio
async def test_realtime_takeover_drops_input_queued_by_replaced_attachment():
    class DelayedReplacementCloseWebSocket(TimedWebSocket):
        def __init__(self) -> None:
            super().__init__(receive_timeout_s=1.0)
            self.replacement_close_started = asyncio.Event()
            self.release_replacement_close = asyncio.Event()
            self.stale_heartbeat_read = asyncio.Event()

        async def receive_text(self) -> str:
            payload = await super().receive_text()
            if json.loads(payload).get("type") == "session.heartbeat":
                self.stale_heartbeat_read.set()
            return payload

        async def close(self, code: int = 1000, reason: str | None = None) -> None:
            if not self.replacement_close_started.is_set():
                self.replacement_close_started.set()
                await self.release_replacement_close.wait()
            await super().close(code=code, reason=reason)

    engine = FakeEngineClient()
    handler = OmniDuplexSessionHandler(
        chat_service=FakeChatService(engine),
        config_timeout_s=0.1,
        idle_timeout_s=1,
    )
    first = DelayedReplacementCloseWebSocket()
    first.put(_native_realtime_session_update("sid-takeover-stale-input"))
    first_task = asyncio.create_task(handler.handle_realtime_session(first))
    for _ in range(100):
        created = next((message for message in first.sent if message.get("type") == "session.created"), None)
        if created is not None:
            break
        await asyncio.sleep(0.001)
    else:
        pytest.fail("first attachment did not create the session")

    second = TimedWebSocket(receive_timeout_s=1.0)
    second.put(
        {
            "type": "session.resume",
            "session_id": "sid-takeover-stale-input",
            "incarnation": created["incarnation"],
            "resume_token": created["resume_token"],
            "last_received_server_event_seq": 0,
        }
    )
    second_task = asyncio.create_task(handler.handle_realtime_session(second))
    await asyncio.wait_for(first.replacement_close_started.wait(), timeout=1)

    first.put({"type": "session.heartbeat"})
    await asyncio.wait_for(first.stale_heartbeat_read.wait(), timeout=1)
    first.release_replacement_close.set()
    await asyncio.wait_for(first_task, timeout=1)

    second.put({"type": "session.close"})
    await asyncio.wait_for(second_task, timeout=2)
    assert all(activity is not DuplexLeaseActivity.HEARTBEAT for _, activity in engine.touched)
    assert "session.heartbeat_ack" not in second.sent_types()


@pytest.mark.asyncio
async def test_realtime_resume_rejects_invalid_token_before_engine_control():
    engine = FakeEngineClient()
    handler = OmniDuplexSessionHandler(
        chat_service=FakeChatService(engine),
        config_timeout_s=0.1,
        idle_timeout_s=1,
    )
    first = TimedWebSocket(receive_timeout_s=0.01)
    first.put(_native_realtime_session_update("sid-invalid-resume"))
    await handler.handle_realtime_session(first)

    created = next(message for message in first.sent if message.get("type") == "session.created")
    second = TimedWebSocket(receive_timeout_s=0.01)
    second.put(
        {
            "type": "session.resume",
            "session_id": "sid-invalid-resume",
            "incarnation": created["incarnation"],
            "resume_token": "invalid-token",
            "last_received_server_event_seq": 0,
        }
    )

    await handler.handle_realtime_session(second)

    error = next(message for message in second.sent if message.get("type") == "error")
    assert error["error"]["code"] == "invalid_resume_token"
    assert engine.resumed == []
    assert engine.closed == []


@pytest.mark.asyncio
async def test_realtime_resume_recovers_when_rotated_token_delivery_is_lost():
    class FailingResumeWebSocket(TimedWebSocket):
        async def send_json(self, data: dict[str, Any]):
            if data.get("type") == "session.resumed":
                raise RuntimeError("resume transport disappeared")
            await super().send_json(data)

    engine = FakeEngineClient()
    handler = OmniDuplexSessionHandler(
        chat_service=FakeChatService(engine),
        config_timeout_s=0.1,
        idle_timeout_s=1,
    )
    first = TimedWebSocket(receive_timeout_s=0.01)
    first.put(_native_realtime_session_update("sid-token-delivery"))
    await handler.handle_realtime_session(first)
    created = next(message for message in first.sent if message.get("type") == "session.created")
    resume_event = {
        "type": "session.resume",
        "session_id": "sid-token-delivery",
        "incarnation": created["incarnation"],
        "resume_token": created["resume_token"],
        "last_received_server_event_seq": 0,
    }

    failed_transport = FailingResumeWebSocket(receive_timeout_s=0.01)
    failed_transport.put(resume_event)
    await handler.handle_realtime_session(failed_transport)

    recovered_transport = TimedWebSocket(receive_timeout_s=0.1)
    recovered_transport.put(resume_event)
    recovered_transport.put({"type": "session.close"})
    await handler.handle_realtime_session(recovered_transport)

    assert "session.resumed" in recovered_transport.sent_types()
    assert engine.resumed == [
        ("sid-token-delivery", 0),
        ("sid-token-delivery", 1),
    ]
    assert engine.closed == [("sid-token-delivery", "session_close")]


@pytest.mark.asyncio
async def test_realtime_disconnect_grace_cancels_only_orphan_response():
    engine = FakeEngineClient()
    handler = OmniDuplexSessionHandler(
        chat_service=FakeChatService(engine),
        config_timeout_s=0.1,
        idle_timeout_s=1,
        duplex_session_config=DuplexSessionRuntimeConfig(disconnect_grace_s=0.02),
    )
    ws = TimedWebSocket(receive_timeout_s=0.01)
    ws.put(_native_realtime_session_update("sid-grace-response"))

    await handler.handle_realtime_session(ws)
    session = handler._registry.get("sid-grace-response")
    assert session is not None
    session.begin_response()
    session.bind_request("orphan-request")
    orphan_task = asyncio.create_task(asyncio.sleep(10))
    handler._session_tasks[session.session_id].active_response_task = orphan_task
    assert engine.closed == []

    await asyncio.sleep(0.05)

    assert engine.closed == []
    assert engine.aborted == ["orphan-request"]
    assert orphan_task.cancelled()
    assert session.active_response_id is None
    assert "sid-grace-response" in handler._minicpmo_sessions


@pytest.mark.asyncio
async def test_realtime_heartbeat_control_failure_is_non_terminal():
    engine = FakeEngineClient(fail_touch=True)
    handler = OmniDuplexSessionHandler(
        chat_service=FakeChatService(engine),
        config_timeout_s=0.1,
        idle_timeout_s=1,
    )
    ws = TimedWebSocket(receive_timeout_s=0.1)
    ws.put(_native_realtime_session_update("sid-heartbeat-failure"))
    ws.put({"type": "session.heartbeat"})
    ws.put({"type": "session.close"})

    await handler.handle_realtime_session(ws)

    error = next(message for message in ws.sent if message.get("type") == "error")
    assert error["error"]["code"] == "runtime_touch_failed"
    assert engine.closed == [("sid-heartbeat-failure", "session_close")]


@pytest.mark.asyncio
async def test_realtime_idle_ttl_lifecycle_removes_detached_serving_projection():
    engine = FakeEngineClient()
    handler = OmniDuplexSessionHandler(
        chat_service=FakeChatService(engine),
        config_timeout_s=0.1,
        idle_timeout_s=1,
    )
    ws = TimedWebSocket(receive_timeout_s=0.01)
    ws.put(_native_realtime_session_update("sid-idle-expired"))

    await handler.handle_realtime_session(ws)
    fence = engine.opened_fences[0]
    assert fence is not None
    await engine.duplex_lifecycle_events.put(
        DuplexSessionLifecycleMessage(
            fence=fence,
            session_id="sid-idle-expired",
            event="expired",
            reason="idle_ttl_expired",
            lease_generation=1,
            submitted_request_ids=[],
            reserved_request_ids=[],
        )
    )
    await asyncio.sleep(0.05)

    assert handler._registry.get("sid-idle-expired") is None
    assert "sid-idle-expired" not in handler._minicpmo_sessions
    assert "sid-idle-expired" not in handler._session_tasks
    assert engine.closed == []


@pytest.mark.asyncio
async def test_realtime_resume_reports_resync_required_after_journal_ttl_gap():
    engine = FakeEngineClient()
    handler = OmniDuplexSessionHandler(
        chat_service=FakeChatService(engine),
        config_timeout_s=0.1,
        idle_timeout_s=1,
        duplex_session_config=DuplexSessionRuntimeConfig(resume_replay_ttl_s=0.01),
    )
    first = TimedWebSocket(receive_timeout_s=0.01)
    first.put(_native_realtime_session_update("sid-journal-gap"))
    first.put({"type": "session.heartbeat"})
    await handler.handle_realtime_session(first)
    created = next(message for message in first.sent if message.get("type") == "session.created")
    await asyncio.sleep(0.02)

    second = TimedWebSocket(receive_timeout_s=0.01)
    second.put(
        {
            "type": "session.resume",
            "session_id": "sid-journal-gap",
            "incarnation": created["incarnation"],
            "resume_token": created["resume_token"],
            "last_received_server_event_seq": 0,
        }
    )
    await handler.handle_realtime_session(second)

    assert "session.resync_required" in second.sent_types()
    assert engine.resumed == []
    assert engine.closed == []


@pytest.mark.asyncio
async def test_realtime_journal_overflow_degrades_to_live_resync_without_closing_runtime():
    engine = FakeEngineClient()
    handler = OmniDuplexSessionHandler(
        chat_service=FakeChatService(engine),
        config_timeout_s=0.1,
        idle_timeout_s=1,
        duplex_session_config=DuplexSessionRuntimeConfig(resume_replay_max_bytes_per_session=128),
    )
    ws = TimedWebSocket(receive_timeout_s=0.01)
    ws.put(_native_realtime_session_update("sid-journal-overflow"))

    await handler.handle_realtime_session(ws)

    assert "session.created" in ws.sent_types()
    assert "session.resync_required" in ws.sent_types()
    assert engine.closed == []
    assert handler._registry.get("sid-journal-overflow") is not None


@pytest.mark.asyncio
async def test_realtime_input_backpressure_is_per_session_and_rejects_before_mutation():
    engine = FakeEngineClient()
    handler = OmniDuplexSessionHandler(
        chat_service=FakeChatService(engine),
        config_timeout_s=0.1,
        idle_timeout_s=1,
        duplex_session_config=DuplexSessionRuntimeConfig(
            max_pending_input_bytes_per_session=100,
        ),
    )
    first = TimedWebSocket(receive_timeout_s=0.1)
    first.put(_native_realtime_session_update("sid-backpressure-a"))
    for _ in range(2):
        first.put(
            {
                "type": "input_audio_buffer.append",
                "audio": _pcm16_b64(20),
                "format": "pcm16",
                "sample_rate_hz": 16000,
            }
        )
    first.put({"type": "session.close"})

    await handler.handle_realtime_session(first)

    backpressure = [
        event
        for event in first.sent
        if event.get("type") == "error"
        and isinstance(event.get("error"), dict)
        and event["error"].get("code") == "input_backpressure"
    ]
    assert len(backpressure) == 1

    second = TimedWebSocket(receive_timeout_s=0.1)
    second.put(_native_realtime_session_update("sid-backpressure-b"))
    second.put(
        {
            "type": "input_audio_buffer.append",
            "audio": _pcm16_b64(20),
            "format": "pcm16",
            "sample_rate_hz": 16000,
        }
    )
    second.put({"type": "session.close"})

    await handler.handle_realtime_session(second)

    assert not any(
        event.get("type") == "error"
        and isinstance(event.get("error"), dict)
        and event["error"].get("code") == "input_backpressure"
        for event in second.sent
    )


@pytest.mark.asyncio
async def test_duplex_handler_idle_timeout_close_does_not_emit_runtime_control():
    control_result = {
        "operation": "close",
        "session_id": "sid-disconnect",
        "ok": True,
        "unsupported_count": 1,
        "error_count": 0,
        "stage_results": [{"stage_id": 0, "replica_id": 0, "result": {"supported": False}}],
    }
    engine = FakeEngineClient(append_result=control_result)
    chat_service = FakeChatService(engine)
    handler = OmniDuplexSessionHandler(chat_service=chat_service, config_timeout_s=0.1, idle_timeout_s=0.1)
    ws = TimedWebSocket()
    ws.put(_session_create("sid-disconnect"))

    await handler.handle_session(ws)

    assert engine.closed == [("sid-disconnect", "timeout")]


@pytest.mark.asyncio
async def test_idle_output_audio_clear_does_not_advance_runtime_fence():
    engine = FakeEngineClient()
    handler = OmniDuplexSessionHandler(
        chat_service=FakeChatService(engine),
        config_timeout_s=0.1,
        idle_timeout_s=1,
    )
    ws = TimedWebSocket()
    ws.put(_session_create("sid-idle-output-clear"))
    ws.put({"type": "output_audio_buffer.clear"})
    ws.put({"type": "session.close"})

    await handler.handle_session(ws)

    cancelled = next(event for event in ws.sent if event.get("type") == "audio.cancelled")
    assert cancelled["cancelled_epoch"] == cancelled["epoch"] == 0
    assert engine.signals == []


@pytest.mark.asyncio
async def test_duplex_handler_runtime_close_failure_is_reported_without_closed_ack():
    engine = FakeEngineClient(fail_close=True)
    chat_service = FakeChatService(engine)
    handler = OmniDuplexSessionHandler(chat_service=chat_service, config_timeout_s=0.1, idle_timeout_s=1)
    ws = TimedWebSocket()
    ws.put(_session_create("sid-close-fail"))
    ws.put({"type": "session.close"})

    await handler.handle_session(ws)

    assert ws.sent_types() == ["session.created", "error"]
    assert ws.sent[-1]["code"] == "runtime_close_failed"


@pytest.mark.asyncio
async def test_duplex_handler_control_close_failure_is_reported_without_closed_ack():
    control_result = {
        "operation": "close",
        "session_id": "sid-control-close-fail",
        "ok": False,
        "unsupported_count": 0,
        "error_count": 1,
        "stage_results": [
            {
                "stage_id": 0,
                "replica_id": 0,
                "result": {"supported": False, "error": "stage close failed"},
            }
        ],
    }
    engine = FakeEngineClient(close_result=control_result)
    chat_service = FakeChatService(engine)
    handler = OmniDuplexSessionHandler(chat_service=chat_service, config_timeout_s=0.1, idle_timeout_s=1)
    ws = TimedWebSocket()
    ws.put(_session_create("sid-control-close-fail"))
    ws.put({"type": "session.close"})

    await handler.handle_session(ws)

    assert ws.sent_types() == ["session.created", "error"]
    assert ws.sent[-1]["code"] == "runtime_close_failed"
    assert ws.sent[-1]["runtime_control"]["error_count"] == 1


@pytest.mark.asyncio
async def test_duplex_cancel_without_active_response_clears_pending_input_and_acks():
    engine = FakeEngineClient()
    chat_service = FakeChatService(engine)
    handler = OmniDuplexSessionHandler(chat_service=chat_service, config_timeout_s=0.1, idle_timeout_s=1)
    ws = TimedWebSocket()
    ws.put(_session_create("sid-pending-cancel"))
    ws.put({"type": "input.text.append", "text": "hello"})
    ws.put({"type": "input.cancel", "reason": "user_cancel"})
    ws.put({"type": "input.commit"})

    await handler.handle_session(ws)

    cancelled = next(m for m in ws.sent if m.get("type") == "input.cancelled")
    assert cancelled["cancelled"] == {"text_chunks": 1, "audio_chunks": 0}
    assert cancelled["epoch"] == 1
    assert chat_service.seen_request_ids == []
    assert "response.created" not in ws.sent_types()


@pytest.mark.asyncio
async def test_turn_signal_input_cancel_uses_epoch_fence_transition():
    engine = FakeEngineClient()
    handler = OmniDuplexSessionHandler(
        chat_service=FakeChatService(engine),
        config_timeout_s=0.1,
        idle_timeout_s=1,
    )
    ws = TimedWebSocket()
    ws.put(_session_create("sid-turn-signal-cancel"))
    ws.put({"type": "input.text.append", "text": "discard me"})
    ws.put({"type": "turn.signal", "event": "input.cancel"})
    ws.put({"type": "session.close"})

    await handler.handle_session(ws)

    assert engine.signal_fences[-1] == DuplexFence("sid-turn-signal-cancel")
    assert engine.signal_next_fences[-1] == DuplexFence("sid-turn-signal-cancel", epoch=1)
    assert any(message.get("type") == "input.cancelled" for message in ws.sent)


@pytest.mark.asyncio
async def test_duplex_handler_local_turn_signal_does_not_round_trip_runtime():
    engine = FakeEngineClient(fail_signal=True)
    chat_service = FakeChatService(engine)
    handler = OmniDuplexSessionHandler(chat_service=chat_service, config_timeout_s=0.1, idle_timeout_s=1)
    ws = TimedWebSocket()
    ws.put(_session_create("sid-signal-fail"))
    ws.put({"type": "turn.signal", "event": "user_started"})
    ws.put({"type": "session.close"})

    await handler.handle_session(ws)

    turn_event = next(m for m in ws.sent if m.get("type") == "turn.event")
    assert turn_event["event"] == "user_started"
    assert not engine.signals
    assert "runtime_signal_failed" not in {m.get("code") for m in ws.sent}


@pytest.mark.asyncio
async def test_duplex_barge_in_aborts_active_response_when_runtime_signal_fails():
    engine = FakeEngineClient(fail_signal_events={"barge_in"})
    chat_service = FakeChatService(engine)
    handler = OmniDuplexSessionHandler(chat_service=chat_service, config_timeout_s=0.1, idle_timeout_s=1)

    def on_send(ws: TimedWebSocket, data: dict[str, Any]) -> None:
        if data.get("type") == "response.created":
            ws.put({"type": "input.cancel", "reason": "test_barge_in"})

    ws = TimedWebSocket(on_send=on_send)
    ws.put(_session_create("sid-barge-signal-fail"))
    ws.put({"type": "input.text.append", "text": "hello"})
    ws.put({"type": "input.commit"})

    await handler.handle_session(ws)

    assert "audio.cancelled" in ws.sent_types()
    assert engine.aborted == ["chatcmpl-duplex-sid-barge-signal-fail-0-1"]
    assert engine.internal_abort_batches == [["chatcmpl-duplex-sid-barge-signal-fail-0-1"]]
    error = next(m for m in ws.sent if m.get("type") == "error")
    assert error["code"] == "runtime_signal_failed"


@pytest.mark.asyncio
async def test_duplex_handler_surfaces_stage_unsupported_result_to_client():
    control_result = {
        "operation": "open",
        "session_id": "sid-unsupported",
        "ok": True,
        "unsupported_count": 1,
        "error_count": 0,
        "stage_results": [
            {
                "stage_id": 0,
                "replica_id": 0,
                "result": {"supported": False, "reason": "not implemented"},
            }
        ],
    }
    engine = FakeEngineClient(open_result=control_result)
    chat_service = FakeChatService(engine)
    handler = OmniDuplexSessionHandler(chat_service=chat_service, config_timeout_s=0.1, idle_timeout_s=1)
    ws = TimedWebSocket()
    ws.put(_session_create("sid-unsupported"))
    ws.put({"type": "session.close"})

    await handler.handle_session(ws)

    created = next(m for m in ws.sent if m.get("type") == "session.created")
    assert created["runtime_control"]["unsupported_count"] == 1


@pytest.mark.asyncio
async def test_minicpmo_native_duplex_append_without_format_defaults_to_pcm16():
    engine = FakeEngineClient()
    chat_service = FakeChatService(engine)
    handler = OmniDuplexSessionHandler(chat_service=chat_service, config_timeout_s=0.1, idle_timeout_s=1)
    ws = TimedWebSocket()
    ws.put(_native_session_create("sid-native-default-pcm16"))
    ws.put(
        {
            "type": "input_audio_buffer.append",
            "audio": _pcm16_b64(16000),
            "sample_rate_hz": 16000,
            "is_speech": True,
        }
    )
    ws.put({"type": "session.close"})

    await handler.handle_session(ws)

    assert len(engine.appended) == 1
    _, mode, payload, final = engine.appended[0]
    assert mode == "append_audio_chunk"
    assert final is False
    assert isinstance(payload, dict)
    assert payload["format"] == "pcm_f32le"
    assert payload["sample_rate_hz"] == 16000
    samples = np.frombuffer(base64.b64decode(payload["audio"]), dtype="<f4")
    assert samples.shape == (16000,)
    assert samples[:4].tolist() == pytest.approx([1000 / 32768.0] * 4)


@pytest.mark.asyncio
async def test_minicpmo_native_duplex_rejects_invalid_sample_rate_without_append():
    engine = FakeEngineClient()
    handler = OmniDuplexSessionHandler(
        chat_service=FakeChatService(engine),
        config_timeout_s=0.1,
        idle_timeout_s=1,
    )
    ws = TimedWebSocket()
    ws.put(_native_session_create("sid-native-invalid-rate"))
    ws.put(
        {
            "type": "input_audio_buffer.append",
            "audio": _pcm16_b64(1),
            "sample_rate_hz": 16_000.5,
            "is_speech": True,
        }
    )
    ws.put({"type": "session.close"})

    await handler.handle_session(ws)

    assert engine.appended == []
    error = next(event for event in ws.sent if event.get("type") == "error")
    assert error["code"] == "bad_event"


@pytest.mark.asyncio
async def test_native_input_commit_does_not_send_noop_runtime_signal():
    engine = FakeEngineClient()
    handler = OmniDuplexSessionHandler(
        chat_service=FakeChatService(engine),
        config_timeout_s=0.1,
        idle_timeout_s=1,
    )
    ws = TimedWebSocket()
    ws.put(_native_session_create("sid-commit-without-signal"))
    ws.put(
        {
            "type": "input_audio_buffer.append",
            "audio": _pcm16_b64(16000),
            "sample_rate_hz": 16000,
            "is_speech": True,
        }
    )
    ws.put({"type": "input_audio_buffer.commit"})
    ws.put({"type": "session.close"})

    await handler.handle_session(ws)

    assert ("sid-commit-without-signal", "input.commit") not in engine.signals


@pytest.mark.asyncio
async def test_audio_clear_preserves_later_wire_order_append():
    engine = FakeEngineClient()
    handler = OmniDuplexSessionHandler(
        chat_service=FakeChatService(engine),
        config_timeout_s=0.1,
        idle_timeout_s=1,
    )
    ws = TimedWebSocket()
    ws.put(_native_session_create("sid-clear-then-append"))
    ws.put({"type": "input_audio_buffer.clear"})
    ws.put(
        {
            "type": "input_audio_buffer.append",
            "audio": _pcm16_b64(16000),
            "sample_rate_hz": 16000,
            "is_speech": True,
        }
    )
    ws.put({"type": "session.close"})

    await handler.handle_session(ws)

    assert len(engine.appended) == 1


@pytest.mark.asyncio
async def test_native_append_is_cancelled_by_later_wire_order_input_cancel():
    class SlowAppendEngine(FakeEngineClient):
        def __init__(self) -> None:
            super().__init__()
            self.append_started = asyncio.Event()
            self.append_cancelled = asyncio.Event()
            self.release_append = asyncio.Event()

        async def append_duplex_input_async(
            self,
            session_id: str,
            *,
            mode: str,
            payload: object,
            final: bool = False,
            timeout: float | None = None,
            collect_outputs: bool = True,
            fence: DuplexFence | None = None,
        ):
            self.append_started.set()
            try:
                await self.release_append.wait()
            except asyncio.CancelledError:
                self.append_cancelled.set()
                raise
            return await super().append_duplex_input_async(
                session_id,
                mode=mode,
                payload=payload,
                final=final,
                timeout=timeout,
                collect_outputs=collect_outputs,
                fence=fence,
            )

    engine = SlowAppendEngine()
    handler = OmniDuplexSessionHandler(
        chat_service=FakeChatService(engine),
        config_timeout_s=0.1,
        idle_timeout_s=1,
    )
    ws = TimedWebSocket()
    ws.put(_native_session_create("sid-cancel-slow-append"))
    ws.put(
        {
            "type": "input_audio_buffer.append",
            "audio": _pcm16_b64(16000),
            "sample_rate_hz": 16000,
            "is_speech": True,
        }
    )
    ws.put({"type": "input.cancel"})
    ws.put({"type": "session.close"})

    handler_task = asyncio.create_task(handler.handle_session(ws))
    await asyncio.wait_for(engine.append_started.wait(), timeout=1)
    try:
        await asyncio.wait_for(engine.append_cancelled.wait(), timeout=0.2)
    except TimeoutError:
        pass
    finally:
        engine.release_append.set()
    await asyncio.wait_for(handler_task, timeout=1)

    assert engine.append_cancelled.is_set()
    assert engine.signal_fences[-1] == DuplexFence("sid-cancel-slow-append")
    assert engine.signal_next_fences[-1] == DuplexFence("sid-cancel-slow-append", epoch=1)


@pytest.mark.asyncio
async def test_minicpmo_native_duplex_append_control_error_does_not_emit_model_delta():
    control_result = {
        "operation": "append",
        "session_id": "sid-native-append-error",
        "ok": False,
        "unsupported_count": 0,
        "error_count": 1,
        "stage_results": [
            {
                "stage_id": -1,
                "replica_id": -1,
                "result": {
                    "supported": False,
                    "error": "duplex_stage_handoff_target_missing:tts",
                },
            },
        ],
    }
    engine = FakeEngineClient(append_result=control_result)
    chat_service = FakeChatService(engine)
    handler = OmniDuplexSessionHandler(chat_service=chat_service, config_timeout_s=0.1, idle_timeout_s=1)
    ws = TimedWebSocket()
    ws.put(_native_session_create("sid-native-append-error"))
    ws.put({"type": "input_audio_buffer.append", "audio": "AAAA", "format": "pcm_f32le"})
    ws.put({"type": "session.close"})

    await handler.handle_session(ws)

    assert "response.output_audio.delta" not in ws.sent_types()
    error = next(m for m in ws.sent if m.get("type") == "error")
    assert error["code"] == "runtime_append_failed"
    assert error["runtime_control"]["error_count"] == 1


@pytest.mark.asyncio
async def test_duplex_handler_session_update_control_failure_rolls_back_config():
    control_result = {
        "operation": "signal",
        "session_id": "sid-control-signal-fail",
        "ok": False,
        "unsupported_count": 0,
        "error_count": 1,
        "stage_results": [
            {
                "stage_id": 0,
                "replica_id": 0,
                "result": {"supported": False, "error": "stage signal failed"},
            }
        ],
    }
    engine = FakeEngineClient(signal_result=control_result)
    chat_service = FakeChatService(engine)
    handler = OmniDuplexSessionHandler(chat_service=chat_service, config_timeout_s=0.1, idle_timeout_s=1)
    ws = TimedWebSocket()
    ws.put(_session_create("sid-control-signal-fail"))
    ws.put(
        {
            "type": "turn.signal",
            "event": "session.update",
            "payload": {"instructions": "must roll back"},
        }
    )
    ws.put({"type": "session.close"})

    await handler.handle_session(ws)

    assert "session.updated" not in ws.sent_types()
    error = next(m for m in ws.sent if m.get("type") == "error")
    assert error["code"] == "runtime_signal_failed"
    assert error["runtime_control"]["error_count"] == 1
    assert engine.signal_session_configs[-1]["instructions"] == "must roll back"


@pytest.mark.asyncio
async def test_native_session_update_commits_config_only_after_runtime_ack():
    class BlockingUpdateEngine(FakeEngineClient):
        def __init__(self):
            super().__init__()
            self.update_started = asyncio.Event()
            self.release_update = asyncio.Event()

        async def signal_duplex_turn_async(self, session_id: str, **kwargs):
            if kwargs.get("event") == "session.update":
                self.update_started.set()
                await self.release_update.wait()
            return await super().signal_duplex_turn_async(session_id, **kwargs)

    engine = BlockingUpdateEngine()
    handler = OmniDuplexSessionHandler(
        chat_service=FakeChatService(engine),
        config_timeout_s=0.1,
        idle_timeout_s=1,
    )
    ws = TimedWebSocket()
    ws.put(_session_create("sid-update-two-phase"))
    ws.put(
        {
            "type": "turn.signal",
            "event": "session.update",
            "payload": {"instructions": "candidate instructions"},
        }
    )

    handler_task = asyncio.create_task(handler.handle_session(ws))
    await asyncio.wait_for(engine.update_started.wait(), timeout=1)
    live_session = handler._registry.get("sid-update-two-phase")
    assert live_session is not None
    assert live_session.config.instructions != "candidate instructions"

    engine.release_update.set()
    ws.put({"type": "session.close"})
    await asyncio.wait_for(handler_task, timeout=2)

    updated = next(message for message in ws.sent if message.get("type") == "session.updated")
    assert updated["session"]["instructions"] == "candidate instructions"


@pytest.mark.asyncio
async def test_native_session_update_waits_for_prior_append_effect():
    class OrderedUpdateEngine(FakeEngineClient):
        def __init__(self):
            super().__init__()
            self.append_started = asyncio.Event()
            self.release_append = asyncio.Event()
            self.update_started = asyncio.Event()
            self.effects: list[str] = []

        async def append_duplex_input_async(self, session_id: str, **kwargs):
            self.effects.append("append.started")
            self.append_started.set()
            await self.release_append.wait()
            kwargs.pop("expected_epoch", None)
            result = await super().append_duplex_input_async(session_id, **kwargs)
            self.effects.append("append.done")
            return result

        async def signal_duplex_turn_async(self, session_id: str, **kwargs):
            if kwargs.get("event") == "session.update":
                self.effects.append("session.update")
                self.update_started.set()
            return await super().signal_duplex_turn_async(session_id, **kwargs)

    engine = OrderedUpdateEngine()
    handler = OmniDuplexSessionHandler(
        chat_service=FakeChatService(engine),
        config_timeout_s=0.1,
        idle_timeout_s=1,
    )
    ws = TimedWebSocket()
    ws.put(_native_session_create("sid-ordered-update"))
    ws.put(
        {
            "type": "input_audio_buffer.append",
            "audio": _pcm_f32_b64(16000, value=0.05),
            "format": "pcm_f32le",
        }
    )
    ws.put(
        {
            "type": "turn.signal",
            "event": "session.update",
            "payload": {"temperature": 0.2},
        }
    )

    handler_task = asyncio.create_task(handler.handle_session(ws))
    await asyncio.wait_for(engine.append_started.wait(), timeout=1)
    try:
        await asyncio.sleep(0.05)
        assert not engine.update_started.is_set()
    finally:
        engine.release_append.set()
        ws.put({"type": "session.close"})
        await asyncio.wait_for(handler_task, timeout=2)

    assert engine.effects.index("append.done") < engine.effects.index("session.update")


@pytest.mark.asyncio
async def test_failed_native_append_prevents_queued_append_from_running():
    failed_result = {
        "operation": "append",
        "session_id": "sid-append-chain-failure",
        "ok": False,
        "unsupported_count": 0,
        "error_count": 1,
        "stage_results": [
            {
                "stage_id": 0,
                "replica_id": 0,
                "result": {"supported": False, "error": "append failed"},
            }
        ],
    }

    class BlockingFailureEngine(FakeEngineClient):
        def __init__(self):
            super().__init__(append_result=failed_result)
            self.first_started = asyncio.Event()
            self.release_first = asyncio.Event()

        async def append_duplex_input_async(self, session_id: str, **kwargs):
            if not self.appended:
                self.first_started.set()
                await self.release_first.wait()
            kwargs.pop("expected_epoch", None)
            return await super().append_duplex_input_async(session_id, **kwargs)

    engine = BlockingFailureEngine()
    handler = OmniDuplexSessionHandler(
        chat_service=FakeChatService(engine),
        config_timeout_s=0.1,
        idle_timeout_s=1,
    )
    ws = TimedWebSocket()
    create = _native_session_create("sid-append-chain-failure")
    create["session"]["extra_body"]["auto_response"] = True
    ws.put(create)
    for _ in range(2):
        ws.put(
            {
                "type": "input_audio_buffer.append",
                "audio": _pcm_f32_b64(16_000, value=0.05),
                "format": "pcm_f32le",
            }
        )

    handler_task = asyncio.create_task(handler.handle_session(ws))
    await asyncio.wait_for(engine.first_started.wait(), timeout=1)
    engine.release_first.set()
    await asyncio.sleep(0.1)
    ws.put({"type": "session.close"})
    await asyncio.wait_for(handler_task, timeout=2)

    assert len(engine.appended) == 1


@pytest.mark.asyncio
async def test_failed_final_append_keeps_committed_audio_for_response_retry():
    failed_result = {
        "operation": "append",
        "session_id": "sid-final-append-retry",
        "ok": False,
        "unsupported_count": 0,
        "error_count": 1,
        "stage_results": [],
    }
    successful_result = {
        "operation": "append",
        "session_id": "sid-final-append-retry",
        "ok": True,
        "unsupported_count": 0,
        "error_count": 0,
        "stage_results": [],
    }
    engine = FakeEngineClient(append_results=[failed_result, successful_result])
    handler = OmniDuplexSessionHandler(
        chat_service=FakeChatService(engine),
        config_timeout_s=0.1,
        idle_timeout_s=1,
    )

    def retry_after_runtime_error(ws: TimedWebSocket, data: dict[str, Any]) -> None:
        if data.get("type") == "error" and data.get("code") == "runtime_append_failed":
            ws.put({"type": "response.create"})

    ws = TimedWebSocket(on_send=retry_after_runtime_error)
    ws.put(_native_session_create("sid-final-append-retry"))
    ws.put(
        {
            "type": "input_audio_buffer.append",
            "audio": _pcm_f32_b64(8_000, value=0.05),
            "format": "pcm_f32le",
        }
    )
    ws.put({"type": "input_audio_buffer.commit", "final": True})
    ws.put({"type": "response.create"})

    handler_task = asyncio.create_task(handler.handle_session(ws))
    for _ in range(100):
        if len(engine.appended) >= 2:
            break
        await asyncio.sleep(0.01)
    ws.put({"type": "session.close"})
    await asyncio.wait_for(handler_task, timeout=2)

    assert len(engine.appended) == 2
    assert engine.appended[0][2]["audio"] == engine.appended[1][2]["audio"]
    assert engine.append_operation_ids[0] == engine.append_operation_ids[1]


@pytest.mark.asyncio
async def test_duplex_handler_signal_unsupported_workers_with_data_plane_ack_is_not_fatal():
    control_result = {
        "operation": "signal",
        "session_id": "sid-control-signal-data-plane",
        "ok": False,
        "unsupported_count": 2,
        "error_count": 0,
        "stage_results": [
            {
                "stage_id": 0,
                "replica_id": 0,
                "result": {"supported": False, "reason": "worker_duplex_signal_not_implemented"},
            },
            {
                "stage_id": 1,
                "replica_id": 0,
                "result": {"supported": False, "reason": "worker_duplex_signal_not_implemented"},
            },
            {
                "stage_id": -1,
                "replica_id": -1,
                "result": {
                    "supported": True,
                    "data_plane_signal": True,
                    "event": "barge_in",
                    "epoch": 1,
                },
            },
        ],
    }
    engine = FakeEngineClient(signal_result=control_result)
    chat_service = FakeChatService(engine)
    handler = OmniDuplexSessionHandler(chat_service=chat_service, config_timeout_s=0.1, idle_timeout_s=1)
    ws = TimedWebSocket()
    ws.put(_session_create("sid-control-signal-data-plane"))
    ws.put({"type": "turn.signal", "event": "barge_in"})
    ws.put({"type": "session.close"})

    await handler.handle_session(ws)

    assert "runtime_signal_failed" not in {m.get("code") for m in ws.sent}
    assert "input.cancelled" in ws.sent_types()
    runtime_control = next(m for m in ws.sent if m.get("type") == "runtime.control")
    assert runtime_control["result"]["unsupported_count"] == 2


@pytest.mark.asyncio
async def test_minicpmo_native_duplex_open_unsupported_fails_session_create():
    control_result = {
        "operation": "open",
        "session_id": "sid-native-busy",
        "ok": True,
        "unsupported_count": 1,
        "error_count": 0,
        "stage_results": [
            {
                "stage_id": 0,
                "replica_id": 0,
                "result": {"supported": False, "reason": "native_duplex_session_busy"},
            }
        ],
    }
    engine = FakeEngineClient(control_result=control_result)
    chat_service = FakeChatService(engine)
    handler = OmniDuplexSessionHandler(chat_service=chat_service, config_timeout_s=0.1, idle_timeout_s=1)
    ws = TimedWebSocket()
    ws.put(_native_session_create("sid-native-busy"))
    ws.put({"type": "session.close"})

    await handler.handle_session(ws)

    assert ws.sent_types() == ["error"]
    assert ws.sent[0]["code"] == "runtime_open_unsupported"
    assert ws.sent[0]["runtime_control"]["unsupported_count"] == 1


@pytest.mark.asyncio
async def test_minicpmo_native_duplex_rejects_engine_without_fence_contract():
    class LegacyEngineClient(FakeEngineClient):
        async def open_duplex_session_async(
            self,
            session_id: str,
            *,
            session_mode: str = "duplex",
            capabilities: dict[str, object] | None = None,
            session_config: dict[str, object] | None = None,
            timeout: float | None = None,
        ) -> None:
            return await super().open_duplex_session_async(
                session_id,
                session_mode=session_mode,
                capabilities=capabilities,
                session_config=session_config,
                timeout=timeout,
            )

    engine = LegacyEngineClient()
    handler = OmniDuplexSessionHandler(
        chat_service=FakeChatService(engine),
        config_timeout_s=0.1,
        idle_timeout_s=1,
    )
    ws = TimedWebSocket()
    ws.put(_native_session_create("sid-native-no-fence-contract"))
    ws.put({"type": "session.close"})

    await handler.handle_session(ws)

    assert ws.sent_types() == ["error"]
    assert ws.sent[0]["code"] == "runtime_contract_invalid"
    assert "open_duplex_session_async" in ws.sent[0]["error"]
    assert "fence" in ws.sent[0]["error"]
    assert engine.opened == []


@pytest.mark.asyncio
async def test_minicpmo_native_duplex_audio_append_does_not_retain_pending_audio():
    engine = FakeEngineClient()
    chat_service = FakeChatService(engine)
    handler = OmniDuplexSessionHandler(chat_service=chat_service, config_timeout_s=0.1, idle_timeout_s=1)
    ws = TimedWebSocket()
    ws.put(_native_session_create("sid-native-no-pending"))
    short_chunk = base64.b64encode(b"\x00" * (800 * 4)).decode("ascii")
    ws.put({"type": "input_audio_buffer.append", "audio": short_chunk, "format": "pcm_f32le"})
    ws.put({"type": "input.cancel", "reason": "barge_in"})
    ws.put({"type": "session.close"})

    await handler.handle_session(ws)

    cancelled = next(m for m in ws.sent if m.get("type") == "input.cancelled")
    assert cancelled["cancelled"] == {"text_chunks": 0, "audio_chunks": 0}
    assert cancelled["epoch"] == 1


@pytest.mark.asyncio
async def test_minicpmo_native_duplex_rejects_text_append_before_runtime_call():
    engine = FakeEngineClient()
    chat_service = FakeChatService(engine)
    handler = OmniDuplexSessionHandler(chat_service=chat_service, config_timeout_s=0.1, idle_timeout_s=1)
    ws = TimedWebSocket()
    ws.put(_native_session_create("sid-native-text"))
    ws.put({"type": "input.text.append", "text": "hello"})
    ws.put({"type": "session.close"})

    await handler.handle_session(ws)

    error = next(m for m in ws.sent if m.get("type") == "error")
    assert error["code"] == "native_text_append_unsupported"
    assert engine.appended == []


@pytest.mark.asyncio
async def test_minicpmo_native_duplex_separates_public_and_runtime_config(monkeypatch):
    engine = FakeEngineClient()
    chat_service = FakeChatService(engine)
    handler = OmniDuplexSessionHandler(chat_service=chat_service, config_timeout_s=0.1, idle_timeout_s=1)

    async def fake_resolve_ref_audio(_, *, model_config):
        del model_config
        return [0.25] * 1600, 16000

    monkeypatch.setattr(
        "vllm_omni.experimental.fullduplex.minicpmo45.MiniCPMO45NativeDuplexServingAdapter.resolve_ref_audio",
        fake_resolve_ref_audio,
    )
    event = _native_session_create("sid-native-ref-audio", modalities=["text", "audio"])
    event["session"]["ref_audio"] = "data:audio/wav;base64,AAAA"
    ws = TimedWebSocket()
    ws.put(event)
    ws.put({"type": "session.close"})

    await handler.handle_session(ws)

    assert ws.sent_types()[0] == "session.created"
    opened_config = engine.opened_configs[0]
    opened_runtime_config = engine.opened_runtime_configs[0]
    extra_body = opened_config["extra_body"]
    assert "ref_audio_data" not in extra_body
    assert "duplex_stage_max_tokens" not in extra_body
    assert opened_runtime_config["ref_audio_format"] == "pcm_f32le"
    assert opened_runtime_config["ref_audio_sample_rate_hz"] == 16000
    assert opened_runtime_config["duplex_stage_max_tokens"] == {"0": 20, "1": 8192}
    assert opened_runtime_config["duplex_stage_sampling_params"]["0"]["top_k"] == 20
    assert opened_runtime_config["duplex_stage_sampling_params"]["0"]["top_p"] == 0.8
    assert base64.b64decode(opened_runtime_config["ref_audio_data"]) == struct.pack("<1600f", *([0.25] * 1600))


@pytest.mark.asyncio
async def test_minicpmo_native_duplex_rejects_client_runtime_config():
    engine = FakeEngineClient()
    handler = OmniDuplexSessionHandler(
        chat_service=FakeChatService(engine),
        config_timeout_s=0.1,
        idle_timeout_s=1,
    )
    event = _native_session_create("sid-native-private-config")
    event["session"]["extra_body"]["duplex_stage_sampling_params"] = {"0": {"temperature": 0.0}}
    ws = TimedWebSocket()
    ws.put(event)

    await handler.handle_session(ws)

    assert ws.sent_types() == ["error"]
    assert ws.sent[0]["code"] == "invalid_duplex_runtime_config"
    assert engine.opened == []


@pytest.mark.asyncio
async def test_minicpmo_native_duplex_audio_output_requires_ref_audio(monkeypatch):
    monkeypatch.setattr(
        MiniCPMO45NativeDuplexServingAdapter,
        "_load_native_tokenizer",
        staticmethod(lambda model_config: None),
    )

    event = _native_session_create(
        "sid-native-ref-required",
        modalities=["text", "audio"],
    )
    config = DuplexSessionConfig.from_event(event)

    with pytest.raises(ServingRuntimeConfigError) as exc_info:
        await MiniCPMO45NativeDuplexServingAdapter.prepare_runtime_config(
            config,
            model_config=SimpleNamespace(model="openbmb/MiniCPM-o-4_5"),
        )

    assert exc_info.value.code == "ref_audio_required"


@pytest.mark.asyncio
async def test_minicpmo_native_duplex_open_session_maps_missing_ref_audio_to_typed_error():
    engine = FakeEngineClient()
    handler = OmniDuplexSessionHandler(
        chat_service=FakeChatService(engine),
        config_timeout_s=0.1,
        idle_timeout_s=1,
    )
    ws = TimedWebSocket()
    ws.put(
        _native_session_create(
            "sid-native-ref-required-open",
            modalities=["text", "audio"],
        )
    )

    await handler.handle_session(ws)

    assert ws.sent_types() == ["error"]
    assert ws.sent[0]["code"] == "ref_audio_required"
    assert engine.opened == []


@pytest.mark.asyncio
async def test_minicpmo_native_duplex_text_only_omits_ref_audio_when_client_does_not_provide_it(monkeypatch):
    monkeypatch.setattr(
        MiniCPMO45NativeDuplexServingAdapter,
        "_load_native_tokenizer",
        staticmethod(lambda model_config: None),
    )

    event = _native_session_create("sid-native-default-ref", modalities=["text"])
    config = DuplexSessionConfig.from_event(event)

    runtime_config = await MiniCPMO45NativeDuplexServingAdapter.prepare_runtime_config(
        config,
        model_config=SimpleNamespace(model="openbmb/MiniCPM-o-4_5"),
    )

    assert "ref_audio_data" not in runtime_config
    assert "ref_audio_format" not in runtime_config
    assert "ref_audio_sample_rate_hz" not in runtime_config


@pytest.mark.asyncio
async def test_minicpmo_native_duplex_preserves_ref_audio_channels_until_normalize(monkeypatch):
    class FakeMediaConnector:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def fetch_audio_async(self, ref_audio):
            assert ref_audio == "data:audio/wav;base64,AAAA"
            return np.tile(np.array([[0.25, -0.25], [0.5, -0.5]], dtype=np.float32), (800, 1)), 16000

    monkeypatch.setattr(
        "vllm_omni.experimental.fullduplex.minicpmo45.adapter.MediaConnector",
        FakeMediaConnector,
    )

    engine = FakeEngineClient()
    chat_service = FakeChatService(engine)
    handler = OmniDuplexSessionHandler(chat_service=chat_service, config_timeout_s=0.1, idle_timeout_s=1)
    event = _native_session_create("sid-native-ref-audio-stereo", modalities=["text", "audio"])
    event["session"]["ref_audio"] = "data:audio/wav;base64,AAAA"
    ws = TimedWebSocket()
    ws.put(event)
    ws.put({"type": "session.close"})

    await handler.handle_session(ws)

    runtime_config = engine.opened_runtime_configs[0]
    assert base64.b64decode(runtime_config["ref_audio_data"]) == struct.pack("<1600f", *([0.0] * 1600))


@pytest.mark.asyncio
async def test_minicpmo_native_duplex_rejects_ref_audio_path():
    engine = FakeEngineClient()
    chat_service = FakeChatService(engine)
    handler = OmniDuplexSessionHandler(chat_service=chat_service, config_timeout_s=0.1, idle_timeout_s=1)
    event = _native_session_create("sid-native-ref-path")
    event["session"]["extra_body"] = {
        "minicpmo45_native_duplex": True,
        "ref_audio_path": "/tmp/ref.wav",
    }
    ws = TimedWebSocket()
    ws.put(event)

    await handler.handle_session(ws)

    assert ws.sent_types() == ["error"]
    assert ws.sent[0]["code"] == "unsupported_ref_audio_path"
    assert engine.opened == []


def test_minicpmo_native_duplex_explicit_barge_in_request_is_deferred_as_listen():
    engine = FakeEngineClient()
    handler = OmniDuplexSessionHandler(
        chat_service=FakeChatService(engine),
        config_timeout_s=0.1,
        idle_timeout_s=1,
    )
    session = DuplexSession(
        session_id="sid-native-explicit-barge-disabled",
        config=DuplexSessionConfig(overlap_policy=DuplexOverlapPolicy.BARGE_IN_ON_SPEECH.value),
        capabilities=DuplexCapabilities.minicpmo45_native(),
    )
    payload = {
        "type": "audio",
        "audio": _pcm_f32_b64(16000, value=0.05),
        "format": "pcm_f32le",
        "sample_rate_hz": 16000,
        "is_speech": True,
    }

    decision = handler._overlap_decision(
        session,
        {"force_barge_in": True},
        payload,
    )

    assert decision == {
        "action": "listen",
        "reason": "barge_in_unsupported",
        "duration_ms": 1000,
        "overlap_speech_ms": 1000,
        "buffer_audio": True,
        "defer_runtime_append": True,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "barge_in_event",
    [
        {"type": "barge_in"},
        {"type": "turn.signal", "event": "barge_in"},
    ],
)
async def test_minicpmo_native_duplex_rejects_unadvertised_barge_in_control(
    barge_in_event: dict[str, object],
):
    engine = FakeEngineClient()
    handler = OmniDuplexSessionHandler(
        chat_service=FakeChatService(engine),
        config_timeout_s=0.1,
        idle_timeout_s=1,
    )
    ws = TimedWebSocket()
    ws.put(_native_session_create("sid-native-barge-control-disabled"))
    ws.put(barge_in_event)
    ws.put({"type": "session.close"})

    await handler.handle_session(ws)

    error = next(message for message in ws.sent if message.get("code") == "barge_in_unsupported")
    assert error["session_id"] == "sid-native-barge-control-disabled"
    assert ("sid-native-barge-control-disabled", "barge_in") not in engine.signals


@pytest.mark.asyncio
async def test_minicpmo_native_duplex_explicit_non_speech_stays_listening_without_append():
    engine = FakeEngineClient()
    chat_service = FakeChatService(engine)
    handler = OmniDuplexSessionHandler(chat_service=chat_service, config_timeout_s=0.1, idle_timeout_s=1)
    ws = TimedWebSocket()
    ws.put(_native_session_create("sid-native-silence"))
    ws.put(
        {
            "type": "input_audio_buffer.append",
            "audio": "AAAA",
            "format": "pcm_f32le",
            "is_speech": False,
        }
    )
    ws.put({"type": "session.close"})

    await handler.handle_session(ws)

    assert engine.appended == []
    assert "response.listen" in ws.sent_types()
    assert "response.created" not in ws.sent_types()
    assert "response.output_audio.delta" not in ws.sent_types()


@pytest.mark.asyncio
async def test_minicpmo_native_duplex_drains_data_plane_stream_until_done():
    def _stage_output(samples: int, *, finished: bool):
        return SimpleNamespace(
            request_id="duplex-sid-native-stream-e0-stage0-s1",
            finished=finished,
            outputs=[
                SimpleNamespace(
                    text="",
                    multimodal_output={
                        "audio": np.zeros(samples, dtype=np.float32),
                        "sr": 24000,
                    },
                )
            ],
        )

    control_result = {
        "operation": "append",
        "session_id": "sid-native-stream",
        "ok": True,
        "unsupported_count": 0,
        "error_count": 0,
        "stage_results": [
            {
                "stage_id": 0,
                "replica_id": 0,
                "result": {
                    "supported": True,
                    "implementation_level": "model_native_duplex",
                    "data_plane_append": True,
                    "request_id": "duplex-sid-native-stream-e0-stage0-s1",
                    "response_stage_id": 1,
                },
            }
        ],
        "data_plane_outputs": [_stage_output(10, finished=False)],
    }
    engine = FakeEngineClient(
        append_result=control_result,
        collect_outputs=[
            [_stage_output(20, finished=False)],
            [_stage_output(30, finished=True)],
        ],
    )
    chat_service = FakeChatService(engine)
    handler = OmniDuplexSessionHandler(chat_service=chat_service, config_timeout_s=0.1, idle_timeout_s=1)

    def close_on_second_created(ws: TimedWebSocket, data: dict[str, Any]) -> None:
        if ws.sent_types().count("response.created") >= 2:
            ws.put({"type": "session.close"})

    ws = TimedWebSocket(on_send=close_on_second_created)
    ws.put(_native_session_create("sid-native-stream"))
    ws.put({"type": "input_audio_buffer.append", "audio": "AAAA", "format": "pcm_f32le"})

    await handler.handle_session(ws)

    deltas = [m for m in ws.sent if m.get("type") == "response.output_audio.delta"]
    assert [m["audio"] for m in deltas] == ["wav-10", "wav-10", "wav-10"]
    assert [m["end_of_turn"] for m in deltas] == [False, False, True]
    assert len([m for m in ws.sent if m.get("type") == "response.done"]) == 1
    assert engine.collected == [
        ("duplex-sid-native-stream-e0-stage0-s1", 1),
        ("duplex-sid-native-stream-e0-stage0-s1", 1),
    ]


@pytest.mark.asyncio
async def test_minicpmo_auto_response_restarts_drain_when_append_races_idle_exit(
    monkeypatch: pytest.MonkeyPatch,
):
    request_id = "duplex-sid-native-late-output-e0-stage0"
    late_output = object()
    engine = FakeEngineClient(
        collect_outputs=[[], [], [], [late_output]],
        collect_delay_s=0.05,
    )
    handler = OmniDuplexSessionHandler(
        chat_service=FakeChatService(engine),
        config_timeout_s=0.1,
        idle_timeout_s=1,
    )
    session = DuplexSession(
        session_id="sid-native-late-output",
        config=DuplexSessionConfig(extra_body={"auto_response": True}),
    )
    session.capabilities = DuplexCapabilities.minicpmo45_native()
    session.bind_request(request_id)
    native = handler._minicpmo_session_state(session)
    projected_batches: list[object] = []
    projected = asyncio.Event()

    async def project_late_output(_send_json, result, **_kwargs):
        projected_batches.append(result)
        projected.set()
        session.close()
        return None, True

    monkeypatch.setattr(handler, "_send_native_duplex_events", project_late_output)
    result = {
        "stage_results": [
            {
                "result": {
                    "data_plane_append": True,
                    "request_id": request_id,
                    "response_stage_id": 1,
                }
            }
        ]
    }

    assert (
        await handler._start_native_data_plane_stream_task(
            None,
            result,
            session=session,
            expected_epoch=session.epoch,
        )
        is True
    )
    first_drain = native.data_plane_task
    assert first_drain is not None
    await asyncio.sleep(0.01)

    assert (
        await handler._start_native_data_plane_stream_task(
            None,
            result,
            session=session,
            expected_epoch=session.epoch,
        )
        is False
    )
    await asyncio.wait_for(projected.wait(), timeout=1)
    await asyncio.wait_for(first_drain, timeout=1)

    assert len(engine.collected) == 4
    assert projected_batches == [{"data_plane_outputs": [late_output]}]


@pytest.mark.asyncio
async def test_minicpmo_native_duplex_ignores_outputs_after_data_plane_turn_done():
    request_id = "duplex-sid-native-post-done-e0-stage0-s1"

    def _stage_output(samples: int, *, turn_end: bool = False):
        return SimpleNamespace(
            request_id=request_id,
            finished=False,
            outputs=[
                SimpleNamespace(
                    text="hello",
                    multimodal_output={
                        "audio": np.zeros(samples, dtype=np.float32),
                        "sr": 24000,
                        "meta": {
                            "turn_end": turn_end,
                            "duplex_turn_id": 0,
                            "duplex_epoch": 0,
                        },
                    },
                )
            ],
        )

    control_result = {
        "operation": "append",
        "session_id": "sid-native-post-done",
        "ok": True,
        "unsupported_count": 0,
        "error_count": 0,
        "stage_results": [
            {
                "stage_id": 0,
                "replica_id": 0,
                "result": {
                    "supported": True,
                    "implementation_level": "model_native_duplex",
                    "data_plane_append": True,
                    "request_id": request_id,
                    "response_stage_id": 1,
                },
            }
        ],
        "data_plane_outputs": [_stage_output(10)],
    }
    engine = FakeEngineClient(
        append_result=control_result,
        collect_outputs=[
            [_stage_output(20, turn_end=True)],
            [_stage_output(30)],
        ],
    )
    chat_service = FakeChatService(engine)
    handler = OmniDuplexSessionHandler(chat_service=chat_service, config_timeout_s=0.1, idle_timeout_s=1)

    def close_on_done(ws: TimedWebSocket, data: dict[str, Any]) -> None:
        if data.get("type") == "response.done":
            ws.put({"type": "session.close"})

    event = _native_session_create("sid-native-post-done")
    event["session"]["extra_body"]["auto_response"] = True
    ws = TimedWebSocket(on_send=close_on_done)
    ws.put(event)
    ws.put({"type": "input_audio_buffer.append", "audio": "AAAA", "format": "pcm_f32le"})

    await handler.handle_session(ws)

    assert ws.sent_types().count("response.created") == 1
    assert ws.sent_types().count("response.done") == 1
    deltas = [m for m in ws.sent if m.get("type") == "response.output_audio.delta"]
    assert [m["audio"] for m in deltas] == ["wav-10", "wav-10"]
    assert engine.collected
    assert all(collected == (request_id, 1) for collected in engine.collected)


@pytest.mark.asyncio
async def test_minicpmo_native_duplex_accepts_next_append_after_response_done():
    def _stage_output(*, finished: bool):
        return SimpleNamespace(
            request_id="duplex-sid-native-next-turn-e0-stage0-s1",
            finished=finished,
            outputs=[
                SimpleNamespace(
                    text="",
                    multimodal_output={
                        "audio": np.zeros(10, dtype=np.float32),
                        "sr": 24000,
                    },
                )
            ],
        )

    control_result = {
        "operation": "append",
        "session_id": "sid-native-next-turn",
        "ok": True,
        "unsupported_count": 0,
        "error_count": 0,
        "stage_results": [
            {
                "stage_id": 0,
                "replica_id": 0,
                "result": {
                    "supported": True,
                    "implementation_level": "model_native_duplex",
                    "data_plane_append": True,
                    "request_id": "duplex-sid-native-next-turn-e0-stage0-s1",
                    "response_stage_id": 1,
                },
            }
        ],
        "data_plane_outputs": [_stage_output(finished=True)],
    }
    engine = FakeEngineClient(append_result=control_result)
    chat_service = FakeChatService(engine)
    handler = OmniDuplexSessionHandler(chat_service=chat_service, config_timeout_s=0.1, idle_timeout_s=1)
    done_count = 0

    def send_next_turn_or_close(ws: TimedWebSocket, data: dict[str, Any]) -> None:
        nonlocal done_count
        if data.get("type") != "response.done":
            return
        done_count += 1
        if done_count == 1:
            ws.put({"type": "input_audio_buffer.append", "audio": "BBBB", "format": "pcm_f32le"})
            ws.put({"type": "input_audio_buffer.commit", "final": True})
            ws.put({"type": "response.create"})
        else:
            ws.put({"type": "session.close"})

    ws = TimedWebSocket(on_send=send_next_turn_or_close, receive_timeout_s=5)
    create = _native_session_create("sid-native-next-turn")
    create["session"]["idle_timeout_s"] = 5
    ws.put(create)
    ws.put({"type": "input_audio_buffer.append", "audio": "AAAA", "format": "pcm_f32le"})

    await handler.handle_session(ws)

    assert len(engine.appended) >= 2, [
        (message.get("type"), message.get("code"), message.get("reason")) for message in ws.sent
    ]
    assert len([m for m in ws.sent if m.get("type") == "response.done"]) == 2


@pytest.mark.asyncio
async def test_minicpmo_native_auto_response_commit_finalizes_after_streamed_full_chunk():
    control_result = {
        "operation": "append",
        "session_id": "sid-native-auto-finalize",
        "ok": True,
        "unsupported_count": 0,
        "error_count": 0,
        "stage_results": [],
    }
    engine = FakeEngineClient(control_result=control_result)
    handler = OmniDuplexSessionHandler(
        chat_service=FakeChatService(engine),
        config_timeout_s=0.1,
        idle_timeout_s=1,
    )
    ws = TimedWebSocket()
    event = _native_session_create("sid-native-auto-finalize")
    event["session"]["extra_body"]["auto_response"] = True
    ws.put(event)
    ws.put(
        {
            "type": "input_audio_buffer.append",
            "audio": _pcm_f32_b64(16000),
            "format": "pcm_f32le",
            "sample_rate_hz": 16000,
        }
    )
    ws.put({"type": "input_audio_buffer.commit", "final": True})
    ws.put({"type": "session.close"})

    await handler.handle_session(ws)

    assert len(engine.appended) == 1
    assert engine.appended[0][3] is False
    assert "response.created" not in ws.sent_types()


@pytest.mark.asyncio
async def test_minicpmo_native_auto_response_commit_finalizes_with_active_data_plane_stream():
    request_id = "duplex-sid-native-auto-active-stream-stage0"
    control_result = {
        "operation": "append",
        "session_id": "sid-native-auto-active-stream",
        "ok": True,
        "unsupported_count": 0,
        "error_count": 0,
        "stage_results": [
            {
                "result": {
                    "data_plane_append": True,
                    "request_id": request_id,
                    "response_stage_id": 1,
                }
            }
        ],
    }
    engine = FakeEngineClient(
        control_result=control_result,
        collect_outputs=[[], [], []],
        collect_delay_s=0.05,
    )
    handler = OmniDuplexSessionHandler(
        chat_service=FakeChatService(engine),
        config_timeout_s=0.1,
        idle_timeout_s=1,
    )
    ws = TimedWebSocket()
    event = _native_session_create("sid-native-auto-active-stream")
    event["session"]["extra_body"]["auto_response"] = True
    ws.put(event)
    ws.put(
        {
            "type": "input_audio_buffer.append",
            "audio": _pcm_f32_b64(16000),
            "format": "pcm_f32le",
            "sample_rate_hz": 16000,
        }
    )
    ws.put({"type": "input_audio_buffer.commit", "final": True})
    ws.put({"type": "session.close"})

    await handler.handle_session(ws)

    assert len(engine.appended) == 1
    assert engine.appended[0][3] is False
    assert "response.created" not in ws.sent_types()


@pytest.mark.asyncio
async def test_minicpmo_native_auto_response_accepts_realtime_commit_after_stream_consumed():
    engine = FakeEngineClient()
    handler = OmniDuplexSessionHandler(
        chat_service=FakeChatService(engine),
        config_timeout_s=0.1,
        idle_timeout_s=1,
    )
    ws = TimedWebSocket()
    event = _native_session_create("sid-native-auto-consumed-stream")
    event["session"]["extra_body"]["auto_response"] = True
    ws.put(event)
    ws.put(
        {
            "type": "input_audio_buffer.append",
            "audio": _pcm_f32_b64(16000),
            "format": "pcm_f32le",
            "sample_rate_hz": 16000,
        }
    )
    ws.put(
        {
            "type": "input_audio_buffer.commit",
            "final": True,
            "realtime_item_id": "item-consumed-stream",
        }
    )
    ws.put({"type": "session.close"})

    await handler.handle_session(ws)

    assert not any(message.get("code") == "input_audio_buffer_empty" for message in ws.sent)
    committed = next(message for message in ws.sent if message.get("type") == "input.committed")
    assert committed["realtime_item_id"] == "item-consumed-stream"
    assert committed["native_audio"] is True


@pytest.mark.asyncio
async def test_minicpmo_native_auto_response_post_response_commit_ignores_stale_progress_state():
    session_id = "sid-native-auto-post-response"
    engine = FakeEngineClient()
    handler = OmniDuplexSessionHandler(
        chat_service=FakeChatService(engine),
        config_timeout_s=0.1,
        idle_timeout_s=1,
    )

    def on_send(ws: TimedWebSocket, data: dict[str, Any]) -> None:
        if data.get("type") != "session.created":
            return
        session = handler._registry.get(session_id)
        assert session is not None
        ws.put(
            {
                "type": "input_audio_buffer.append",
                "audio": _pcm_f32_b64(8000),
                "format": "pcm_f32le",
                "sample_rate_hz": 16000,
            }
        )
        ws.put({"type": "input_audio_buffer.commit", "final": True})
        ws.put({"type": "session.close"})

    ws = TimedWebSocket(on_send=on_send)
    create = _native_session_create(session_id)
    create["session"]["extra_body"]["auto_response"] = True
    ws.put(create)

    await handler.handle_session(ws)

    assert len(engine.appended) == 1
    assert engine.appended[0][3] is True
    assert "input.committed" in ws.sent_types()
    assert not any(message.get("code") == "response_already_active" for message in ws.sent)


def test_realtime_overlap_commit_uses_current_generation_identity():
    handler, session = _auto_response_context("sid-realtime-reserved-input-turn")
    session.turn_id = 1
    session.bind_response_turn(1)
    session.turn_id = 3
    committed = handler._commit_native_audio_input(session)

    assert committed.turn_id == 3
    assert session.turn_id == 3


@pytest.mark.asyncio
async def test_minicpmo_native_auto_response_keeps_request_bound_for_segment_continuation():
    request_id = "duplex-sid-native-auto-continuation-e0-stage0"
    control_result = {
        "operation": "append",
        "session_id": "sid-native-auto-continuation",
        "ok": True,
        "unsupported_count": 0,
        "error_count": 0,
        "stage_results": [
            {
                "result": {
                    "data_plane_append": True,
                    "request_id": request_id,
                    "response_stage_id": 1,
                }
            }
        ],
    }
    terminal_segment = _duplex_tts_output(
        request_id=request_id,
        samples=0,
        finished=True,
        tts_is_last_chunk=True,
        token_ids=[151645],
    )
    engine = FakeEngineClient(
        control_result=control_result,
        collect_outputs=[[terminal_segment], [], []],
        collect_delay_s=0.05,
    )
    handler = OmniDuplexSessionHandler(
        chat_service=FakeChatService(engine),
        config_timeout_s=0.1,
        idle_timeout_s=1,
    )
    ws = TimedWebSocket()
    event = _native_session_create("sid-native-auto-continuation")
    event["session"]["extra_body"]["auto_response"] = True
    ws.put(event)
    ws.put(
        {
            "type": "input_audio_buffer.append",
            "audio": _pcm_f32_b64(16000),
            "format": "pcm_f32le",
            "sample_rate_hz": 16000,
        }
    )
    ws.put({"type": "input_audio_buffer.commit", "final": True})

    await handler.handle_session(ws)

    assert len(engine.appended) >= 3
    _, mode, payload, final = engine.appended[2]
    assert mode == "append_audio_chunk"
    assert payload["duplex_turn_id"] == 0
    assert final is False


@pytest.mark.asyncio
async def test_minicpmo_native_auto_response_real_input_waits_for_submitted_silence_tail():
    request_id = "duplex-sid-native-silence-tail-e0-stage0"

    class SubmittedSilenceEngine(FakeEngineClient):
        def __init__(self) -> None:
            control_result = {
                "operation": "append",
                "session_id": "sid-native-silence-tail",
                "ok": True,
                "unsupported_count": 0,
                "error_count": 0,
                "stage_results": [
                    {
                        "result": {
                            "data_plane_append": True,
                            "request_id": request_id,
                            "response_stage_id": 1,
                        }
                    }
                ],
            }
            terminal_segment = _duplex_tts_output(
                request_id=request_id,
                samples=0,
                finished=True,
                tts_is_last_chunk=True,
                token_ids=[151645],
            )
            super().__init__(
                control_result=control_result,
                collect_outputs=[[terminal_segment], [], []],
                collect_delay_s=0.01,
            )
            self.silence_append_started = asyncio.Event()
            self.release_silence_append = asyncio.Event()
            self.append_sequence: list[str] = []
            self.real_append_count = 0

        async def append_duplex_input_async(self, session_id: str, **kwargs):
            kwargs.pop("expected_epoch", None)
            payload = kwargs.get("payload")
            is_silence_continuation = (
                isinstance(payload, dict)
                and payload.get("format") == "pcm_f32le"
                and payload.get("audio") == OmniDuplexSessionHandler._NATIVE_SILENCE_UNIT_PAYLOAD_AUDIO
            )
            if is_silence_continuation:
                result = await super().append_duplex_input_async(session_id, **kwargs)
                self.append_sequence.append("silence-started")
                self.silence_append_started.set()
                await self.release_silence_append.wait()
                self.append_sequence.append("silence-returned")
                return result
            result = await super().append_duplex_input_async(session_id, **kwargs)
            if (
                isinstance(payload, dict)
                and payload.get("audio") != OmniDuplexSessionHandler._NATIVE_SILENCE_UNIT_PAYLOAD_AUDIO
            ):
                self.real_append_count += 1
                self.append_sequence.append(f"real-{self.real_append_count}")
            return result

    engine = SubmittedSilenceEngine()
    handler = OmniDuplexSessionHandler(
        chat_service=FakeChatService(engine),
        config_timeout_s=0.1,
        idle_timeout_s=1,
    )
    ws = TimedWebSocket(receive_timeout_s=2.0)
    event = _native_session_create("sid-native-silence-tail")
    event["session"]["extra_body"]["auto_response"] = True
    ws.put(event)
    ws.put(
        {
            "type": "input_audio_buffer.append",
            "audio": _pcm_f32_b64(16000, value=0.05),
            "format": "pcm_f32le",
            "sample_rate_hz": 16000,
        }
    )
    handler_task = asyncio.create_task(handler.handle_session(ws))

    await asyncio.wait_for(engine.silence_append_started.wait(), timeout=1)
    ws.put(
        {
            "type": "input_audio_buffer.append",
            "audio": _pcm_f32_b64(16000, value=0.07),
            "format": "pcm_f32le",
            "sample_rate_hz": 16000,
        }
    )
    await asyncio.sleep(0.05)

    assert engine.real_append_count == 1
    assert engine.append_sequence == ["real-1", "silence-started"]

    engine.release_silence_append.set()
    ws.put({"type": "session.close"})
    await asyncio.wait_for(handler_task, timeout=2)

    assert engine.real_append_count == 2
    assert engine.append_sequence == ["real-1", "silence-started", "silence-returned", "real-2"]


@pytest.mark.asyncio
async def test_minicpmo_native_duplex_uses_segment_text_metadata_for_transcript_cursor():
    request_id = "duplex-sid-native-repeat-text-e0-stage0"

    def _stage_output(
        text: str,
        samples: int,
        *,
        segment_text: str,
        turn_id: int = 0,
        turn_end: bool = False,
        token_ids: list[int] | None = None,
    ):
        return SimpleNamespace(
            request_id=request_id,
            finished=False,
            outputs=[
                SimpleNamespace(
                    text=text,
                    token_ids=list(token_ids or []),
                    multimodal_output={
                        "audio": np.zeros(samples, dtype=np.float32),
                        "sr": 24000,
                        "meta.llm_output_text_utf8": np.frombuffer(segment_text.encode("utf-8"), dtype=np.uint8),
                        "meta": {"duplex_turn_id": turn_id, "turn_end": turn_end},
                    },
                )
            ],
        )

    handler = OmniDuplexSessionHandler(chat_service=FakeChatService(FakeEngineClient()))
    session = DuplexSession(
        session_id="sid-native-repeat-text",
        config=DuplexSessionConfig(extra_body={"auto_response": True}),
    )
    sent: list[dict[str, Any]] = []

    async def send_json(payload: dict[str, Any]) -> None:
        sent.append(payload)

    async def emit(output: object) -> None:
        result = {"data_plane_outputs": [output]}
        for native_result in handler._minicpmo_data_plane.project(
            result,
            context=_data_plane_context(session),
        ):
            await handler._send_one_native_duplex_event(send_json, native_result, session=session)

    await emit(_stage_output("same reply", 24000, segment_text="same reply"))
    await emit(
        _stage_output("same replysame reply", 24000, segment_text="same reply", turn_end=True, token_ids=[151645])
    )

    await emit(_stage_output("same replysame replysame reply", 48000, segment_text="same reply", turn_id=1))
    await emit(
        _stage_output(
            "same replysame replysame replysame reply",
            48000,
            segment_text="same reply",
            turn_id=1,
            turn_end=True,
            token_ids=[151645],
        )
    )

    created = [m for m in sent if m.get("type") == "response.created"]
    done = [m for m in sent if m.get("type") == "response.done"]
    deltas = [m for m in sent if m.get("type") == "response.output_audio.delta" and m.get("audio")]
    protocol = NativeRealtimeSessionProtocol(TimedWebSocket())  # type: ignore[arg-type]
    realtime_events = [event for payload in sent for event in protocol._from_duplex_event(payload)]
    transcript_deltas = [
        m for m in realtime_events if m.get("type") == "response.audio_transcript.delta" and m.get("delta")
    ]
    assert len(created) == 2
    assert len(done) == 2
    assert len(deltas) == 2
    assert [m["text"] for m in deltas] == ["same reply", "same reply"]
    assert [m["audio"] for m in deltas] == ["wav-24000", "wav-24000"]
    assert [m["delta"] for m in transcript_deltas] == ["same reply", "same reply"]
    assert [m["response_id"] for m in deltas] == [m["response_id"] for m in created]
    assert [m["response_id"] for m in done] == [m["response_id"] for m in created]


@pytest.mark.asyncio
async def test_minicpmo_native_duplex_continuous_speak_reuses_active_response_until_turn_end():
    request_id = "duplex-sid-native-continuous-speak-e0-stage0"

    def _stage_output(
        text: str,
        samples: int,
        *,
        turn_end: bool = False,
        token_ids: list[int] | None = None,
    ):
        return SimpleNamespace(
            request_id=request_id,
            finished=turn_end,
            outputs=[
                SimpleNamespace(
                    text=text,
                    token_ids=list(token_ids or []),
                    multimodal_output={},
                )
            ],
            multimodal_output={
                "audio": np.zeros(samples, dtype=np.float32),
                "sr": 24000,
                "meta.duplex_turn_id": np.array([0], dtype=np.int32),
                "meta.duplex_epoch": np.array([0], dtype=np.int32),
                "meta.turn_end": np.array([int(turn_end)], dtype=np.int32),
                "meta.tts_is_last_chunk": np.array([int(turn_end)], dtype=np.int32),
            },
        )

    handler = OmniDuplexSessionHandler(chat_service=FakeChatService(FakeEngineClient()))
    session = DuplexSession(
        session_id="sid-native-continuous-speak",
        config=DuplexSessionConfig(extra_body={"auto_response": True}),
    )
    sent: list[dict[str, Any]] = []

    async def send_json(payload: dict[str, Any]) -> None:
        sent.append(payload)

    async def emit(output: object) -> None:
        result = {"data_plane_outputs": [output]}
        for native_result in handler._minicpmo_data_plane.project(
            result,
            context=_data_plane_context(session),
        ):
            await handler._send_one_native_duplex_event(send_json, native_result, session=session)

    await emit(_stage_output("hello", 24000))
    await emit(_stage_output("hello again", 48000))

    created = [m for m in sent if m.get("type") == "response.created"]
    deltas = [m for m in sent if m.get("type") == "response.output_audio.delta" and m.get("audio")]
    assert len(created) == 1
    assert len(deltas) == 2
    assert {m["response_id"] for m in deltas} == {created[0]["response_id"]}
    assert "response.done" not in [m.get("type") for m in sent]

    await emit(_stage_output("hello again", 48000, turn_end=True, token_ids=[151645]))

    done = [m for m in sent if m.get("type") == "response.done"]
    assert len(done) == 1
    assert done[0]["response_id"] == created[0]["response_id"]
    assert session.active_response_id is None
    assert session.turn_id == 1


@pytest.mark.asyncio
async def test_continuous_response_metrics_accumulate_only_owned_model_units():
    handler = OmniDuplexSessionHandler(chat_service=FakeChatService(FakeEngineClient()))
    session = DuplexSession(
        session_id="sid-response-metrics",
        config=DuplexSessionConfig(extra_body={"auto_response": True}),
    )
    sent: list[dict[str, Any]] = []

    async def send_json(payload: dict[str, Any]) -> None:
        sent.append(payload)

    async def emit(*, text: str, audio: str, tokens: int, ttft_ms: float, itls: list[float]) -> None:
        await handler._send_one_native_duplex_event(
            send_json,
            {
                "supported": True,
                "stage_role": "tts",
                "is_listen": False,
                "data_plane_request_id": "duplex-sid-response-metrics-e0-stage0",
                "text": text,
                "audio_data": audio,
                "audio_format": "pcm16",
                "audio_duration_ms": 100,
                "end_of_turn": False,
                "model_turn_id": 0,
                "stage_metrics": {
                    "0": {
                        "num_tokens_out": tokens,
                        "stage_gen_time_ms": 80.0,
                        "vllm_ttft_ms": ttft_ms,
                        "vllm_tpot_ms": 9.0,
                        "vllm_itls_ms": itls,
                    }
                },
            },
            session=session,
        )

    await emit(text="first", audio="audio-a", tokens=4, ttft_ms=50.0, itls=[10.0, 11.0])
    await emit(text="second", audio="audio-b", tokens=6, ttft_ms=70.0, itls=[12.0, 13.0, 14.0])

    deltas = [payload for payload in sent if payload.get("type") == "response.output_audio.delta"]
    assert len(deltas) == 2
    first_metrics = deltas[0]["vllm_omni"]["stage_metrics"]["0"]
    second_metrics = deltas[1]["vllm_omni"]["stage_metrics"]["0"]
    assert first_metrics["num_tokens_out"] == 4
    assert second_metrics["num_tokens_out"] == 10
    assert second_metrics["vllm_ttft_ms"] == 50.0
    assert second_metrics["vllm_itls_ms"] == [10.0, 11.0, 12.0, 13.0, 14.0]

    first_response_id = deltas[0]["response_id"]
    session.end_response()
    await emit(text="third", audio="audio-c", tokens=3, ttft_ms=90.0, itls=[15.0, 16.0])

    third_delta = [payload for payload in sent if payload.get("type") == "response.output_audio.delta"][-1]
    third_metrics = third_delta["vllm_omni"]["stage_metrics"]["0"]
    assert third_delta["response_id"] != first_response_id
    assert third_metrics["num_tokens_out"] == 3
    assert third_metrics["vllm_ttft_ms"] == 90.0
    assert third_metrics["vllm_itls_ms"] == [15.0, 16.0]


@pytest.mark.asyncio
async def test_minicpmo_native_duplex_drops_old_turn_audio_while_new_response_is_active():
    request_id = "duplex-sid-native-old-turn-audio-e0-stage0"
    handler = OmniDuplexSessionHandler(chat_service=FakeChatService(FakeEngineClient()))
    session = DuplexSession(
        session_id="sid-native-old-turn-audio",
        config=DuplexSessionConfig(extra_body={"auto_response": True}),
    )
    response_id = session.begin_response(turn_id=1)
    session.turn_id = 2
    session.bind_request(request_id)
    ws = TimedWebSocket()

    close_reason, emitted = await handler._send_one_native_duplex_event(
        ws.send_json,
        {
            "supported": True,
            "stage_role": "tts",
            "is_listen": False,
            "data_plane_request_id": request_id,
            "text": "stale",
            "audio_data": "old-audio",
            "audio_format": "pcm16",
            "audio_duration_ms": 100,
            "end_of_turn": False,
            "model_turn_id": 0,
            "uses_model_runner_scheduler": True,
            "runner_kv_backed": True,
            "runtime_impl": "scheduler_data_plane",
        },
        session=session,
        expected_epoch=session.epoch,
    )

    assert close_reason is None
    assert emitted is False
    assert session.active_response_id == response_id
    assert ws.sent == []


@pytest.mark.asyncio
async def test_minicpmo_native_duplex_future_turn_audio_starts_next_response():
    request_id = "duplex-sid-native-future-turn-audio-e0-stage0"
    handler = OmniDuplexSessionHandler(chat_service=FakeChatService(FakeEngineClient()))
    session = DuplexSession(
        session_id="sid-native-future-turn-audio",
        config=DuplexSessionConfig(extra_body={"auto_response": True}),
    )
    first_response_id = session.begin_response(turn_id=0)
    session.bind_request(request_id)
    ws = TimedWebSocket()

    close_reason, emitted = await handler._send_one_native_duplex_event(
        ws.send_json,
        {
            "supported": True,
            "stage_role": "tts",
            "is_listen": False,
            "data_plane_request_id": request_id,
            "text": "next",
            "audio_data": "next-audio",
            "audio_format": "pcm16",
            "audio_duration_ms": 100,
            "end_of_turn": False,
            "model_turn_id": 1,
            "uses_model_runner_scheduler": True,
            "runner_kv_backed": True,
            "runtime_impl": "scheduler_data_plane",
        },
        session=session,
        expected_epoch=session.epoch,
    )

    assert close_reason is None
    assert emitted is True
    sent_types = [event["type"] for event in ws.sent]
    assert sent_types == [
        "response.done",
        "response.created",
        "response.speak",
        "response.output_audio.delta",
    ]
    assert ws.sent[0]["response_id"] == first_response_id
    assert ws.sent[1]["response_id"] != first_response_id
    assert ws.sent[3]["response_id"] == ws.sent[1]["response_id"]
    assert ws.sent[3]["vllm_omni"]["model_turn_id"] == 1
    assert session.active_response_turn_id == 1
    assert session.turn_id == 1


@pytest.mark.asyncio
async def test_minicpmo_native_duplex_drops_old_turn_listen_while_new_response_is_active():
    request_id = "duplex-sid-native-old-turn-listen-e0-stage0"
    handler = OmniDuplexSessionHandler(chat_service=FakeChatService(FakeEngineClient()))
    session = DuplexSession(
        session_id="sid-native-old-turn-listen",
        config=DuplexSessionConfig(extra_body={"auto_response": True}),
    )
    response_id = session.begin_response(turn_id=1)
    session.turn_id = 2
    session.bind_request(request_id)
    ws = TimedWebSocket()

    close_reason, emitted = await handler._send_one_native_duplex_event(
        ws.send_json,
        {
            "supported": True,
            "stage_role": "llm",
            "is_listen": True,
            "model_listen": True,
            "data_plane_request_id": request_id,
            "end_of_turn": False,
            "model_turn_id": 0,
            "uses_model_runner_scheduler": True,
            "runner_kv_backed": True,
            "runtime_impl": "scheduler_data_plane",
        },
        session=session,
        expected_epoch=session.epoch,
    )

    assert close_reason is None
    assert emitted is False
    assert session.active_response_id == response_id
    assert ws.sent == []


@pytest.mark.asyncio
async def test_minicpmo_native_duplex_silence_continuation_requires_active_request():
    engine = FakeEngineClient()
    chat_service = FakeChatService(engine)
    handler = OmniDuplexSessionHandler(chat_service=chat_service, config_timeout_s=0.1, idle_timeout_s=1)
    session = DuplexSession(
        session_id="sid-native-stale-continuation",
        config=DuplexSessionConfig(extra_body={"auto_response": True}),
    )
    session.begin_response()
    session.bind_request("duplex-sid-native-stale-continuation-stage0")
    session.mark_audio_sent(100)
    sent: list[dict[str, Any]] = []

    async def send_json(data: dict[str, Any]) -> None:
        sent.append(data)

    _install_direct_silence_scheduler(handler, session)

    await handler._maybe_continue_native_response(send_json, session=session, expected_epoch=session.epoch)
    session.clear_request()
    await asyncio.sleep(0.01)

    assert engine.appended == []


@pytest.mark.asyncio
async def test_minicpmo_auto_response_segment_complete_continues_until_model_listen():
    engine = FakeEngineClient()
    chat_service = FakeChatService(engine)
    handler = OmniDuplexSessionHandler(chat_service=chat_service, config_timeout_s=0.1, idle_timeout_s=1)
    session = DuplexSession(
        session_id="sid-native-segment-complete",
        config=DuplexSessionConfig(extra_body={"auto_response": True}),
    )
    session.capabilities = DuplexCapabilities.minicpmo45_native()
    response_id = session.begin_response()
    request_id = "duplex-sid-native-segment-complete-stage0"
    session.bind_request(request_id)
    session.mark_audio_sent(1000)
    _install_direct_silence_scheduler(handler, session)
    sent: list[dict[str, Any]] = []

    async def send_json(data: dict[str, Any]) -> None:
        sent.append(data)

    close_reason, emitted = await handler._send_one_native_duplex_event(
        send_json,
        {
            "supported": True,
            "stage_role": "llm",
            "is_listen": True,
            "model_listen": False,
            "listen_source": "auto_response_segment_complete",
            "reason": "auto_response_segment_complete",
            "data_plane_request_id": request_id,
            "end_of_turn": False,
            "uses_model_runner_scheduler": True,
            "runner_kv_backed": True,
            "runtime_impl": "scheduler_data_plane",
            "owned_runtime": False,
        },
        session=session,
        expected_epoch=session.epoch,
    )
    await asyncio.sleep(0.01)

    assert close_reason is None
    assert emitted is False
    assert session.active_response_id == response_id
    assert session.active_request_id == request_id
    assert not handler._minicpmo_data_plane.is_terminal(request_id)
    assert sent == []
    assert len(engine.appended) == 1
    _, mode, payload, final = engine.appended[0]
    assert mode == "append_audio_chunk"
    assert payload["type"] == "audio"
    assert payload.get("force_listen") is not True
    assert final is False


@pytest.mark.parametrize(
    ("steps", "expected_deltas"),
    [
        (
            [("你好呀", 1), ("你好呀", 1), ("你好呀", 2)],
            ["你好呀", "", "你好呀"],
        ),
        (
            [("你好呀", 1), ("你好呀", 2)],
            ["你好呀", "你好呀"],
        ),
    ],
    ids=["turn-scoped", "exact-repeated-turn-text"],
)
def test_minicpmo_data_plane_text_cursor(
    steps: list[tuple[str, int]],
    expected_deltas: list[str],
):
    data_plane = _test_data_plane()
    request_id = "duplex-sid-native-e0-stage0"

    deltas = [data_plane.segment_text_delta(request_id, text, turn_id=turn_id) for text, turn_id in steps]

    assert deltas == expected_deltas


def test_minicpmo_data_plane_text_cursor_does_not_clip_prior_turn_overlap():
    data_plane = _test_data_plane()
    request_id = "duplex-sid-native-e0-stage0"

    assert data_plane.segment_text_delta(request_id, "你好呀。你有什么想聊的吗？", turn_id=1)
    assert (
        data_plane.segment_text_delta(request_id, "你有什么想聊的吗？你好呀。", turn_id=2)
        == "你有什么想聊的吗？你好呀。"
    )


@pytest.mark.asyncio
async def test_minicpmo_native_duplex_cancel_interrupts_background_data_plane_stream():
    control_result = {
        "operation": "append",
        "session_id": "sid-native-cancel-stream",
        "ok": True,
        "unsupported_count": 0,
        "error_count": 0,
        "stage_results": [
            {
                "stage_id": 0,
                "replica_id": 0,
                "result": {
                    "supported": True,
                    "implementation_level": "model_native_duplex",
                    "data_plane_append": True,
                    "request_id": "duplex-sid-native-cancel-stream-e0-stage0-s1",
                    "response_stage_id": 1,
                },
            }
        ],
    }
    engine = FakeEngineClient(
        append_result=control_result,
        collect_delay_s=0.2,
        collect_outputs=[
            [
                SimpleNamespace(
                    request_id="duplex-sid-native-cancel-stream-e0-stage0-s1",
                    finished=True,
                    outputs=[
                        SimpleNamespace(
                            text="",
                            multimodal_output={"audio": np.zeros(10, dtype=np.float32), "sr": 24000},
                        )
                    ],
                )
            ]
        ],
    )
    chat_service = FakeChatService(engine)
    handler = OmniDuplexSessionHandler(chat_service=chat_service, config_timeout_s=0.1, idle_timeout_s=1)
    ws = TimedWebSocket()
    ws.put(_native_session_create("sid-native-cancel-stream"))
    ws.put({"type": "input_audio_buffer.append", "audio": "AAAA", "format": "pcm_f32le"})
    ws.put({"type": "input.cancel"})
    ws.put({"type": "session.close"})

    await handler.handle_session(ws)

    assert "audio.cancelled" in ws.sent_types()
    assert "response.output_audio.delta" not in ws.sent_types()
    assert engine.signals == [("sid-native-cancel-stream", "barge_in")]
    assert engine.signal_fences == [DuplexFence("sid-native-cancel-stream")]


@pytest.mark.asyncio
async def test_minicpmo_native_duplex_session_close_discards_partial_pcm_tail():
    engine = FakeEngineClient()
    chat_service = FakeChatService(engine)
    handler = OmniDuplexSessionHandler(chat_service=chat_service, config_timeout_s=0.1, idle_timeout_s=1)
    ws = TimedWebSocket()
    ws.put(_native_session_create("sid-native-short-chunks"))
    chunk = _pcm_f32_b64(3200)
    for _ in range(6):
        ws.put(
            {
                "type": "input_audio_buffer.append",
                "audio": chunk,
                "format": "pcm_f32le",
                "sample_rate_hz": 16000,
            }
        )
    ws.put({"type": "session.close"})

    await handler.handle_session(ws)

    assert len(engine.appended) == 1
    _, mode, payload, final = engine.appended[0]
    assert mode == "append_audio_chunk"
    assert final is False
    assert isinstance(payload, dict)
    assert len(base64.b64decode(payload["audio"])) == 16000 * 4
    assert "duplex_num_input_tokens" not in payload


@pytest.mark.asyncio
async def test_minicpmo_native_duplex_idle_timeout_closes_runtime_with_timeout_reason():
    engine = FakeEngineClient()
    chat_service = FakeChatService(engine)
    handler = OmniDuplexSessionHandler(chat_service=chat_service, config_timeout_s=0.1, idle_timeout_s=0.1)
    ws = TimedWebSocket()
    ws.put(_native_session_create("sid-native-timeout"))

    await handler.handle_session(ws)

    assert ws.sent_types() == ["session.created", "session.closed"]
    assert ws.sent[-1]["reason"] == "timeout"
    assert engine.closed == [("sid-native-timeout", "timeout")]
