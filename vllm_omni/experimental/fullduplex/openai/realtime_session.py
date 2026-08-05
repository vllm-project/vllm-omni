from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from fastapi import WebSocket

from vllm_omni.experimental.fullduplex.openai import realtime_state as _state
from vllm_omni.experimental.fullduplex.openai.realtime_input import (
    RealtimeInputTranslator,
)
from vllm_omni.experimental.fullduplex.openai.realtime_output import (
    RealtimeOutputProjector,
)

REALTIME_ERROR_TYPES_BY_CODE = _state.REALTIME_ERROR_TYPES_BY_CODE
REALTIME_INPUT_AUDIO_FORMATS = _state.REALTIME_INPUT_AUDIO_FORMATS
REALTIME_OUTPUT_AUDIO_FORMATS = _state.REALTIME_OUTPUT_AUDIO_FORMATS
RealtimeSessionState = _state.RealtimeSessionState
_RealtimeResponseState = _state._RealtimeResponseState


class NativeRealtimeSessionProtocol(
    RealtimeInputTranslator,
    RealtimeOutputProjector,
    _state.RealtimeStateOwner,
):
    """Compatibility facade for one native Realtime protocol session.

    Input translation, output projection, and mutable state have separate
    owners. The facade retains the established import path and private helper
    surface used by serving and protocol contract tests.
    """

    def __init__(self, query_params: Any) -> None:
        self._state = RealtimeSessionState.from_query_params(query_params)

    def bind_sender(self, send_realtime_json) -> None:
        self._send_realtime_json = send_realtime_json

    def _default_session_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self._default_model,
            "session_id": self._default_session_id,
        }
        if self._default_extra_body:
            payload["extra_body"] = dict(self._default_extra_body)
        return payload

    async def receive_internal_event_text(self, websocket: WebSocket) -> str:
        if not self._pending_outbound.empty():
            return json.dumps(await self._pending_outbound.get())
        if not self._opened and not self._resume_only and not self._autostarted_default_session and self._default_model:
            self._opened = True
            self._autostarted_default_session = True
            return json.dumps(self._session_create_from_realtime(self._default_session_payload()))
        while True:
            if not self._pending_outbound.empty():
                return json.dumps(await self._pending_outbound.get())
            raw = await websocket.receive_text()
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                return raw
            if not isinstance(event, dict):
                return raw
            if not self._opened and event.get("type") == "session.resume":
                self._opened = True
                translated = await self._to_duplex_event(event)
                if translated is None:
                    if not self._pending_outbound.empty():
                        return json.dumps(await self._pending_outbound.get())
                    continue
                return json.dumps(translated)
            if not self._opened and self._resume_only:
                translated = await self._to_duplex_event(event)
                if translated is None:
                    if not self._pending_outbound.empty():
                        return json.dumps(await self._pending_outbound.get())
                    continue
                return json.dumps(translated)
            if not self._opened and event.get("type") != "session.update":
                self._opened = True
                await self._pending_outbound.put(self._session_create_from_realtime(self._default_session_payload()))
                translated = await self._to_duplex_event(event)
                if translated is not None:
                    await self._send_realtime_input_ack(event)
                    await self._pending_outbound.put(translated)
                return json.dumps(await self._pending_outbound.get())
            translated = await self._to_duplex_event(event)
            if translated is None:
                if not self._pending_outbound.empty():
                    return json.dumps(await self._pending_outbound.get())
                continue
            await self._send_realtime_input_ack(event)
            return json.dumps(translated)

    def encode_outbound_event(self, data: dict[str, Any]) -> list[dict[str, object]]:
        payloads = self._from_duplex_event(data)
        for payload in payloads:
            self._attach_event_id(payload)
        return payloads

    @staticmethod
    def _attach_event_id(payload: dict[str, object]) -> None:
        payload.setdefault("event_id", f"event_{uuid4().hex}")

    async def _send_realtime_payload(self, payload: dict[str, object]) -> None:
        self._attach_event_id(payload)
        if (
            self._hold_realtime_output_until_session_created
            and payload.get("type") != "error"
            and payload.get("type") != "session.created"
        ):
            self._held_realtime_payloads.append(payload)
            return
        if self._send_realtime_json is None:
            raise RuntimeError("Native Realtime sender is not bound")
        await self._send_realtime_json(payload)

    @staticmethod
    def _realtime_error_payload(
        code: str,
        message: str,
        *,
        event_id: object | None = None,
        param: object | None = None,
    ) -> dict[str, object]:
        error: dict[str, object] = {
            "type": REALTIME_ERROR_TYPES_BY_CODE.get(code, "invalid_request_error"),
            "code": code,
            "message": message,
        }
        if isinstance(event_id, str) and event_id:
            error["event_id"] = event_id
        if isinstance(param, str) and param:
            error["param"] = param
        return {"type": "error", "error": error}
