# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio

import pytest
from fastapi import WebSocket
from starlette.datastructures import QueryParams

from vllm_omni.entrypoints.duplex.realtime_session import NativeRealtimeSessionProtocol
from vllm_omni.entrypoints.duplex.realtime_state import RealtimeSessionState as LegacySessionState
from vllm_omni.entrypoints.duplex.server_vad import ServerVADConfig as LegacyVADConfig
from vllm_omni.entrypoints.realtime.adapters.base import RealtimeModelAdapter
from vllm_omni.entrypoints.realtime.config import ServerVADConfig
from vllm_omni.entrypoints.realtime.runner import run_realtime_session
from vllm_omni.entrypoints.realtime.session import RealtimeSessionProtocol
from vllm_omni.entrypoints.realtime.state import RealtimeSessionState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_legacy_imports_share_the_same_implementation() -> None:
    # Existing duplex suites keep their imports and assertions unchanged.
    assert NativeRealtimeSessionProtocol is RealtimeSessionProtocol
    assert LegacySessionState is RealtimeSessionState
    assert LegacyVADConfig is ServerVADConfig


@pytest.mark.asyncio
async def test_runner_passes_a_connection_local_codec_to_the_adapter(mocker) -> None:
    websocket = mocker.Mock(spec=WebSocket)
    websocket.query_params = QueryParams("model=test-model&session_id=example")
    adapter = mocker.Mock(spec=RealtimeModelAdapter)
    adapter.handle_session = mocker.AsyncMock()

    await run_realtime_session(websocket, adapter)
    await run_realtime_session(websocket, adapter)

    calls = adapter.handle_session.await_args_list
    assert len(calls) == 2
    first = calls[0].kwargs["realtime_protocol"]
    second = calls[1].kwargs["realtime_protocol"]
    assert calls[0].args == calls[1].args == (websocket,)
    assert isinstance(first, RealtimeSessionProtocol)
    assert first is not second
    assert first._state is not second._state
    assert first._default_session_payload() == {"model": "test-model", "session_id": "example"}
    first._conversation_items["item-a"] = {"id": "item-a"}
    assert not second._conversation_items


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [RuntimeError, asyncio.CancelledError])
async def test_runner_preserves_adapter_failure_and_cancellation(mocker, error_type) -> None:
    websocket = mocker.Mock(spec=WebSocket)
    websocket.query_params = QueryParams()
    adapter = mocker.Mock(spec=RealtimeModelAdapter)
    error = error_type("adapter stopped")
    adapter.handle_session = mocker.AsyncMock(side_effect=error)

    with pytest.raises(error_type) as caught:
        await run_realtime_session(websocket, adapter)

    assert caught.value is error
    adapter.handle_session.assert_awaited_once()


@pytest.mark.asyncio
async def test_shared_codec_keeps_response_create_and_cancel_as_execution_intents() -> None:
    protocol = RealtimeSessionProtocol({})
    options = {"instructions": "hello", "output_modalities": ["audio"]}
    create = await protocol._to_duplex_event({"type": "response.create", "response": options})
    assert create == {"type": "response.create", "response": options}
    # Decoding an intent does not itself begin a model response.
    assert protocol._active_response_id is None

    projected = protocol.encode_outbound_event({"type": "response.created", "response_id": "response-a"})
    assert projected[0]["type"] == "response.created"
    cancel = await protocol._to_duplex_event({"type": "response.cancel"})
    assert cancel == {"type": "response.cancel", "reason": "response.cancel", "response_id": "response-a"}


def test_shared_codec_preserves_native_opt_in_alias_precedence() -> None:
    protocol = RealtimeSessionProtocol({})
    command = protocol._session_create_from_realtime(
        {
            "model": "test-model",
            "extra_body": {"native_duplex": False, "minicpmo45_native_duplex": True},
        }
    )
    extra = command["session"]["extra_body"]
    assert extra["native_duplex"] is False
    assert "minicpmo45_native_duplex" not in extra
