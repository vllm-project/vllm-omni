# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from fastapi import WebSocket
from vllm.logger import init_logger
from vllm.utils import random_uuid

from vllm_omni.entrypoints.duplex.warmup import wait_for_duplex_warmup

logger = init_logger(__name__)

# Keep OpenAI connection imports in the handlers: openai/__init__.py imports
# api_server, which imports this module.


async def reject_realtime_websocket(websocket: WebSocket, message: str) -> None:
    await websocket.accept()
    await websocket.send_json(
        {
            "event_id": f"evt_{random_uuid()}",
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "code": "unsupported_model",
                "message": message,
                "param": "model",
                "event_id": None,
            },
        }
    )
    await websocket.close(code=1008)


async def _reject_unavailable_realtime_websocket(websocket: WebSocket) -> None:
    await websocket.accept()
    await websocket.send_json({"type": "error", "error": "Realtime API is not available", "code": "unsupported"})
    await websocket.close()


async def dispatch_realtime_websocket(websocket: WebSocket) -> None:
    """Route a turn-based Realtime session to the OpenAI connection handler."""
    state = websocket.app.state
    if (
        getattr(state, "diffusion_engine", None) is not None
        or getattr(state, "engine_client", None) is None
        or getattr(state, "openai_serving_chat", None) is None
    ):
        await _reject_unavailable_realtime_websocket(websocket)
        return

    model_name = state.openai_serving_models.base_model_paths[0].name
    # Some OpenAI-compatible clients always send ``model=`` even when the
    # caller leaves model selection to this single-model Omni server.
    requested_model = websocket.query_params.get("model")
    if requested_model and requested_model != model_name:
        await reject_realtime_websocket(websocket, f"Model '{requested_model}' is not available")
        return

    await wait_for_duplex_warmup(websocket)

    from vllm_omni.entrypoints.openai.realtime.connection import OpenAIFullDuplexConnection

    connection = OpenAIFullDuplexConnection(
        websocket=websocket,
        engine=state.engine_client,
        model_name=model_name,
        chat_handler=state.openai_serving_chat,
        tool_call_parser=getattr(state.args, "tool_call_parser", None),
        enable_auto_tool_choice=getattr(state.args, "enable_auto_tool_choice", False),
    )
    await connection.handle_connection()
