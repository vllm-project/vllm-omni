# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import TYPE_CHECKING

from fastapi import WebSocket
from vllm.logger import init_logger
from vllm.utils import random_uuid

from vllm_omni.entrypoints.duplex.warmup import DUPLEX_WARMUP_CLIENT_WAIT_S

if TYPE_CHECKING:
    from vllm_omni.config.omni_config import BaseVllmOmniStageConfig

logger = init_logger(__name__)

_QWEN3_OMNI_REALTIME_ARCH = "Qwen3OmniMoeForConditionalGeneration"
_QWEN3_OMNI_REALTIME_STAGES = {"thinker", "talker", "code2wav"}

# Keep OpenAI connection imports in the handlers: openai/__init__.py imports
# api_server, which imports this module.


def supports_qwen3_omni_realtime(stage_configs: Sequence[BaseVllmOmniStageConfig] | None) -> bool:
    stages = stage_configs or ()
    return (
        len(stages) == len(_QWEN3_OMNI_REALTIME_STAGES)
        and all(stage.model_config.model_arch == _QWEN3_OMNI_REALTIME_ARCH for stage in stages)
        and {stage.model_stage for stage in stages} == _QWEN3_OMNI_REALTIME_STAGES
    )


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


async def dispatch_generic_realtime_websocket(websocket: WebSocket) -> None:
    """Fall back to the generic OpenAI Realtime API for non-Qwen3-Omni models."""
    serving = getattr(websocket.app.state, "openai_serving_realtime", None)
    if serving is None:
        await websocket.accept()
        await websocket.send_json({"type": "error", "error": "Realtime API is not available", "code": "unsupported"})
        await websocket.close()
        return
    from vllm_omni.entrypoints.openai.realtime_connection import RealtimeConnection

    connection = RealtimeConnection(websocket, serving)
    await connection.handle_connection()


async def dispatch_realtime_websocket(websocket: WebSocket) -> None:
    """Handle Realtime sessions not already routed to the proprietary duplex
    handler by the caller: Qwen3-Omni gets a conformant full-duplex connection,
    everything else falls back to the generic turn-based Realtime API."""
    state = websocket.app.state
    if not supports_qwen3_omni_realtime(getattr(state, "stage_configs", None)):
        await dispatch_generic_realtime_websocket(websocket)
        return

    model_name = state.openai_serving_models.base_model_paths[0].name
    # Some OpenAI-compatible clients always send ``model=`` even when the
    # caller leaves model selection to this single-model Omni server.
    requested_model = websocket.query_params.get("model")
    if requested_model and requested_model != model_name:
        await reject_realtime_websocket(websocket, f"Model '{requested_model}' is not available")
        return

    # Hold real clients until the startup duplex warmup finishes (the warmup
    # connection marks itself with vllm_omni_warmup=1 and passes through).
    warmup_done = getattr(state, "duplex_warmup_done", None)
    if warmup_done is not None and not warmup_done.is_set() and websocket.query_params.get("vllm_omni_warmup") != "1":
        try:
            await asyncio.wait_for(warmup_done.wait(), timeout=DUPLEX_WARMUP_CLIENT_WAIT_S)
        except (TimeoutError, asyncio.TimeoutError):
            logger.warning(
                "Duplex warmup still running after %d s; admitting the client anyway.",
                DUPLEX_WARMUP_CLIENT_WAIT_S,
            )

    tokenizer = await state.engine_client.get_tokenizer()
    from vllm_omni.entrypoints.openai.realtime.connection import OpenAIFullDuplexConnection

    connection = OpenAIFullDuplexConnection(
        websocket=websocket,
        engine=state.engine_client,
        model_name=model_name,
        tokenizer=tokenizer,
        tool_call_parser=getattr(state.args, "tool_call_parser", None),
        enable_auto_tool_choice=getattr(state.args, "enable_auto_tool_choice", False),
    )
    await connection.handle_connection()
