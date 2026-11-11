# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Omni's implementation of the Positronic offboard protocol.

Protocol reference: Positronic-Robotics/positronic at d5f2a47 (Apache-2.0):
https://github.com/Positronic-Robotics/positronic/blob/d5f2a47/LICENSE
This adapter uses Omni's existing transport codec; it does not vendor the
Positronic server or require its package at runtime.
"""

from fastapi import APIRouter, Request, WebSocket

router = APIRouter(prefix="/api/v1")


@router.get("/models")
async def positronic_models(request: Request) -> dict[str, list[str]]:
    from vllm_omni.entrypoints.positronic.connection import list_models

    return await list_models(request)


@router.websocket("/session")
@router.websocket("/session/{model_id:path}")
async def positronic_session(websocket: WebSocket, model_id: str | None = None) -> None:
    from vllm_omni.entrypoints.positronic.connection import handle_session

    await handle_session(websocket, model_id)
