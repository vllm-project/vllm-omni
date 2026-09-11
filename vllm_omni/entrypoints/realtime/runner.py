# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from fastapi import WebSocket

from vllm_omni.entrypoints.realtime.adapters.base import RealtimeModelAdapter
from vllm_omni.entrypoints.realtime.session import RealtimeSessionProtocol


async def run_realtime_session(websocket: WebSocket, adapter: RealtimeModelAdapter) -> None:
    """Give one connection's codec to its existing execution owner."""
    await adapter.handle_session(
        websocket,
        realtime_protocol=RealtimeSessionProtocol(websocket.query_params),
    )
