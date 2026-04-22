"""WebSocket API for real-time streaming inference."""

import asyncio
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


class ConnectionState(Enum):
    """WebSocket connection states."""

    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTING = "disconnecting"
    DISCONNECTED = "disconnected"


@dataclass
class WebSocketConnection:
    """WebSocket connection."""

    connection_id: str
    websocket: Any
    state: ConnectionState = ConnectionState.CONNECTING
    metadata: dict[str, Any] = None  # type: ignore[assignment]


class WebSocketAPI:
    """WebSocket API for real-time inference."""

    def __init__(self, omni_engine=None):
        self._engine = omni_engine
        self._connections: dict[str, WebSocketConnection] = {}
        self._connection_tasks: dict[str, asyncio.Task] = {}

    async def handle_connection(self, websocket) -> str:
        """Handle new WebSocket connection."""
        connection_id = str(uuid.uuid4())

        connection = WebSocketConnection(
            connection_id=connection_id, websocket=websocket, state=ConnectionState.CONNECTED
        )
        self._connections[connection_id] = connection

        task = asyncio.create_task(self._connection_loop(connection_id, websocket))
        self._connection_tasks[connection_id] = task

        await websocket.send_json({"type": "connection", "connection_id": connection_id, "status": "connected"})

        logger.info(f"WebSocket connection established: {connection_id}")
        return connection_id

    async def _connection_loop(self, connection_id: str, websocket) -> None:
        """Main connection loop."""
        try:
            while connection_id in self._connections:
                message = await websocket.receive_json()
                await self._handle_message(connection_id, message)

        except asyncio.CancelledError:
            logger.info(f"Connection {connection_id} cancelled")
        except Exception as e:
            logger.error(f"Connection error: {e}")
        finally:
            await self._cleanup_connection(connection_id)

    async def _handle_message(self, connection_id: str, message: dict[str, Any]) -> None:
        """Handle incoming message."""
        message_type = message.get("type")

        if message_type == "infer":
            await self._handle_inference(connection_id, message)
        elif message_type == "cancel":
            await self._handle_cancel(connection_id, message)
        elif message_type == "ping":
            await self._handle_ping(connection_id)
        elif message_type == "status":
            await self._handle_status(connection_id)

    async def _handle_inference(self, connection_id: str, message: dict[str, Any]) -> None:
        """Handle inference request."""
        connection = self._connections.get(connection_id)
        if not connection:
            return

        prompt = message.get("prompt", "")
        multi_modal_data = message.get("multi_modal_data", {})
        sampling_params = message.get("sampling_params", {})
        stream = message.get("stream", True)

        if stream and self._engine:
            async for output in self._engine.stream_generate(
                prompt=prompt, multi_modal_data=multi_modal_data, sampling_params=sampling_params
            ):
                await connection.websocket.send_json(
                    {
                        "type": "chunk",
                        "content": output.text if hasattr(output, "text") else str(output),
                        "final": getattr(output, "is_final", False),
                    }
                )
        else:
            await connection.websocket.send_json({"type": "error", "message": "Non-streaming not implemented"})

    async def _handle_cancel(self, connection_id: str, message: dict[str, Any]) -> None:
        """Handle cancellation request."""
        request_id = message.get("request_id")
        if request_id and self._engine:
            await self._engine.cancel(request_id)

    async def _handle_ping(self, connection_id: str) -> None:
        """Handle ping."""
        connection = self._connections.get(connection_id)
        if connection:
            await connection.websocket.send_json({"type": "pong"})

    async def _handle_status(self, connection_id: str) -> None:
        """Handle status request."""
        connection = self._connections.get(connection_id)
        if connection:
            await connection.websocket.send_json(
                {
                    "type": "status",
                    "connection_id": connection_id,
                    "state": connection.state.value,
                    "active_connections": len(self._connections),
                }
            )

    async def _cleanup_connection(self, connection_id: str) -> None:
        """Clean up connection resources."""
        self._connections.pop(connection_id, None)
        task = self._connection_tasks.pop(connection_id, None)
        if task and not task.done():
            task.cancel()

        logger.info(f"WebSocket connection closed: {connection_id}")

    def get_active_connections(self) -> int:
        """Get number of active connections."""
        return len(self._connections)
