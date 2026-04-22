"""Unit tests for WebSocket API."""

import pytest

from vllm_omni.entrypoints.openai.websocket_api import (
    ConnectionState,
    WebSocketAPI,
    WebSocketConnection,
)


class TestWebSocketAPI:
    """Tests for WebSocketAPI."""

    @pytest.mark.asyncio
    async def test_connection_state(self):
        """Test connection state management."""
        api = WebSocketAPI()

        assert api.get_active_connections() == 0

    def test_connection_creation(self):
        """Test WebSocket connection creation."""
        connection = WebSocketConnection(connection_id="test-123", websocket=None, state=ConnectionState.CONNECTED)

        assert connection.connection_id == "test-123"
        assert connection.state == ConnectionState.CONNECTED
        assert connection.metadata is None

    def test_connection_with_metadata(self):
        """Test connection with metadata."""
        metadata = {"user": "test_user", "origin": "test"}
        connection = WebSocketConnection(connection_id="test-456", websocket=None, metadata=metadata)

        assert connection.metadata == metadata

    def test_get_active_connections(self):
        """Test active connections count."""
        api = WebSocketAPI()

        api._connections["conn1"] = WebSocketConnection("conn1", None)
        api._connections["conn2"] = WebSocketConnection("conn2", None)

        assert api.get_active_connections() == 2


class TestConnectionState:
    """Tests for ConnectionState enum."""

    def test_connection_states(self):
        """Test all connection states."""
        assert ConnectionState.CONNECTING.value == "connecting"
        assert ConnectionState.CONNECTED.value == "connected"
        assert ConnectionState.DISCONNECTING.value == "disconnecting"
        assert ConnectionState.DISCONNECTED.value == "disconnected"


class TestWebSocketConnection:
    """Tests for WebSocketConnection."""

    def test_default_state(self):
        """Test default connection state."""
        conn = WebSocketConnection("id", None)
        assert conn.state == ConnectionState.CONNECTING

    def test_connected_state(self):
        """Test connected state."""
        conn = WebSocketConnection("id", None, state=ConnectionState.CONNECTED)
        assert conn.state == ConnectionState.CONNECTED
