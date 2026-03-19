# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""ZMQ-based queue utilities for Omni IPC."""

from __future__ import annotations

import queue
from typing import Any

import zmq
from vllm.utils.network_utils import make_zmq_socket


class ZmqQueue:
    """Queue-like wrapper on a ZMQ socket."""

    def __init__(
        self,
        ctx: zmq.Context,
        socket_type: int,
        *,
        bind: str | None = None,
        connect: str | None = None,
        recv_timeout_ms: int | None = None,
        send_timeout_ms: int | None = None,
    ) -> None:
        # Determine path and bind mode
        path = bind if bind is not None else connect
        if path is None:
            raise ValueError("Either bind or connect must be specified")
        bind_mode = bind is not None

        self._socket = make_zmq_socket(ctx, path, socket_type, bind=bind_mode, linger=5000)

        # Reusable poller for efficient polling operations
        self._poller = zmq.Poller()
        self._poller.register(self._socket, zmq.POLLIN)

        # Store default timeout settings
        self._default_recv_timeout = recv_timeout_ms
        self._default_send_timeout = send_timeout_ms

        # Apply timeout settings if specified
        if recv_timeout_ms is not None:
            self._socket.rcvtimeo = recv_timeout_ms
        if send_timeout_ms is not None:
            self._socket.sndtimeo = send_timeout_ms

        self.endpoint = path

    def put(self, obj: Any) -> None:
        """Send an object to the queue. Blocks until sent or timeout."""
        try:
            self._socket.send_pyobj(obj)
        except zmq.Again as e:
            raise queue.Full() from e

    def put_nowait(self, obj: Any) -> None:
        """Send an object to the queue without blocking."""
        try:
            self._socket.send_pyobj(obj, flags=zmq.NOBLOCK)
        except zmq.Again as e:
            raise queue.Full() from e

    def get(self, timeout: float | None = None) -> Any:
        """Receive an object from the queue with optional timeout in seconds."""
        if timeout is None:
            return self._socket.recv_pyobj()

        # Use the reusable poller for timeout handling
        events = dict(self._poller.poll(int(timeout * 1000)))
        if events.get(self._socket) == zmq.POLLIN:
            return self._socket.recv_pyobj()
        raise queue.Empty()

    def get_nowait(self) -> Any:
        """Receive an object from the queue without blocking."""
        try:
            return self._socket.recv_pyobj(flags=zmq.NOBLOCK)
        except zmq.Again as e:
            raise queue.Empty() from e

    def empty(self) -> bool:
        """Check if the queue is empty without blocking."""
        events = dict(self._poller.poll(0))
        return events.get(self._socket) != zmq.POLLIN

    def close(self) -> None:
        self._socket.close(0)


def create_zmq_queue(ctx: zmq.Context, endpoint: str, socket_type: int) -> ZmqQueue:
    """Create a ZmqQueue from an endpoint string and socket type."""
    return ZmqQueue(ctx, socket_type, connect=endpoint)
