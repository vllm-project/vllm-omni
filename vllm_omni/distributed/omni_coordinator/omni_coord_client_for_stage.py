# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import logging
import threading
import time
from dataclasses import asdict

import zmq

from .messages import InstanceEvent, StageStatus

logger = logging.getLogger(__name__)


class OmniCoordClientForStage:
    """Client used by stage instances to send events to OmniCoordinator.

    This client maintains a DEALER socket connected to OmniCoordinator's
    ROUTER endpoint and sends JSON-encoded events describing instance status.
    """

    def __init__(
        self,
        coord_zmq_addr: str,
        input_addr: str,
        output_addr: str,
        stage_id: int,
    ) -> None:
        """Initialize client and send initial registration / status-up event."""
        self._coord_zmq_addr = coord_zmq_addr
        self._input_addr = input_addr
        self._output_addr = output_addr
        self._stage_id = stage_id

        self._ctx = zmq.Context()
        self._socket = self._ctx.socket(zmq.DEALER)
        try:
            self._socket.connect(self._coord_zmq_addr)
        except zmq.ZMQError as e:
            self._socket.close()
            raise RuntimeError(f"Failed to connect to coordinator at {self._coord_zmq_addr}: {e}") from e

        self._status = StageStatus.UP
        self._queue_length = 0
        self._closed = False
        self._closing = False
        self._heartbeat_interval = 5.0
        self._stop_event = threading.Event()
        self._send_lock = threading.RLock()

        self._send_event("update")

        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
        )
        self._heartbeat_thread.start()

    def _reconnect(self, max_retries: int = 3, retry_interval: float = 5.0) -> bool:
        """Best-effort reconnect with up to ``max_retries`` attempts.

        Each attempt closes the current socket/context, sleeps ``retry_interval``
        seconds, then creates a new DEALER socket and reconnects to the coordinator.
        Returns True on success, False if all attempts fail.
        """
        if max_retries <= 0:
            return False

        for attempt in range(1, max_retries + 1):
            with self._send_lock:
                if self._stop_event.is_set() or self._closed:
                    return False
                try:
                    self._socket.close(0)
                except zmq.ZMQError:
                    pass
                try:
                    self._ctx.term()
                except zmq.ZMQError:
                    pass

                try:
                    self._ctx = zmq.Context()
                    self._socket = self._ctx.socket(zmq.DEALER)
                    self._socket.connect(self._coord_zmq_addr)
                    return True
                except zmq.ZMQError as e:
                    logger.error(
                        "Stage client reconnect failed (attempt=%d/%d, coord=%s)",
                        attempt,
                        max_retries,
                        self._coord_zmq_addr,
                        exc_info=e,
                    )

            if retry_interval > 0:
                time.sleep(retry_interval)
        return False

    def _send_event(self, event_type: str) -> None:
        """Send an InstanceEvent to OmniCoordinator.

        Wire format: input_addr, output_addr, stage_id, status, queue_length, event_type.
        For "update": includes status and queue_length from instance state.
        For "heartbeat": status and queue_length are null.

        On send failure (ZMQError / RuntimeError), attempts to reconnect up
        to 3 times (5s sleep each) and retries the send once after a
        successful reconnect. Raises if reconnect or the retry send fails.
        """
        with self._send_lock:
            if self._closed:
                raise RuntimeError("Client already closed")

            event = InstanceEvent(
                input_addr=self._input_addr,
                output_addr=self._output_addr,
                stage_id=self._stage_id,
                event_type=event_type,
                status=self._status,
                queue_length=self._queue_length,
            )
            data = json.dumps(asdict(event)).encode("utf-8")

            try:
                self._socket.send(data, flags=zmq.NOBLOCK)
                return
            except zmq.Again:
                logger.debug("Send buffer full, dropping event")
                return
            except (RuntimeError, zmq.ZMQError) as e:
                # First send failed; try reconnecting a few times.
                if not self._reconnect(max_retries=3):
                    logger.error("Failed to send event and reconnect to coordinator", exc_info=e)
                    raise

                # Reconnected successfully; try sending once more.
                try:
                    self._socket.send(data, flags=zmq.NOBLOCK)
                except zmq.Again:
                    logger.debug("Send buffer full after reconnect, dropping event")
                except (RuntimeError, zmq.ZMQError) as e2:
                    logger.error("Failed to send event after successful reconnect", exc_info=e2)
                    raise

    def update_info(
        self,
        status: StageStatus | None = None,
        queue_length: int | None = None,
    ) -> None:
        """Update instance information and notify OmniCoordinator.

        At least one of ``status`` or ``queue_length`` must be provided.
        """
        if status is None and queue_length is None:
            raise ValueError("At least one of status or queue_length must be provided")

        with self._send_lock:
            if self._closed or self._closing:
                raise RuntimeError("Client is closing or already closed")

            if status is not None:
                self._status = status
            if queue_length is not None:
                self._queue_length = queue_length

            self._send_event("update")

    def _heartbeat_loop(self) -> None:
        """Periodically send heartbeat events while the client is alive."""
        while not self._stop_event.wait(timeout=self._heartbeat_interval):
            if self._closed:
                break

            try:
                self._send_event("heartbeat")
            except (RuntimeError, zmq.ZMQError) as e:
                if self._closed or self._stop_event.is_set():
                    break
                logger.warning("Heartbeat send failed; will retry on next interval", exc_info=e)
                continue

    def close(self) -> None:
        """Send a final down event and close the underlying socket."""
        if self._closed:
            raise RuntimeError("Client already closed")

        # Stop heartbeat thread first to avoid concurrent sends during shutdown.
        self._stop_event.set()
        if hasattr(self, "_heartbeat_thread"):
            self._heartbeat_thread.join(timeout=1.0)

        with self._send_lock:
            if self._closed:
                raise RuntimeError("Client already closed")

            self._closing = True

            # Mark status as DOWN and send one last update.
            self._status = StageStatus.DOWN
            try:
                self._send_event("update")
            except (RuntimeError, zmq.ZMQError):
                pass  # Socket may already be broken, proceed with close

            # Close DEALER socket and terminate this client's context.
            self._socket.close(0)
            try:
                self._ctx.term()
            except zmq.ZMQError:
                pass
            self._closed = True
