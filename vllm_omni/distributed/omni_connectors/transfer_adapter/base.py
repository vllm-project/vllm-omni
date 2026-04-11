# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from collections import deque
from typing import Any

from ..utils.logging import get_connector_logger

logger = get_connector_logger(__name__)


class OmniTransferAdapterBase:
    """Base class for managing data transfer via OmniConnector.

    This class handles the core loop logic and connector interactions, but
    leaves the specific data processing (chunks, KV cache, etc.) to subclasses.
    """

    def __init__(self, config: Any):
        self.config = config
        if not hasattr(self, "connector"):
            self.connector = None
        # Requests that are waiting to be polled
        self._pending_load_reqs = deque()
        # Requests that have successfully retrieved data
        self._finished_load_reqs = set()
        self._cancelled_load_reqs: set[str] = set()

        # Requests that are waiting to be saved
        self._pending_save_reqs = deque()
        # Requests that have successfully saved data
        self._finished_save_reqs = set()

        self.stop_event = threading.Event()
        self._recv_cond = threading.Condition()
        self._save_cond = threading.Condition()

        self.recv_thread = threading.Thread(target=self.recv_loop, daemon=True)
        self.recv_thread.start()

        self.save_thread = threading.Thread(target=self.save_loop, daemon=True)
        self.save_thread.start()

    @classmethod
    def create_connector(cls, model_config: Any):
        raise NotImplementedError

    def recv_loop(self):
        """Loop to poll for incoming data.

        Process each pending request exactly once per pass.  When no request
        made progress, back off 1 ms instead of tight-spinning on failed
        shm_open syscalls (which can burn a full CPU core).
        """
        while not self.stop_event.is_set():
            n = len(self._pending_load_reqs)
            any_success = False
            for _ in range(n):
                if not self._pending_load_reqs:
                    break
                request = self._pending_load_reqs.popleft()
                request_id = request.request_id
                if request_id in self._cancelled_load_reqs:
                    self._cancelled_load_reqs.discard(request_id)
                    continue
                self.request_ids_mapping[request_id] = request.external_req_id
                try:
                    is_success = self._poll_single_request(request)
                    if is_success:
                        any_success = True
                    else:
                        self._pending_load_reqs.append(request)
                except Exception as e:
                    self._pending_load_reqs.append(request)
                    logger.warning(f"Error receiving data for {request_id}: {e}")

            # Timeout is the fallback for lock-free append/notify races.
            with self._recv_cond:
                if not self._pending_load_reqs and not self.stop_event.is_set():
                    self._recv_cond.wait(timeout=0.1)
                elif not any_success and not self.stop_event.is_set():
                    self._recv_cond.wait(timeout=0.001)

    def save_loop(self):
        """Loop to send outgoing data."""
        while not self.stop_event.is_set():
            while self._pending_save_reqs:
                task = self._pending_save_reqs.popleft()
                try:
                    self._send_single_request(task)
                except Exception as e:
                    logger.warning(f"Error saving data for {task.get('request_id')}: {e}")

            with self._save_cond:
                if not self._pending_save_reqs and not self.stop_event.is_set():
                    self._save_cond.wait(timeout=0.1)

    def _poll_single_request(self, *args, **kwargs):
        """Poll connector for a single request task.
        Subclasses should implement request-specific receive behavior."""
        raise NotImplementedError

    def _send_single_request(self, *args, **kwargs):
        """Send one pending save request task to the connector.
        Subclasses should implement task-specific handling logic."""
        raise NotImplementedError

    def load_async(self, *args, **kwargs):
        """Register a request to load data. To be implemented by subclasses."""
        raise NotImplementedError

    def save_async(self, *args, **kwargs):
        """Submit data to be saved. To be implemented by subclasses."""
        raise NotImplementedError

    def load(self, *args, **kwargs):
        """Load request data from connector synchronously. To be implemented by subclasses."""
        raise NotImplementedError

    def save(self, *args, **kwargs):
        """Save data to connector synchronously. To be implemented by subclasses."""
        raise NotImplementedError

    def get_finished_requests(self):
        """Get finished loaded or saved requests"""
        raise NotImplementedError

    def shutdown(self):
        """Stop background loops and close the connector."""
        self.stop_event.set()
        with self._recv_cond:
            self._recv_cond.notify_all()
        with self._save_cond:
            self._save_cond.notify_all()
        if self.connector is not None:
            try:
                self.connector.close()
            except Exception:
                pass
