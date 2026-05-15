# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import json
import logging
import threading
from dataclasses import asdict
from time import time
from typing import Any

import zmq

from .messages import InstanceEvent, InstanceInfo, InstanceList, StageStatus

logger = logging.getLogger(__name__)


class OmniCoordinator:
    """Coordinator for stage instances and hub clients.

    This service receives instance events from :class:`OmniCoordClientForStage`
    via a ZMQ ROUTER socket and publishes active instance lists to
    :class:`OmniCoordClientForHub` via a PUB socket.

    The coordinator maintains an in-memory registry of all known instances,
    including their status, queue length, and heartbeat timestamps. A
    background thread periodically checks for heartbeat timeouts and marks
    unhealthy instances as ``StageStatus.ERROR``.
    """

    def __init__(
        self,
        router_zmq_addr: str,
        pub_zmq_addr: str,
        heartbeat_timeout: float = 30.0,
    ) -> None:
        """Initialize coordinator and start background service loops.

        Args:
            router_zmq_addr: ZMQ address to bind the ROUTER socket.
            pub_zmq_addr: ZMQ address to bind the PUB socket.
            heartbeat_timeout: Seconds before an instance is considered
                unhealthy if no heartbeat / update is received.
        """
        self._router_zmq_addr = router_zmq_addr
        self._pub_zmq_addr = pub_zmq_addr
        self._heartbeat_timeout = heartbeat_timeout

        # Dedicated ZMQ context for this coordinator instance.
        self._ctx = zmq.Context()
        self._router = self._ctx.socket(zmq.ROUTER)
        self._router.bind(self._router_zmq_addr)

        self._pub = self._ctx.socket(zmq.PUB)
        self._pub.bind(self._pub_zmq_addr)

        self._instances: dict[str, InstanceInfo] = {}
        self._lock = threading.Lock()
        self._pub_lock = threading.Lock()

        self._publish_min_interval: float = 0.1  # seconds
        self._pending_broadcast: bool = False
        self._pending_lock = threading.Lock()

        self._running = True
        self._closed = False
        self._stop_event = threading.Event()

        self._router.setsockopt(zmq.RCVTIMEO, 100)

        self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._recv_thread.start()

        self._periodic_thread = threading.Thread(target=self._periodic_loop, daemon=True)
        self._periodic_thread.start()

    def get_active_instances(self) -> InstanceList:
        """Return an :class:`InstanceList` of active (UP) instances only."""
        with self._lock:
            active = [inst for inst in self._instances.values() if inst.status == StageStatus.UP]
        return InstanceList(instances=active, timestamp=time())

    def add_new_instance(self, event: InstanceEvent) -> None:
        """Add a new instance based on an incoming event."""
        with self._lock:
            self._add_new_instance_locked(event)
        self._schedule_broadcast()

    def update_instance_info(self, event: InstanceEvent) -> None:
        """Update an existing instance based on an incoming event."""
        with self._lock:
            self._update_instance_info_locked(event)
        self._schedule_broadcast()

    def remove_instance(self, event: InstanceEvent) -> None:
        """Mark an instance as removed / down based on an incoming event.

        This marks the instance's status as DOWN or ERROR (depending on the
        event) but keeps it in the internal registry. It is removed from the
        *active* instance list published to hubs.
        """
        with self._lock:
            self._remove_instance_locked(event)
        self._schedule_broadcast()

    def publish_instance_list_update(self) -> bool:
        """Publish the current active instance list to all subscribers.

        Returns:
            True if the PUB send succeeded, False if it was dropped (e.g.
            socket not ready when using ``zmq.NOBLOCK``).
        """
        active_list = self.get_active_instances()
        payload = asdict(active_list)
        data = json.dumps(payload).encode("utf-8")

        with self._pub_lock:
            try:
                # PUB socket is best-effort; drop update if not ready.
                self._pub.send(data, flags=zmq.NOBLOCK)
                return True
            except (zmq.Again, zmq.ZMQError):
                # Silently ignore send failures; next update will catch up.
                return False

    def _schedule_broadcast(self) -> None:
        """Request a broadcast to be flushed by the periodic loop.

        All broadcast requests are coalesced via ``_pending_broadcast`` and
        flushed at most once per ``_publish_min_interval``.
        """
        with self._pending_lock:
            self._pending_broadcast = True

    def _mark_instance_error_locked(self, info: InstanceInfo) -> None:
        """Mark instance as ERROR (e.g. after heartbeat timeout)."""
        info.status = StageStatus.ERROR

    def _check_heartbeat_timeouts(self) -> None:
        """Mark instances as ERROR if their heartbeat has timed out."""
        now = time()
        timed_out = False
        gc_ttl = 600.0  # 10 minutes

        with self._lock:
            to_delete: list[str] = []

            for input_addr, info in self._instances.items():
                if info.status == StageStatus.UP and now - info.last_heartbeat > self._heartbeat_timeout:
                    self._mark_instance_error_locked(info)
                    timed_out = True
                elif info.status in (StageStatus.DOWN, StageStatus.ERROR) and now - info.last_heartbeat > gc_ttl:
                    to_delete.append(input_addr)

            for input_addr in to_delete:
                del self._instances[input_addr]
        if timed_out:
            # Instance liveness changed; request broadcast.
            self._schedule_broadcast()

    def close(self) -> None:
        """Shut down background threads and close all ZMQ sockets."""
        if self._closed:
            raise RuntimeError("Coordinator already closed")

        self._closed = True
        self._running = False
        self._stop_event.set()

        # Wait for threads to exit before closing sockets.
        for thread in (self._recv_thread, self._periodic_thread):
            thread.join(timeout=1.0)

        try:
            self._router.close(0)
        except zmq.ZMQError:
            pass

        try:
            self._pub.close(0)
        except zmq.ZMQError:
            pass

        try:
            self._ctx.term()
        except zmq.ZMQError:
            pass

    def _parse_instance_event(self, data: dict[str, Any]) -> InstanceEvent | None:
        """Parse wire payload dict into InstanceEvent. Returns None if invalid."""
        try:
            return InstanceEvent(
                input_addr=str(data["input_addr"]),
                output_addr=str(data["output_addr"]),
                stage_id=int(data["stage_id"]),
                event_type=str(data["event_type"]),
                status=StageStatus(data.get("status")),
                queue_length=data.get("queue_length"),
            )
        except (KeyError, ValueError, TypeError):
            return None

    def _recv_loop(self) -> None:
        """Background loop that receives and processes instance events."""
        while self._running:
            try:
                frames = self._router.recv_multipart()
            except zmq.Again:
                # RCVTIMEO expired, loop to recheck _running.
                continue
            except zmq.ZMQError:
                # Socket likely closed or context terminated.
                break

            if not frames:
                continue

            payload = frames[-1]
            try:
                data = json.loads(payload.decode("utf-8"))
                event = self._parse_instance_event(data)
            except json.JSONDecodeError as e:
                logger.warning("Invalid JSON in instance event, dropping: %s", e)
                continue
            if event is None:
                logger.warning("Malformed instance event, dropping")
                continue

            self._handle_event(event)

    def _periodic_loop(self) -> None:
        """Periodic loop to check heartbeat timeouts and flush broadcasts.

        Heartbeat timeouts are checked on their original cadence, while all
        broadcast requests are coalesced and flushed at most once per
        ``_publish_min_interval``.
        """
        heartbeat_interval = max(1.0, min(self._heartbeat_timeout / 2.0, 5.0))
        loop_interval = self._publish_min_interval

        last_heartbeat_check = 0.0
        while self._running:
            now = time()

            if now - last_heartbeat_check >= heartbeat_interval:
                self._check_heartbeat_timeouts()
                last_heartbeat_check = now

            with self._pending_lock:
                has_pending_broadcast = self._pending_broadcast

            if not has_pending_broadcast:
                if self._stop_event.wait(timeout=loop_interval):
                    break
                continue

            # Publish outside lock. Clear pending only on success.
            if self.publish_instance_list_update():
                with self._pending_lock:
                    self._pending_broadcast = False

            if self._stop_event.wait(timeout=loop_interval):
                break

    def _handle_event(self, event: InstanceEvent) -> None:
        """Dispatch an incoming event to the appropriate handler."""
        try:
            input_addr = event.input_addr

            # Heartbeat: only update last_heartbeat; if previously ERROR,
            # promote back to UP and broadcast once.
            if event.event_type == "heartbeat":
                promote = False
                with self._lock:
                    info = self._instances.get(input_addr)
                    if info is not None:
                        info.last_heartbeat = time()
                        if info.status == StageStatus.ERROR:
                            info.status = StageStatus.UP
                            promote = True
                if promote:
                    self._schedule_broadcast()
                return

            # Check-and-act under single lock to avoid TOCTOU race (duplicate
            # registration when concurrent events arrive for the same instance).
            with self._lock:
                if input_addr not in self._instances:
                    self._add_new_instance_locked(event)
                else:
                    if event.status == StageStatus.DOWN:
                        self._remove_instance_locked(event)
                    else:
                        self._update_instance_info_locked(event)

            # Any non-heartbeat state change that affects the active list
            # is coalesced and flushed via the periodic loop.
            self._schedule_broadcast()
        except (KeyError, ValueError, TypeError) as e:
            logger.warning("Dropping malformed event: %s", e)

    def _add_new_instance_locked(self, event: InstanceEvent) -> None:
        input_addr = event.input_addr
        if not input_addr:
            raise KeyError("input_addr required")
        stage_id = event.stage_id
        if stage_id < 0:
            raise KeyError("stage_id required and must be non-negative")

        now = time()
        info = InstanceInfo(
            input_addr=input_addr,
            output_addr=event.output_addr,
            stage_id=stage_id,
            status=event.status,
            queue_length=event.queue_length,
            last_heartbeat=now,
            registered_at=now,
        )
        self._instances[input_addr] = info

    def _update_instance_info_locked(self, event: InstanceEvent) -> None:
        input_addr = event.input_addr
        info = self._instances[input_addr]

        if event.status is not None:
            info.status = event.status

        if event.queue_length is not None:
            info.queue_length = event.queue_length

    def _remove_instance_locked(self, event: InstanceEvent) -> None:
        input_addr = event.input_addr
        info = self._instances.get(input_addr)
        if info is None:
            return

        info.status = StageStatus.DOWN
