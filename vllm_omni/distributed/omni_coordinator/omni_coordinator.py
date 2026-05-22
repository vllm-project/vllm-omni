# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import json
import logging
import threading
from dataclasses import asdict
from time import time
from typing import Any

import zmq

from .messages import ReplicaEvent, ReplicaInfo, ReplicaList, ReplicaStatus

logger = logging.getLogger(__name__)


class OmniCoordinator:
    """Coordinator for stage replicas and hub clients.

    This service receives replica events from :class:`OmniCoordClientForStage`
    via a ZMQ ROUTER socket and publishes active replica lists to
    :class:`OmniCoordClientForHub` via a PUB socket.

    The coordinator maintains an in-memory registry of all known replicas,
    including their status, queue length, and heartbeat timestamps. A
    background thread periodically checks for heartbeat timeouts and marks
    unhealthy replicas as ``ReplicaStatus.ERROR``.
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
            heartbeat_timeout: Seconds before a replica is considered
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

        self._replicas: dict[str, ReplicaInfo] = {}
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

    def get_active_replicas(self) -> ReplicaList:
        """Return a :class:`ReplicaList` of active (UP) replicas only."""
        with self._lock:
            active = [rep for rep in self._replicas.values() if rep.status == ReplicaStatus.UP]
        return ReplicaList(replicas=active, timestamp=time())

    def add_new_replica(self, event: ReplicaEvent) -> None:
        """Add a new replica based on an incoming event."""
        with self._lock:
            self._add_new_replica_locked(event)
        self._schedule_broadcast()

    def update_replica_info(self, event: ReplicaEvent) -> None:
        """Update an existing replica based on an incoming event."""
        with self._lock:
            self._update_replica_info_locked(event)
        self._schedule_broadcast()

    def remove_replica(self, event: ReplicaEvent) -> None:
        """Mark a replica as removed / down based on an incoming event.

        This marks the replica's status as DOWN or ERROR (depending on the
        event) but keeps it in the internal registry. It is removed from the
        *active* replica list published to hubs.
        """
        with self._lock:
            self._remove_replica_locked(event)
        self._schedule_broadcast()

    def publish_replica_list_update(self) -> bool:
        """Publish the current active replica list to all subscribers.

        Returns:
            True if the PUB send succeeded, False if it was dropped (e.g.
            socket not ready when using ``zmq.NOBLOCK``).
        """
        active_list = self.get_active_replicas()
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

    def _mark_replica_error_locked(self, info: ReplicaInfo) -> None:
        """Mark replica as ERROR (e.g. after heartbeat timeout)."""
        info.status = ReplicaStatus.ERROR

    def _check_heartbeat_timeouts(self) -> None:
        """Mark replicas as ERROR if their heartbeat has timed out."""
        now = time()
        timed_out = False
        gc_ttl = 600.0  # 10 minutes

        with self._lock:
            to_delete: list[str] = []

            for input_addr, info in self._replicas.items():
                if info.status == ReplicaStatus.UP and now - info.last_heartbeat > self._heartbeat_timeout:
                    self._mark_replica_error_locked(info)
                    timed_out = True
                elif info.status in (ReplicaStatus.DOWN, ReplicaStatus.ERROR) and now - info.last_heartbeat > gc_ttl:
                    to_delete.append(input_addr)

            for input_addr in to_delete:
                del self._replicas[input_addr]
        if timed_out:
            # Replica liveness changed; request broadcast.
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

    def _parse_replica_event(self, data: dict[str, Any]) -> ReplicaEvent | None:
        """Parse wire payload dict into ReplicaEvent. Returns None if invalid."""
        try:
            return ReplicaEvent(
                input_addr=str(data["input_addr"]),
                output_addr=str(data["output_addr"]),
                stage_id=int(data["stage_id"]),
                event_type=str(data["event_type"]),
                status=ReplicaStatus(data.get("status")),
                queue_length=data.get("queue_length"),
            )
        except (KeyError, ValueError, TypeError):
            return None

    def _recv_loop(self) -> None:
        """Background loop that receives and processes replica events."""
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
                event = self._parse_replica_event(data)
            except json.JSONDecodeError as e:
                logger.warning("Invalid JSON in replica event, dropping: %s", e)
                continue
            if event is None:
                logger.warning("Malformed replica event, dropping")
                continue

            self._handle_event(event)

    def _periodic_loop(self) -> None:
        """Periodic loop to check heartbeat timeouts and flush broadcasts.

        Heartbeat timeouts are checked on their original cadence, while all
        broadcast requests are coalesced and flushed at most once per
        ``_publish_min_interval``. The heartbeat-check tick also schedules a
        keepalive broadcast so late-joining hubs (which miss any PUB sends
        that happened before their SUB connected) catch up within at most
        ``heartbeat_interval`` seconds.
        """
        heartbeat_interval = max(1.0, min(self._heartbeat_timeout / 2.0, 5.0))
        loop_interval = self._publish_min_interval

        last_heartbeat_check = 0.0
        while self._running:
            now = time()

            if now - last_heartbeat_check >= heartbeat_interval:
                self._check_heartbeat_timeouts()
                # Keepalive broadcast: ZMQ PUB doesn't queue for late
                # subscribers, so an OmniCoordClientForHub that connects
                # after the initial UP events miss them entirely and would
                # never see the current replica list otherwise. Scheduling a
                # broadcast on every heartbeat tick caps that staleness at
                # ``heartbeat_interval`` without flooding the wire.
                self._schedule_broadcast()
                last_heartbeat_check = now

            with self._pending_lock:
                has_pending_broadcast = self._pending_broadcast

            if not has_pending_broadcast:
                if self._stop_event.wait(timeout=loop_interval):
                    break
                continue

            # Publish outside lock. Clear pending only on success.
            if self.publish_replica_list_update():
                with self._pending_lock:
                    self._pending_broadcast = False

            if self._stop_event.wait(timeout=loop_interval):
                break

    def _handle_event(self, event: ReplicaEvent) -> None:
        """Dispatch an incoming event to the appropriate handler."""
        try:
            input_addr = event.input_addr

            # Heartbeat: refresh last_heartbeat and queue_length. The stage
            # client refreshes queue_length just-in-time via its
            # ``_on_heartbeat`` hook, so heartbeats are the only periodic
            # source of live load for LeastQueueLengthBalancer; failing to
            # propagate it here would let the policy route on stale data.
            # If previously ERROR, promote back to UP and broadcast once.
            if event.event_type == "heartbeat":
                promote = False
                queue_changed = False
                with self._lock:
                    info = self._replicas.get(input_addr)
                    if info is not None:
                        info.last_heartbeat = time()
                        if event.queue_length is not None and info.queue_length != event.queue_length:
                            info.queue_length = event.queue_length
                            queue_changed = True
                        if info.status == ReplicaStatus.ERROR:
                            info.status = ReplicaStatus.UP
                            promote = True
                if promote or queue_changed:
                    self._schedule_broadcast()
                return

            # Check-and-act under single lock to avoid TOCTOU race (duplicate
            # registration when concurrent events arrive for the same replica).
            with self._lock:
                if input_addr not in self._replicas:
                    self._add_new_replica_locked(event)
                else:
                    if event.status == ReplicaStatus.DOWN:
                        self._remove_replica_locked(event)
                    else:
                        self._update_replica_info_locked(event)

            # Any non-heartbeat state change that affects the active list
            # is coalesced and flushed via the periodic loop.
            self._schedule_broadcast()
        except (KeyError, ValueError, TypeError) as e:
            logger.warning("Dropping malformed event: %s", e)

    def _add_new_replica_locked(self, event: ReplicaEvent) -> None:
        input_addr = event.input_addr
        if not input_addr:
            raise KeyError("input_addr required")
        stage_id = event.stage_id
        if stage_id < 0:
            raise KeyError("stage_id required and must be non-negative")

        now = time()
        info = ReplicaInfo(
            input_addr=input_addr,
            output_addr=event.output_addr,
            stage_id=stage_id,
            status=event.status,
            queue_length=event.queue_length,
            last_heartbeat=now,
            registered_at=now,
        )
        self._replicas[input_addr] = info

    def _update_replica_info_locked(self, event: ReplicaEvent) -> None:
        input_addr = event.input_addr
        info = self._replicas[input_addr]

        if event.status is not None:
            info.status = event.status

        if event.queue_length is not None:
            info.queue_length = event.queue_length

    def _remove_replica_locked(self, event: ReplicaEvent) -> None:
        input_addr = event.input_addr
        info = self._replicas.get(input_addr)
        if info is None:
            return

        info.status = ReplicaStatus.DOWN
