# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import logging
import threading
from time import sleep, time
from typing import Any

import zmq

from .messages import ReplicaInfo, ReplicaList, ReplicaStatus

logger = logging.getLogger(__name__)


class OmniCoordClientForHub:
    """Client for AsyncOmni side to receive replica list updates.

    This client maintains a SUB socket connected to OmniCoordinator's PUB
    endpoint and caches the latest :class:`ReplicaList` in memory for use by
    the load balancer and routing logic.
    """

    def __init__(self, coord_zmq_addr: str) -> None:
        """Initialize client and start receive thread (socket created in thread)."""
        self._coord_zmq_addr = coord_zmq_addr

        self._ctx = zmq.Context()
        self._lock = threading.Lock()
        self._replica_list: ReplicaList | None = None
        self._closed = False
        self._stop_event = threading.Event()
        self._init_done = threading.Event()
        self._init_error: list[BaseException] = []

        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

        self._init_done.wait(timeout=5.0)
        if self._init_error:
            raise RuntimeError(f"Failed to connect to coordinator at {self._coord_zmq_addr}") from self._init_error[0]

    def _decode_replica_list(self, payload: dict[str, Any]) -> ReplicaList:
        """Convert a JSON-decoded dict into a :class:`ReplicaList`."""
        replicas_payload = payload.get("replicas", [])
        replicas: list[ReplicaInfo] = []
        for rep in replicas_payload:
            replicas.append(
                ReplicaInfo(
                    input_addr=rep["input_addr"],
                    output_addr=rep["output_addr"],
                    stage_id=int(rep["stage_id"]),
                    status=ReplicaStatus(rep["status"]),
                    queue_length=int(rep["queue_length"]),
                    last_heartbeat=float(rep["last_heartbeat"]),
                    registered_at=float(rep["registered_at"]),
                )
            )

        timestamp = float(payload.get("timestamp", time()))
        return ReplicaList(replicas=replicas, timestamp=timestamp)

    def _recv_loop(self) -> None:
        """Background loop that receives and caches replica lists."""
        sub: zmq.Socket | None = None
        try:
            sub = self._ctx.socket(zmq.SUB)
            sub.setsockopt(zmq.SUBSCRIBE, b"")
            sub.setsockopt(zmq.RCVTIMEO, 100)  # 100ms timeout, avoids busy-wait
            sub.connect(self._coord_zmq_addr)
        except (zmq.ZMQError, OSError) as e:
            self._init_error.append(e)
            sub = None
        finally:
            self._init_done.set()

        try:
            while not self._stop_event.is_set():
                # (Re)create and connect SUB socket if needed.
                if sub is None:
                    try:
                        sub = self._ctx.socket(zmq.SUB)
                        sub.setsockopt(zmq.SUBSCRIBE, b"")
                        sub.setsockopt(zmq.RCVTIMEO, 100)  # 100ms timeout, avoids busy-wait
                        sub.connect(self._coord_zmq_addr)
                    except (zmq.ZMQError, OSError) as e:
                        logger.error(
                            "Hub client failed to connect to coordinator at %s, will retry",
                            self._coord_zmq_addr,
                            exc_info=e,
                        )
                        if sub is not None:
                            try:
                                sub.close()
                            except zmq.ZMQError:
                                pass
                            sub = None
                        sleep(1.0)
                        continue

                try:
                    data = sub.recv()
                except zmq.Again:
                    continue
                except zmq.ZMQError as e:
                    logger.error("Hub client recv failed, will reconnect", exc_info=e)
                    try:
                        sub.close()
                    except zmq.ZMQError:
                        pass
                    sub = None
                    sleep(1.0)
                    continue

                try:
                    payload = json.loads(data.decode("utf-8"))
                    rep_list = self._decode_replica_list(payload)
                    with self._lock:
                        self._replica_list = rep_list
                except (
                    json.JSONDecodeError,
                    KeyError,
                    ValueError,
                    TypeError,
                    AttributeError,
                ) as e:
                    logger.warning("Invalid replica list message, skipping: %s", e)
        finally:
            try:
                if sub is not None:
                    sub.close()
            except zmq.ZMQError:
                pass
            try:
                self._ctx.term()
            except zmq.ZMQError:
                pass

    def get_replica_list(self) -> ReplicaList:
        """Return the latest cached :class:`ReplicaList`.

        If no update has been received yet, returns an empty list with
        ``timestamp=0.0``.
        """
        with self._lock:
            if self._replica_list is None:
                return ReplicaList(replicas=[], timestamp=0.0)
            return self._replica_list

    def get_replicas_for_stage(self, stage_id: int) -> ReplicaList:
        """Return replicas filtered by ``stage_id``."""
        base = self.get_replica_list()
        filtered = [rep for rep in base.replicas if rep.stage_id == stage_id]
        return ReplicaList(replicas=filtered, timestamp=base.timestamp)

    def close(self) -> None:
        """Close the SUB socket and stop the background thread."""
        if self._closed:
            raise RuntimeError("Client already closed")

        self._closed = True
        self._stop_event.set()
        self._thread.join(timeout=1.0)
