"""DiT-stage load state aggregator (Orchestrator-side).

``DitLoadState`` aggregates each DiT replica's scheduler queue depth
(waiting / running counts + request-id sets) polled by the Orchestrator
every loop tick. The aggregated snapshot is pushed to the AR scheduler
via the UTILITY ZMQ channel. Per-replica ``waiting``/``running``
request-id sets are unioned so the AR scheduler can de-duplicate its blind
in-flight set. This object only exposes raw aggregates; ``DTPSScheduler``
owns the threshold and decides the idle/busy phase.
"""

from __future__ import annotations

import threading
from typing import TypedDict


class DitLoadSnapshot(TypedDict):
    min_waiting: int
    max_waiting: int
    total_waiting: int
    total_running: int
    num_replicas: int
    fresh: bool
    waiting_ids: frozenset[str]
    running_ids: frozenset[str]


class DitLoadState:
    """Thread-safe aggregator of per-replica DiT waiting/running counts + ids."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._per_replica: dict[tuple[int, int], tuple[int, int, frozenset[str], frozenset[str]]] = {}

    def update(
        self,
        stage_id: int,
        replica_id: int,
        waiting: int,
        running: int,
        waiting_ids: frozenset[str] | set[str] | list[str] | None = None,
        running_ids: frozenset[str] | set[str] | list[str] | None = None,
    ) -> bool:
        """Update per-replica load data.

        Returns ``True`` if the new data differs from the previously stored
        data for this replica, ``False`` if it is unchanged. Callers can use
        the return value to skip redundant downstream propagation when the
        load snapshot has not changed.
        """
        try:
            w = int(waiting)
        except (TypeError, ValueError):
            w = 0
        try:
            r = int(running)
        except (TypeError, ValueError):
            r = 0
        if w < 0:
            w = 0
        if r < 0:
            r = 0
        w_ids = frozenset(waiting_ids) if waiting_ids else frozenset()
        r_ids = frozenset(running_ids) if running_ids else frozenset()
        key = (int(stage_id), int(replica_id))
        new_entry = (w, r, w_ids, r_ids)
        with self._lock:
            old_entry = self._per_replica.get(key)
            if old_entry is not None and old_entry == new_entry:
                return False
            self._per_replica[key] = new_entry
            return True

    def remove(self, stage_id: int, replica_id: int) -> None:
        key = (int(stage_id), int(replica_id))
        with self._lock:
            self._per_replica.pop(key, None)

    def snapshot(self) -> DitLoadSnapshot:
        with self._lock:
            if not self._per_replica:
                return self._empty_snapshot()
            waitings = [v[0] for v in self._per_replica.values()]
            runnings = [v[1] for v in self._per_replica.values()]
            waiting_ids: frozenset[str] = frozenset()
            running_ids: frozenset[str] = frozenset()
            for v in self._per_replica.values():
                waiting_ids = waiting_ids | v[2]
                running_ids = running_ids | v[3]
            return {
                "min_waiting": min(waitings),
                "max_waiting": max(waitings),
                "total_waiting": sum(waitings),
                "total_running": sum(runnings),
                "num_replicas": len(waitings),
                "fresh": True,
                "waiting_ids": waiting_ids,
                "running_ids": running_ids,
            }

    @staticmethod
    def _empty_snapshot() -> DitLoadSnapshot:
        return {
            "min_waiting": 0,
            "max_waiting": 0,
            "total_waiting": 0,
            "total_running": 0,
            "num_replicas": 0,
            "fresh": False,
            "waiting_ids": frozenset(),
            "running_ids": frozenset(),
        }
