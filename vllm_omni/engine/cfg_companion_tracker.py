"""CFG companion request tracker for the Omni orchestrator.

Encapsulates all bookkeeping for Classifier-Free Guidance companion
requests (parent/companion ID mapping, completion tracking,
deferred forwarding, and cleanup).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class CfgCompanionTracker:
    """Manages CFG companion request lifecycle in the orchestrator scheduling loop."""

    def __init__(self) -> None:
        self._companion_map: dict[str, dict[str, str]] = {}  # parent -> {role: companion_id}
        self._companion_ids: set[str] = set()
        self._companion_to_parent: dict[str, str] = {}  # companion -> parent
        self._done: dict[str, set[str]] = {}  # parent -> completed companion ids
        self._pending_parents: dict[str, dict[str, Any]] = {}  # parent -> deferred result

    def is_companion(self, req_id: str) -> bool:
        return req_id in self._companion_ids

    def has_companions(self, parent_id: str) -> bool:
        return parent_id in self._companion_map

    def get_parent_id(self, req_id: str) -> str | None:
        """Return the parent request id for a companion, or None."""
        return self._companion_to_parent.get(req_id)

    def all_companions_done(self, parent_id: str) -> bool:
        role_map = self._companion_map.get(parent_id, {})
        done_set = self._done.get(parent_id, set())
        return all(cid in done_set for cid in role_map.values())

    def get_companion_request_ids(self, parent_id: str) -> dict[str, str]:
        """Return ``{role: companion_request_id}`` for a parent."""
        return self._companion_map.get(parent_id, {})

    def register_parent(self, parent_id: str) -> None:
        self._companion_map.setdefault(parent_id, {})
        self._done.setdefault(parent_id, set())

    def register_companion(self, parent_id: str, role: str, companion_id: str) -> None:
        self.register_parent(parent_id)
        self._companion_map[parent_id][role] = companion_id
        self._companion_ids.add(companion_id)
        self._companion_to_parent[companion_id] = parent_id

    def on_companion_completed(self, companion_id: str) -> str | None:
        """Mark done. Returns parent_id only if parent is pending and all companions finished."""
        parent_id = self._companion_to_parent.get(companion_id)
        if parent_id is None:
            return None
        done_set = self._done.get(parent_id)
        assert done_set is not None, f"Companion {companion_id} completed before parent {parent_id} was registered"
        if companion_id in done_set:
            return None
        done_set.add(companion_id)
        logger.debug("CFG companion %s completed (parent=%s)", companion_id, parent_id)
        if parent_id in self._pending_parents and self.all_companions_done(parent_id):
            return parent_id
        return None

    def defer_parent(self, parent_id: str, engine_outputs: Any, stage_id: int) -> None:
        """Hold parent result while waiting for companions to finish."""
        # TODO: Add timeout/error recovery when the orchestrator grows a
        # companion-failure path. Today deferred parents are released only when
        # companions finish or the external layer aborts the request.
        self._pending_parents[parent_id] = {
            "engine_outputs": engine_outputs,
            "stage_id": stage_id,
        }
        logger.debug("Parent %s deferred, waiting for CFG companions", parent_id)

    def pop_pending_parent(self, parent_id: str) -> dict[str, Any] | None:
        return self._pending_parents.pop(parent_id, None)

    def cleanup_parent(self, parent_id: str) -> list[str]:
        companion_ids = list(self._companion_map.pop(parent_id, {}).values())
        for companion_id in companion_ids:
            self._companion_ids.discard(companion_id)
            self._companion_to_parent.pop(companion_id, None)
        self._done.pop(parent_id, None)
        self._pending_parents.pop(parent_id, None)
        return companion_ids

    def abort_parents(self, request_ids: list[str]) -> list[str]:
        all_request_ids = list(request_ids)
        seen = set(all_request_ids)
        parents_to_cleanup: set[str] = set()

        for req_id in request_ids:
            # The orchestrator calls this with parent request IDs. If a raw
            # companion ID is passed here, keep it as a direct abort target and
            # avoid tearing down parent tracking state implicitly.
            if req_id not in self._companion_ids:
                parents_to_cleanup.add(req_id)

        for parent_id in parents_to_cleanup:
            companion_ids = self.cleanup_parent(parent_id)
            for companion_id in companion_ids:
                if companion_id not in seen:
                    seen.add(companion_id)
                    all_request_ids.append(companion_id)

        return all_request_ids
