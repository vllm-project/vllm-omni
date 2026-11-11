# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Which replica owns a session, and what happens when it is gone.

A Noisy KV session keeps its history on the workers of one replica. A follow-up
request for that session therefore has to reach the same replica; sending it
anywhere else would either start cold or, worse, read a different session's
state. This module owns that decision so the engine, the stage pool and the
worker agree on one rule instead of each guessing.

The contract, from the T1 roadmap (2.1/2.2):

* The first request of a session binds it to a replica.
* Every later request must route to the replica that holds it.
* A replica that is no longer live cannot serve the session. The caller either
  refuses the request (``allow_cold=False``) or explicitly accepts a cold
  restart, which rebinds the session and drops the history it cannot reach.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from vllm.logger import init_logger

logger = init_logger(__name__)


class SessionAffinityError(RuntimeError):
    """The session cannot be served by any candidate replica."""


@dataclass
class SessionRouter:
    """Session id -> replica id, with an explicit cold-restart path."""

    #: Sessions bound to a replica that is currently serving them.
    _owners: dict[str, int] = field(default_factory=dict)
    #: Bindings whose replica disappeared, kept so a cold restart is visible.
    _stale: dict[str, int] = field(default_factory=dict)

    def owner(self, session_id: str) -> int | None:
        return self._owners.get(session_id)

    def bind(self, session_id: str, replica_id: int) -> int:
        """Bind a session to a replica; re-binding to the same one is a no-op."""
        owner = self._owners.get(session_id)
        if owner is not None and owner != replica_id:
            raise SessionAffinityError(f"session {session_id!r} is bound to replica {owner}, not {replica_id}")
        self._owners[session_id] = replica_id
        # `_stale` is not cleared here: it records that this session lost its
        # history at some point, and only release() forgets that.
        return replica_id

    def route(self, session_id: str, candidates: Sequence[int], *, allow_cold: bool = False) -> int:
        """Pick the replica that must serve ``session_id``.

        ``candidates`` is what the stage pool can actually reach right now. An
        unbound session binds to the first candidate; a bound one must find its
        owner among them.
        """
        if not candidates:
            raise SessionAffinityError(f"no live replica can serve session {session_id!r}")
        owner = self._owners.get(session_id)
        if owner is None:
            return self.bind(session_id, candidates[0])
        if owner in candidates:
            return owner
        if not allow_cold:
            raise SessionAffinityError(
                f"session {session_id!r} is bound to replica {owner}, which is not live "
                f"(live: {list(candidates)}); pass allow_cold to restart it"
            )
        # The caller accepted losing the history: rebind and record the drop.
        logger.warning(
            "Noisy KV session %s lost replica %d; restarting cold on replica %d",
            session_id,
            owner,
            candidates[0],
        )
        self._stale[session_id] = owner
        # Drop the old binding first: `bind` refuses to move a session that
        # still has one, and the point of this branch is that it no longer does.
        self._owners.pop(session_id, None)
        return self.bind(session_id, candidates[0])

    def release(self, session_id: str) -> None:
        """Forget a session (close/reset by the owner)."""
        self._owners.pop(session_id, None)
        self._stale.pop(session_id, None)

    def release_replica(self, replica_id: int) -> list[str]:
        """Forget every session bound to a replica that went away."""
        dropped = [sid for sid, owner in self._owners.items() if owner == replica_id]
        for session_id in dropped:
            self._stale[session_id] = self._owners.pop(session_id)
        return dropped

    def is_stale(self, session_id: str) -> bool:
        return session_id in self._stale

    def __len__(self) -> int:
        return len(self._owners)
