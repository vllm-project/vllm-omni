# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""In-memory session store for RL rollout serving (RFC #3747, P0).

Each RolloutSession holds a per-session asyncio.Lock so that concurrent step
requests for the same session are serialised. committed_step_id advances only
on successful step completion; failed or timed-out steps leave it unchanged,
preserving the atomic-context-commit guarantee from section 6.5 of the RFC.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Literal


class RolloutSessionNotFoundError(KeyError):
    pass


class RolloutSessionClosedError(RuntimeError):
    pass


class RolloutSessionStepError(ValueError):
    pass


@dataclass
class RolloutSession:
    session_id: str
    model: str
    mode: Literal["world_model_env"]
    created_at: float = field(default_factory=time.time)
    committed_step_id: int = -1
    context_length: int = 0
    closed: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class RolloutSessionStore:
    """Thread-safe (asyncio) store for active rollout sessions."""

    def __init__(self) -> None:
        self._sessions: dict[str, RolloutSession] = {}
        self._store_lock = asyncio.Lock()

    async def create(self, session_id: str, model: str, mode: str) -> RolloutSession:
        async with self._store_lock:
            session = RolloutSession(
                session_id=session_id,
                model=model,
                mode=mode,  # type: ignore[arg-type]
            )
            self._sessions[session_id] = session
            return session

    async def get(self, session_id: str) -> RolloutSession:
        session = self._sessions.get(session_id)
        if session is None:
            raise RolloutSessionNotFoundError(session_id)
        if session.closed:
            raise RolloutSessionClosedError(session_id)
        return session

    async def reset(self, session_id: str) -> RolloutSession:
        session = await self.get(session_id)
        async with session.lock:
            session.committed_step_id = -1
            session.context_length = 0
        return session

    async def close(self, session_id: str) -> None:
        session = await self.get(session_id)
        session.closed = True

    async def advance(self, session_id: str, step_id: int) -> None:
        """Commit a successfully completed step. Called only on success."""
        session = await self.get(session_id)
        expected_step_id = session.committed_step_id + 1
        if step_id != expected_step_id:
            raise RolloutSessionStepError(f"Expected step_id {expected_step_id}, got {step_id}.")
        session.committed_step_id = step_id
        session.context_length += 1
