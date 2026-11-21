# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Ownership lease for one storage prefix.

Startup reconciliation is destructive: it fails `queued`/`in_progress` jobs and
deletes their artifacts. That is only safe when no other live API process is
using the same prefix. Every process therefore refreshes a small lease record
while it runs, and only the owner (or the process that finds the lease stale)
may run destructive recovery. Callers must fail closed while the lease is not
held: no new video jobs (otherwise the process would own work it may later
"recover") and no destructive cleanup.

One API process per storage prefix stays the supported deployment; the lease
turns an accidental overlap into a loud warning instead of deleting a live job.
Acquisition is atomic only where the backend supports a conditional write (S3);
the local filesystem has no compare-and-swap, so there a takeover is settled
with a randomized re-check that both contenders run before writing.
"""

import asyncio
import contextlib
import json
import os
import random
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.storage.base import StorageBackend

logger = init_logger(__name__)

#: Control record holding the current owner. Dotted names are skipped by
#: ``record_keys``/``list_artifacts``, so it never shows up as a job.
OWNER_RECORD_KEY = ".owner"

#: A lease older than this is treated as abandoned (process died).
OWNER_LEASE_TTL_SECONDS = 90

#: Randomized pause both contenders take before taking over a stale lease.
TAKEOVER_SETTLE_SECONDS = 1.0


class BackendLease:
    """Best-effort ownership lease over the backend's record layer."""

    def __init__(
        self,
        backend: StorageBackend,
        *,
        ttl_seconds: int = OWNER_LEASE_TTL_SECONDS,
        owner_id: str | None = None,
        on_acquired: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("`ttl_seconds` must be greater than or equal to 1.")
        self._backend = backend
        self._ttl_seconds = ttl_seconds
        self._owner = owner_id or uuid.uuid4().hex
        self._on_acquired = on_acquired
        self._task: asyncio.Task[None] | None = None
        self._held = False
        self._reported_foreign_owner = False

    @property
    def owner_id(self) -> str:
        return self._owner

    @property
    def held(self) -> bool:
        return self._held

    async def _read(self) -> tuple[dict[str, Any] | None, bool]:
        """Return ``(lease, known)``; ``known`` is False when the read failed.

        An uncertain read must never be treated as "prefix is free": that is how
        two processes end up owning the same prefix.
        """
        try:
            raw = await self._backend.record_get(OWNER_RECORD_KEY)
        except Exception:
            logger.warning("Could not read the storage ownership lease", exc_info=True)
            return None, False
        if raw is None:
            return None, True
        try:
            lease = json.loads(raw)
        except ValueError:
            return None, True
        return (lease if isinstance(lease, dict) else None), True

    def _payload(self) -> bytes:
        payload = {"owner": self._owner, "heartbeat_at": time.time(), "pid": os.getpid()}
        return json.dumps(payload).encode()

    async def _write(self) -> None:
        await self._backend.record_set(OWNER_RECORD_KEY, self._payload())

    async def acquire(self) -> bool:
        """Take the lease unless another live owner holds it."""
        lease, known = await self._read()
        if not known:
            # Cannot tell whether the prefix is owned: do not write anything.
            return False
        if lease is None:
            # Free prefix: create-if-absent, atomic on backends with a
            # conditional write.
            try:
                created = await self._backend.record_create(OWNER_RECORD_KEY, self._payload())
            except Exception:
                logger.warning("Could not write the storage ownership lease", exc_info=True)
                return False
            if not created:
                logger.warning("Another process acquired the storage lease first")
                return False
            self._held = True
            return True

        if lease.get("owner") != self._owner:
            heartbeat_at = lease.get("heartbeat_at")
            fresh = isinstance(heartbeat_at, int | float) and time.time() - heartbeat_at < self._ttl_seconds
            if fresh:
                if not self._reported_foreign_owner:
                    self._reported_foreign_owner = True
                    logger.error(
                        "Storage prefix is owned by another live API process (owner=%s, pid=%s, heartbeat %.0fs ago). "
                        "Skipping destructive startup recovery; run one API process per storage prefix. "
                        "Recovery retries in the background and runs once that lease goes stale.",
                        lease.get("owner"),
                        lease.get("pid"),
                        time.time() - float(heartbeat_at),
                    )
                return False
            # Both contenders for a stale lease settle for a random moment and
            # re-read: the first writer refreshes the heartbeat, so the other one
            # sees a live lease and backs off instead of double-owning.
            await asyncio.sleep(random.uniform(0, TAKEOVER_SETTLE_SECONDS))
            current, still_known = await self._read()
            if not still_known or current is None or current.get("owner") != lease.get("owner"):
                logger.warning("Storage lease changed while taking it over; retrying later")
                return False
            logger.warning("Taking over an abandoned storage lease (owner=%s)", lease.get("owner"))

        try:
            await self._write()
        except Exception:
            logger.warning("Could not write the storage ownership lease", exc_info=True)
            return False
        confirmed, confirmed_known = await self._read()
        if not confirmed_known or (confirmed or {}).get("owner") != self._owner:
            logger.warning("Lost the storage lease race; backing off")
            return False
        self._held = True
        return True

    async def start(self) -> None:
        """Keep the lease fresh, and take it over once a foreign owner goes stale."""
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._heartbeat_loop())

    async def release(self) -> None:
        """Stop the heartbeat and hand the lease back."""
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        if not self._held:
            return
        self._held = False
        lease, known = await self._read()
        if not known or lease is None or lease.get("owner") != self._owner:
            return
        with contextlib.suppress(Exception):
            await self._backend.record_pop(OWNER_RECORD_KEY)

    async def _heartbeat_loop(self) -> None:
        interval = max(0.2, self._ttl_seconds / 3)
        while True:
            await asyncio.sleep(interval)
            if not self._held:
                # A previous owner may have died: retry the takeover so its
                # interrupted jobs still get recovered without a manual restart.
                if not await self.acquire():
                    continue
                logger.warning("Took over the abandoned storage lease; running startup recovery")
                if self._on_acquired is not None:
                    try:
                        await self._on_acquired()
                    except Exception:
                        logger.exception("Recovery after taking over the storage lease failed")
                continue
            try:
                await self._write()
            except Exception:
                logger.warning("Failed to refresh the storage ownership lease", exc_info=True)
