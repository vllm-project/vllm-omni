# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""TTL sweeping for storage backends.

Three strategies, picked once per backend:

``LAST_MODIFIED``
    Backends that expose a listing ``last_modified`` (fs, S3): an artifact is
    expired when ``last_modified + ttl`` has passed. Zero record reads, and
    equivalent to the pre-record ``st_mtime`` sweep. The backend's configured
    default TTL is used, as timestamps carry no per-object TTL.
``RECORD_EXPIRES_AT``
    Backends that can list but expose no timestamp (memory): an artifact is
    expired when the ``expires_at`` of its job record has passed. This honours
    per-object TTLs, at the cost of one record read per artifact.
``NATIVE_TTL``
    Backends that cannot enumerate objects (redis): expiry is the backend's
    own business, so the server-side sweeper stays disabled.
"""

import asyncio
import contextlib
import json
import time
from collections.abc import Awaitable, Callable
from enum import Enum

from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.storage.base import StorageBackend

logger = init_logger(__name__)


class SweepStrategy(str, Enum):
    #: Backend expires objects itself (native ``default_ttl``); nothing to sweep.
    NATIVE_TTL = "native_ttl"
    #: Fast path: artifact ``last_modified`` + ttl, zero record reads.
    LAST_MODIFIED = "last_modified"
    #: Fallback: read ``expires_at`` from each job record.
    RECORD_EXPIRES_AT = "record_expires_at"


def _record_expires_at(data: bytes | None) -> float | None:
    if data is None:
        return None
    try:
        record = json.loads(data)
    except ValueError:
        logger.warning("Ignoring unreadable job record during TTL sweep")
        return None
    if not isinstance(record, dict):
        return None
    expires_at = record.get("expires_at")
    return float(expires_at) if isinstance(expires_at, int | float) else None


class TtlSweeper:
    """Delete artifacts (and their records) once their TTL elapsed."""

    def __init__(
        self,
        backend: StorageBackend,
        *,
        ttl_seconds: int,
        sweep_interval_seconds: int,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("`ttl_seconds` must be greater than or equal to 1.")
        if sweep_interval_seconds <= 0:
            raise ValueError("`sweep_interval_seconds` must be greater than or equal to 1.")

        self._backend = backend
        self._ttl_seconds = ttl_seconds
        self._sweep_interval_seconds = sweep_interval_seconds
        self._task: asyncio.Task[None] | None = None
        self._strategy: SweepStrategy | None = None
        self._on_expire: Callable[[list[str]], Awaitable[None]] | None = None

    async def strategy(self) -> SweepStrategy:
        """Detect (once) how expiry is evaluated for this backend."""
        if self._strategy is None:
            if not self._backend.supports_list:
                self._strategy = SweepStrategy.NATIVE_TTL
            elif await self._backend.probe_last_modified():
                self._strategy = SweepStrategy.LAST_MODIFIED
            else:
                self._strategy = SweepStrategy.RECORD_EXPIRES_AT
            logger.info("Storage TTL sweep strategy: %s", self._strategy.value)
        return self._strategy

    async def sweep_once(self) -> int:
        strategy = await self.strategy()
        if strategy is SweepStrategy.NATIVE_TTL:
            return 0

        now = time.time()
        cutoff = now - self._ttl_seconds
        deleted_keys: list[str] = []
        async for key, last_modified in self._backend.list_artifacts():
            if await self._is_expired(key, last_modified, strategy, cutoff=cutoff, now=now):
                try:
                    await self._backend.delete(key)
                except Exception:
                    logger.warning("TTL sweep failed to delete expired artifact %s", key, exc_info=True)
                    continue
                deleted_keys.append(key)

        if deleted_keys and self._on_expire is not None:
            try:
                await self._on_expire(deleted_keys)
            except Exception:
                logger.warning("Expiry handler failed for %s", deleted_keys, exc_info=True)
        return len(deleted_keys)

    async def _is_expired(
        self,
        key: str,
        last_modified: float | None,
        strategy: SweepStrategy,
        *,
        cutoff: float,
        now: float,
    ) -> bool:
        if strategy is SweepStrategy.LAST_MODIFIED:
            return last_modified is not None and last_modified < cutoff

        expires_at = _record_expires_at(await self._backend.record_get(key))
        if expires_at is None:
            # No record (or no ttl): nothing tells us this artifact expired.
            logger.debug("TTL sweep skipped %s; no expiry information", key)
            return False
        return expires_at <= now

    async def start(self, *, on_expire: Callable[[list[str]], Awaitable[None]] | None = None) -> None:
        if self._task is None or self._task.done():
            self._on_expire = on_expire
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        self._task = None

    async def _loop(self) -> None:
        while True:
            try:
                await self.sweep_once()
            except Exception:
                logger.exception("Storage TTL sweep failed")
            await asyncio.sleep(self._sweep_interval_seconds)
