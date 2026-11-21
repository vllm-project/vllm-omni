# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Apache OpenDAL backed storage (local filesystem).

Layout under the backend root::

    artifacts/{key}        # artifact payload (large object)
    meta/{key}.json        # job record (small JSON document)

Only bytes move through this layer; record serialization stays in the caller.
"""

import asyncio
import contextlib
import json
import os
import time
from collections.abc import AsyncIterator, Awaitable, Callable

import opendal
from opendal.types import Metadata
from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.storage.base import (
    DEFAULT_CHUNK_SIZE,
    FileStorageHandle,
    SaveContext,
    StorageBackend,
)
from vllm_omni.entrypoints.openai.storage.ttl import TtlSweeper

logger = init_logger(__name__)

ARTIFACT_PREFIX = "artifacts"
RECORD_PREFIX = "meta"
RECORD_SUFFIX = ".json"


def _validate_key(key: str) -> str:
    """Reject keys that could escape the backend root or alias another object."""
    if not key or key.startswith("/") or key.endswith("/"):
        raise ValueError(f"Illegal storage key: {key!r}")
    if any(part in ("", ".", "..") for part in key.split("/")):
        raise ValueError(f"Illegal storage key: {key!r}")
    return key


class OpenDALBackend(StorageBackend):
    """Store artifacts and job records through a single OpenDAL operator."""

    def __init__(
        self,
        operator: opendal.AsyncOperator,
        *,
        artifact_prefix: str = ARTIFACT_PREFIX,
        record_prefix: str = RECORD_PREFIX,
        max_concurrency: int = 4,
        local_root: str | None = None,
        default_ttl: int | None = None,
        sweep_interval: int | None = None,
    ) -> None:
        if default_ttl is not None and sweep_interval is None:
            raise ValueError("`sweep_interval` is required when `default_ttl` is set.")

        self._op = operator
        self._artifact_prefix = artifact_prefix.strip("/")
        self._record_prefix = record_prefix.strip("/")
        self._local_root = os.path.realpath(local_root) if local_root is not None else None
        self._default_ttl = default_ttl

        self._io_semaphore = asyncio.Semaphore(max(1, max_concurrency))
        self._record_lock = asyncio.Lock()
        self._capability = operator.capability()
        self._has_last_modified: bool | None = None

        self._sweeper = (
            TtlSweeper(self, ttl_seconds=default_ttl, sweep_interval_seconds=sweep_interval)
            if default_ttl is not None and sweep_interval is not None
            else None
        )

    # ---- key layout ---------------------------------------------------
    def artifact_path(self, key: str) -> str:
        return f"{self._artifact_prefix}/{_validate_key(key)}"

    def record_path(self, key: str) -> str:
        return f"{self._record_prefix}/{_validate_key(key)}{RECORD_SUFFIX}"

    @property
    def storage_path(self) -> str | None:
        """Root directory for a filesystem backend, None otherwise."""
        return self._local_root

    @storage_path.setter
    def storage_path(self, path: str) -> None:
        """Point a filesystem backend at another root (tests, deployment tools)."""
        root = os.path.realpath(path)
        atomic_write_dir = os.path.join(root, ".tmp")
        os.makedirs(atomic_write_dir, exist_ok=True)
        self._op = opendal.AsyncOperator("fs", root=root, atomic_write_dir=atomic_write_dir)
        self._capability = self._op.capability()
        self._local_root = root
        self._has_last_modified = None

    def get_full_file_path(self, key: str) -> str:
        """Absolute path of an artifact; only valid for filesystem backends."""
        if self._local_root is None:
            raise ValueError("Storage backend does not expose filesystem paths")
        return os.path.join(self._local_root, self._artifact_prefix, _validate_key(key))

    # ---- artifact -----------------------------------------------------
    async def save(self, key: str, data: bytes, *, ttl: int | None = None) -> SaveContext:
        created_at = int(time.time())
        effective_ttl = self._default_ttl if ttl is None else ttl
        expires_at = created_at + effective_ttl if effective_ttl else None

        async with self._io_semaphore:
            await self._op.write(self.artifact_path(key), data)
        if expires_at is not None:
            await self._merge_record_expiry(key, expires_at)
        return SaveContext(key=key, created_at=created_at, expires_at=expires_at)

    async def _merge_record_expiry(self, key: str, expires_at: int) -> None:
        """Record ``expires_at`` without clobbering the rest of the job record."""
        async with self._record_lock:
            record: dict = {}
            existing = await self._read_record(key)
            if existing is not None:
                try:
                    record = json.loads(existing)
                except ValueError:
                    logger.warning("Replacing unreadable job record for %s", key)
            if not isinstance(record, dict):
                record = {}
            record["expires_at"] = expires_at
            await self._write_record(key, json.dumps(record).encode())

    async def open(self, key: str) -> FileStorageHandle | None:
        path = self.artifact_path(key)
        try:
            metadata = await self._op.stat(path)
        except opendal.exceptions.NotFound:
            return None
        if not metadata.is_file:
            return None
        if self._local_root is None:
            # Non-filesystem backends cannot hand out a path; they must stream.
            return None
        return FileStorageHandle(path=self.get_full_file_path(key))

    async def iter_chunks(self, key: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> AsyncIterator[bytes]:
        reader = await self._op.open(self.artifact_path(key), "rb")
        try:
            while True:
                chunk = await reader.read(chunk_size)
                if not chunk:
                    return
                yield chunk
        finally:
            with contextlib.suppress(Exception):
                await reader.close()

    async def delete(self, key: str) -> bool:
        async with self._io_semaphore:
            removed = await self._remove_artifact(key)
            await self._remove(self.record_path(key))
        return removed

    async def delete_artifact(self, key: str) -> bool:
        async with self._io_semaphore:
            return await self._remove_artifact(key)

    async def _remove_artifact(self, key: str) -> bool:
        return await self._remove(self.artifact_path(key))

    async def _remove(self, path: str) -> bool:
        """Delete one object; returns whether it existed.

        Only "not found" is swallowed: an unreachable or unauthorised backend
        must surface instead of silently leaking objects.
        """
        try:
            existed = await self._op.exists(path)
            await self._op.delete(path)
        except opendal.exceptions.NotFound:
            return False
        return bool(existed)

    # ---- record -------------------------------------------------------
    async def record_get(self, key: str) -> bytes | None:
        return await self._read_record(key)

    async def record_set(self, key: str, data: bytes) -> None:
        async with self._record_lock:
            await self._write_record(key, data)

    async def record_pop(self, key: str) -> bytes | None:
        async with self._record_lock:
            data = await self._read_record(key)
            if data is None:
                return None
            await self._remove(self.record_path(key))
            return data

    async def _read_record(self, key: str) -> bytes | None:
        try:
            return await self._op.read(self.record_path(key))
        except opendal.exceptions.NotFound:
            return None

    async def _write_record(self, key: str, data: bytes) -> None:
        await self._op.write(self.record_path(key), data)

    async def record_keys(self) -> list[str]:
        prefix = f"{self._record_prefix}/"
        keys: list[str] = []
        async for path, _ in self._iter_files(self._record_prefix):
            if path.startswith(prefix) and path.endswith(RECORD_SUFFIX):
                keys.append(path[len(prefix) : -len(RECORD_SUFFIX)])
        return sorted(keys)

    # ---- enumeration --------------------------------------------------
    @property
    def supports_list(self) -> bool:
        return bool(self._capability.list)

    async def list_artifacts(self) -> AsyncIterator[tuple[str, float | None]]:
        if not self.supports_list:
            return
        prefix = f"{self._artifact_prefix}/"
        async for path, metadata in self._iter_files(self._artifact_prefix):
            if not path.startswith(prefix):
                continue
            key = path[len(prefix) :]
            if key.startswith("."):
                # Internal probe objects; never swept.
                continue
            last_modified = metadata.last_modified
            yield key, last_modified.timestamp() if last_modified is not None else None

    async def _iter_files(self, prefix: str) -> AsyncIterator[tuple[str, Metadata]]:
        """Yield ``(path, metadata)`` for every file under ``prefix``."""
        lister = await self._op.list(f"{prefix}/", recursive=True)
        async for entry in lister:
            if entry.metadata.is_file:
                yield entry.path, entry.metadata

    async def probe_last_modified(self) -> bool:
        if self._has_last_modified is not None:
            return self._has_last_modified
        if not self.supports_list:
            self._has_last_modified = False
            return False

        probe_path = f"{self._artifact_prefix}/.last_modified_probe"
        try:
            await self._op.write(probe_path, b"probe")
            metadata = await self._op.stat(probe_path)
        except Exception:
            logger.warning("Could not probe last_modified support; falling back to record expiry", exc_info=True)
            self._has_last_modified = False
        else:
            self._has_last_modified = metadata.last_modified is not None
        finally:
            with contextlib.suppress(Exception):
                await self._op.delete(probe_path)
        return self._has_last_modified

    # ---- lifecycle ----------------------------------------------------
    @property
    def sweeper(self) -> TtlSweeper | None:
        """TTL sweeper, present when a TTL is configured."""
        return self._sweeper

    async def start(self, *, on_expire: Callable[[list[str]], Awaitable[None]] | None = None) -> None:
        if self._sweeper is not None:
            await self._sweeper.start(on_expire=on_expire)

    async def stop(self) -> None:
        if self._sweeper is not None:
            await self._sweeper.stop()
