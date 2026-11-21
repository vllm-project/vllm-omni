# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Base abstractions for the storage backend layer."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Literal

DEFAULT_CHUNK_SIZE = 1 << 20


@dataclass
class SaveContext:
    key: str
    created_at: int
    expires_at: int | None = None


@dataclass(frozen=True)
class BaseStorageHandle:
    kind: str


@dataclass(frozen=True)
class FileStorageHandle(BaseStorageHandle):
    """The artifact is readable through a local filesystem path."""

    path: str
    kind: Literal["path"] = field(default="path", init=False)


class StorageBackend(ABC):
    """Common contract for storing generated artifacts and their job records.

    Two object families live under one backend root: the artifact payload
    (large, opaque bytes) and the job record (small JSON document, written by
    the caller). Backends only move bytes; serialization stays in the caller.

    Concrete backends may expose extra convenience helpers beyond the
    contract (e.g. ``OpenDALBackend.get_full_file_path``).
    """

    # ---- lifecycle ----------------------------------------------------
    async def start(self, *, on_expire: Callable[[list[str]], Awaitable[None]] | None = None) -> None:
        """Optional resource startup (e.g. background TTL sweeper tasks).

        ``on_expire`` receives the keys of artifacts/records the backend
        expired on its own, so callers can drop cached metadata for them.
        """

    async def stop(self) -> None:
        """Optional resource shutdown."""

    # ---- artifact -----------------------------------------------------
    @abstractmethod
    async def save(self, key: str, data: bytes, *, ttl: int | None = None) -> SaveContext:
        """Persist ``data`` under ``key``.

        ``ttl`` overrides the backend's configured lifetime for this object;
        ``None`` falls back to the backend default. Per-object values are only
        observed by the record-based TTL sweep (see ``TtlSweeper``).
        """

    @abstractmethod
    async def open(self, key: str) -> BaseStorageHandle | None:
        """Return a handle usable to read back the artifact, or None if missing."""

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Remove the artifact and its record. Returns True if the artifact existed."""

    @abstractmethod
    async def delete_artifact(self, key: str) -> bool:
        """Remove only the artifact, keeping the job record.

        Used when the artifact is known to be unusable (e.g. a generation that
        was interrupted by a restart) but the record must survive as history.
        """

    async def iter_chunks(self, key: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> AsyncIterator[bytes]:
        """Stream the artifact bytes.

        Backends whose artifacts cannot be served from a local path must
        implement this for ``/v1/videos/{id}/content``.
        """
        raise NotImplementedError(f"{type(self).__name__} cannot stream artifacts")
        yield b""  # pragma: no cover - keeps this an async generator

    # ---- record (raw bytes; serialization lives in the caller) --------
    async def record_get(self, key: str) -> bytes | None:
        """Return the stored record bytes for ``key``, or None if absent."""
        raise NotImplementedError(f"{type(self).__name__} does not support records")

    async def record_set(self, key: str, data: bytes) -> None:
        """Create or replace the record for ``key``."""
        raise NotImplementedError(f"{type(self).__name__} does not support records")

    async def record_create(self, key: str, data: bytes) -> bool:
        """Create the record only if it does not exist yet.

        Returns True when this call created it. Backends without a conditional
        write fall back to a check-then-write, which is not atomic.
        """
        if await self.record_get(key) is not None:
            return False
        await self.record_set(key, data)
        return True

    async def record_pop(self, key: str) -> bytes | None:
        """Remove and return the record for ``key``, or None if absent."""
        raise NotImplementedError(f"{type(self).__name__} does not support records")

    async def record_keys(self) -> list[str]:
        """List the keys that currently have a record.

        Requires the backend to enumerate objects; backends without that
        capability return an empty list (see ``supports_list``).
        """
        return []

    # ---- enumeration (used by the TTL sweeper) ------------------------
    @property
    def supports_list(self) -> bool:
        """Whether the backend can enumerate artifacts and records."""
        return False

    async def list_artifacts(self) -> AsyncIterator[tuple[str, float | None]]:
        """Yield ``(key, last_modified)`` for every stored artifact.

        ``last_modified`` is a Unix timestamp, or None when the backend does
        not expose it (memory, redis, ...).
        """
        return
        yield "", None  # pragma: no cover - keeps this an async generator

    async def probe_last_modified(self) -> bool:
        """Whether the backend exposes ``last_modified`` for artifacts."""
        return False
