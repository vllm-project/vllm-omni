# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
import time
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ValidationError
from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.protocol.videos import VideoError, VideoGenerationStatus, VideoResponse
from vllm_omni.entrypoints.openai.storage import STORAGE_MANAGER
from vllm_omni.entrypoints.openai.storage.base import StorageBackend
from vllm_omni.entrypoints.openai.video.paths import lidar_storage_key

logger = init_logger(__name__)

T = TypeVar("T", bound=BaseModel)
U = TypeVar("U")

ORPHANED_JOB_MESSAGE = "The server restarted before this job finished; please submit the request again."


class TaskRegistry:
    def __init__(self) -> None:
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> asyncio.Task[None] | None:
        async with self._lock:
            return self._tasks.get(key)

    async def pop(self, key: str) -> asyncio.Task[None] | None:
        async with self._lock:
            return self._tasks.pop(key, None)

    async def upsert(self, key: str, task: asyncio.Task[None]) -> None:
        def _cleanup(_: asyncio.Task[None]) -> None:
            asyncio.create_task(self.pop(key))

        task.add_done_callback(_cleanup)

        async with self._lock:
            self._tasks[key] = task


class AsyncDictStore(Generic[T]):
    """A small async-safe in-memory key-value store for dict items.

    This encapsulates the usual pattern of a module-level dict guarded by
    an asyncio.Lock and provides simple CRUD methods that are safe to call
    concurrently from FastAPI request handlers and background tasks.
    """

    def __init__(self) -> None:
        self._items: dict[str, T] = {}
        self._lock = asyncio.Lock()

    async def upsert(self, key: str, value: T) -> None:
        async with self._lock:
            self._items[key] = value

    async def update_fields(self, key: str, updates: dict[str, Any]) -> T | None:
        async with self._lock:
            item: T | None = self._items.get(key)
            if item is None:
                return None
            new_item = item.model_copy(update=updates)
            self._items[key] = new_item
            return new_item

    async def get(self, key: str) -> T | None:
        async with self._lock:
            return self._items.get(key)

    async def pop(self, key: str) -> T | None:
        async with self._lock:
            return self._items.pop(key, None)

    async def remove_many(self, keys: list[str]) -> None:
        async with self._lock:
            for key in keys:
                self._items.pop(key, None)

    async def list_values(self) -> list[T]:
        async with self._lock:
            return list(self._items.values())

    async def reconcile_orphans(self) -> list[str]:
        """Fail records that a previous process left unfinished.

        Stores that persist their items override this; the in-memory store has
        nothing that can outlive the process, so it is a no-op.
        """
        return []


class VideoStore(AsyncDictStore[VideoResponse]):
    """Video job records, mirrored into the storage backend's record layer.

    The in-memory copy is the read/write path for the running server; every
    mutation is also written through to ``meta/{video_id}.json`` so that job
    metadata (and therefore ``GET /v1/videos/{id}``) survives a restart.

    Persistence is best-effort: a failing record write must not break video
    generation, so errors are logged and the in-memory record stays
    authoritative for this process.
    """

    def __init__(self, backend: StorageBackend | None = None) -> None:
        super().__init__()
        self._backend = backend
        self._loaded = False

    async def upsert(self, key: str, value: VideoResponse) -> None:
        await super().upsert(key, value)
        await self._persist_best_effort(key, value)

    async def update_fields(
        self,
        key: str,
        updates: dict[str, Any],
        *,
        durable: bool = False,
    ) -> VideoResponse | None:
        """Apply ``updates``, optionally publishing them only once durable.

        ``durable=True`` is used for terminal states: the record is written
        first and the in-memory copy is published only if that write succeeds,
        so an acknowledged completion cannot be lost by a later restart.
        """
        if not durable:
            item = await super().update_fields(key, updates)
            if item is not None:
                await self._persist_best_effort(key, item)
            return item

        async with self._lock:
            current = self._items.get(key)
            if current is None:
                return None
            item = current.model_copy(update=updates)
            await self._persist(key, item)
            self._items[key] = item
            return item

    async def pop(self, key: str) -> VideoResponse | None:
        item = await super().pop(key)
        if self._backend is not None:
            try:
                await self._backend.record_pop(key)
            except Exception:
                logger.warning("Failed to drop job record for %s", key, exc_info=True)
        return item

    async def get(self, key: str) -> VideoResponse | None:
        item = await super().get(key)
        if item is not None or self._backend is None:
            return item
        return await self._load_one(key)

    async def load(self) -> int:
        """Warm the in-memory store from persisted job records."""
        if self._backend is None or self._loaded:
            return 0

        try:
            keys = await self._backend.record_keys()
        except Exception:
            logger.warning("Failed to enumerate persisted job records", exc_info=True)
            return 0
        self._loaded = True

        loaded = 0
        for key in keys:
            if await self._load_one(key) is not None:
                loaded += 1
        if loaded:
            logger.info("Recovered %d video job record(s) from storage", loaded)
        return loaded

    async def reconcile_orphans(self) -> list[str]:
        """Fail jobs that were interrupted by a restart.

        A ``queued``/``in_progress`` record can only linger after the owning
        process died: nothing will ever finish it, so it is marked ``failed``
        instead of looking stuck forever.
        """
        await self.load()
        orphans = [
            job
            for job in await self.list_values()
            if job.status in (VideoGenerationStatus.QUEUED, VideoGenerationStatus.IN_PROGRESS)
        ]
        for job in orphans:
            # The artifact of an interrupted job can only be partial; the
            # record survives as history, the payload does not.
            if self._backend is not None:
                for storage_key in (job.id, lidar_storage_key(job.id)):
                    try:
                        await self._backend.delete_artifact(storage_key)
                    except Exception:
                        logger.warning("Failed to remove interrupted artifact for %s", storage_key, exc_info=True)
            await self.update_fields(
                job.id,
                {
                    "status": VideoGenerationStatus.FAILED,
                    "completed_at": int(time.time()),
                    "error": VideoError(code=500, message=ORPHANED_JOB_MESSAGE),
                },
            )
        if orphans:
            logger.warning("Marked %d interrupted video job(s) as failed after restart", len(orphans))
        return [job.id for job in orphans]

    async def _load_one(self, key: str) -> VideoResponse | None:
        if self._backend is None:
            return None
        try:
            data = await self._backend.record_get(key)
        except Exception:
            logger.warning("Failed to read job record for %s", key, exc_info=True)
            return None
        if data is None:
            return None
        try:
            item = VideoResponse.model_validate_json(data)
        except ValidationError:
            logger.warning("Discarding unreadable job record for %s", key, exc_info=True)
            return None
        async with self._lock:
            self._items.setdefault(key, item)
        return item

    async def _persist(self, key: str, item: VideoResponse) -> None:
        """Write the record through; raises when the backend rejects the write."""
        if self._backend is None:
            return
        await self._backend.record_set(key, item.model_dump_json().encode())

    async def _persist_best_effort(self, key: str, item: VideoResponse) -> None:
        """Write the record through, logging instead of raising on failure."""
        try:
            await self._persist(key, item)
        except Exception:
            logger.warning("Failed to persist job record for %s", key, exc_info=True)


VIDEO_STORE: VideoStore = VideoStore(STORAGE_MANAGER)
VIDEO_TASKS: TaskRegistry = TaskRegistry()
