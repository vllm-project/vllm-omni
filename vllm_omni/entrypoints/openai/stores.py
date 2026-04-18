import asyncio
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from vllm_omni.entrypoints.openai.protocol.videos import VideoResponse

T = TypeVar("T", bound=BaseModel)
U = TypeVar("U")


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

    async def list_values(self) -> list[T]:
        async with self._lock:
            return list(self._items.values())


class VideoMetadataStore(ABC):
    """Pluggable persistence interface for VideoResponse records."""

    @abstractmethod
    async def upsert(self, key: str, value: VideoResponse) -> None: ...

    @abstractmethod
    async def get(self, key: str) -> VideoResponse | None: ...

    @abstractmethod
    async def pop(self, key: str) -> VideoResponse | None: ...

    @abstractmethod
    async def update_fields(self, key: str, updates: dict[str, Any]) -> VideoResponse | None: ...

    @abstractmethod
    async def list_values(self) -> list[VideoResponse]: ...

    def close(self) -> None:
        """Optional teardown hook. Override if the backend holds resources."""


class InMemoryVideoMetadataStore(AsyncDictStore[VideoResponse], VideoMetadataStore):
    """In-memory VideoMetadataStore backed by AsyncDictStore."""


class DiskCacheVideoMetadataStore(VideoMetadataStore):
    """Persistent backend backed by diskcache.Cache."""

    def __init__(self, directory: str) -> None:
        import diskcache

        self._cache = diskcache.Cache(directory)

    async def upsert(self, key: str, value: VideoResponse) -> None:
        await asyncio.to_thread(self._cache.set, key, value.model_dump_json())

    async def get(self, key: str) -> VideoResponse | None:
        raw = await asyncio.to_thread(self._cache.get, key)
        return VideoResponse.model_validate_json(raw) if raw is not None else None

    async def pop(self, key: str) -> VideoResponse | None:
        def _pop() -> str | None:
            with self._cache.transact():
                raw = self._cache.get(key)
                if raw is None:
                    return None
                del self._cache[key]
                return raw

        raw = await asyncio.to_thread(_pop)
        return VideoResponse.model_validate_json(raw) if raw is not None else None

    async def update_fields(self, key: str, updates: dict[str, Any]) -> VideoResponse | None:
        def _update() -> VideoResponse | None:
            with self._cache.transact():
                raw = self._cache.get(key)
                if raw is None:
                    return None
                new = VideoResponse.model_validate_json(raw).model_copy(update=updates)
                self._cache.set(key, new.model_dump_json())
                return new

        return await asyncio.to_thread(_update)

    async def list_values(self) -> list[VideoResponse]:
        def _list() -> list[VideoResponse]:
            result = []
            for k in self._cache:
                raw = self._cache.get(k)  # returns None if key was concurrently deleted
                if raw is not None:
                    result.append(VideoResponse.model_validate_json(raw))
            return result

        return await asyncio.to_thread(_list)

    def close(self) -> None:
        self._cache.close()


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


VIDEO_STORE: VideoMetadataStore = InMemoryVideoMetadataStore()
VIDEO_TASKS: TaskRegistry = TaskRegistry()


def init_video_store(backend: str, directory: str = None) -> None:
    if directory is None and backend == "diskcache":
        raise ValueError("when backend is diskcache, directory cannot be empty.")
    if backend == "diskcache":
        VIDEO_STORE._swap(DiskCacheVideoMetadataStore(directory=directory))
    elif backend == "memory":
        VIDEO_STORE._swap(InMemoryVideoMetadataStore())
    else:
        raise ValueError(f"Unknown video-metadata-store backend: {backend!r}. Choose 'memory' or 'diskcache'.")
