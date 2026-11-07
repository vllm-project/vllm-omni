"""Base abstractions for the storage backend layer."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

from vllm.logger import init_logger

logger = init_logger(__name__)


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
    path: str
    kind: Literal["path"] = field(default="path", init=False)


class StorageBackend(ABC):
    """Common contract for storing generated artifacts (e.g. video files).

    Concrete backends may expose extra convenience helpers beyond the
    contract (e.g. ``LocalStorageManager.exists``).
    """

    @abstractmethod
    async def save(self, data: bytes, file_name: str) -> SaveContext:
        """Persist ``data`` under ``file_name`` (relative to the backend root)."""

    @abstractmethod
    async def delete(self, file_name: str) -> bool:
        """Remove the object stored under ``file_name``. Returns True if removed."""

    @abstractmethod
    async def open(self, storage_key: str) -> BaseStorageHandle | None:
        """Return a handle usable to read back the object, or None if missing."""

    async def start(self) -> None:
        """Optional resource startup (e.g. background TTL sweeper tasks)."""

    async def stop(self) -> None:
        """Optional resource shutdown."""
