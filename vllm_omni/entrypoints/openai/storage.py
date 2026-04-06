import asyncio
import contextlib
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from tempfile import NamedTemporaryFile
from typing import Generic, Literal, TypeVar

from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.config import CONFIG, STORAGE_BACKENDS, FileBackend

logger = init_logger(__name__)


K = TypeVar("K", bound=str, covariant=True)


@dataclass
class SaveContext:
    key: str
    created_at: int
    expires_at: int | None = None


@dataclass
class BaseStorageHandle(Generic[K]):
    kind: K


@dataclass
class FileStorageHandle(BaseStorageHandle[Literal["path"]]):
    path: str
    kind: Literal["path"] = "path"


class StorageBaseManager(ABC):
    @abstractmethod
    async def save(self, *args, **kwargs) -> SaveContext:
        pass

    @abstractmethod
    async def delete(self, *args, **kwargs) -> bool:
        pass

    def start(self, *args, **kwargs):
        pass

    @abstractmethod
    async def open(self, storage_key: str) -> BaseStorageHandle | None:
        pass


class LocalStorageManager(StorageBaseManager):
    def __init__(self, storage_path: str, max_concurrency: int = 4):
        self.storage_path = storage_path
        os.makedirs(self.storage_path, exist_ok=True)

        self._io_semaphore = asyncio.Semaphore(max(1, max_concurrency))

    async def open(self, storage_key: str) -> FileStorageHandle | None:
        local_file = self.get_full_file_path(storage_key)
        if not os.path.exists(local_file):
            return None
        return FileStorageHandle(path=local_file)

    def _save_sync(self, data: bytes, file_name: str) -> SaveContext:
        filename = self.get_full_file_path(file_name)
        tmp_name: str | None = None

        response = SaveContext(key=file_name, created_at=int(time.time()))
        try:
            with NamedTemporaryFile("wb", dir=self.storage_path, delete=False) as f:
                tmp_name = f.name
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_name, filename)
            return response
        except Exception:
            if tmp_name is not None:
                try:
                    os.remove(tmp_name)
                except OSError:
                    pass
            raise

    async def save(self, data: bytes, file_name: str) -> SaveContext:
        async with self._io_semaphore:
            return await asyncio.to_thread(self._save_sync, data, file_name)

    def _delete_sync(self, file_name: str) -> bool:
        try:
            os.remove(self.get_full_file_path(file_name))
        except FileNotFoundError:
            return False
        return True

    async def delete(self, file_name: str) -> bool:
        async with self._io_semaphore:
            return await asyncio.to_thread(self._delete_sync, file_name)

    def exists(self, file_name: str) -> bool:
        return os.path.exists(self.get_full_file_path(file_name))

    def get_full_file_path(self, file_name: str) -> str:
        return os.path.join(self.storage_path, file_name)


class LocalStorageTTLManager(LocalStorageManager):
    def __init__(self, ttl_seconds: int, sweep_interval_seconds: int, *args, **kwargs):
        if ttl_seconds <= 0:
            raise ValueError("`ttl_seconds` must be greater than or equal to 1.")
        if sweep_interval_seconds <= 0:
            raise ValueError("`sweep_interval_seconds` must be greater than or equal to 1.")

        self._ttl_seconds = ttl_seconds
        self._sweep_interval_seconds = sweep_interval_seconds
        self._sweeper_task: asyncio.Task[None] | None = None

        super().__init__(*args, **kwargs)

    async def save(self, data: bytes, file_name: str) -> SaveContext:
        result = await super().save(data, file_name)
        result.expires_at = result.created_at + self._ttl_seconds
        return result

    async def _sweep_loop(self) -> None:
        def _sweep_once(cutoff: float) -> int:
            deleted = 0
            for entry in os.scandir(self.storage_path):
                if not entry.is_file(follow_symlinks=False):
                    continue
                try:
                    if entry.stat(follow_symlinks=False).st_mtime < cutoff:
                        os.remove(entry.path)
                        deleted += 1
                except FileNotFoundError:
                    pass
                except OSError:
                    logger.warning("TTL sweep failed to delete expired file %s", entry.path, exc_info=True)
            return deleted

        while True:
            try:
                cutoff = time.time() - self._ttl_seconds
                async with self._io_semaphore:
                    await asyncio.to_thread(_sweep_once, cutoff)
            except Exception:
                logger.exception("TTL sweep failed for storage path %s", self.storage_path)
            await asyncio.sleep(self._sweep_interval_seconds)

    def start(self) -> None:
        if self._sweeper_task is None or self._sweeper_task.done():
            self._sweeper_task = asyncio.create_task(self._sweep_loop())

    async def stop(self) -> None:
        if self._sweeper_task is None:
            return
        self._sweeper_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._sweeper_task
        self._sweeper_task = None


def get_storage_manager(storage_config: STORAGE_BACKENDS) -> StorageBaseManager:
    if isinstance(storage_config, FileBackend):
        if storage_config.file_ttl is not None and storage_config.ttl_sweep_interval is not None:
            manager = LocalStorageTTLManager(
                storage_path=storage_config.storage_path,
                max_concurrency=storage_config.file_concurrency,
                ttl_seconds=storage_config.file_ttl,
                sweep_interval_seconds=storage_config.ttl_sweep_interval,
            )
        else:
            manager = LocalStorageManager(
                storage_path=storage_config.storage_path, max_concurrency=storage_config.file_concurrency
            )
    else:
        raise ValueError("No supported storage managers")

    return manager


STORAGE_MANAGER = get_storage_manager(CONFIG.storage)
