import asyncio
import os
from abc import ABC, abstractmethod
from tempfile import NamedTemporaryFile


class StorageBaseManager(ABC):
    @abstractmethod
    async def save(self, *args, **kwargs) -> str:
        pass

    @abstractmethod
    async def delete(self, *args, **kwargs) -> bool:
        pass


class LocalStorageManager(StorageBaseManager):
    def __init__(self, storage_path: str | None = None):
        if storage_path is None:
            storage_path = os.getenv("VLLM_OMNI_STORAGE_PATH", "/tmp/storage")
        self.storage_path = storage_path
        os.makedirs(self.storage_path, exist_ok=True)

        max_concurrency = int(os.getenv("VLLM_OMNI_STORAGE_MAX_CONCURRENCY", "4"))
        self._io_semaphore = asyncio.Semaphore(max(1, max_concurrency))

    def _save_sync(self, data: bytes, file_name: str) -> str:
        filename = self.get_full_file_path(file_name)
        tmp_name: str | None = None
        try:
            with NamedTemporaryFile("wb", dir=self.storage_path, delete=False) as f:
                tmp_name = f.name
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_name, filename)
            return filename
        except Exception:
            if tmp_name is not None:
                try:
                    os.remove(tmp_name)
                except OSError:
                    pass
            raise

    async def save(self, data: bytes, file_name: str) -> str:
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


# Should implement TTL management on local file storage
# and recovery on restart in the *STORE's
STORAGE_MANAGER = LocalStorageManager()
