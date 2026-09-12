"""Storage backend factory and the application-wide instance."""

from vllm_omni.config.server_settings import SERVER_SETTINGS_CONFIG, FileBackend
from vllm_omni.entrypoints.openai.storage.base import StorageBackend
from vllm_omni.entrypoints.openai.storage.local import (
    LocalStorageManager,
    LocalStorageTTLManager,
)


def get_storage_manager(storage_config: FileBackend) -> StorageBackend:
    if isinstance(storage_config, FileBackend):
        if storage_config.file_ttl is not None and storage_config.ttl_sweep_interval is not None:
            manager: StorageBackend = LocalStorageTTLManager(
                storage_path=storage_config.path,
                max_concurrency=storage_config.file_concurrency,
                ttl_seconds=storage_config.file_ttl,
                sweep_interval_seconds=storage_config.ttl_sweep_interval,
            )
        else:
            manager = LocalStorageManager(
                storage_path=storage_config.path, max_concurrency=storage_config.file_concurrency
            )
    else:
        raise ValueError("No supported storage managers")

    return manager


STORAGE_MANAGER = get_storage_manager(SERVER_SETTINGS_CONFIG.storage)

__all__ = ["get_storage_manager", "STORAGE_MANAGER"]
