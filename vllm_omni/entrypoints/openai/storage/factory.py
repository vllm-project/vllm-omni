# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Storage backend factory and the application-wide instance."""

import os

import opendal

from vllm_omni.config.server_settings import SERVER_SETTINGS_CONFIG, FileBackend
from vllm_omni.entrypoints.openai.storage.base import StorageBackend
from vllm_omni.entrypoints.openai.storage.opendal_backend import OpenDALBackend


def get_storage_manager(storage_config: FileBackend) -> StorageBackend:
    storage_path = os.path.realpath(storage_config.path)
    # Stage writes in a sibling directory and rename them into place, so a
    # crashed or interrupted save never leaves a truncated artifact behind.
    atomic_write_dir = os.path.join(storage_path, ".tmp")
    os.makedirs(atomic_write_dir, exist_ok=True)
    operator = opendal.AsyncOperator("fs", root=storage_path, atomic_write_dir=atomic_write_dir)

    return OpenDALBackend(
        operator,
        max_concurrency=storage_config.file_concurrency,
        local_root=storage_path,
        default_ttl=storage_config.file_ttl,
        sweep_interval=storage_config.ttl_sweep_interval,
    )


STORAGE_MANAGER = get_storage_manager(SERVER_SETTINGS_CONFIG.storage)

__all__ = ["get_storage_manager", "STORAGE_MANAGER"]
