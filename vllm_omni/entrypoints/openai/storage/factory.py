# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Storage backend factory and the application-wide instance."""

import os

import opendal

from vllm_omni.config.server_settings import SERVER_SETTINGS_CONFIG, StorageConfig
from vllm_omni.entrypoints.openai.storage.base import StorageBackend
from vllm_omni.entrypoints.openai.storage.opendal_backend import OpenDALBackend


def _build_file_operator(storage_config: StorageConfig) -> tuple[opendal.AsyncOperator, str]:
    storage_path = os.path.realpath(storage_config.path)
    # Stage writes in a sibling directory and rename them into place, so a
    # crashed or interrupted save never leaves a truncated artifact behind.
    atomic_write_dir = os.path.join(storage_path, ".tmp")
    os.makedirs(atomic_write_dir, exist_ok=True)
    return opendal.AsyncOperator("fs", root=storage_path, atomic_write_dir=atomic_write_dir), storage_path


def _build_s3_operator(storage_config: StorageConfig) -> opendal.AsyncOperator:
    bucket = storage_config.bucket
    if not bucket:
        raise ValueError("`storage.bucket` is required when `storage.type` is 's3'.")

    kwargs: dict[str, str] = {"bucket": bucket, "root": storage_config.prefix}
    if storage_config.region:
        kwargs["region"] = storage_config.region
    if storage_config.endpoint_url:
        kwargs["endpoint"] = storage_config.endpoint_url
    # Credentials are never part of the server config: OpenDAL resolves them
    # through the standard chain (env vars, ~/.aws/credentials, IAM role).
    return opendal.AsyncOperator("s3", **kwargs)


def get_storage_manager(storage_config: StorageConfig) -> StorageBackend:
    if storage_config.type == "file":
        operator, local_root = _build_file_operator(storage_config)
    elif storage_config.type == "s3":
        operator, local_root = _build_s3_operator(storage_config), None
    else:  # pragma: no cover - guarded by StorageConfig validation
        raise ValueError(f"Unsupported storage backend: {storage_config.type!r}")

    return OpenDALBackend(
        operator,
        max_concurrency=storage_config.file_concurrency,
        local_root=local_root,
        default_ttl=storage_config.file_ttl,
        sweep_interval=storage_config.ttl_sweep_interval,
        presign=storage_config.type == "s3" and storage_config.presign,
        presign_expires=storage_config.presign_expires,
        proxy=storage_config.proxy,
    )


STORAGE_MANAGER = get_storage_manager(SERVER_SETTINGS_CONFIG.storage)

__all__ = ["get_storage_manager", "STORAGE_MANAGER"]
