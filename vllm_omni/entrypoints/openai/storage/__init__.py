# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Pluggable storage backend for generated artifacts and job records.

Public surface: the ``StorageBackend`` contract, the OpenDAL-backed
implementations (local filesystem and S3-compatible object stores), the TTL
sweeper, and the shared ``STORAGE_MANAGER`` instance.
"""

from vllm_omni.entrypoints.openai.storage.base import (
    BaseStorageHandle,
    FileStorageHandle,
    SaveContext,
    StorageBackend,
    StreamStorageHandle,
    UrlStorageHandle,
)
from vllm_omni.entrypoints.openai.storage.factory import STORAGE_MANAGER, get_storage_manager
from vllm_omni.entrypoints.openai.storage.opendal_backend import OpenDALBackend
from vllm_omni.entrypoints.openai.storage.ttl import SweepStrategy, TtlSweeper

__all__ = [
    # base
    "BaseStorageHandle",
    "FileStorageHandle",
    "SaveContext",
    "StorageBackend",
    "StreamStorageHandle",
    "UrlStorageHandle",
    # implementations
    "OpenDALBackend",
    "SweepStrategy",
    "TtlSweeper",
    # factory / instance
    "get_storage_manager",
    "STORAGE_MANAGER",
]
