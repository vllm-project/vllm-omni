"""Pluggable storage backend for generated artifacts (e.g. video files).

Public surface: the ``StorageBackend`` contract, local filesystem
implementations, and the shared ``STORAGE_MANAGER`` instance.
"""

# Re-exported for ``vllm_omni.entrypoints.openai.storage`` callers that
# patch ``storage_module.time.time`` in tests.
import time  # noqa: F401

from vllm_omni.entrypoints.openai.storage.base import (
    BaseStorageHandle,
    FileStorageHandle,
    SaveContext,
    StorageBackend,
)
from vllm_omni.entrypoints.openai.storage.factory import STORAGE_MANAGER, get_storage_manager
from vllm_omni.entrypoints.openai.storage.local import (
    LocalStorageManager,
    LocalStorageTTLManager,
)

__all__ = [
    # base
    "BaseStorageHandle",
    "FileStorageHandle",
    "SaveContext",
    "StorageBackend",
    # local implementations
    "LocalStorageManager",
    "LocalStorageTTLManager",
    # factory / instance
    "get_storage_manager",
    "STORAGE_MANAGER",
]

# ``time`` stays out of ``__all__``: it is exported for the test patch
# mentioned above, not part of the public API.
