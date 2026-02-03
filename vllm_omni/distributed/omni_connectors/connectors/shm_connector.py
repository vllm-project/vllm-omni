# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import fcntl
import os
import time
from collections import defaultdict
from typing import Any

from vllm_omni.entrypoints.stage_utils import shm_read_bytes, shm_write_bytes

from ..utils.logging import get_connector_logger
from .base import OmniConnectorBase

logger = get_connector_logger(__name__)


class SharedMemoryConnector(OmniConnectorBase):
    """
    Connector that uses SharedMemory for large objects and inline data for small objects.
    Acts as a unified replacement for the legacy IPC fallback logic.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.stage_id = config.get("stage_id", -1)
        self.device = config.get("device", "cuda:0")
        self.put_requests: dict[str, int] = defaultdict(int)
        self.get_requests: dict[str, int] = defaultdict(int)
        self.finished_requests: set[str] = set()
        self.request_payload = {}
        self.request_prompt_token_ids: dict[str, list[int]] = defaultdict(list)
        self.code_prompt_token_ids: dict[str, list[list[int]]] = defaultdict(list)
        self.request_ids_mapping: dict[str, str] = {}
        # Default threshold matches legacy behavior (64KB)
        self.threshold = int(config.get("shm_threshold_bytes", 65536))
        self._metrics = {
            "puts": 0,
            "gets": 0,
            "bytes_transferred": 0,
            "shm_writes": 0,
            "inline_writes": 0,
        }

    def put(self, from_stage: str, to_stage: str, put_key: str, data: Any) -> tuple[bool, int, dict[str, Any] | None]:
        try:
            # Always serialize first to check size (and for SHM writing)
            # Note: For extremely large objects in "inline" mode (e.g. Ray),
            # we might double-serialize if we're not careful, but here we assume
            # if it's huge we use SHM, or if Ray, threshold is maxsize.
            payload = self.serialize_obj(data)
            size = len(payload)

            metadata = {}
            # if size > self.threshold:
            if True:  # TODO: correct put & get logic
                # Use Shared Memory
                lock_file = f"/dev/shm/shm_{put_key}_lockfile.lock"
                with open(lock_file, "w") as lockf:
                    fcntl.flock(lockf, fcntl.LOCK_EX)
                    meta = shm_write_bytes(payload, name=put_key)
                    fcntl.flock(lockf, fcntl.LOCK_UN)

                # meta contains {'name': ..., 'size': ...}
                metadata[put_key] = {"shm": meta, "size": size}
                self._metrics["shm_writes"] += 1
            else:
                # Inline - pass bytes directly to avoid double serialization of the object
                # We already serialized it to check size, so we pass the bytes.
                # The Queue will pickle these bytes (fast), avoiding re-serializing the complex object.
                metadata[put_key] = {"inline_bytes": payload, "size": size}
                self._metrics["inline_writes"] += 1

            self._metrics["puts"] += 1
            self._metrics["bytes_transferred"] += size

            return True, size, metadata

        except Exception as e:
            logger.error(f"SharedMemoryConnector put failed for req {put_key}: {e}")
            return False, 0, None

    def get(self, from_stage: str, to_stage: str, get_key: str, metadata=None) -> tuple[Any, int] | None:
        from multiprocessing import shared_memory as shm_pkg

        # Wait for shared memory to be available (with retry logic)
        max_retries = 30
        retry_delay = 0.1  # 100ms between retries
        shm = None

        for attempt in range(max_retries):
            try:
                shm = shm_pkg.SharedMemory(name=get_key)
                break  # Successfully opened, exit retry loop
            except FileNotFoundError:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    # Max retries reached, return None
                    logger.warning(f"Shared memory '{get_key}' not found after {max_retries} retries")
                    return None, 0

        if shm is None:
            return None, 0

        try:
            lock_file = f"/dev/shm/shm_{get_key}_lockfile.lock"
            with open(lock_file) as lockf:
                fcntl.flock(lockf, fcntl.LOCK_SH)
                data_bytes = shm_read_bytes({"name": get_key, "size": shm.size})
                fcntl.flock(lockf, fcntl.LOCK_UN)
            # Clean up the temporary file if it still exists.
            if os.path.exists(lock_file):
                os.remove(lock_file)
            obj = self.deserialize_obj(data_bytes)
            return obj, shm.size
        finally:
            shm.close()

        # TODO: update another read method

    def cleanup(self, request_id: str) -> None:
        # SHM segments are automatically unlinked during 'get' (shm_read_bytes).
        # If 'get' is never called (e.g. error flow), the SHM segment might leak.
        # A robust implementation might track created segments and unlink them here
        # if they haven't been consumed.
        # For now, we rely on the consumer to read and unlink.
        pass

    def health(self) -> dict[str, Any]:
        return {"status": "healthy", "threshold": self.threshold, **self._metrics}
