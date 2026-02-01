# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
        # Default threshold matches legacy behavior (64KB)
        self.threshold = int(config.get("shm_threshold_bytes", 65536))
        self._metrics = {
            "puts": 0,
            "gets": 0,
            "bytes_transferred": 0,
            "shm_writes": 0,
            "inline_writes": 0,
        }

    def put(
        self, from_stage: str, to_stage: str, request_id: str, data: Any
    ) -> tuple[bool, int, dict[str, Any] | None]:
        try:
            # Always serialize first to check size (and for SHM writing)
            # Note: For extremely large objects in "inline" mode (e.g. Ray),
            # we might double-serialize if we're not careful, but here we assume
            # if it's huge we use SHM, or if Ray, threshold is maxsize.
            payload = self.serialize_obj(data)
            size = len(payload)

            if size > self.threshold:
                # Use Shared Memory
                meta = shm_write_bytes(payload)
                # meta contains {'name': ..., 'size': ...}
                metadata = {"shm": meta, "size": size}
                self._metrics["shm_writes"] += 1
            else:
                # Inline - pass bytes directly to avoid double serialization of the object
                # We already serialized it to check size, so we pass the bytes.
                # The Queue will pickle these bytes (fast), avoiding re-serializing the complex object.
                metadata = {"inline_bytes": payload, "size": size}
                self._metrics["inline_writes"] += 1

            self._metrics["puts"] += 1
            self._metrics["bytes_transferred"] += size

            return True, size, metadata

        except Exception as e:
            logger.error(f"SharedMemoryConnector put failed for req {request_id}: {e}")
            return False, 0, None

    # Robust shared memory protocol constants
    # Layout: [MAGIC(4)] [SIZE(8)] [CRC32(4)] [PAYLOAD(N)]
    # Total header: 16 bytes
    _SHM_MAGIC = b"VLLM"  # 4-byte magic marker
    _SHM_HEADER_SIZE = 16  # 4 (magic) + 8 (size) + 4 (crc32)

    def put_chunk(
        self, from_stage: str, to_stage: str, put_key: str, data: Any
    ) -> tuple[bool, int, dict[str, Any] | None]:
        """Write chunk data to shared memory with integrity checks.

        Protocol:
        - Layout: [MAGIC(4)][SIZE(8)][CRC32(4)][PAYLOAD(N)]
        - Write order: clear header, write payload, then write header last (atomicity)
        - Magic marker ensures reader knows data is complete
        - CRC32 checksum ensures data integrity
        """
        import struct
        import zlib
        from multiprocessing import shared_memory as shm_pkg

        try:
            # Serialize data
            payload = self.serialize_obj(data)
            payload_size = len(payload)

            # Calculate CRC32 checksum of payload
            crc32 = zlib.crc32(payload) & 0xFFFFFFFF

            # Total size: header + payload
            total_size = self._SHM_HEADER_SIZE + payload_size

            # Create shared memory
            shm = shm_pkg.SharedMemory(create=True, size=total_size, name=put_key)

            try:
                # Step 1: Clear header to indicate "not ready"
                shm.buf[: self._SHM_HEADER_SIZE] = b"\x00" * self._SHM_HEADER_SIZE

                # Step 2: Write payload FIRST
                shm.buf[self._SHM_HEADER_SIZE : self._SHM_HEADER_SIZE + payload_size] = payload

                # Step 3: Write header LAST (signals data is ready)
                # Format: magic(4) + size(8, little-endian) + crc32(4, little-endian)
                header = self._SHM_MAGIC + struct.pack("<Q", payload_size) + struct.pack("<I", crc32)
                shm.buf[: self._SHM_HEADER_SIZE] = header

            finally:
                shm.close()

            metadata = {put_key: {"shm": {"name": put_key, "size": payload_size}, "size": payload_size}}
            self._metrics["shm_writes"] += 1
            self._metrics["puts"] += 1
            self._metrics["bytes_transferred"] += payload_size

            return True, payload_size, metadata

        except Exception as e:
            logger.error(f"put_chunk failed for key {put_key}: {e}")
            return False, 0, None

    def get(
        self, from_stage: str, to_stage: str, request_id: str, metadata: dict[str, Any] | None = None
    ) -> tuple[Any, int] | None:
        if not metadata:
            logger.error(f"SharedMemoryConnector get called without metadata for req {request_id}")
            return None

        try:
            obj = None
            size = 0

            if "shm" in metadata:
                meta = metadata["shm"]
                # shm_read_bytes handles reading and unlinking
                data_bytes = shm_read_bytes(meta)
                obj = self.deserialize_obj(data_bytes)
                size = metadata.get("size", len(data_bytes))
            elif "inline_bytes" in metadata:
                # Deserialize bytes back to object
                payload = metadata["inline_bytes"]
                obj = self.deserialize_obj(payload)
                size = metadata.get("size", len(payload))
            elif "inline" in metadata:
                obj = metadata["inline"]
                size = metadata.get("size", 0)
                if size == 0:
                    # Fallback if size wasn't recorded
                    try:
                        size = len(self.serialize_obj(obj))
                    except Exception:
                        pass
            else:
                logger.error(
                    f"Unknown metadata format in SharedMemoryConnector for req {request_id}: {list(metadata.keys())}"
                )
                return None

            self._metrics["gets"] += 1
            return obj, size

        except Exception as e:
            logger.error(f"SharedMemoryConnector get failed for req {request_id}: {e}")
            return None

    def get_chunk(self, from_stage: str, to_stage: str, get_key: str, metadata=None) -> tuple[Any, int] | None:
        """Read chunk data from shared memory with integrity checks.

        Protocol:
        - Layout: [MAGIC(4)][SIZE(8)][CRC32(4)][PAYLOAD(N)]
        - Validates magic marker to ensure data is complete
        - Verifies CRC32 checksum for data integrity
        - Only unlinks on successful verified read
        """
        import struct
        import zlib
        from multiprocessing import shared_memory as shm_pkg

        shm = None
        success = False
        try:
            shm = shm_pkg.SharedMemory(name=get_key)
            buf = shm.buf

            # Step 1: Read and validate header
            header = bytes(buf[: self._SHM_HEADER_SIZE])

            # Check magic marker (ensures data is complete)
            magic = header[:4]
            if magic != self._SHM_MAGIC:
                # Data not ready - header not written yet
                return None, 0

            # Parse size and CRC32
            payload_size = struct.unpack("<Q", header[4:12])[0]
            expected_crc32 = struct.unpack("<I", header[12:16])[0]

            # Validate size is reasonable
            total_expected = self._SHM_HEADER_SIZE + payload_size
            if total_expected > shm.size:
                logger.warning(f"get_chunk: size mismatch for {get_key}: expected {total_expected}, got {shm.size}")
                return None, 0

            # Step 2: Read payload
            payload = bytes(buf[self._SHM_HEADER_SIZE : self._SHM_HEADER_SIZE + payload_size])

            # Step 3: Verify CRC32 checksum
            actual_crc32 = zlib.crc32(payload) & 0xFFFFFFFF
            if actual_crc32 != expected_crc32:
                logger.warning(
                    f"get_chunk: CRC32 mismatch for {get_key}: expected {expected_crc32:08x}, got {actual_crc32:08x}"
                )
                return None, 0

            # Step 4: Deserialize
            obj = self.deserialize_obj(payload)

            self._metrics["gets"] += 1
            success = True
            return obj, payload_size

        except FileNotFoundError:
            # Shared memory doesn't exist yet
            return None, 0
        except Exception as e:
            # Unexpected error - log but don't unlink (allow retry)
            logger.warning(f"get_chunk failed for key {get_key}: {e}")
            return None, 0
        finally:
            if shm is not None:
                shm.close()
                # ONLY unlink if read was fully successful and verified
                if success:
                    try:
                        shm.unlink()
                    except FileNotFoundError:
                        pass  # Already unlinked

    def cleanup(self, request_id: str) -> None:
        # SHM segments are automatically unlinked during 'get' (shm_read_bytes).
        # If 'get' is never called (e.g. error flow), the SHM segment might leak.
        # A robust implementation might track created segments and unlink them here
        # if they haven't been consumed.
        # For now, we rely on the consumer to read and unlink.
        pass

    def health(self) -> dict[str, Any]:
        return {"status": "healthy", "threshold": self.threshold, **self._metrics}
