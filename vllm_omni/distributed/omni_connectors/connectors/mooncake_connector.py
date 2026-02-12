# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from typing import Any

from ..utils.logging import get_connector_logger
from .base import OmniConnectorBase

logger = get_connector_logger(__name__)

try:
    from mooncake.store import MooncakeDistributedStore, ReplicateConfig
except ImportError:
    try:
        from mooncake import MooncakeDistributedStore, ReplicateConfig
    except ImportError:
        MooncakeDistributedStore = None
        ReplicateConfig = None


class MooncakeConnector(OmniConnectorBase):
    """Mooncake-based distributed connector for OmniConnector."""

    def __init__(self, config: dict[str, Any]):
        if MooncakeDistributedStore is None or ReplicateConfig is None:
            raise ImportError(
                "Mooncake components (MooncakeDistributedStore/ReplicateConfig) are not available. "
                "Please ensure the 'mooncake' package is installed in your environment."
            )

        self.config = config
        self.host = config.get("host", "127.0.0.1")
        self.metadata = config.get("metadata_server", "http://127.0.0.1:8080/metadata")
        self.master = config.get("master", "127.0.0.1:50051")
        self.segment = config.get("segment", 512 * 1024 * 1024)  # 512MB
        self.localbuf = config.get("localbuf", 64 * 1024 * 1024)  # 64MB
        self.proto = config.get("proto", "tcp")
        self.rdma = config.get("rdma", "")

        self.store: MooncakeDistributedStore | None = None
        self.pin: ReplicateConfig | None = None

        self._metrics = {
            "puts": 0,
            "gets": 0,
            "bytes_transferred": 0,
            "errors": 0,
            "timeouts": 0,
        }

        self._init_store()

    def _make_key(self, rid: str, from_stage: str, to_stage: str) -> str:
        """Generate store key for request between stages."""
        return f"{rid}/{from_stage}_{to_stage}"

    def _init_store(self):
        """Initialize Mooncake store."""
        try:
            self.store = MooncakeDistributedStore()
            rc = self.store.setup(
                self.host, self.metadata, self.segment, self.localbuf, self.proto, self.rdma, self.master
            )
            if rc != 0:
                raise RuntimeError(f"Mooncake setup failed: {rc}")

            self.pin = ReplicateConfig()
            self.pin.with_soft_pin = True
            logger.info("MooncakeConnector initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize Mooncake store: %s", e)
            raise

    # Use base class serialization methods for consistency

    def put(self, from_stage: str, to_stage: str, put_key: str, data: Any) -> tuple[bool, int, dict[str, Any] | None]:
        if not self.store:
            logger.error("Store not initialized")
            return False, 0, None

        try:
            serialized_data = self.serialize_obj(data)
            key = self._make_key(put_key, from_stage, to_stage)
            self.store.put(key, serialized_data, self.pin)

            self._metrics["puts"] += 1
            self._metrics["bytes_transferred"] += len(serialized_data)

            logger.debug(
                "MooncakeConnector: stored %s (%s -> %s) %d bytes",
                key,
                from_stage,
                to_stage,
                len(serialized_data),
            )
            return True, len(serialized_data), None

        except Exception as e:
            self._metrics["errors"] += 1
            logger.error("MooncakeConnector put failed: %s", e)
            return False, 0, None

    def get(
        self, from_stage: str, to_stage: str, get_key: str, metadata: dict[str, Any] | None = None
    ) -> tuple[Any, int] | None:
        if not self.store:
            logger.error("Store not initialized")
            return None

        retries = 20
        sleep_s = 0.05
        key = self._make_key(get_key, from_stage, to_stage)

        for attempt in range(retries):
            try:
                raw_data = self.store.get(key)

                if raw_data:
                    data = self.deserialize_obj(raw_data)
                    self._metrics["gets"] += 1
                    payload_size = len(raw_data)
                    logger.debug(
                        "MooncakeConnector: retrieved %s (%s -> %s) %d bytes",
                        key,
                        from_stage,
                        to_stage,
                        payload_size,
                    )
                    return data, payload_size

            except Exception as e:
                logger.debug("MooncakeConnector get attempt %s failed: %s", attempt, e)

            if attempt < retries - 1:
                time.sleep(sleep_s)

        self._metrics["timeouts"] += 1
        logger.warning("MooncakeConnector: timeout waiting for %s", key)
        return None

    def cleanup(self, request_id: str) -> None:
        if not self.store:
            return

        # Note: Mooncake doesn't have explicit delete, data will be garbage collected
        # We could implement a cleanup mechanism by storing deletion markers
        logger.debug("MooncakeConnector: cleanup requested for %s (no-op)", request_id)

    def health(self) -> dict[str, Any]:
        if not self.store:
            return {"status": "unhealthy", "error": "Store not initialized"}

        return {
            "status": "healthy",
            "host": self.host,
            "metadata_server": self.metadata,
            "master": self.master,
            **self._metrics,
        }

    def close(self):
        """Clean shutdown."""
        if self.store:
            try:
                self.store.close()
                self.store = None
                logger.info("MooncakeConnector closed")
            except Exception as e:
                logger.error("Error closing Mooncake store: %s", e)
