# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

from ..utils.logging import get_connector_logger
from .base import OmniConnectorBase

logger = get_connector_logger(__name__)

try:
    from datasystem.kv_client import KVClient, SetParam, WriteMode
except ImportError:
    KVClient = None
    SetParam = None
    WriteMode = None


class YuanrongConnector(OmniConnectorBase):
    """Datasystem-based distributed connector for OmniConnector."""

    def __init__(self, config: dict[str, Any]):
        if KVClient is None or SetParam is None or WriteMode is None:
            raise ImportError(
                "Datasystem components (KVClient/SetParam/WriteMode) are not available. "
                "Please ensure the 'datasystem' package is installed in your environment."
            )

        self.config = config
        self.client = None
        self.set_param = SetParam()
        self.set_param.write_mode = WriteMode.NONE_L2_CACHE_EVICT
        self.get_sub_timeout_ms = max(0, int(self.config.get("get_sub_timeout_ms", 1000)))

        self._metrics = {
            "puts": 0,
            "gets": 0,
            "bytes_transferred": 0,
            "errors": 0,
            "timeouts": 0,
        }

        self._init_client()

    def _make_key(self, rid: str, from_stage: str, to_stage: str) -> str:
        """Generate key for request between stages."""
        return f"{rid}:{from_stage}_{to_stage}"

    def _init_client(self):
        """Initialize Datasystem client."""
        try:
            self.host = self.config.get("host", "127.0.0.1")
            self.port = int(self.config.get("port", "35001"))
            self.client = KVClient(self.host, self.port)
            self.client.init()

            logger.info("YuanrongConnector initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize Datasystem client: %s", e)
            raise

    def put(self, from_stage: str, to_stage: str, put_key: str, data: Any) -> tuple[bool, int, dict[str, Any] | None]:
        if not self.client:
            logger.error("Datasystem client not initialized")
            return False, 0, None

        try:
            serialized_data = self.serialize_obj(data)
            key = self._make_key(put_key, from_stage, to_stage)
            self.client.set(key, serialized_data, self.set_param.write_mode)

            self._metrics["puts"] += 1
            self._metrics["bytes_transferred"] += len(serialized_data)

            logger.debug(
                "YuanrongConnector: stored %s (%s -> %s) %d bytes",
                key,
                from_stage,
                to_stage,
                len(serialized_data),
            )
            return True, len(serialized_data), None

        except Exception as exc:
            self._metrics["errors"] += 1
            logger.error("YuanrongConnector put failed: %s", exc)
            return False, 0, None

    def get(
        self, from_stage: str, to_stage: str, get_key: str, metadata: dict[str, Any] | None = None
    ) -> tuple[Any, int] | None:
        if not self.client:
            logger.error("Datasystem client not initialized")
            return None

        key = self._make_key(get_key, from_stage, to_stage)
        try:
            raw_list = self.client.get([key], False, self.get_sub_timeout_ms)
            raw_data = raw_list[0] if raw_list else None
            if raw_data is not None:
                data = self.deserialize_obj(raw_data)
                self._metrics["gets"] += 1
                payload_size = len(raw_data)
                logger.debug(
                    "YuanrongConnector: retrieved %s (%s -> %s) %d bytes",
                    key,
                    from_stage,
                    to_stage,
                    payload_size,
                )
                return data, payload_size

        except Exception as exc:
            self._metrics["timeouts"] += 1
            logger.error("YuanrongConnector get failed: %s", exc)
            return None

    def cleanup(self, request_id: str) -> None:
        if not self.client:
            return

        # Note: Datasystem doesn't have explicit delete, data will be garbage collected
        logger.debug("YuanrongConnector: cleanup requested for %s (no-op)", request_id)

    def health(self) -> dict[str, Any]:
        if not self.client:
            return {"status": "unhealthy", "error": "Datasystem client not initialized"}

        return {"status": "healthy", "host": self.host, "port": self.port, **self._metrics}

    def close(self) -> None:
        if not self.client:
            return

        self.client = None
        logger.info("YuanrongConnector closed")
