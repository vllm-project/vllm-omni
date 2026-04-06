# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import base64
import time
from typing import Any

from ..utils.logging import get_connector_logger
from .base import OmniConnectorBase

logger = get_connector_logger(__name__)

try:
    from priskv import PriskvClient
    import priskv._priskv as _priskv_raw
except ImportError:
    PriskvClient = None
    _priskv_raw = None


class PrisKVConnector(OmniConnectorBase):
    """PrisKV-based high-performance connector for OmniConnector.

    Uses PrisKV (RDMA/TCP/SHM KV store) as the transport backend for
    inter-stage data transfer.  Supports two data paths:

    - **setstr/getstr**: For serialized Python objects (dict, KV cache, etc.).
      Data is serialized via OmniSerializer, stored as opaque bytes in PrisKV,
      and deserialized on retrieval.

    - **SGL (scatter-gather list)**: For raw memory buffers when the caller
      provides pre-registered memory regions.  This path enables RDMA/GDR
      zero-copy transfers when PrisKV is configured with RDMA transport.

    Configuration keys (passed via ``config`` dict):
        host (str): PrisKV server address.  Default ``"127.0.0.1"``.
        port (int): PrisKV server port.  Default ``6379``.
        password (str): Connection password.  Default ``"kvcache-redis"``.
        get_retries (int): Max retry attempts for get().  Default ``20``.
        get_retry_interval (float): Sleep between retries in seconds.
            Default ``0.05``.
        key_timeout (int): TTL for stored keys in milliseconds.
            Default ``PRISKV_KEY_MAX_TIMEOUT`` (no expiry).
    """

    def __init__(self, config: dict[str, Any]):
        if PriskvClient is None:
            raise ImportError(
                "PrisKV Python client is not available. "
                "Please build and install pypriskv: "
                "cd PrisKV && make all && cd pypriskv && pip install -e ."
            )

        self.config = config
        self.host = config.get("host", "127.0.0.1")
        self.port = int(config.get("port", 6379))
        self.password = config.get("password", "kvcache-redis")
        self.get_retries = int(config.get("get_retries", 20))
        self.get_retry_interval = float(config.get("get_retry_interval", 0.05))
        self.key_timeout = int(
            config.get("key_timeout", _priskv_raw.PRISKV_KEY_MAX_TIMEOUT)
        )

        self.client: PriskvClient | None = None
        self._metrics = {
            "puts": 0,
            "gets": 0,
            "bytes_transferred": 0,
            "errors": 0,
            "timeouts": 0,
        }

        self._init_client()

    def _init_client(self):
        try:
            self.client = PriskvClient(self.host, self.port, self.password)
            logger.info(
                "PrisKVConnector initialized: %s:%s", self.host, self.port
            )
        except Exception as e:
            logger.error("Failed to initialize PrisKV client: %s", e)
            raise

    def put(
        self,
        from_stage: str,
        to_stage: str,
        put_key: str,
        data: Any,
    ) -> tuple[bool, int, dict[str, Any] | None]:
        if not self.client:
            logger.error("PrisKV client not initialized")
            return False, 0, None

        try:
            serialized_data = self.serialize_obj(data)
            key = self._make_key(put_key, from_stage, to_stage)

            encoded = base64.b64encode(serialized_data).decode("ascii")
            ret = self.client.setstr(key, encoded, self.key_timeout)
            if ret != 0:
                self._metrics["errors"] += 1
                logger.error(
                    "PrisKVConnector put failed for %s: status=%d", key, ret
                )
                return False, 0, None

            size = len(serialized_data)
            self._metrics["puts"] += 1
            self._metrics["bytes_transferred"] += size

            logger.debug(
                "PrisKVConnector: stored %s (%s -> %s) %d bytes",
                key, from_stage, to_stage, size,
            )
            return True, size, None

        except Exception as e:
            self._metrics["errors"] += 1
            logger.error("PrisKVConnector put failed: %s", e)
            return False, 0, None

    def get(
        self,
        from_stage: str,
        to_stage: str,
        get_key: str,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[Any, int] | None:
        if not self.client:
            logger.error("PrisKV client not initialized")
            return None

        key = self._make_key(get_key, from_stage, to_stage)
        t0 = time.perf_counter()

        for attempt in range(self.get_retries):
            try:
                t_fetch_start = time.perf_counter()
                raw_data = self.client.getstr(key)
                t_fetch_end = time.perf_counter()

                if raw_data is not None:
                    fetch_ms = (t_fetch_end - t_fetch_start) * 1000

                    t_deser_start = time.perf_counter()
                    decoded = base64.b64decode(raw_data)
                    data = self.deserialize_obj(decoded)
                    t_deser_end = time.perf_counter()
                    deser_ms = (t_deser_end - t_deser_start) * 1000

                    self._metrics["gets"] += 1
                    payload_size = len(decoded)

                    total_ms = (t_deser_end - t0) * 1000
                    mbps = (
                        (payload_size / 1024 / 1024) / (total_ms / 1000)
                        if total_ms > 0
                        else 0
                    )
                    logger.info(
                        "[PrisKV GET] %s: fetch=%.1fms, deser=%.1fms, "
                        "total=%.1fms, %d bytes, %.1f MB/s",
                        get_key, fetch_ms, deser_ms,
                        total_ms, payload_size, mbps,
                    )
                    return data, payload_size

            except Exception as e:
                logger.debug(
                    "PrisKVConnector get attempt %d failed: %s", attempt, e
                )

            if attempt < self.get_retries - 1:
                time.sleep(self.get_retry_interval)

        self._metrics["timeouts"] += 1
        logger.warning("PrisKVConnector: timeout waiting for %s", key)
        return None

    def cleanup(self, request_id: str) -> None:
        if not self.client:
            return

        try:
            self.client.delete(request_id)
        except Exception as e:
            logger.debug(
                "PrisKVConnector: cleanup failed for %s: %s", request_id, e
            )

    def health(self) -> dict[str, Any]:
        if not self.client:
            return {"status": "unhealthy", "error": "Client not initialized"}

        return {
            "status": "healthy",
            "host": self.host,
            "port": self.port,
            **self._metrics,
        }

    def close(self) -> None:
        if self.client:
            try:
                self.client.close()
                self.client = None
                logger.info("PrisKVConnector closed")
            except Exception as e:
                logger.error("Error closing PrisKV client: %s", e)
