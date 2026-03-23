# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from typing import Any

from ..utils.logging import get_connector_logger

logger = get_connector_logger(__name__)


class OmniConnectorBase(ABC):
    """Base class for all OmniConnectors."""

    # Whether the connector can handle raw bytes/torch.Tensor natively
    # without going through OmniSerializer.  Connectors that copy raw
    # payloads directly (e.g. RDMA) should override this to True.
    supports_raw_data: bool = False

    @abstractmethod
    def put(self, from_stage: str, to_stage: str, put_key: str, data: Any) -> tuple[bool, int, dict[str, Any] | None]:
        """Store Python object, internal serialization handled by connector.

        Args:
            from_stage: Source stage identifier
            to_stage: Destination stage identifier
            put_key: Unique request identifier
            data: Python object to store

        Returns:
            tuple: (success: bool, serialized_size: int, metadata: Optional[dict])
                   Metadata may contain transport-specific handles or inline data.
        """
        pass

    @abstractmethod
    def get(self, from_stage: str, to_stage: str, get_key: str, metadata=None) -> tuple[Any, int] | None:
        """Retrieve Python object and payload size (bytes).

        Args:
            from_stage: Source stage identifier
            to_stage: Destination stage identifier
            get_key: Unique request identifier

        Returns:
            Tuple of (Python object, serialized byte size) if found, None otherwise
        """
        pass

    @abstractmethod
    def cleanup(self, request_id: str) -> None:
        """Clean up resources for a request."""
        pass

    @abstractmethod
    def health(self) -> dict[str, Any]:
        """Return health status and metrics."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Release resources held by this connector.

        Subclasses must implement this to clean up transport-specific
        resources (connections, memory pools, threads, etc.).
        Implementations should be idempotent (safe to call multiple times).
        """
        pass

    # --- Default resource-management protocol ---
    # Subclasses get context-manager and destructor support for free;
    # they only need to implement close().

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @staticmethod
    def serialize_obj(obj: Any) -> bytes:
        """Serialize a Python object to bytes using centralized serializer."""
        from ..utils.serialization import OmniSerializer

        return OmniSerializer.serialize(obj)

    @staticmethod
    def deserialize_obj(data: bytes) -> Any:
        """Deserialize bytes to Python object using centralized serializer."""
        from ..utils.serialization import OmniSerializer

        return OmniSerializer.deserialize(data)

    @staticmethod
    def _make_key(key: str, from_stage: str, to_stage: str, separator: str = "@") -> str:
        """Generate internal key with stage routing info.

        Default format: ``{key}@{from_stage}_{to_stage}``.
        Connectors with different key conventions can override this method.
        """
        return f"{key}{separator}{from_stage}_{to_stage}"
