# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from typing import Any

from ..utils.logging import get_connector_logger

logger = get_connector_logger(__name__)


class OmniConnectorBase(ABC):
    """Base class for all OmniConnectors."""

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
