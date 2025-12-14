# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

from vllm import envs

try:
    import cloudpickle

    _has_cloudpickle = True
except ImportError:
    _has_cloudpickle = False

from .logging import get_connector_logger

logger = get_connector_logger(__name__)


def _log_insecure_serialization_warning():
    logger.warning_once("Allowing insecure serialization using pickle due to "
                        "VLLM_ALLOW_INSECURE_SERIALIZATION=1")


class OmniSerializer:
    """
    Centralized serialization handler for OmniConnectors.
    """

    @staticmethod
    def serialize(obj: Any) -> bytes:
        """
        Serialize an object to bytes using cloudpickle.

        Args:
            obj: The object to serialize.

        Returns:
            Serialized bytes.

        Raises:
            TypeError: If cloudpickle is not available and insecure serialization
                       is not explicitly allowed.
        """
        if _has_cloudpickle:
            return cloudpickle.dumps(obj)

        if not envs.VLLM_ALLOW_INSECURE_SERIALIZATION:
            raise TypeError(
                f"Object of type {type(obj)} is not serializable. "
                "Set VLLM_ALLOW_INSECURE_SERIALIZATION=1 to allow "
                "fallback to pickle-based serialization."
            )
        _log_insecure_serialization_warning()
        import pickle
        return pickle.dumps(obj)

    @staticmethod
    def deserialize(data: bytes) -> Any:
        """
        Deserialize bytes to an object using cloudpickle.

        Args:
            data: The bytes to deserialize.

        Returns:
            Deserialized object.

        Raises:
            TypeError: If cloudpickle is not available and insecure serialization
                       is not explicitly allowed.
        """
        if _has_cloudpickle:
            return cloudpickle.loads(data)

        if not envs.VLLM_ALLOW_INSECURE_SERIALIZATION:
            raise TypeError(
                "Data is not deserializable. "
                "Set VLLM_ALLOW_INSECURE_SERIALIZATION=1 to allow "
                "fallback to pickle-based deserialization."
            )
        _log_insecure_serialization_warning()
        import pickle
        return pickle.loads(data)
