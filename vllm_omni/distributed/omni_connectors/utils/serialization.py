# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import io
from typing import Any

try:
    import torch

    _has_torch = True
except ImportError:
    _has_torch = False

try:
    import cloudpickle

    _has_cloudpickle = True
except ImportError:
    _has_cloudpickle = False

from .logging import get_connector_logger

logger = get_connector_logger(__name__)


class OmniSerializer:
    """
    Centralized serialization handler for OmniConnectors.
    Supports multiple backends (torch, cloudpickle) to ensure consistency across connectors.
    """

    @staticmethod
    def serialize(obj: Any, method: str = "cloudpickle") -> bytes:
        """
        Serialize an object to bytes.

        Args:
            obj: The object to serialize.
            method: Serialization method ("cloudpickle" or "torch").
                    Defaults to "cloudpickle" (consistent with legacy stage_utils).
        """
        if method == "torch" and _has_torch:
            buf = io.BytesIO()
            torch.save(obj, buf)
            return buf.getvalue()
        elif method == "cloudpickle" or (method == "torch" and not _has_torch):
            if not _has_cloudpickle:
                # Fallback to standard pickle if cloudpickle is missing
                import pickle

                return pickle.dumps(obj)
            return cloudpickle.dumps(obj)
        else:
            # Default fallback
            import pickle

            return pickle.dumps(obj)

    @staticmethod
    def deserialize(data: bytes, method: str = "cloudpickle") -> Any:
        """
        Deserialize bytes to an object.

        Args:
            data: The bytes to deserialize.
            method: Serialization method used ("cloudpickle" or "torch").
        """
        if method == "torch" and _has_torch:
            return torch.load(io.BytesIO(data), map_location="cpu", weights_only=False)
        elif method == "cloudpickle" or (method == "torch" and not _has_torch):
            if not _has_cloudpickle:
                import pickle

                return pickle.loads(data)
            return cloudpickle.loads(data)
        else:
            import pickle

            return pickle.loads(data)
