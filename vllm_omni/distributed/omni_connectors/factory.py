# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from typing import Any

from .utils.logging import get_connector_logger

try:
    from .connectors.base import OmniConnectorBase
    from .utils.config import ConnectorSpec
except ImportError:
    # Fallback for direct execution
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from omni_connectors.connectors.base import OmniConnectorBase
    from omni_connectors.utils.config import ConnectorSpec

logger = get_connector_logger(__name__)


class OmniConnectorFactory:
    """Factory for creating OmniConnectors."""

    _registry: dict[str, Callable[[dict[str, Any]], OmniConnectorBase]] = {}

    @classmethod
    def register_connector(cls, name: str, constructor: Callable[[dict[str, Any]], OmniConnectorBase]) -> None:
        """Register a connector constructor."""
        if name in cls._registry:
            raise ValueError(f"Connector '{name}' is already registered.")
        cls._registry[name] = constructor
        logger.debug(f"Registered connector: {name}")

    @classmethod
    def create_connector(cls, spec: ConnectorSpec) -> OmniConnectorBase:
        """Create a connector from specification."""
        if spec.name not in cls._registry:
            raise ValueError(f"Unknown connector: {spec.name}. Available: {list(cls._registry.keys())}")

        constructor = cls._registry[spec.name]
        try:
            connector = constructor(spec.extra)
            logger.info(f"Created connector: {spec.name}")
            return connector
        except Exception as e:
            logger.error(f"Failed to create connector {spec.name}: {e}")
            raise ValueError(f"Failed to create connector {spec.name}: {e}")

    @classmethod
    def list_registered_connectors(cls) -> list[str]:
        """List all registered connector names."""
        return list(cls._registry.keys())


# Register built-in connectors with lazy imports
def _create_mooncake_connector(config: dict[str, Any]) -> OmniConnectorBase:
    try:
        from .connectors.mooncake_connector import MooncakeConnector
    except ImportError:
        # Fallback import
        import os
        import sys

        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from omni_connectors.connectors.mooncake_connector import MooncakeConnector
    return MooncakeConnector(config)


def _create_shm_connector(config: dict[str, Any]) -> OmniConnectorBase:
    try:
        from .connectors.shm_connector import SharedMemoryConnector
    except ImportError:
        # Fallback import
        import os
        import sys

        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from omni_connectors.connectors.shm_connector import SharedMemoryConnector
    return SharedMemoryConnector(config)


# Register connectors
OmniConnectorFactory.register_connector("MooncakeConnector", _create_mooncake_connector)
OmniConnectorFactory.register_connector("SharedMemoryConnector", _create_shm_connector)
