# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import os
from collections.abc import Callable
from typing import Any

from .utils.logging import get_connector_logger

try:
    from .connectors import base as connector_base
    from .utils import config as connector_config
except ImportError:
    # Fallback for direct execution
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from omni_connectors.connectors import base as fallback_connector_base
    from omni_connectors.utils import config as fallback_connector_config

    connector_base = fallback_connector_base
    connector_config = fallback_connector_config

OmniConnectorBase = connector_base.OmniConnectorBase
ConnectorSpec = connector_config.ConnectorSpec

logger = get_connector_logger(__name__)


class OmniConnectorFactory:
    """Factory for creating OmniConnectors."""

    _registry: dict[str, Callable[[dict[str, Any]], OmniConnectorBase]] = {}
    _deadline_aware_connectors: set[str] = set()

    @classmethod
    def register_connector(
        cls,
        name: str,
        constructor: Callable[[dict[str, Any]], OmniConnectorBase],
        *,
        supports_deadline_receive: bool = False,
    ) -> None:
        """Register a lazy constructor and its deadline-aware receive capability.

        Set supports_deadline_receive only when the backend implements
        get_with_deadline with bounded backend-controlled waits.
        """
        if name in cls._registry:
            raise ValueError(f"Connector '{name}' is already registered.")
        cls._registry[name] = constructor
        if supports_deadline_receive:
            cls._deadline_aware_connectors.add(name)
        logger.debug(f"Registered connector: {name}")

    @classmethod
    def supports_deadline_receive(cls, name: str) -> bool:
        """Inspect capability without importing or constructing the backend."""
        return name in cls._registry and name in cls._deadline_aware_connectors

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
def _create_mooncake_store_connector(config: dict[str, Any]) -> OmniConnectorBase:
    try:
        from .connectors import mooncake_store_connector as connector_module
    except ImportError:
        # Fallback import
        import sys

        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from omni_connectors.connectors import mooncake_store_connector as fallback_connector_module

        connector_module = fallback_connector_module
    return connector_module.MooncakeStoreConnector(config)


def _create_shm_connector(config: dict[str, Any]) -> OmniConnectorBase:
    try:
        from .connectors import shm_connector as connector_module
    except ImportError:
        # Fallback import
        import sys

        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from omni_connectors.connectors import shm_connector as fallback_connector_module

        connector_module = fallback_connector_module
    return connector_module.SharedMemoryConnector(config)


def _create_yuanrong_connector(config: dict[str, Any]) -> OmniConnectorBase:
    try:
        from .connectors import yuanrong_connector as connector_module
    except ImportError:
        import sys

        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from omni_connectors.connectors import yuanrong_connector as fallback_connector_module

        connector_module = fallback_connector_module
    return connector_module.YuanrongConnector(config)


def _create_yuanrong_transfer_engine_connector(config: dict[str, Any]) -> OmniConnectorBase:
    try:
        from vllm_omni.platforms.npu.omni_connectors import YuanrongTransferEngineConnector
    except ImportError as exc:
        raise ImportError(
            "YuanrongTransferEngineConnector is only available in the NPU platform "
            "environment. Install the Ascend/Yuanrong runtime dependencies before "
            "using this connector."
        ) from exc
    return YuanrongTransferEngineConnector(config)


def _create_mooncake_transfer_engine_connector(config: dict[str, Any]) -> OmniConnectorBase:
    try:
        from .connectors import mooncake_transfer_engine_connector as connector_module
    except ImportError:
        import sys

        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from omni_connectors.connectors import mooncake_transfer_engine_connector as fallback_connector_module

        connector_module = fallback_connector_module
    return connector_module.MooncakeTransferEngineConnector(config)


def _create_mori_transfer_engine_connector(config: dict[str, Any]) -> OmniConnectorBase:
    try:
        from .connectors import mori_transfer_engine_connector as connector_module
    except ImportError:
        import sys

        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from omni_connectors.connectors import mori_transfer_engine_connector as fallback_connector_module

        connector_module = fallback_connector_module
    return connector_module.MoriTransferEngineConnector(config)


def _create_nixl_connector(config: dict[str, Any]) -> OmniConnectorBase:
    try:
        from .connectors import nixl_connector as connector_module
    except ImportError:
        import sys

        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from omni_connectors.connectors import nixl_connector as fallback_connector_module

        connector_module = fallback_connector_module
    return connector_module.NixlConnector(config)


# Register connectors
OmniConnectorFactory.register_connector("MooncakeStoreConnector", _create_mooncake_store_connector)
OmniConnectorFactory.register_connector("MooncakeTransferEngineConnector", _create_mooncake_transfer_engine_connector)
OmniConnectorFactory.register_connector("SharedMemoryConnector", _create_shm_connector, supports_deadline_receive=True)
OmniConnectorFactory.register_connector("YuanrongConnector", _create_yuanrong_connector)
OmniConnectorFactory.register_connector("YuanrongTransferEngineConnector", _create_yuanrong_transfer_engine_connector)
OmniConnectorFactory.register_connector("MoriTransferEngineConnector", _create_mori_transfer_engine_connector)
OmniConnectorFactory.register_connector("NixlConnector", _create_nixl_connector, supports_deadline_receive=True)
# Backward-compatible aliases – will be removed in the future
OmniConnectorFactory.register_connector("MooncakeConnector", _create_mooncake_store_connector)
