# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from typing import Any

from .omni_connectors import (
    ConnectorSpec,
    OmniConnectorBase,
    OmniConnectorFactory,
    OmniTransferConfig,
    SharedMemoryConnector,
    YuanrongConnector,
    load_omni_transfer_config,
)

__all__ = [
    # Config
    "ConnectorSpec",
    "OmniTransferConfig",
    # Connectors
    "OmniConnectorBase",
    "OmniConnectorFactory",
    "MooncakeConnector",  # compat alias
    "MooncakeStoreConnector",
    "MooncakeTransferEngineConnector",
    "MoriTransferEngineConnector",
    "NixlConnector",
    "SharedMemoryConnector",
    "YuanrongConnector",
    "YuanrongTransferEngineConnector",
    # Utilities
    "load_omni_transfer_config",
]


def __getattr__(name: str) -> Any:
    """Forward to ``omni_connectors`` without eagerly importing it here.

    ``from .omni_connectors import MooncakeStoreConnector`` would defeat the lazy
    resolution in that package and pull in optional native transports on every
    ``import vllm_omni``; see ``vllm_omni/distributed/omni_connectors/__init__.py``.
    """
    from . import omni_connectors

    try:
        value = getattr(omni_connectors, name)
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    globals()[name] = value  # cache so later lookups skip this function
    return value
