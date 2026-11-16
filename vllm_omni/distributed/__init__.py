# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from .omni_connectors import (
    ConnectorSpec,
    MoriTransferEngineConnector,
    NixlConnector,
    OmniConnectorBase,
    OmniConnectorFactory,
    OmniTransferConfig,
    SharedMemoryConnector,
    YuanrongConnector,
    YuanrongTransferEngineConnector,
    load_omni_transfer_config,
)

# The Mooncake names stay in ``__all__`` for API compatibility, but note that
# ``from vllm_omni.distributed import *`` resolves them through ``__getattr__``
# and therefore does import ``mooncake``. Star imports are not free of the
# optional dependency; import the specific names you need instead.
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

# Re-exported lazily so that ``import vllm_omni.distributed`` does not pull in
# the optional native ``mooncake`` extension. See
# ``vllm_omni.distributed.omni_connectors.__getattr__`` for details.
_LAZY_CONNECTOR_EXPORTS = frozenset({"MooncakeConnector", "MooncakeStoreConnector", "MooncakeTransferEngineConnector"})


def __getattr__(name: str):
    if name in _LAZY_CONNECTOR_EXPORTS:
        from . import omni_connectors

        value = getattr(omni_connectors, name)
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    # Lazy names are absent from the module dict until first access; include them
    # so that dir() and autocompletion match the documented API.
    return sorted(set(globals()) | _LAZY_CONNECTOR_EXPORTS)
