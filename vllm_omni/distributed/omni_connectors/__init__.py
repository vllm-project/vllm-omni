# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from .connectors.base import OmniConnectorBase
from .connectors.shm_connector import SharedMemoryConnector
from .connectors.yuanrong_connector import YuanrongConnector

try:
    from vllm_omni.platforms.npu.omni_connectors.yuanrong_transfer_engine_connector import (
        YuanrongTransferEngineConnector,
    )
except ImportError:
    YuanrongTransferEngineConnector = None

try:
    from .connectors.mori_transfer_engine_connector import MoriTransferEngineConnector
except ImportError:
    MoriTransferEngineConnector = None  # RDMA deps (msgspec/zmq/mori) not installed

try:
    from .connectors.nixl_connector import NixlConnector
except ImportError:
    NixlConnector = None  # NIXL deps not installed
from .factory import OmniConnectorFactory
from .utils.config import ConnectorSpec, OmniTransferConfig
from .utils.initialization import (
    build_stage_connectors,
    get_connectors_config_for_stage,
    get_stage_connector_config,
    initialize_connectors_from_config,
    initialize_orchestrator_connectors,
    load_omni_transfer_config,
)

# Mooncake-backed connectors are resolved lazily through the module-level
# ``__getattr__`` below. Importing ``mooncake_store_connector`` or
# ``mooncake_transfer_engine_connector`` loads the native ``mooncake``
# extension, which is an optional dependency that is only needed when a stage
# explicitly selects one of these connectors (see ``factory.py``, which already
# imports each connector lazily). Keeping them out of the package import path
# avoids loading a large native library in every process that happens to import
# vllm_omni, including the short-lived subprocesses vLLM uses to inspect model
# architectures.
_LAZY_MOONCAKE_EXPORTS = frozenset({"MooncakeConnector", "MooncakeStoreConnector", "MooncakeTransferEngineConnector"})


def __getattr__(name: str):
    if name in _LAZY_MOONCAKE_EXPORTS:
        if name == "MooncakeTransferEngineConnector":
            try:
                from .connectors.mooncake_transfer_engine_connector import (
                    MooncakeTransferEngineConnector,
                )
            except ImportError:
                MooncakeTransferEngineConnector = None  # RDMA deps not installed
            globals()[name] = MooncakeTransferEngineConnector
            return MooncakeTransferEngineConnector

        from .connectors.mooncake_store_connector import MooncakeStoreConnector

        # Backward-compatible alias: MooncakeConnector was renamed to
        # MooncakeStoreConnector. Keep it for at least one release cycle.
        globals()["MooncakeStoreConnector"] = MooncakeStoreConnector
        globals()["MooncakeConnector"] = MooncakeStoreConnector
        return globals()[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    # Lazy names are absent from the module dict until first access; include them
    # so that dir() and autocompletion match the documented API.
    return sorted(set(globals()) | _LAZY_MOONCAKE_EXPORTS)


# The Mooncake names stay in ``__all__`` for API compatibility, but note that
# ``from vllm_omni.distributed.omni_connectors import *`` resolves them through
# ``__getattr__`` and therefore does import ``mooncake``. Star imports are not
# free of the optional dependency; import the specific names you need instead.
__all__ = [
    # Config
    "ConnectorSpec",
    "OmniTransferConfig",
    # Base classes and implementations
    "OmniConnectorBase",
    # Factory
    "OmniConnectorFactory",
    # Specific implementations
    "MooncakeConnector",  # compat alias → MooncakeStoreConnector
    "MooncakeStoreConnector",
    "MooncakeTransferEngineConnector",
    "MoriTransferEngineConnector",
    "NixlConnector",
    "SharedMemoryConnector",
    "YuanrongConnector",
    "YuanrongTransferEngineConnector",
    # Utilities
    "load_omni_transfer_config",
    "initialize_connectors_from_config",
    "get_connectors_config_for_stage",
    # Manager helpers
    "initialize_orchestrator_connectors",
    "get_stage_connector_config",
    "build_stage_connectors",
]
