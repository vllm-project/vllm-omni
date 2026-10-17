# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import importlib
from typing import Any

from .connectors.base import OmniConnectorBase
from .connectors.shm_connector import SharedMemoryConnector
from .connectors.yuanrong_connector import YuanrongConnector
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

# Connectors backed by optional native transports are resolved on first attribute
# access rather than at package import time.
#
# Their dependencies are heavy, and merely loading one can be harmful when the
# connector is never used: ``mooncake.store`` and ``mooncake.engine`` pull in
# CANN's LLM-DataDist runtime (``libruntime_v100.so``), whose exit-time destructor
# frees a corrupted heap block and makes the interpreter abort on shutdown
# (``corrupted size vs. prev_size`` followed by SIGABRT).  That kills any process
# which so much as imports ``vllm_omni`` -- including short-lived subprocesses --
# even though the deployment may only ever build a ``SharedMemoryConnector``.
#
# ``OmniConnectorFactory`` already resolves connectors by name at construction
# time (see ``factory.py``), so nothing needs these classes at import time.
#
# name -> (module, attribute, missing_dependency_yields_none)
_LAZY_CONNECTORS: dict[str, tuple[str, str, bool]] = {
    "MooncakeConnector": (
        f"{__name__}.connectors.mooncake_store_connector",
        "MooncakeStoreConnector",
        False,
    ),
    "MooncakeStoreConnector": (
        f"{__name__}.connectors.mooncake_store_connector",
        "MooncakeStoreConnector",
        False,
    ),
    "MooncakeTransferEngineConnector": (
        f"{__name__}.connectors.mooncake_transfer_engine_connector",
        "MooncakeTransferEngineConnector",
        True,  # RDMA deps (msgspec/zmq/mooncake) not installed
    ),
    "MoriTransferEngineConnector": (
        f"{__name__}.connectors.mori_transfer_engine_connector",
        "MoriTransferEngineConnector",
        True,  # RDMA deps (msgspec/zmq/mori) not installed
    ),
    "NixlConnector": (
        f"{__name__}.connectors.nixl_connector",
        "NixlConnector",
        True,  # NIXL deps not installed
    ),
    "YuanrongTransferEngineConnector": (
        "vllm_omni.platforms.npu.omni_connectors.yuanrong_transfer_engine_connector",
        "YuanrongTransferEngineConnector",
        True,
    ),
}


def __getattr__(name: str) -> Any:
    """Resolve the optional connector implementations lazily (PEP 562)."""
    entry = _LAZY_CONNECTORS.get(name)
    if entry is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr, optional = entry
    try:
        value = getattr(importlib.import_module(module_name), attr)
    except ImportError:
        if not optional:
            raise
        value = None
    globals()[name] = value  # cache so later lookups skip this function
    return value


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
