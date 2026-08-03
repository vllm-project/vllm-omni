# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from .connectors.base import OmniConnectorBase
from .connectors.mooncake_store_connector import MooncakeStoreConnector
from .connectors.shm_connector import SharedMemoryConnector
from .connectors.yuanrong_connector import YuanrongConnector

try:
    from vllm_omni.platforms.npu.omni_connectors.yuanrong_transfer_engine_connector import (
        YuanrongTransferEngineConnector,
    )
except ImportError:
    YuanrongTransferEngineConnector = None

try:
    from .connectors.mooncake_transfer_engine_connector import MooncakeTransferEngineConnector
except ImportError:
    MooncakeTransferEngineConnector = None  # RDMA deps (msgspec/zmq/mooncake) not installed

try:
    from .connectors.mori_transfer_engine_connector import MoriTransferEngineConnector
except ImportError:
    MoriTransferEngineConnector = None  # RDMA deps (msgspec/zmq/mori) not installed
from .factory import OmniConnectorFactory, StageConnectorSet
from .utils.config import ConnectorSpec, OmniTransferConfig, StageConnectorPlan, StageConnectorSpec
from .utils.initialization import (
    build_stage_connectors,
    default_stage_connector_plan,
    get_connectors_config_for_stage,
    get_stage_connector_config,
    initialize_connectors_from_config,
    initialize_orchestrator_connectors,
    load_omni_transfer_config,
    resolve_stage_connector_plan,
)

# Backward-compatible alias: MooncakeConnector was renamed to MooncakeStoreConnector.
# Keep this alias for at least one release cycle.
MooncakeConnector = MooncakeStoreConnector

__all__ = [
    # Config
    "ConnectorSpec",
    "OmniTransferConfig",
    "StageConnectorPlan",
    "StageConnectorSpec",
    # Base classes and implementations
    "OmniConnectorBase",
    # Factory
    "OmniConnectorFactory",
    "StageConnectorSet",
    # Specific implementations
    "MooncakeConnector",  # compat alias → MooncakeStoreConnector
    "MooncakeStoreConnector",
    "MooncakeTransferEngineConnector",
    "MoriTransferEngineConnector",
    "SharedMemoryConnector",
    "YuanrongConnector",
    "YuanrongTransferEngineConnector",
    # Utilities
    "load_omni_transfer_config",
    "initialize_connectors_from_config",
    "get_connectors_config_for_stage",
    "default_stage_connector_plan",
    "resolve_stage_connector_plan",
    # Manager helpers
    "initialize_orchestrator_connectors",
    "get_stage_connector_config",
    "build_stage_connectors",
]
