# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, field
from typing import Any

from .logging import get_connector_logger

logger = get_connector_logger(__name__)

TRANSFER_ENGINE_CONNECTOR_NAMES = frozenset(
    {
        "MooncakeTransferEngineConnector",
        "MoriTransferEngineConnector",
        "YuanrongTransferEngineConnector",
    }
)


def get_stage_connector_role(model_config: Any) -> str | None:
    """Return the configured stage connector direction, if explicit."""
    connector_config = getattr(model_config, "stage_connector_config", None)
    if isinstance(connector_config, dict):
        extra = connector_config.get("extra")
    else:
        extra = getattr(connector_config, "extra", None)
    if isinstance(extra, dict):
        role = extra.get("role")
        return role if isinstance(role, str) else None
    return None


def stage_receives_chunks(model_config: Any) -> bool:
    """Whether connector chunks, rather than the orchestrator, feed a stage."""
    return get_stage_connector_role(model_config) != "sender"


def stage_sends_async_output(model_config: Any) -> bool:
    """Whether async output should be partitioned for connector transport."""
    role = get_stage_connector_role(model_config)
    if role is not None:
        return role == "sender"
    # Preserve legacy partitioning while keeping stage-0 orchestrator bridges
    # on the normal RequestOutput path.
    return getattr(model_config, "stage_id", None) != 0


@dataclass
class ConnectorSpec:
    """Specification for a connector instance."""

    name: str  # e.g., "MooncakeStoreConnector", "SharedMemoryConnector", "YuanrongConnector"
    extra: dict[str, Any] = field(default_factory=dict)  # backend-specific config


@dataclass
class OmniTransferConfig:
    """
    Top-level configuration for OmniConnector system.
    Members:
        connectors: A dictionary of connectors, keyed by (from_stage, to_stage).
        default_connector: The default connector to use if no connector is specified for an edge.
    """

    # Direct mapping: (from_stage, to_stage) -> connector
    connectors: dict[tuple[str, str], ConnectorSpec] = field(default_factory=dict)
    default_connector: ConnectorSpec | None = None

    def get_connector_for_edge(self, from_stage: str, to_stage: str) -> ConnectorSpec | None:
        """Get connector spec for a specific edge."""
        edge_key = (from_stage, to_stage)
        return self.connectors.get(edge_key, self.default_connector)

    def has_connector_for_edge(self, from_stage: str, to_stage: str) -> bool:
        """Check if there's a connector configured for the edge."""
        return self.get_connector_for_edge(from_stage, to_stage) is not None
