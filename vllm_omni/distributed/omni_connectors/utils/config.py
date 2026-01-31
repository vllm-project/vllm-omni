# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, field
from typing import Any

from .logging import get_connector_logger

logger = get_connector_logger(__name__)


@dataclass
class ConnectorSpec:
    """Specification for a connector instance."""

    name: str  # e.g., "MooncakeConnector", "SharedMemoryConnector", "YuanrongConnector"
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
