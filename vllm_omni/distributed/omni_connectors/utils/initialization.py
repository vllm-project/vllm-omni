# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Utilities for OmniConnector configuration and validation."""

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..factory import OmniConnectorFactory
from .config import ConnectorSpec, OmniTransferConfig
from .logging import get_connector_logger

if TYPE_CHECKING:
    from ..connectors.base import OmniConnectorBase
else:
    OmniConnectorBase = Any

logger = get_connector_logger(__name__)


def initialize_connectors_from_config(
    config_path: str | Path | None = None, default_shm_threshold: int = 65536
) -> tuple[OmniTransferConfig | None, dict[tuple[str, str], OmniConnectorBase]]:
    """
    Initialize connectors from configuration file.

    Returns:
        tuple: (OmniTransferConfig, dict of {(from, to): connector_instance})
    """
    transfer_config = load_omni_transfer_config(config_path, default_shm_threshold=default_shm_threshold)

    if not transfer_config:
        logger.info("No OmniTransferConfig provided")
        return None, {}

    # create connectors from config
    connectors = create_connectors_from_config(transfer_config.connectors)
    return transfer_config, connectors


def create_connectors_from_config(
    connectors_config: dict[tuple[str, str], ConnectorSpec],
) -> dict[tuple[str, str], OmniConnectorBase]:
    """
    Create connectors from config.

    Args:
        connectors_config: A dictionary of connector configurations.

    Returns:
        A dictionary of connectors.
    """
    connectors = {}
    for edge_key, connector_spec in connectors_config.items():
        try:
            connector = OmniConnectorFactory.create_connector(connector_spec)
            connectors[edge_key] = connector
            logger.info(f"Created connector for {edge_key[0]} -> {edge_key[1]}: {type(connector).__name__}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize connector for edge {edge_key}: {e}") from e

    return connectors


def get_connectors_config_for_stage(transfer_config: OmniTransferConfig | None, stage_id: str | int) -> dict[str, Any]:
    """
    Extract connector configurations relevant for a specific stage worker.

    Returns a dict compatible with worker initialization:
    {
        "from_stage_X": {
            "spec": {
                "name": "ConnectorName",
                "extra": {...}
            }
        },
        ...
    }
    """
    if not transfer_config:
        return {}

    stage_connectors_config = {}
    target_stage = str(stage_id)

    # Iterate through all configured edges
    for (from_stage, to_stage), spec in transfer_config.connectors.items():
        # We only care about incoming edges for the worker process
        # (Worker needs to create connectors to receive data)
        if to_stage == target_stage:
            stage_connectors_config[f"from_stage_{from_stage}"] = {"spec": {"name": spec.name, "extra": spec.extra}}

    return stage_connectors_config


def load_omni_transfer_config(
    config_path: str | Path | None = None,
    config_dict: dict[str, Any] | None = None,
    default_shm_threshold: int = 65536,
) -> OmniTransferConfig | None:
    """Load OmniTransferConfig from file or dict."""
    if config_path is None and config_dict is None:
        # Even if no config provided, we might want to return a default config with SHM connectors
        # But without stage info we can't do much.
        return None

    if config_path is not None:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, encoding="utf-8") as f:
            if config_path.suffix.lower() == ".json":
                config_dict = json.load(f)
            elif config_path.suffix.lower() in [".yaml", ".yml"]:
                try:
                    import yaml

                    config_dict = yaml.safe_load(f)
                except ImportError:
                    raise ImportError("PyYAML required for YAML config files")
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")

    if config_dict is None:
        return None

    # Parse connectors
    connectors = {}
    runtime_config = config_dict.get("runtime", {})

    # Parse global connectors (from runtime.connectors)
    global_connectors = runtime_config.get("connectors", {})

    # Parse stage-level connectors
    stage_args = config_dict.get("stage_args", [])
    expected_edges: set[tuple[str, str]] = set()
    for stage_config in stage_args:
        stage_id = str(stage_config["stage_id"])

        # Input connectors
        for input_key, conn_ref in stage_config.get("input_connectors", {}).items():
            if isinstance(conn_ref, str):
                # Reference to global connector
                if conn_ref in global_connectors:
                    conn_config = global_connectors[conn_ref]
                    connector = ConnectorSpec(name=conn_config["name"], extra=conn_config.get("extra", {}))
                else:
                    raise ValueError(f"Undefined connector reference: {conn_ref}")
            else:
                # Inline connector definition
                connector = ConnectorSpec(name=conn_ref["name"], extra=conn_ref.get("extra", {}))

            # Parse from_stage from key (e.g., "from_stage_0" -> "0")
            from_stage = input_key.replace("from_stage_", "")
            edge_key = (from_stage, stage_id)
            connectors[edge_key] = connector
            expected_edges.add(edge_key)

        # Output connectors
        for output_key, conn_ref in stage_config.get("output_connectors", {}).items():
            if isinstance(conn_ref, str):
                # Reference to global connector
                if conn_ref in global_connectors:
                    conn_config = global_connectors[conn_ref]
                    connector = ConnectorSpec(name=conn_config["name"], extra=conn_config.get("extra", {}))
                else:
                    raise ValueError(f"Undefined connector reference: {conn_ref}")
            else:
                # Inline connector definition
                connector = ConnectorSpec(name=conn_ref["name"], extra=conn_ref.get("extra", {}))

            # Parse to_stage from key (e.g., "to_stage_1" -> "1")
            to_stage = output_key.replace("to_stage_", "")
            edge_key = (stage_id, to_stage)
            connectors[edge_key] = connector
            expected_edges.add(edge_key)

    # Auto-configure SharedMemoryConnector for missing edges based on runtime edges / engine_input_source
    if stage_args:
        try:
            # Prefer explicit runtime edges if provided
            runtime_edges = runtime_config.get("edges", [])
            if isinstance(runtime_edges, list) and runtime_edges:
                for edge in runtime_edges:
                    from_stage = edge.get("from")
                    to_stage = edge.get("to")
                    if from_stage is None or to_stage is None:
                        continue
                    edge_key = (str(from_stage), str(to_stage))
                    expected_edges.add(edge_key)
                    if edge_key not in connectors:
                        logger.info(f"Auto-configuring SharedMemoryConnector for edge {edge_key}")
                        connectors[edge_key] = ConnectorSpec(
                            name="SharedMemoryConnector",
                            extra={"shm_threshold_bytes": default_shm_threshold},
                        )

            # Fallback: infer edges from engine_input_source for each stage
            for stage_config in stage_args:
                to_stage = str(stage_config["stage_id"])
                # Check explicit input sources
                sources = stage_config.get("engine_input_source", [])

                for from_stage in sources:
                    from_stage_str = str(from_stage)
                    edge_key = (from_stage_str, to_stage)
                    expected_edges.add(edge_key)

                    if edge_key not in connectors:
                        logger.info(f"Auto-configuring SharedMemoryConnector for edge {edge_key}")
                        connectors[edge_key] = ConnectorSpec(
                            name="SharedMemoryConnector", extra={"shm_threshold_bytes": default_shm_threshold}
                        )

        except Exception as e:
            logger.warning(f"Failed to auto-configure SHM connectors: {e}")

    # Fail fast if any expected edge is still missing a connector
    missing_edges = [edge for edge in expected_edges if edge not in connectors]
    if missing_edges:
        missing_str = ", ".join([f"{f}->{t}" for f, t in missing_edges])
        raise ValueError(
            "Connector configuration missing for edges: "
            f"{missing_str}. Define connectors or allow auto SHM creation for these edges."
        )

    config = OmniTransferConfig(connectors=connectors)

    logger.info(f"Loaded OmniTransferConfig with {len(connectors)} connector configurations")
    return config


# High-level management functions


def initialize_orchestrator_connectors(
    config_path: str | None, worker_backend: str | None = "multi_process", shm_threshold_bytes: int = 65536
) -> tuple[OmniTransferConfig | None, dict[tuple[str, str], OmniConnectorBase]]:
    """Initialize connectors shared at orchestrator level.
    Args:
        config_path: The path to the configuration file.
        worker_backend: The backend to use for the worker.
    Returns:
        A tuple containing the OmniTransferConfig and a dictionary of connectors.
    """
    if worker_backend == "ray":
        default_shm_threshold = sys.maxsize
    else:
        default_shm_threshold = max(0, shm_threshold_bytes)
    transfer_config, connectors = initialize_connectors_from_config(
        config_path, default_shm_threshold=default_shm_threshold
    )
    return transfer_config, connectors


def get_stage_connector_config(
    transfer_config: OmniTransferConfig | None,
    stage_id: int,
) -> dict[str, Any]:
    """Return the serialized connector config payload for a specific stage."""
    if transfer_config is None:
        return {}

    try:
        return get_connectors_config_for_stage(transfer_config, stage_id)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning(
            "Failed to build connector config for stage %s: %s. Using IPC fallback.",
            stage_id,
            exc,
        )
        return {}


def build_stage_connectors(
    stage_id: int,
    connectors_config: dict[str, Any],
) -> dict[tuple[str, str], Any] | None:
    """Instantiate OmniConnectors for a stage based on config."""
    if not connectors_config:
        return {}

    logger.info(
        "[Stage-%s] Initializing OmniConnectors with config keys: %s",
        stage_id,
        list(connectors_config.keys()),
    )

    from .config import ConnectorSpec

    connectors: dict[tuple[str, str], Any] = {}
    # Convert dictionary-formatted config to ConnectorSpec objects
    stage_connector_specs = {}
    for input_key, config in connectors_config.items():
        if not input_key.startswith("from_stage_"):
            continue

        from_stage = input_key.replace("from_stage_", "")
        spec_dict = config.get("spec", {})
        if not spec_dict:
            continue

        connector_spec = ConnectorSpec(
            name=spec_dict.get("name", "SharedMemoryConnector"),
            extra=spec_dict.get("extra", {}),
        )
        stage_connector_specs[(str(from_stage), str(stage_id))] = connector_spec

    try:
        # Use unified connector creation logic
        connectors = create_connectors_from_config(stage_connector_specs)
    except Exception as exc:  # pragma: no cover - defensive logging
        # Fail fast so the stage does not start with missing connectors.
        logger.exception("[Stage-%s] Failed to initialize connectors: %s", stage_id, exc)
        raise

    return connectors


def resolve_omni_kv_config_for_stage(
    transfer_cfg: OmniTransferConfig | None, stage_id: int | str
) -> tuple[dict[str, Any] | None, str | None, str | None]:
    """Resolve connector configuration for a specific stage (Sender/Receiver).

    This determines the primary connector configuration to be injected into the
    engine arguments, prioritizing outgoing edges (Sender role).
    """
    if not transfer_cfg or not getattr(transfer_cfg, "connectors", None):
        return None, None, None

    stage_id_str = str(stage_id)

    # Find outgoing edges (Sender logic)
    outgoing = [
        (to_stage, spec)
        for (from_stage, to_stage), spec in transfer_cfg.connectors.items()
        if from_stage == stage_id_str
    ]

    # Find incoming edges (Receiver logic)
    incoming = [
        (from_stage, spec)
        for (from_stage, to_stage), spec in transfer_cfg.connectors.items()
        if to_stage == stage_id_str
    ]

    omni_conn_cfg = None
    omni_from = None
    omni_to = None

    # Prioritize outgoing (Sender) if exists, else check incoming (Receiver)
    if outgoing:
        if len(outgoing) > 1:
            logger.debug(
                "Stage-%s has %d outgoing edges; using the smallest to_stage",
                stage_id,
                len(outgoing),
            )
        outgoing.sort(key=lambda x: int(x[0]) if str(x[0]).isdigit() else str(x[0]))
        to_s, spec = outgoing[0]
        omni_conn_cfg = {"type": spec.name, **(spec.extra or {})}
        omni_from = stage_id_str
        omni_to = str(to_s)
    elif incoming:
        # For receiver, pick one incoming edge to configure the connector
        incoming.sort(key=lambda x: int(x[0]) if str(x[0]).isdigit() else str(x[0]))
        from_s, spec = incoming[0]
        omni_conn_cfg = {"type": spec.name, **(spec.extra or {})}
        omni_from = str(from_s)
        omni_to = stage_id_str

    return omni_conn_cfg, omni_from, omni_to
