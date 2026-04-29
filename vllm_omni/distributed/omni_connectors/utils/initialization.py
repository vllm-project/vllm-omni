# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Utilities for OmniConnector configuration and validation."""

import json
import socket
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

# Reserve a separate port range for KV-transfer sockets so they do not
# collide with request-forwarding endpoints that share the same base port.
KV_TRANSFER_PORT_OFFSET = 100

# Port stride between TP ranks so each worker binds a unique ZMQ port
# when TP > 1.  Must be larger than the maximum number of pipeline stages.
# Formula: zmq_port = base + KV_TRANSFER_PORT_OFFSET + rank * STRIDE + stage
KV_RANK_PORT_STRIDE = 16

# Connector types that carry a ZMQ-based sender/receiver handshake and
# therefore need per-edge endpoint derivation on the
# ``OmniChunkTransferAdapter`` path.  Without this, a 3-stage intranode
# pipeline would instantiate three co-located senders trying to bind the
# same base ``zmq_port`` from yaml, and every receiver / dual-role
# instance would start with no ``sender_host`` / ``sender_zmq_port`` and
# therefore no way to reach its upstream listener.
#
# The orchestrator-level path (``create_connectors_from_config`` below)
# has its own Mooncake-specific port adjustment for PD disaggregation
# and is intentionally independent of this set.
_ROLE_BOUND_ZMQ_CONNECTORS = frozenset({
    "MoriTransferEngineConnector",
    "MooncakeTransferEngineConnector",
})


def _detect_local_ip() -> str:
    """Best-effort local IP for framework-level endpoint derivation.

    Mirrors the connector-side ``_get_local_ip`` behaviour (used by
    Mori/Mooncake when ``host: "auto"``) so that, on an intranode
    pipeline, a sender's advertised listener host and its downstream
    receiver's precomputed upstream host agree without an explicit
    yaml setting.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        try:
            return socket.gethostbyname(socket.gethostname())
        except Exception:
            return "127.0.0.1"


def initialize_connectors_from_config(
    config_path: str | Path | None = None,
    default_shm_threshold: int = 65536,
    purpose: str = "request_forwarding",
    caller_stage_id: int | str | None = None,
    is_sender: bool | None = None,
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
    connectors = create_connectors_from_config(
        transfer_config.connectors,
        purpose=purpose,
        caller_stage_id=caller_stage_id,
        is_sender=is_sender,
    )
    return transfer_config, connectors


def create_connectors_from_config(
    connectors_config: dict[tuple[str, str], ConnectorSpec],
    purpose: str = "request_forwarding",
    caller_stage_id: int | str | None = None,
    is_sender: bool | None = None,
) -> dict[tuple[str, str], OmniConnectorBase]:
    """
    Create connectors from config.

    Args:
        connectors_config: A dictionary of connector configurations.

    Returns:
        A dictionary of connectors.
    """
    purpose_port_offsets = {
        "request_forwarding": 0,
        "kv_transfer": KV_TRANSFER_PORT_OFFSET,
    }
    port_offset = purpose_port_offsets.get(purpose, 0)
    orchestrator_port_offset = 200

    connectors = {}
    for edge_key, connector_spec in connectors_config.items():
        from_stage, to_stage = edge_key
        try:
            if connector_spec.name == "MooncakeTransferEngineConnector":
                extra = dict(connector_spec.extra) if connector_spec.extra else {}
                base_port = extra.get("zmq_port", 50051)
                try:
                    stage_offset = int(from_stage)
                except (TypeError, ValueError):
                    stage_offset = 0

                if str(caller_stage_id) == "orchestrator":
                    adjusted_port = base_port + orchestrator_port_offset + stage_offset
                else:
                    adjusted_port = base_port + port_offset + stage_offset
                extra["zmq_port"] = adjusted_port

                if is_sender is not None:
                    extra["role"] = "sender" if is_sender else "receiver"
                    if not is_sender:
                        extra.setdefault("sender_host", extra.get("host", "127.0.0.1"))
                        extra.setdefault("sender_zmq_port", adjusted_port)
                elif caller_stage_id is not None:
                    caller_str = str(caller_stage_id)
                    if caller_str == from_stage:
                        extra["role"] = "sender"
                    elif caller_str == to_stage:
                        extra["role"] = "receiver"
                        extra.setdefault("sender_host", extra.get("host", "127.0.0.1"))
                        extra.setdefault("sender_zmq_port", adjusted_port)
                    else:
                        extra["role"] = "sender"
                else:
                    extra["role"] = extra.get("role", "auto")

                connector = OmniConnectorFactory.create_connector(ConnectorSpec(name=connector_spec.name, extra=extra))
            else:
                connector = OmniConnectorFactory.create_connector(connector_spec)
            connectors[edge_key] = connector
            logger.info(
                "Created connector for %s -> %s: %s",
                from_stage,
                to_stage,
                type(connector).__name__,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize connector for edge {edge_key}: {e}") from e

    return connectors


def _primary_upstream_stage(incoming_edges: list[tuple[str, ConnectorSpec]]) -> str | None:
    """Return the lowest-stage-id upstream edge id, or *None* if there are no
    incoming edges.

    Used by ``_inject_chunk_path_endpoints`` to pick which upstream sender a
    dual / receiver instance should point at when a (future) fan-in topology
    has more than one.  Linear pipelines have exactly one incoming edge so
    the choice is unambiguous today; picking the sorted-first keeps behaviour
    deterministic if fan-in is introduced later.
    """
    if not incoming_edges:
        return None
    try:
        return sorted(incoming_edges, key=lambda e: int(e[0]))[0][0]
    except (TypeError, ValueError):
        return sorted(incoming_edges, key=lambda e: str(e[0]))[0][0]


def _inject_chunk_path_endpoints(
    extra: dict[str, Any],
    connector_name: str,
    role: str,
    own_stage: str,
    upstream_stage: str | None,
) -> None:
    """Compute per-edge ZMQ endpoints in-place for chunk-adapter connectors.

    Populates, for role-bound ZMQ connectors listed in
    ``_ROLE_BOUND_ZMQ_CONNECTORS``:

    * ``zmq_port`` (sender / dual only): bind port for the local listener
      = base_port + own_stage_id.  Offsetting by stage id prevents multiple
      co-located stage listeners from all binding the same base port on a
      single-node pipeline.
    * ``sender_zmq_port`` (receiver / dual only): upstream sender's bind
      port = base_port + upstream_stage_id.
    * ``sender_host`` (receiver / dual only): defaults to the
      framework-detected local IP when the yaml uses ``host: "auto"``,
      or to the yaml's explicit ``host`` value otherwise -- so an
      intranode receiver can locate its upstream sender without an
      external handshake.

    ``base_port`` is read from ``extra["zmq_port"]`` (default 50051).
    Values already present in *extra* (explicit yaml override) are kept,
    so cross-node deployments that already know the peer endpoint can
    set ``sender_host`` / ``sender_zmq_port`` directly and are unaffected.

    For non-role-bound connectors (``SharedMemoryConnector``,
    ``YuanrongConnector``, ...) this function is a no-op.  Non-integer
    stage ids (unusual pipeline keys such as ``"prefill"``) also return
    without side effects.
    """
    if connector_name not in _ROLE_BOUND_ZMQ_CONNECTORS:
        return
    try:
        own_stage_id = int(own_stage)
    except (TypeError, ValueError):
        return
    base_port = int(extra.get("zmq_port", 50051))

    if role in ("sender", "dual"):
        extra["zmq_port"] = base_port + own_stage_id

    if role in ("receiver", "dual") and upstream_stage is not None:
        try:
            upstream_stage_id = int(upstream_stage)
        except (TypeError, ValueError):
            upstream_stage_id = None
        if upstream_stage_id is not None:
            if "sender_zmq_port" not in extra or extra["sender_zmq_port"] is None:
                extra["sender_zmq_port"] = base_port + upstream_stage_id
        if not extra.get("sender_host"):
            host_cfg = extra.get("host", "auto")
            if isinstance(host_cfg, str) and host_cfg.lower() == "auto":
                extra["sender_host"] = _detect_local_ip()
            else:
                extra["sender_host"] = host_cfg


def get_connectors_config_for_stage(transfer_config: OmniTransferConfig | None, stage_id: str | int) -> dict[str, Any]:
    """Extract connector configurations relevant for a specific stage worker.

    Returns a dict shape compatible with worker initialization::

        {
            "from_stage_X": {"spec": {"name": ..., "extra": {..., "role": ...}}},
            "to_stage_Y":   {"spec": {"name": ..., "extra": {..., "role": ...}}},
            ...
        }

    The per-stage role is chosen by looking at which edges the stage
    participates in:

    * stage has only outgoing edges -> ``role="sender"``
    * stage has only incoming edges -> ``role="receiver"``
    * stage has both (middle stage in a 3+ stage pipeline) ->
      ``role="dual"``, and *both* the ``from_stage_*`` and
      ``to_stage_*`` entries share the same composite extra (same
      ``role``, same ``zmq_port``, same ``sender_host`` /
      ``sender_zmq_port``) so that downstream flattening (e.g.
      ``engine/stage_init_utils.get_stage_connector_spec`` returning
      the first spec it sees) always recovers a self-consistent
      per-stage connector config.

    For role-bound ZMQ connectors (see ``_ROLE_BOUND_ZMQ_CONNECTORS``)
    per-edge ``zmq_port`` / ``sender_host`` / ``sender_zmq_port`` are
    pre-computed from the sender's stage id so a co-located intranode
    pipeline does not need an external handshake.  Role-neutral
    connectors (``SharedMemoryConnector``, ``YuanrongConnector``) are
    unaffected by the endpoint injection but still receive the
    direction-specific ``role`` key (connectors that ignore it keep
    working verbatim).

    The orchestrator-level ``build_stage_connectors`` only filters
    ``from_stage_*`` keys, so orchestrator instantiation behaviour is
    bit-for-bit unchanged.
    """
    if not transfer_config:
        return {}

    target_stage = str(stage_id)

    incoming_edges: list[tuple[str, ConnectorSpec]] = [
        (from_s, spec)
        for (from_s, to_s), spec in transfer_config.connectors.items()
        if to_s == target_stage
    ]
    outgoing_edges: list[tuple[str, ConnectorSpec]] = [
        (to_s, spec)
        for (from_s, to_s), spec in transfer_config.connectors.items()
        if from_s == target_stage
    ]

    if not incoming_edges and not outgoing_edges:
        return {}

    if incoming_edges and outgoing_edges:
        effective_role = "dual"
    elif outgoing_edges:
        effective_role = "sender"
    else:
        effective_role = "receiver"

    upstream_stage = _primary_upstream_stage(incoming_edges)

    stage_connectors_config: dict[str, Any] = {}

    # Incoming edges -> this stage is at the ``to`` end.
    for from_s, spec in incoming_edges:
        extra = dict(spec.extra) if spec.extra else {}
        extra.setdefault("role", effective_role)
        _inject_chunk_path_endpoints(
            extra,
            connector_name=spec.name,
            role=effective_role,
            own_stage=target_stage,
            upstream_stage=upstream_stage,
        )
        stage_connectors_config[f"from_stage_{from_s}"] = {"spec": {"name": spec.name, "extra": extra}}

    # Outgoing edges -> this stage is at the ``from`` end.
    for to_s, spec in outgoing_edges:
        extra = dict(spec.extra) if spec.extra else {}
        extra.setdefault("role", effective_role)
        _inject_chunk_path_endpoints(
            extra,
            connector_name=spec.name,
            role=effective_role,
            own_stage=target_stage,
            upstream_stage=upstream_stage,
        )
        stage_connectors_config[f"to_stage_{to_s}"] = {"spec": {"name": spec.name, "extra": extra}}

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

    # Normalize new-schema (top-level ``connectors`` + ``stages``) into the
    # legacy ``runtime.connectors`` + ``stage_args`` shape the parser reads.
    if "stages" in config_dict and "stage_args" not in config_dict:
        normalized: dict[str, Any] = dict(config_dict)
        runtime = dict(normalized.get("runtime") or {})
        if "connectors" in normalized and "connectors" not in runtime:
            runtime["connectors"] = normalized["connectors"]
        if "edges" in normalized and "edges" not in runtime:
            runtime["edges"] = normalized["edges"]
        normalized["runtime"] = runtime
        normalized["stage_args"] = normalized["stages"]
        config_dict = normalized

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

        # Input connectors (this stage is the receiver)
        # NOTE: role is NOT injected here — the shared edge-level ConnectorSpec
        # must remain role-neutral.  Role is injected per-stage in
        # get_connectors_config_for_stage() / resolve_omni_kv_config_for_stage().
        for input_key, conn_ref in stage_config.get("input_connectors", {}).items():
            if isinstance(conn_ref, str):
                # Reference to global connector
                if conn_ref in global_connectors:
                    conn_config = global_connectors[conn_ref]
                    extra = dict(conn_config.get("extra", {}))
                else:
                    raise ValueError(f"Undefined connector reference: {conn_ref}")
                connector = ConnectorSpec(name=conn_config["name"], extra=extra)
            else:
                # Inline connector definition
                extra = dict(conn_ref.get("extra", {}))
                connector = ConnectorSpec(name=conn_ref["name"], extra=extra)

            # Parse from_stage from key (e.g., "from_stage_0" -> "0")
            from_stage = input_key.replace("from_stage_", "")
            edge_key = (from_stage, stage_id)
            # Both sides of an edge may define the same connector reference;
            # verify consistency if already registered.
            if edge_key in connectors:
                existing = connectors[edge_key]
                if existing.name != connector.name:
                    raise ValueError(
                        f"Connector type mismatch for edge {edge_key[0]}->{edge_key[1]}: "
                        f"previously registered as '{existing.name}', "
                        f"but input_connectors of stage {stage_id} specifies '{connector.name}'"
                    )
            else:
                connectors[edge_key] = connector
            expected_edges.add(edge_key)

        # Output connectors (this stage is the sender)
        for output_key, conn_ref in stage_config.get("output_connectors", {}).items():
            if isinstance(conn_ref, str):
                # Reference to global connector
                if conn_ref in global_connectors:
                    conn_config = global_connectors[conn_ref]
                    extra = dict(conn_config.get("extra", {}))
                else:
                    raise ValueError(f"Undefined connector reference: {conn_ref}")
                connector = ConnectorSpec(name=conn_config["name"], extra=extra)
            else:
                # Inline connector definition
                extra = dict(conn_ref.get("extra", {}))
                connector = ConnectorSpec(name=conn_ref["name"], extra=extra)

            # Parse to_stage from key (e.g., "to_stage_1" -> "1")
            to_stage = output_key.replace("to_stage_", "")
            edge_key = (stage_id, to_stage)
            if edge_key in connectors:
                existing = connectors[edge_key]
                if existing.name != connector.name:
                    raise ValueError(
                        f"Connector type mismatch for edge {edge_key[0]}->{edge_key[1]}: "
                        f"previously registered as '{existing.name}', "
                        f"but output_connectors of stage {stage_id} specifies '{connector.name}'"
                    )
            else:
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
        config_path,
        default_shm_threshold=default_shm_threshold,
        purpose="request_forwarding",
        caller_stage_id="orchestrator",
        is_sender=True,
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
    purpose: str = "request_forwarding",
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
    # Convert dictionary-formatted config to ConnectorSpec objects.
    # Only instantiate INPUT connectors ("from_stage_X") — the stage worker
    # only receives via connectors.  Output connectors are handled at the
    # orchestrator level (try_send_via_connector uses orchestrator connectors).
    stage_connector_specs = {}
    for key, config in connectors_config.items():
        if not key.startswith("from_stage_"):
            continue

        from_stage = key.replace("from_stage_", "")
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
        connectors = create_connectors_from_config(
            stage_connector_specs,
            purpose=purpose,
            caller_stage_id=stage_id,
            is_sender=False,
        )
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

    # Prioritize outgoing (Sender) if exists, else check incoming (Receiver).
    # Inject direction-specific role so the connector initializes correctly.
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
        omni_conn_cfg.setdefault("role", "sender")
        omni_from = stage_id_str
        omni_to = str(to_s)
    elif incoming:
        # For receiver, pick one incoming edge to configure the connector
        incoming.sort(key=lambda x: int(x[0]) if str(x[0]).isdigit() else str(x[0]))
        from_s, spec = incoming[0]
        omni_conn_cfg = {"type": spec.name, **(spec.extra or {})}
        omni_conn_cfg.setdefault("role", "receiver")
        omni_from = str(from_s)
        omni_to = stage_id_str

    return omni_conn_cfg, omni_from, omni_to
