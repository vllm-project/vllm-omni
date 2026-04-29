# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

import pytest

# Use the new import path for initialization utilities
from vllm_omni.distributed.omni_connectors.utils.config import ConnectorSpec, OmniTransferConfig
from vllm_omni.distributed.omni_connectors.utils.initialization import (
    _inject_chunk_path_endpoints,
    get_connectors_config_for_stage,
    load_omni_transfer_config,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def get_config_files():
    """Helper to find config files."""
    # Go up two levels from 'tests/distributed/omni_connectors' (approx) to 'vllm-omni' root
    # Adjust based on file location: vllm-omni/tests/distributed/omni_connectors/test_omni_connector_configs.py
    # This file is 4 levels deep from root if we count from tests?
    # vllm-omni/tests/distributed/omni_connectors -> parent -> distributed -> parent -> tests -> parent -> vllm-omni
    # Let's use resolve to be safe.

    # Path(__file__) = .../vllm-omni/tests/distributed/omni_connectors/test_omni_connector_configs.py
    # .parent = omni_connectors
    # .parent = distributed
    # .parent = tests
    # .parent = vllm-omni

    base_dir = Path(__file__).resolve().parent.parent.parent.parent
    config_dir = base_dir / "vllm_omni" / "model_executor" / "stage_configs"

    if not config_dir.exists():
        return []

    return list(config_dir.glob("qwen*.yaml"))


# Collect files at module level for parametrization
config_files = get_config_files()


@pytest.mark.skipif(len(config_files) == 0, reason="No config files found or directory missing")
@pytest.mark.parametrize("yaml_file", config_files, ids=lambda p: p.name)
def test_load_qwen_yaml_configs(yaml_file):
    """
    Scan and test loading of all qwen*.yaml config files.
    This ensures that existing stage configs are compatible with the OmniConnector system.
    """
    print(f"Testing config load: {yaml_file.name}")
    try:
        # Attempt to load the config
        # default_shm_threshold doesn't matter much for loading correctness, using default
        config = load_omni_transfer_config(yaml_file)

        assert config is not None, "Config should not be None"

        # Basic validation
        # Note: Some configs might not have 'runtime' or 'connectors' section if they rely on auto-shm
        # but the load function should succeed regardless.

        # If the config defines stages, we expect connectors to be populated (either explicit or auto SHM)
        # We can't strictly assert len(config.connectors) > 0 because a single stage pipeline might have 0 edges.

        print(f"  -> Successfully loaded. Connectors: {len(config.connectors)}")

    except Exception as e:
        pytest.fail(f"Failed to load config {yaml_file.name}: {e}")


# ---------------------------------------------------------------------------
# Framework-level per-stage role + endpoint derivation for the chunk
# transfer adapter path.
#
# ``get_connectors_config_for_stage`` is responsible for turning a
# role-neutral edge-level ConnectorSpec into a per-stage view where each
# stage carries:
#
#   * ``role=sender`` if the stage only has outgoing edges;
#   * ``role=receiver`` if the stage only has incoming edges;
#   * ``role=dual`` if the stage has both (middle stage in a 3+ stage
#     pipeline; e.g. the Qwen3-Omni-MoE talker stage).  Dual stages
#     emit ``from_stage_*`` and ``to_stage_*`` entries that share the
#     same composite extra so downstream flattening (engine-side
#     ``get_stage_connector_spec`` returning the first spec) always
#     recovers a self-consistent config.
#
# For role-bound ZMQ connectors (Mori / Mooncake) the function also
# pre-computes ``zmq_port`` / ``sender_host`` / ``sender_zmq_port`` so
# an intranode pipeline can come up without an external handshake.
#
# The orchestrator-level path (``create_connectors_from_config``) has
# its own Mooncake-specific port adjustment and is intentionally NOT
# exercised here.
# ---------------------------------------------------------------------------


def _linear_pipeline_config(
    connector_name: str,
    extra: dict | None = None,
    edges: tuple[tuple[str, str], ...] = (("0", "1"), ("1", "2")),
) -> OmniTransferConfig:
    shared_extra = dict(extra or {})
    specs = {edge: ConnectorSpec(name=connector_name, extra=dict(shared_extra)) for edge in edges}
    return OmniTransferConfig(connectors=specs)


@pytest.fixture
def _stable_local_ip(monkeypatch: pytest.MonkeyPatch) -> str:
    """Pin framework-level local-IP detection so endpoint assertions are deterministic."""
    import vllm_omni.distributed.omni_connectors.utils.initialization as init_mod

    monkeypatch.setattr(init_mod, "_detect_local_ip", lambda: "10.20.30.40")
    return "10.20.30.40"


@pytest.mark.parametrize(
    "connector_name",
    ["MoriTransferEngineConnector", "MooncakeTransferEngineConnector"],
)
def test_stage_0_is_sender_only(connector_name, _stable_local_ip):
    """Stage 0 has only outgoing edges → role=sender, listener port = base + 0."""
    cfg = _linear_pipeline_config(connector_name, extra={"zmq_port": 50051, "host": "auto"})

    stage0 = get_connectors_config_for_stage(cfg, 0)

    assert list(stage0.keys()) == ["to_stage_1"], "Sender-only stage should emit only to_stage_*"
    extra = stage0["to_stage_1"]["spec"]["extra"]
    assert extra["role"] == "sender"
    assert extra["zmq_port"] == 50051
    assert "sender_host" not in extra
    assert "sender_zmq_port" not in extra


@pytest.mark.parametrize(
    "connector_name",
    ["MoriTransferEngineConnector", "MooncakeTransferEngineConnector"],
)
def test_final_stage_is_receiver_only(connector_name, _stable_local_ip):
    """Stage 2 has only incoming edges → role=receiver, points at upstream sender."""
    cfg = _linear_pipeline_config(connector_name, extra={"zmq_port": 50051, "host": "auto"})

    stage2 = get_connectors_config_for_stage(cfg, 2)

    assert list(stage2.keys()) == ["from_stage_1"], "Receiver-only stage should emit only from_stage_*"
    extra = stage2["from_stage_1"]["spec"]["extra"]
    assert extra["role"] == "receiver"
    assert extra["sender_zmq_port"] == 50052  # base + upstream stage id (1)
    assert extra["sender_host"] == "10.20.30.40"


@pytest.mark.parametrize(
    "connector_name",
    ["MoriTransferEngineConnector", "MooncakeTransferEngineConnector"],
)
def test_middle_stage_is_dual(connector_name, _stable_local_ip):
    """Middle stage has both → role=dual, both entries share composite spec."""
    cfg = _linear_pipeline_config(connector_name, extra={"zmq_port": 50051, "host": "auto"})

    stage1 = get_connectors_config_for_stage(cfg, 1)

    # Both directions exposed, so whichever one get_stage_connector_spec
    # (which does "return first") picks, it recovers the same dual spec.
    assert set(stage1.keys()) == {"from_stage_0", "to_stage_2"}
    incoming = stage1["from_stage_0"]["spec"]["extra"]
    outgoing = stage1["to_stage_2"]["spec"]["extra"]

    for extra in (incoming, outgoing):
        assert extra["role"] == "dual"
        assert extra["zmq_port"] == 50052  # this stage's listener, base + own stage id (1)
        assert extra["sender_host"] == "10.20.30.40"
        assert extra["sender_zmq_port"] == 50051  # upstream sender at base + 0

    # Composite must be identical so order of iteration does not matter.
    assert incoming == outgoing


def test_shm_connector_is_untouched_by_endpoint_injection(_stable_local_ip):
    """SharedMemoryConnector is not role-bound -> no port/host fields appear."""
    cfg = _linear_pipeline_config(
        "SharedMemoryConnector",
        extra={"shm_threshold_bytes": 65536},
    )

    for sid in (0, 1, 2):
        stage_cfg = get_connectors_config_for_stage(cfg, sid)
        for entry in stage_cfg.values():
            extra = entry["spec"]["extra"]
            assert "zmq_port" not in extra
            assert "sender_host" not in extra
            assert "sender_zmq_port" not in extra
            assert extra["shm_threshold_bytes"] == 65536
            # Role is still injected (passive connectors ignore it, which
            # is fine -- the string is just metadata to them).
            assert extra["role"] in {"sender", "receiver", "dual"}


def test_explicit_sender_host_and_port_override_win(_stable_local_ip):
    """User-provided ``sender_host`` / ``sender_zmq_port`` beat framework derivation."""
    cfg = _linear_pipeline_config(
        "MoriTransferEngineConnector",
        extra={
            "zmq_port": 50051,
            "host": "auto",
            "sender_host": "192.168.1.10",
            "sender_zmq_port": 60000,
        },
    )

    stage1 = get_connectors_config_for_stage(cfg, 1)
    recv_extra = stage1["from_stage_0"]["spec"]["extra"]
    assert recv_extra["sender_host"] == "192.168.1.10"
    assert recv_extra["sender_zmq_port"] == 60000

    # Sender-side zmq_port still offsets so co-located stage listeners do
    # not collide; the override is the upstream peer's address, not this
    # stage's own bind port.
    assert recv_extra["zmq_port"] == 50052


def test_explicit_non_auto_host_cascades_to_sender_host(_stable_local_ip):
    """Non-auto ``host`` is reused as ``sender_host`` for the receiver side."""
    cfg = _linear_pipeline_config(
        "MoriTransferEngineConnector",
        extra={"zmq_port": 50051, "host": "172.16.0.5"},
    )

    stage1 = get_connectors_config_for_stage(cfg, 1)
    recv_extra = stage1["from_stage_0"]["spec"]["extra"]
    assert recv_extra["sender_host"] == "172.16.0.5"
    assert recv_extra["sender_zmq_port"] == 50051


def test_inject_helper_is_noop_for_unknown_connector(_stable_local_ip):
    """Non-role-bound connectors are left untouched by the helper."""
    extra: dict = {"zmq_port": 50051, "host": "auto"}
    _inject_chunk_path_endpoints(
        extra,
        connector_name="SomeFutureConnector",
        role="dual",
        own_stage="1",
        upstream_stage="0",
    )
    assert extra == {"zmq_port": 50051, "host": "auto"}


def test_inject_helper_is_noop_for_non_integer_stage(_stable_local_ip):
    """Non-integer pipeline keys (e.g. ``"prefill"``) short-circuit safely."""
    extra: dict = {"zmq_port": 50051, "host": "auto"}
    _inject_chunk_path_endpoints(
        extra,
        connector_name="MoriTransferEngineConnector",
        role="sender",
        own_stage="prefill",
        upstream_stage=None,
    )
    assert extra == {"zmq_port": 50051, "host": "auto"}


def test_get_connectors_config_for_stage_none_transfer_config():
    """Callers pass ``None`` when no yaml was loaded; return an empty dict."""
    assert get_connectors_config_for_stage(None, 0) == {}
