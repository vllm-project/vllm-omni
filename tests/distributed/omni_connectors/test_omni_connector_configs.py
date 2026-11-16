# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path
from types import SimpleNamespace

import pytest

# Use the new import path for initialization utilities
from vllm_omni.distributed.omni_connectors.utils.config import (
    stage_receives_chunks,
    stage_sends_async_output,
)
from vllm_omni.distributed.omni_connectors.utils.initialization import load_omni_transfer_config

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


@pytest.mark.parametrize(
    ("role", "stage_id", "receives", "sends"),
    [
        ("sender", 0, False, True),
        ("receiver", 1, True, False),
        (None, 0, True, False),
        (None, None, True, True),
    ],
)
def test_stage_chunk_direction_helpers(role, stage_id, receives, sends):
    extra = {} if role is None else {"role": role}
    model_config = SimpleNamespace(
        stage_id=stage_id,
        stage_connector_config={"extra": extra},
    )

    assert stage_receives_chunks(model_config) is receives
    assert stage_sends_async_output(model_config) is sends


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


def test_receiver_auto_sender_host_resolves_to_local_address():
    from vllm_omni.distributed.omni_connectors.utils.config import ConnectorSpec
    from vllm_omni.distributed.omni_connectors.utils.initialization import (
        resolve_connector_spec,
    )

    spec = ConnectorSpec(
        name="NixlConnector",
        extra={"host": "auto", "zmq_port": 50061, "from_stage": 1},
    )
    resolved = resolve_connector_spec(spec, stage_id=2, role="receiver")

    assert resolved.extra["sender_host"] not in {"auto", "0.0.0.0", "*", "::"}
    assert resolved.extra["sender_zmq_port"] == 50062


def test_receiver_explicit_sender_host_is_preserved():
    from vllm_omni.distributed.omni_connectors.utils.config import ConnectorSpec
    from vllm_omni.distributed.omni_connectors.utils.initialization import (
        resolve_connector_spec,
    )

    spec = ConnectorSpec(
        name="NixlConnector",
        extra={
            "host": "auto",
            "sender_host": "10.1.2.3",
            "zmq_port": 50061,
            "from_stage": 1,
        },
    )
    resolved = resolve_connector_spec(spec, stage_id=2, role="receiver")

    assert resolved.extra["sender_host"] == "10.1.2.3"
    assert resolved.extra["sender_zmq_port"] == 50062


def test_receiver_preserves_concrete_host_without_sender_host():
    from vllm_omni.distributed.omni_connectors.utils.config import ConnectorSpec
    from vllm_omni.distributed.omni_connectors.utils.initialization import (
        resolve_connector_spec,
    )

    spec = ConnectorSpec(
        name="NixlConnector",
        extra={"host": "192.0.2.83", "zmq_port": 50061, "from_stage": 1},
    )
    resolved = resolve_connector_spec(spec, stage_id=2, role="receiver")

    assert resolved.extra["sender_host"] == "192.0.2.83"


def test_receiver_normalizes_explicit_auto_sender_host():
    from vllm_omni.distributed.omni_connectors.utils.config import ConnectorSpec
    from vllm_omni.distributed.omni_connectors.utils.initialization import (
        resolve_connector_spec,
    )

    spec = ConnectorSpec(
        name="NixlConnector",
        extra={
            "host": "auto",
            "sender_host": "auto",
            "zmq_port": 50061,
            "from_stage": 1,
        },
    )
    resolved = resolve_connector_spec(spec, stage_id=2, role="receiver")

    assert resolved.extra["sender_host"] != "auto"


def test_receiver_auto_uses_producer_route_address(monkeypatch):
    from vllm_omni.distributed.omni_connectors.utils import initialization
    from vllm_omni.distributed.omni_connectors.utils.config import ConnectorSpec

    monkeypatch.setattr(initialization, "_resolve_local_host", lambda: "10.0.0.7")
    spec = ConnectorSpec(
        name="NixlConnector",
        extra={"host": "auto", "zmq_port": 50061, "from_stage": 1},
    )
    resolved = initialization.resolve_connector_spec(spec, stage_id=2, role="receiver")

    assert resolved.extra["sender_host"] == "10.0.0.7"
