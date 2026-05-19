# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""F11: GR00T-N1.7 online-serving smoke test.

CI-friendly: we exercise the websocket-handling logic without a live engine
by stubbing the policy server.  The real end-to-end run is documented in
``examples/online_serving/gr00t/README.md`` and requires GPUs + the
GR00T-N1.7-3B checkpoint.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


REPO_ROOT = Path(__file__).resolve().parents[3]
DEPLOY_YAML = REPO_ROOT / "vllm_omni" / "deploy" / "gr00t.yaml"
EXAMPLE_CLIENT = (
    REPO_ROOT / "examples" / "online_serving" / "gr00t" / "openpi_client.py"
)


def test_example_client_exists_and_is_importable_smoke():
    """The example script must be present and at minimum syntactically valid
    Python — guards against trivial bitrot in the example slice."""
    assert EXAMPLE_CLIENT.exists()
    source = EXAMPLE_CLIENT.read_text()
    compile(source, str(EXAMPLE_CLIENT), "exec")


def test_example_run_server_script_present():
    run_server = REPO_ROOT / "examples" / "online_serving" / "gr00t" / "run_server.sh"
    assert run_server.exists()
    assert run_server.stat().st_mode & 0o111, "run_server.sh must be executable"
    text = run_server.read_text()
    assert "vllm_omni/deploy/gr00t.yaml" in text
    assert "--omni" in text


def test_deploy_yaml_resolves_in_pipeline_registry():
    """Booting through the deploy YAML requires the model_type "gr00t" to be
    resolvable in the central pipeline registry."""
    from vllm_omni.config.pipeline_registry import _OMNI_PIPELINES

    assert "gr00t" in _OMNI_PIPELINES


def test_serving_returns_dict_actions_through_full_extract_path():
    """Stub the engine; verify the OpenPI serving + extraction path passes
    dict-typed actions through unchanged so the websocket layer sends the
    GR00T per-action-key shape on the wire."""
    try:
        from vllm_omni.entrypoints.openai.realtime.robot.openpi_serving import (
            ServingRealtimeRobotOpenPI,
        )
    except ImportError as exc:  # pragma: no cover — env-only fallback
        pytest.skip(f"openpi_serving unavailable in this env: {exc}")

    serving = ServingRealtimeRobotOpenPI.__new__(ServingRealtimeRobotOpenPI)

    actions_dict = {
        "eef_9d": np.zeros((40, 9), dtype=np.float64),
        "gripper_position": np.linspace(0, 1, 40, dtype=np.float64).reshape(40, 1),
        "joint_position": np.arange(40 * 7, dtype=np.float64).reshape(40, 7),
    }
    fake_result = SimpleNamespace(multimodal_output={"actions": actions_dict})
    out = serving._extract_actions(fake_result)
    assert isinstance(out, dict)
    assert set(out.keys()) == {"eef_9d", "gripper_position", "joint_position"}
    for v in out.values():
        assert v.dtype == np.float32


def test_openpi_protocol_msgpack_roundtrip_dict_actions():
    """The OpenPI wire format must round-trip the dict shape so the example
    client at examples/online_serving/gr00t/openpi_client.py can decode
    server responses as a dict."""
    try:
        from openpi_client import msgpack_numpy
    except ImportError:
        pytest.skip("openpi-client not installed")

    actions = {
        "eef_9d": np.zeros((40, 9), dtype=np.float32),
        "joint_position": np.arange(40 * 7, dtype=np.float32).reshape(40, 7),
    }
    blob = msgpack_numpy.packb(actions)
    decoded = msgpack_numpy.unpackb(blob)
    assert set(decoded.keys()) == set(actions.keys())
    for key, value in decoded.items():
        np.testing.assert_array_equal(value, actions[key])
