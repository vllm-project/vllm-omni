# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

# Use the new import path for initialization utilities
from vllm_omni.distributed.omni_connectors.utils.config import (
    stage_receives_chunks,
    stage_sends_async_output,
)
from vllm_omni.distributed.omni_connectors.utils.initialization import (
    get_connectors_config_for_stage,
    load_omni_transfer_config,
    resolve_omni_kv_config_for_stage,
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


def _duplicate_edge_config(
    output_extra: dict | None = None,
    input_extra: dict | None = None,
    *,
    schema: str = "new",
    output_name: str = "TestConnector",
    input_name: str = "TestConnector",
) -> dict:
    """Build a two-sided config for one logical edge."""
    output = {"name": output_name, "extra": output_extra or {}}
    incoming = {"name": input_name, "extra": input_extra or {}}
    if schema == "legacy":
        return {
            "runtime": {"connectors": {}},
            "stage_args": [
                {"stage_id": 0, "output_connectors": {"to_stage_1": output}},
                {"stage_id": 1, "input_connectors": {"from_stage_0": incoming}},
            ],
        }
    return {
        "connectors": {},
        "stages": [
            {"stage_id": 0, "output_connectors": {"to_stage_1": output}},
            {"stage_id": 1, "input_connectors": {"from_stage_0": incoming}},
        ],
    }


def test_duplicate_edge_with_equal_specs_is_registered_once():
    config_dict = _duplicate_edge_config(
        output_extra={"connector_get_max_wait": 300},
        input_extra={"connector_get_max_wait": 300},
    )

    config = load_omni_transfer_config(config_dict=config_dict)

    assert config is not None
    assert list(config.connectors) == [("0", "1")]
    assert config.connectors[("0", "1")].extra == {"connector_get_max_wait": 300}


def test_duplicate_edge_equal_specs_are_order_independent():
    config_dict = _duplicate_edge_config(
        output_extra={"connector_get_max_wait": 300},
        input_extra={"connector_get_max_wait": 300},
    )
    reversed_config_dict = deepcopy(config_dict)
    reversed_config_dict["stages"].reverse()

    first = load_omni_transfer_config(config_dict=config_dict)
    second = load_omni_transfer_config(config_dict=reversed_config_dict)

    assert first is not None and second is not None
    assert first.connectors == second.connectors


def test_duplicate_edge_equal_global_and_inline_specs_are_accepted():
    config_dict = {
        "connectors": {
            "shared": {"name": "TestConnector", "extra": {"buffer_size": 4096}},
        },
        "stages": [
            {"stage_id": 0, "output_connectors": {"to_stage_1": "shared"}},
            {
                "stage_id": 1,
                "input_connectors": {"from_stage_0": {"name": "TestConnector", "extra": {"buffer_size": 4096}}},
            },
        ],
    }

    config = load_omni_transfer_config(config_dict=config_dict)

    assert config is not None
    assert config.connectors[("0", "1")].extra == {"buffer_size": 4096}


@pytest.mark.parametrize("schema", ["new", "legacy"])
@pytest.mark.parametrize("reverse_order", [False, True], ids=["output-first", "input-first"])
def test_duplicate_edge_with_different_extra_fails_fast(schema, reverse_order):
    config_dict = _duplicate_edge_config(
        output_extra={"connector_get_max_wait": 300, "role": "sender"},
        input_extra={"connector_get_max_wait": 600, "role": "receiver"},
        schema=schema,
    )
    if reverse_order:
        config_dict["stages" if schema == "new" else "stage_args"].reverse()

    with pytest.raises(ValueError, match=r"Conflicting connector options for edge 0->1.*connector_get_max_wait") as exc:
        load_omni_transfer_config(config_dict=config_dict)

    message = str(exc.value)
    assert "output_connectors of stage 0 (to_stage_1)" in message
    assert "input_connectors of stage 1 (from_stage_0)" in message
    assert "role" not in message
    assert "300" not in message
    assert "600" not in message


@pytest.mark.parametrize("schema", ["new", "legacy"])
def test_duplicate_edge_with_different_connector_name_fails_fast(schema):
    config_dict = _duplicate_edge_config(
        output_name="FirstConnector",
        input_name="SecondConnector",
        schema=schema,
    )

    with pytest.raises(ValueError, match=r"Connector type mismatch for edge 0->1"):
        load_omni_transfer_config(config_dict=config_dict)


def test_duplicate_edge_with_nested_extra_difference_fails_fast():
    config_dict = _duplicate_edge_config(
        output_extra={"transport": {"host": "sender-a", "port": 50051}},
        input_extra={"transport": {"host": "sender-b", "port": 50051}},
    )

    with pytest.raises(ValueError, match=r"Conflicting connector options for edge 0->1.*transport"):
        load_omni_transfer_config(config_dict=config_dict)


@pytest.mark.parametrize("schema", ["new", "legacy"])
@pytest.mark.parametrize(
    ("output_role", "input_role", "reverse_order", "expected_role"),
    [
        pytest.param("sender", "receiver", False, "sender", id="both-output-first"),
        pytest.param("sender", "receiver", True, "receiver", id="both-input-first"),
        pytest.param("sender", None, False, "sender", id="output-role-output-first"),
        pytest.param("sender", None, True, None, id="output-role-input-first"),
        pytest.param(None, "receiver", False, None, id="input-role-output-first"),
        pytest.param(None, "receiver", True, "receiver", id="input-role-input-first"),
        pytest.param(None, None, False, None, id="no-role-output-first"),
        pytest.param(None, None, True, None, id="no-role-input-first"),
    ],
)
def test_duplicate_edge_preserves_first_role(schema, output_role, input_role, reverse_order, expected_role):
    config_dict = _duplicate_edge_config(
        output_extra={} if output_role is None else {"role": output_role},
        input_extra={} if input_role is None else {"role": input_role},
        schema=schema,
    )
    if reverse_order:
        config_dict["stages" if schema == "new" else "stage_args"].reverse()

    config = load_omni_transfer_config(config_dict=config_dict)

    assert config is not None
    expected_extra = {} if expected_role is None else {"role": expected_role}
    assert list(config.connectors) == [("0", "1")]
    assert config.connectors[("0", "1")].extra == expected_extra

    for stage_id, direction, connector_key in [(0, "sender", "to_stage_1"), (1, "receiver", "from_stage_0")]:
        resolved, _, _ = resolve_omni_kv_config_for_stage(config, stage_id)
        assert resolved is not None
        assert resolved["role"] == (expected_role or direction)
        stage_config = get_connectors_config_for_stage(config, stage_id)
        assert stage_config[connector_key]["spec"]["extra"]["role"] == (expected_role or direction)

    assert config.connectors[("0", "1")].extra == expected_extra


@pytest.mark.parametrize(
    ("stage_key", "stage_id", "expected_role"),
    [("output_connectors", 0, "receiver"), ("input_connectors", 1, "sender")],
)
def test_explicit_role_is_preserved_for_single_sided_edge(stage_key, stage_id, expected_role):
    stage_args: list[dict[str, object]] = []
    config_dict = {"runtime": {"connectors": {}}, "stage_args": stage_args}
    if stage_key == "output_connectors":
        stage_args.append(
            {
                "stage_id": 0,
                "output_connectors": {"to_stage_1": {"name": "TestConnector", "extra": {"role": expected_role}}},
            }
        )
    else:
        stage_args.append(
            {
                "stage_id": 1,
                "input_connectors": {"from_stage_0": {"name": "TestConnector", "extra": {"role": expected_role}}},
            }
        )

    config = load_omni_transfer_config(config_dict=config_dict)

    assert config is not None
    if stage_key == "output_connectors":
        resolved, _, _ = resolve_omni_kv_config_for_stage(config, stage_id)
        assert resolved is not None
        assert resolved["role"] == expected_role
    else:
        resolved = get_connectors_config_for_stage(config, stage_id)
        assert resolved["from_stage_0"]["spec"]["extra"]["role"] == expected_role


def test_duplicate_edge_with_different_wakeup_scope_fails_fast():
    config_dict = _duplicate_edge_config(
        output_extra={"wakeup_scope": "deployment-a"},
        input_extra={"wakeup_scope": "deployment-b"},
    )

    with pytest.raises(ValueError, match=r"Conflicting connector options for edge 0->1.*wakeup_scope"):
        load_omni_transfer_config(config_dict=config_dict)


def test_duplicate_edge_does_not_mutate_config_dict():
    config_dict = _duplicate_edge_config(
        output_extra={"transport": {"host": "sender", "port": 50051}},
        input_extra={"transport": {"host": "sender", "port": 50051}},
    )
    original = deepcopy(config_dict)

    load_omni_transfer_config(config_dict=config_dict)

    assert config_dict == original


@pytest.mark.parametrize("schema", ["new", "legacy"])
@pytest.mark.parametrize("reverse_order", [False, True], ids=["output-first", "input-first"])
def test_duplicate_edge_missing_and_none_extra_values_conflict(schema, reverse_order):
    config_dict = _duplicate_edge_config(
        output_extra={},
        input_extra={"wakeup_scope": None},
        schema=schema,
    )
    if reverse_order:
        config_dict["stages" if schema == "new" else "stage_args"].reverse()

    with pytest.raises(ValueError, match=r"Conflicting connector options for edge 0->1.*wakeup_scope"):
        load_omni_transfer_config(config_dict=config_dict)


@pytest.mark.parametrize("schema", ["new", "legacy"])
def test_duplicate_edge_equal_specs_support_both_schemas(schema):
    config = load_omni_transfer_config(
        config_dict=_duplicate_edge_config(
            output_extra={"codec_chunk_frames": 25},
            input_extra={"codec_chunk_frames": 25},
            schema=schema,
        )
    )

    assert config is not None
    assert config.connectors[("0", "1")].extra == {"codec_chunk_frames": 25}


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
