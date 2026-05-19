# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""F7: OpenPI serving accepts dict-typed actions for GR00T.

Validates that ``ServingRealtimeRobotOpenPI._extract_actions`` passes a
``dict[str, np.ndarray]`` through (per-action-key OpenPI response shape)
without breaking the existing single-ndarray contract DreamZero relies on.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vllm_omni.entrypoints.openai.realtime.robot.openpi_serving import (
    ServingRealtimeRobotOpenPI,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _build_serving_stub() -> ServingRealtimeRobotOpenPI:
    """Bypass __init__ so we don't need a live engine / policy_server_config
    to exercise the extraction method."""
    return ServingRealtimeRobotOpenPI.__new__(ServingRealtimeRobotOpenPI)


def _result_with_actions(actions):
    return SimpleNamespace(multimodal_output={"actions": actions})


def test_extract_actions_ndarray_passes_through_unchanged():
    serving = _build_serving_stub()
    arr = np.arange(24 * 8, dtype=np.float64).reshape(24, 8)
    out = serving._extract_actions(_result_with_actions(arr))
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, arr.astype(np.float32))


def test_extract_actions_dict_passes_through_as_dict():
    serving = _build_serving_stub()
    actions_dict = {
        "joint_position": np.arange(40 * 7, dtype=np.float64).reshape(40, 7),
        "gripper_position": np.linspace(0, 1, 40, dtype=np.float64).reshape(40, 1),
        "eef_9d": np.zeros((40, 9), dtype=np.float64),
    }
    out = serving._extract_actions(_result_with_actions(actions_dict))
    assert isinstance(out, dict)
    assert set(out.keys()) == {"joint_position", "gripper_position", "eef_9d"}
    for key, value in out.items():
        assert isinstance(value, np.ndarray)
        assert value.dtype == np.float32
        np.testing.assert_array_equal(value, actions_dict[key].astype(np.float32))


def test_extract_actions_dict_rejects_empty():
    serving = _build_serving_stub()
    with pytest.raises(RuntimeError, match="empty dict"):
        serving._extract_actions(_result_with_actions({}))


def test_extract_actions_missing_actions_key_raises():
    serving = _build_serving_stub()
    result = SimpleNamespace(multimodal_output={})
    with pytest.raises(RuntimeError, match="Missing multimodal_output\\['actions'\\]"):
        serving._extract_actions(result)


def test_extract_actions_handles_iterable_result():
    """The serving path treats the engine response as an iterable and uses
    the first element; this preserves DreamZero's existing path."""
    serving = _build_serving_stub()
    payload = _result_with_actions(np.zeros((24, 8), dtype=np.float32))
    out = serving._extract_actions([payload])
    assert isinstance(out, np.ndarray)
    assert out.shape == (24, 8)


def test_msgpack_roundtrip_for_dict_actions():
    """openpi-client msgpack_numpy must round-trip the dict shape so the
    websocket layer doesn't need any extra encoding hook."""
    try:
        from openpi_client import msgpack_numpy
    except ImportError:
        pytest.skip("openpi-client not installed")

    actions = {
        "joint_position": np.arange(28, dtype=np.float32).reshape(4, 7),
        "gripper_position": np.linspace(0, 1, 4, dtype=np.float32).reshape(4, 1),
    }
    blob = msgpack_numpy.packb(actions)
    restored = msgpack_numpy.unpackb(blob)
    assert set(restored.keys()) == set(actions.keys())
    for key, value in restored.items():
        np.testing.assert_array_equal(value, actions[key])
