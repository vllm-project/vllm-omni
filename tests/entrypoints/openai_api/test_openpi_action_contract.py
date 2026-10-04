# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import numpy as np
import pytest

from vllm_omni.entrypoints.openpi.action_contract import validate_action_output

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("shape", [(4, 3), (1, 4, 3), (2, 4, 3)])
def test_dense_chunk_contract(shape):
    metadata = {"horizon": 4, "action_dim": 3, "valid_steps": 2, "raw_action_dim": 32, "custom": "kept"}
    before = metadata.copy()
    actions = np.zeros(shape, dtype=np.float32)
    validate_action_output(actions, metadata, {"action_horizon": 4, "action_dim": 3})
    assert actions.shape == shape
    assert metadata == before


@pytest.mark.parametrize("shape", [(4, 2), (1, 4, 2), (2, 4, 2)])
def test_named_chunks_preserve_batch_layout_and_independent_dimensions(shape):
    actions = {"arm": np.zeros(shape), "gripper": np.zeros((*shape[:-1], 1))}
    validate_action_output(
        actions, {"horizon": 4, "valid_steps": 4}, {"action_horizon": 4, "action_keys": ["gripper", "arm"]}
    )
    assert actions["arm"].shape == shape


def test_maximum_horizon_is_an_upper_bound_and_default_is_not_fixed():
    validate_action_output(np.zeros((3, 2)), {"valid_steps": 0}, {"max_action_horizon": 4, "default_action_horizon": 4})


def test_legacy_output_without_metadata_or_shape_config():
    validate_action_output(np.array([0.0]), None, {"action_space": "joint_position"})
    validate_action_output(np.zeros((4, 2)), None, {})


@pytest.mark.parametrize(
    "actions,metadata,config,message",
    [
        (np.zeros((4, 3)), {"horizon": 3}, {}, "horizon"),
        (np.zeros((4, 3)), {}, {"action_horizon": 3}, "horizon"),
        (np.zeros((4, 3)), {}, {"max_action_horizon": 3}, "exceeds"),
        (np.zeros((4, 3)), {}, {"action_horizon": 4, "max_action_horizon": 3}, "exceeds"),
        (np.zeros((4, 3)), {"action_dim": 2}, {}, "dimension"),
        (np.zeros((4, 3)), {}, {"action_dim": 2}, "dimension"),
        (np.zeros((4, 3)), {"valid_steps": 5}, {}, "valid_steps"),
        (np.zeros((4, 3)), {"valid_steps": -1}, {}, "integer"),
        (np.zeros((4, 3)), {"horizon": True}, {}, "integer"),
        (np.zeros((4, 3)), {}, {"action_dim": 3.0}, "integer"),
        (np.zeros((4, 3)), [], {}, "mapping"),
        (np.zeros((4, 3)), {"action_space": "relative"}, {"action_space": "absolute"}, "action_space"),
        (np.zeros((4, 3)), {"action_space": ""}, {}, "action_space"),
        (np.zeros((0, 3)), {}, {}, "non-empty"),
        (np.array([[np.nan]]), {}, {}, "non-finite"),
        (np.array([[np.inf]]), {}, {}, "non-finite"),
        (np.array(1.0), {}, {}, "shape"),
        (np.zeros((1, 1, 4, 3)), {}, {}, "shape"),
        (np.zeros(3), {"horizon": 3}, {}, "shape"),
        ({}, {}, {}, "empty"),
        ({"arm": np.zeros((4, 3))}, {}, {"action_keys": ["gripper"]}, "keys"),
        ({"arm": np.zeros((4, 3))}, {}, {"action_keys": ["arm", "arm"]}, "keys"),
        ({"arm": np.zeros((4, 3))}, {}, {"action_keys": "arm"}, "list"),
        (np.zeros((4, 3)), {}, {"action_keys": ["arm"]}, "keys"),
        ({"arm": np.zeros((4, 3)), "gripper": np.zeros((3, 1))}, {}, {}, "share"),
        ({"arm": np.zeros((1, 4, 3)), "gripper": np.zeros((2, 4, 1))}, {}, {}, "share"),
        ({"arm": np.zeros((4, 3)), "gripper": np.zeros((1, 4, 1))}, {}, {}, "share"),
        ({"arm": np.zeros(3)}, {}, {}, "shape"),
        ({"arm": np.zeros((4, 3))}, {"action_dim": 3}, {}, "dense"),
    ],
)
def test_invalid_action_contract(actions, metadata, config, message):
    with pytest.raises(ValueError, match=message):
        validate_action_output(actions, metadata, config)
