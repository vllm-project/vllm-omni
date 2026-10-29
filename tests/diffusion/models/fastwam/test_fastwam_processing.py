# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""FastWAM observation parsing, image layout, normalisation and LIBERO action convention."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from vllm_omni.diffusion.models.fastwam.processing_fastwam import (
    MinMaxNormalizer,
    build_image,
    center_crop_resize,
    load_stats,
    parse_observation,
    postprocess_actions,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _obs(**overrides):
    obs = {
        "observation/image": np.zeros((16, 16, 3), dtype=np.uint8),
        "observation/wrist_image": np.full((16, 16, 3), 255, dtype=np.uint8),
        "observation/state": np.arange(8, dtype=np.float64),
        "prompt": "pick up the bowl",
    }
    obs.update(overrides)
    return obs


def test_parse_observation_reads_openpi_libero_keys():
    obs = parse_observation(_obs(task_suite="libero_goal"), proprio_dim=8, num_cameras=2, default_suite="x")
    assert obs.task == "pick up the bowl" and obs.suite == "libero_goal"
    assert obs.state.dtype == np.float32 and obs.state.tolist() == list(range(8))
    assert len(obs.images) == 2
    assert parse_observation(_obs(), proprio_dim=8, num_cameras=2, default_suite="d").suite == "d"
    # No silent default: the policy decides whether a missing suite is an error.
    assert parse_observation(_obs(), proprio_dim=8, num_cameras=2, default_suite=None).suite is None


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"prompt": ""}, "prompt"),
        ({"observation/state": np.zeros(7)}, "8 values"),
        ({"observation/image": np.zeros((16, 16, 3), dtype=np.float32)}, "uint8"),
        ({"observation/wrist_image": np.zeros((16, 16), dtype=np.uint8)}, "uint8"),
        ({"task_suite": 3}, "string"),
    ],
)
def test_parse_observation_rejects_bad_inputs(overrides, match):
    with pytest.raises(ValueError, match=match):
        parse_observation(_obs(**overrides), proprio_dim=8, num_cameras=2, default_suite="d")


def test_parse_observation_lists_missing_keys():
    obs = _obs()
    del obs["observation/wrist_image"]
    with pytest.raises(ValueError, match="observation/wrist_image"):
        parse_observation(obs, proprio_dim=8, num_cameras=2, default_suite="d")


def test_center_crop_resize_covers_then_crops():
    image = np.zeros((20, 40, 3), dtype=np.uint8)
    image[:, 5:35] = 200  # covers the cropped centre plus the bilinear support
    out = center_crop_resize(image, width=8, height=8)
    assert out.shape == (8, 8, 3) and out.dtype == np.uint8
    assert np.all(out == 200)
    same = np.random.RandomState(0).randint(0, 256, (8, 8, 3), dtype=np.uint8)
    assert np.array_equal(center_crop_resize(same, 8, 8), same)


def test_build_image_concatenates_cameras_left_to_right():
    left = np.zeros((16, 16, 3), dtype=np.uint8)
    right = np.full((16, 16, 3), 255, dtype=np.uint8)
    image = build_image((left, right), (8, 8))
    assert image.shape == (1, 3, 8, 16) and image.dtype == torch.uint8
    assert int(image[..., :8].max()) == 0 and int(image[..., 8:].min()) == 255


def test_minmax_normalizer_matches_reference_semantics():
    norm = MinMaxNormalizer([0.0, -2.0, 1.0], [2.0, 2.0, 1.0])  # last dim is constant
    x = torch.tensor([[1.0, 2.0, 1.0], [10.0, -2.0, 3.0]])
    y = norm.normalize(x)
    assert y[0].tolist() == [0.0, 1.0, 0.0]
    assert y[1, 0].item() == 5.0  # clamped to [-5, 5]
    assert y[1, 2].item() == 2.0  # constant dim is only centred
    torch.testing.assert_close(norm.denormalize(norm.normalize(x[:1])), x[:1])


def test_load_stats_reads_global_min_max(tmp_path):
    path = tmp_path / "libero_spatial.json"
    stats = {
        "state": {"default": {"global_min": [0.0] * 8, "global_max": [2.0] * 8, "global_mean": [1.0] * 8}},
        "action": {"default": {"global_min": [-1.0] * 7, "global_max": [1.0] * 7}},
        "num_episodes": 1,
    }
    path.write_text(json.dumps(stats))
    loaded = load_stats(str(path), action_dim=7, proprio_dim=8)
    assert loaded.state.normalize(torch.ones(1, 8)).abs().max().item() == 0.0
    with pytest.raises(ValueError, match="expected 8/6"):
        load_stats(str(path), action_dim=6, proprio_dim=8)


def test_libero_postprocess_maps_and_binarises_gripper():
    actions = np.array([[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.9], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2]], dtype=np.float32)
    out = postprocess_actions(actions, "libero")
    # dataset 0..1 (1 = open) -> LIBERO -1 open / +1 close
    assert out[:, -1].tolist() == [-1.0, 1.0]
    assert np.array_equal(out[:, :-1], actions[:, :-1])
    assert np.array_equal(postprocess_actions(actions, "none"), actions)
    with pytest.raises(ValueError, match="action_postprocess"):
        postprocess_actions(actions, "bogus")
