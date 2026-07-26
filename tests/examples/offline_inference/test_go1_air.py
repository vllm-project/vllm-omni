# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Offline smoke coverage for GO-1-Air."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.go1_air.config import Go1AirConfig
from vllm_omni.diffusion.models.go1_air.model_go1_air import scaled_dot_product
from vllm_omni.diffusion.models.go1_air.pipeline_go1_air import (
    build_go1_air_batch_inputs_from_robot_obs,
    get_go1_air_post_process_func,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_SCRIPT = REPO_ROOT / "examples" / "offline_inference" / "go1_air" / "smoke.py"


def _tiny_model_config() -> dict:
    return {
        "action_chunk_size": 2,
        "force_image_size": 28,
        "downsample_ratio": 0.5,
        "img_context_token_id": 3,
        "pad_token_id": 0,
        "vision_config": {
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "patch_size": 14,
            "image_size": 28,
        },
        "llm_config": {
            "vocab_size": 16,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "max_position_embeddings": 64,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
        },
        "action_config": {
            "hidden_size": 8,
            "input_hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "max_position_embeddings": 64,
            "action_dim": 4,
            "state_dim": 4,
            "state_token_num": 1,
            "pad_token_id": 0,
        },
        "noise_scheduler_config": {
            "num_inference_timesteps": 1,
            "num_train_timesteps": 10,
            "beta_schedule": "squaredcos_cap_v2",
            "prediction_type": "sample",
        },
    }


def test_go1_air_offline_smoke_tiny_config() -> None:
    env = os.environ.copy()
    env.pop("GO1_AIR_MODEL_DIR", None)
    result = subprocess.run(
        [
            sys.executable,
            str(EXAMPLE_SCRIPT),
            "--tiny-config",
            "--device",
            "cpu",
            "--dtype",
            "float32",
        ],
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )

    assert "[smoke] OK action shape=(1, 2, 4)" in result.stdout


def test_go1_air_openpi_obs_maps_to_actions_output() -> None:
    config = Go1AirConfig.from_model_config(_tiny_model_config())
    robot_obs = {
        "prompt": "pick up the red block",
        "observation/joint_position": np.array([1.0, 2.0], dtype=np.float32),
        "observation/gripper_position": np.array([0.5], dtype=np.float32),
        "observation/exterior_image_0_left": np.zeros((14, 14, 3), dtype=np.uint8),
    }

    batch = build_go1_air_batch_inputs_from_robot_obs(
        robot_obs,
        config=config,
        device="cpu",
        dtype=torch.float32,
    )

    assert batch["observation.state"].shape == (1, 4)
    assert batch["observation.task"] == ["pick up the red block"]
    assert batch["observation.images.image0"].shape == (1, 1, 3, 28, 28)
    assert batch["observation.images.image0_mask"].shape == (1, 1)

    postprocess = get_go1_air_post_process_func(OmniDiffusionConfig(model_class_name="Go1AirPipeline"))
    actions = torch.zeros((1, config.chunk_size, config.max_action_dim))

    result = postprocess(actions)
    assert set(result) == {"actions", "video"}
    assert result["actions"] is actions
    assert result["video"] == []


def test_go1_air_sdpa_attention_matches_eager_without_mask() -> None:
    q = torch.tensor(
        [[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]],
        dtype=torch.float32,
    )
    k = torch.tensor(
        [[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]],
        dtype=torch.float32,
    )
    v = torch.tensor(
        [[[[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]]]],
        dtype=torch.float32,
    )

    eager = scaled_dot_product(q, k, v, num_kv_groups=1, mask=None, implementation="eager")
    sdpa = scaled_dot_product(q, k, v, num_kv_groups=1, mask=None, implementation="sdpa")

    assert torch.allclose(sdpa, eager, atol=1e-6, rtol=1e-6)
