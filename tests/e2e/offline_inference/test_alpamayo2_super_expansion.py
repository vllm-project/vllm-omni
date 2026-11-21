# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""End-to-end test for Alpamayo 2 Super: reasoning stage, KV handoff, trajectory expert.

Runs the deploy config from ``recipes/NVIDIA/Alpamayo2-Super-H200.md`` on
synthetic inputs (blank camera frames, straight constant-speed history) and
checks that the request returns a plausible trajectory and that a seed
reproduces it.

It calls ``omni.generate`` directly: no ``OfflineOmniClient`` helper yet takes
a token-id prompt with per-stage sampling params and returns ``actions``.
"""

import pytest
import torch
from vllm import SamplingParams

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner
from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_executor.models.alpamayo2_super import Alpamayo2SuperConfig
from vllm_omni.model_executor.models.alpamayo2_super.processing import (
    build_alpamayo2_super_tokenizer,
    build_trajectory_prompt,
)

MODEL_NAME = "nvidia/Alpamayo2-Super"
DEPLOY_CONFIG = get_deploy_config_path("alpamayo2_super.yaml")

pytestmark = [
    pytest.mark.full_model,
    pytest.mark.diffusion,
    pytest.mark.parametrize("omni_runner", [(MODEL_NAME, DEPLOY_CONFIG)], indirect=True),
]


def _request(seed: int) -> tuple[dict, list]:
    config = Alpamayo2SuperConfig.from_pretrained(MODEL_NAME)
    tokenizer = build_alpamayo2_super_tokenizer(MODEL_NAME, config.history_vocab_size, config.future_vocab_size)
    # Four cameras x four frames of mid-gray, and 16 history poses at 10 m/s straight ahead.
    frames = torch.full((4, 4, 3, 320, 576), 128, dtype=torch.uint8)
    history_xyz = torch.zeros(16, 3)
    history_xyz[:, 0] = torch.arange(-15.0, 1.0)
    history_rot = torch.eye(3).expand(16, 3, 3)
    prompt = {
        "prompt_token_ids": build_trajectory_prompt(
            tokenizer, config, frames=frames, camera_indices=[0, 1, 2, 6], ego_history_xyz=history_xyz
        ),
        "multi_modal_data": {"image": list(frames.flatten(0, 1))},
    }
    reasoning = SamplingParams(
        temperature=0.6,
        top_p=0.98,
        max_tokens=256,
        seed=seed,
        detokenize=False,
        stop_token_ids=[config.traj_ids["future_end"]],
    )
    expert = OmniDiffusionSamplingParams(
        seed=seed,
        num_inference_steps=10,
        extra_args={"robot_obs": {"ego_history_xyz": history_xyz.tolist(), "ego_history_rot": history_rot.tolist()}},
    )
    return prompt, [reasoning, expert]


def _trajectory(omni_runner: OmniRunner, seed: int) -> tuple[list[int], torch.Tensor]:
    prompt, params = _request(seed)
    outputs = {output.stage_id: output for output in omni_runner.omni.generate(prompt, params, use_tqdm=False)}
    return list(outputs[0].outputs[0].token_ids), outputs[1].multimodal_output["actions"]


@hardware_test(res={"cuda": "H200"})
def test_alpamayo2_super_trajectory(omni_runner: OmniRunner) -> None:
    """
    Test reasoning, KV handoff and trajectory sampling end to end, and seeded replay.
    Deploy Setting: default yaml (alpamayo2_super.yaml)
    Input Modal: 4 cameras x 4 frames + 16-pose ego history
    Output Modal: reasoning tokens + (64, 3) trajectory
    Input Setting: seed=42, 10 expert steps
    Datasets: single synthetic request
    """
    reasoning, actions = _trajectory(omni_runner, seed=42)
    assert reasoning, "reasoning stage returned no tokens"
    assert actions.shape == (64, 3) and torch.isfinite(actions).all()
    # On an empty road at a steady 10 m/s the policy keeps going, about 64 m in 6.4 s.
    assert 30.0 < actions[-1, 0] < 100.0

    again = _trajectory(omni_runner, seed=42)
    assert again[0] == reasoning
    # The repeat reuses the cached prompt prefix and recomputes only the rest,
    # which changes rounding slightly; the trajectory agrees to millimeters.
    torch.testing.assert_close(again[1], actions, rtol=0, atol=0.05)
