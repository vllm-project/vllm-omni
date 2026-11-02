# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Weight-free consumer check for Qwen-Image's device-local Euler schedule."""

import numpy as np
import pytest
import torch
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from torch import nn
from vllm.platforms import current_platform

from tests.helpers.mark import hardware_marks, hardware_test
from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image import QwenImagePipeline
from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image_edit import QwenImageEditPipeline
from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image_edit_plus import QwenImageEditPlusPipeline

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, *hardware_marks(res={"cuda": "L4"}, num_cards=1)]


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("pipeline_cls", [QwenImagePipeline, QwenImageEditPipeline, QwenImageEditPlusPipeline])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_qwen_image_euler_steps_match_device_local_diffusers(pipeline_cls, dtype):
    # Qwen/Qwen-Image scheduler settings; no checkpoint or model weights.
    config = {
        "num_train_timesteps": 1000,
        "shift": 1.0,
        "use_dynamic_shifting": True,
        "base_shift": 0.5,
        "max_shift": 0.9,
        "base_image_seq_len": 256,
        "max_image_seq_len": 8192,
        "shift_terminal": 0.02,
    }
    device = torch.device(current_platform.device_type)
    pipeline = pipeline_cls.__new__(pipeline_cls)
    nn.Module.__init__(pipeline)
    pipeline.device = device
    pipeline.scheduler = FlowMatchEulerDiscreteScheduler(**config)
    timesteps, steps = pipeline.prepare_timesteps(20, sigmas=None, image_seq_len=1024)

    # Build the Diffusers pipeline's device-local schedule independently, then
    # consume both schedules with the same BF16 predictions and latent state.
    reference = FlowMatchEulerDiscreteScheduler(**config)
    slope = (config["max_shift"] - config["base_shift"]) / (config["max_image_seq_len"] - config["base_image_seq_len"])
    mu = config["base_shift"] + (1024 - config["base_image_seq_len"]) * slope
    reference.set_timesteps(20, sigmas=np.linspace(1.0, 1 / 20, 20), mu=mu, device=device)
    pipeline.scheduler.set_begin_index(0)
    reference.set_begin_index(0)
    prediction = (torch.arange(1, 65, device=device, dtype=torch.float32) / 64).to(dtype)
    prediction = prediction.view(1, 1, 64).expand(1, 1024, 64).contiguous()
    actual = torch.zeros_like(prediction)
    expected = actual.clone()
    assert steps == 20
    for actual_t, reference_t in zip(timesteps, reference.timesteps):
        actual = pipeline.scheduler.step(prediction, actual_t, actual, return_dict=False)[0]
        expected = reference.step(prediction, reference_t, expected, return_dict=False)[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
