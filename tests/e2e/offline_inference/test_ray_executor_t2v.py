# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""E2E tests for Ray executor with Wan T2V model."""

import pytest
import torch

from tests.utils import hardware_test
from vllm_omni import Omni
from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.platforms import current_omni_platform

MODEL = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
HEIGHT = 480
WIDTH = 640
NUM_FRAMES = 5
NUM_STEPS = 2
SEED = 42


def _generate(num_gpus: int, ulysses_degree: int = 1):
    """Run inference with the ray executor."""
    parallel_config = DiffusionParallelConfig(
        sequence_parallel_size=num_gpus,
        ulysses_degree=ulysses_degree,
    )
    m = Omni(
        model=MODEL,
        distributed_executor_backend="ray",
        num_gpus=num_gpus,
        parallel_config=parallel_config,
    )
    try:
        outputs = m.generate(
            prompts="A cat sitting on a table",
            sampling_params_list=OmniDiffusionSamplingParams(
                height=HEIGHT,
                width=WIDTH,
                num_frames=NUM_FRAMES,
                num_inference_steps=NUM_STEPS,
                guidance_scale=1.0,
                generator=torch.Generator(
                    current_omni_platform.device_type
                ).manual_seed(SEED),
            ),
        )
        return outputs
    finally:
        m.close()


@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4"}, num_cards={"cuda": 1})
def test_ray_executor_single_gpu():
    """Single GPU inference via Ray executor."""
    outputs = _generate(num_gpus=1)

    first_output = outputs[0]
    assert first_output.final_output_type == "image"
    assert first_output.request_output

    req_out = first_output.request_output[0]
    assert isinstance(req_out, OmniRequestOutput)

    frames = req_out.images[0]
    assert frames is not None
    assert frames.shape[1] == NUM_FRAMES
    assert frames.shape[2] == HEIGHT
    assert frames.shape[3] == WIDTH


@pytest.mark.diffusion
@pytest.mark.parallel
@hardware_test(res={"cuda": "L4"}, num_cards={"cuda": 2})
def test_ray_executor_parallel_ulysses():
    """Multi-GPU inference via Ray executor with Ulysses SP."""
    outputs = _generate(num_gpus=2, ulysses_degree=2)

    first_output = outputs[0]
    assert first_output.final_output_type == "image"
    assert first_output.request_output

    req_out = first_output.request_output[0]
    assert isinstance(req_out, OmniRequestOutput)

    frames = req_out.images[0]
    assert frames is not None
    assert frames.shape[1] == NUM_FRAMES
    assert frames.shape[2] == HEIGHT
    assert frames.shape[3] == WIDTH
