# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OmniWeaving diffusion feature coverage."""

import pytest
import torch

from tests.helpers.mark import hardware_marks

pytestmark = [pytest.mark.diffusion, pytest.mark.full_model]

MODEL = "Tencent-Hunyuan/OmniWeaving"
PROMPT = "A simple test prompt for CI."
NEGATIVE_PROMPT = "blurry, distorted"
SINGLE_CARD_MARKS = hardware_marks(res={"cuda": "H100"})
PARALLEL_MARKS = hardware_marks(res={"cuda": "H100"}, num_cards=2)


def _assert_has_generated_diffusion_payload(outputs):
    from vllm_omni.outputs import OmniRequestOutput

    assert outputs is not None, "Output should not be None"
    assert len(outputs) > 0, "Output list should not be empty"

    output_data = OmniRequestOutput.unwrap_result(outputs)
    assert output_data is not None, "Generated output should not be None"
    assert output_data.images or output_data.multimodal_output or output_data.latents is not None


@pytest.mark.parametrize(
    "tensor_parallel_size",
    [
        pytest.param(1, marks=SINGLE_CARD_MARKS),
        pytest.param(2, marks=PARALLEL_MARKS),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA GPU for OmniWeaving E2E test.")
def test_omniweaving_t2v_expansion(tensor_parallel_size):
    """
    E2E test to verify OmniWeaving T2V generation works correctly
    under both Single-GPU (TP=1) and Multi-GPU (TP=2) configurations.
    """
    if torch.accelerator.device_count() < tensor_parallel_size:
        pytest.skip(f"Need {tensor_parallel_size} CUDA GPUs for OmniWeaving TP={tensor_parallel_size}.")

    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    omni = Omni(
        model=MODEL,
        tensor_parallel_size=tensor_parallel_size,
        dtype="bfloat16",
        trust_remote_code=True,
    )

    sampling_params = OmniDiffusionSamplingParams(
        num_inference_steps=2,
        guidance_scale=6.0,
        height=128,
        width=128,
        num_frames=5,
        seed=42,
    )

    prompt = {
        "prompt": PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
    }
    outputs = omni.generate(prompts=prompt, sampling_params_list=sampling_params)

    _assert_has_generated_diffusion_payload(outputs)


@pytest.mark.parametrize("tensor_parallel_size", [pytest.param(1, marks=PARALLEL_MARKS)])
@pytest.mark.skipif(torch.accelerator.device_count() < 2, reason="Need 2 CUDA GPUs for OmniWeaving CFG parallelism.")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA GPU for OmniWeaving E2E test.")
def test_omniweaving_cfg_parallel(tensor_parallel_size):
    """
    E2E test to verify CFG Parallelism works for OmniWeaving.
    """
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    omni = Omni(
        model=MODEL,
        tensor_parallel_size=tensor_parallel_size,
        cfg_parallel_size=2,
        dtype="bfloat16",
        trust_remote_code=True,
    )

    sampling_params = OmniDiffusionSamplingParams(
        num_inference_steps=2,
        guidance_scale=6.0,
        height=128,
        width=128,
        num_frames=5,
    )

    prompt = {
        "prompt": "Testing CFG parallel generation.",
        "negative_prompt": "blurry, low quality",
    }
    outputs = omni.generate(prompts=prompt, sampling_params_list=sampling_params)

    _assert_has_generated_diffusion_payload(outputs)


@pytest.mark.parametrize("tensor_parallel_size", [pytest.param(1, marks=SINGLE_CARD_MARKS)])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA GPU for OmniWeaving E2E test.")
def test_omniweaving_i2v_expansion(tensor_parallel_size):
    """E2E test to verify OmniWeaving accepts image-conditioned video requests."""
    from tests.helpers.media import generate_synthetic_image
    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    omni = Omni(
        model=MODEL,
        tensor_parallel_size=tensor_parallel_size,
        dtype="bfloat16",
        trust_remote_code=True,
    )

    sampling_params = OmniDiffusionSamplingParams(
        num_inference_steps=2,
        guidance_scale=6.0,
        height=128,
        width=128,
        num_frames=5,
        seed=42,
    )

    prompt = {
        "prompt": "Animate the geometric shapes with gentle motion.",
        "negative_prompt": NEGATIVE_PROMPT,
        "multi_modal_data": {"image": generate_synthetic_image(128, 128)["file_path"]},
    }
    outputs = omni.generate(prompts=prompt, sampling_params_list=sampling_params)

    _assert_has_generated_diffusion_payload(outputs)
