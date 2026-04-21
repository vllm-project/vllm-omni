import pytest

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


@pytest.mark.parametrize("tensor_parallel_size", [1, 2])
def test_omniweaving_t2v_expansion(tensor_parallel_size):
    """
    E2E test to verify OmniWeaving T2V generation works correctly
    under both Single-GPU (TP=1) and Multi-GPU (TP=2) configurations.
    """
    omni = Omni(
        model="Tencent-Hunyuan/OmniWeaving",
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

    outputs = omni.generate(prompts=["A simple test prompt for CI."], sampling_params=sampling_params)

    assert outputs is not None, "Output should not be None"
    assert len(outputs) > 0, "Output list should not be empty"

    output_data = outputs[0].outputs[0]
    assert output_data is not None, "Generated video tensor should not be None"


@pytest.mark.parametrize("tensor_parallel_size", [1])
def test_omniweaving_cfg_parallel(tensor_parallel_size):
    """
    E2E test to verify CFG Parallelism works for OmniWeaving.
    """
    omni = Omni(
        model="Tencent-Hunyuan/OmniWeaving",
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

    outputs = omni.generate(prompts=["Testing CFG parallel generation."], sampling_params=sampling_params)

    assert outputs is not None
    assert len(outputs) > 0
    assert outputs[0].outputs[0] is not None
