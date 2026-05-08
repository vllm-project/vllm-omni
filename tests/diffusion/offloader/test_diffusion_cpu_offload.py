import gc

import numpy as np
import pytest
import torch
from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

from tests.helpers.env import DeviceMemoryMonitor
from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniRunner
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform

AUDIO_MODEL = {
    "stabilityai/stable-audio-open-1.0": {"cuda": 100, "rocm": None},
}

IMAGE_VIDEO_MODELS = {
    "riverclouds/qwen_image_random": {"cuda": 2500, "rocm": 2100},
}

MODELS = {**AUDIO_MODEL, **IMAGE_VIDEO_MODELS}

AUDIO_MODEL_PARAMS = {
    "runner_params": {},
    "sampler_params": {},
}

IMAGE_VIDEO_MODELS_PARAMS = {
    "runner_params": {},
    "sampler_params": {
        "height": 256,
        "width": 256,
    },
}


def inference(model_name: str, offload: bool = True):
    gc.collect()
    current_omni_platform.empty_cache()
    device_index = current_omni_platform.current_device()
    current_omni_platform.reset_peak_memory_stats()
    monitor = DeviceMemoryMonitor(device_index=device_index, interval=0.02)

    if model_name in AUDIO_MODEL:
        params = AUDIO_MODEL_PARAMS
    else:
        params = IMAGE_VIDEO_MODELS_PARAMS

    with OmniRunner(
        model_name,
        # TODO: we might want to add overlapped feature e2e tests
        # cache_backend="cache_dit",
        enable_cpu_offload=offload,
        **params["runner_params"],
    ) as runner:
        current_omni_platform.reset_peak_memory_stats()
        monitor.start()
        output = runner.omni.generate(
            "a photo of a cat sitting on a laptop keyboard",
            OmniDiffusionSamplingParams(
                num_inference_steps=9,
                guidance_scale=0.0,
                generator=torch.Generator(device=current_omni_platform.device_type).manual_seed(42),
                **params["sampler_params"],
            ),
        )
    peak = monitor.peak_used_mb
    monitor.stop()

    gc.collect()
    current_omni_platform.empty_cache()

    return peak, output


def check_audio_determinism(audio1, audio2, atol=1e-2):
    device = current_omni_platform.device_type
    if isinstance(audio1, np.ndarray):
        audio1 = torch.from_numpy(audio1).to(device)
    if isinstance(audio2, np.ndarray):
        audio2 = torch.from_numpy(audio2).to(device)

    if not torch.allclose(audio1, audio2, atol=atol):
        diff = torch.abs(audio1 - audio2)
        print(f"Max difference: {diff.max().item()}")
        print(f"Mean difference: {diff.mean().item()}")
        raise AssertionError(f"Audio outputs differ beyond tolerance atol={atol}")
    return True


@pytest.mark.core_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4", "rocm": "MI325"})
@pytest.mark.parametrize("model_name", list(MODELS.keys()))
def test_cpu_offload_diffusion_model(model_name: str):
    try:
        offload_peak_memory, output_offload = inference(model_name, offload=True)
        cleanup_dist_env_and_memory()
        no_offload_peak_memory, output_no_offload = inference(model_name, offload=False)
    except Exception:
        pytest.fail("Inference failed")
    print(f"Offload peak memory: {offload_peak_memory} MB")
    print(f"No offload peak memory: {no_offload_peak_memory} MB")

    if model_name == "stabilityai/stable-audio-open-1.0":
        audio_offload = output_offload[0].request_output.multimodal_output.get("audio")
        audio_no_offload = output_no_offload[0].request_output.multimodal_output.get("audio")
        check_audio_determinism(audio_offload, audio_no_offload, atol=1e-2)

    # Set platform-specific VRAM saving thresholds to account
    # for varying runtime memory overhead and fragmentation between CUDA and ROCm.
    is_rocm = torch.version.hip is not None
    platform = "rocm" if is_rocm else "cuda"
    threshold = MODELS[model_name][platform]
    if threshold is None:
        pytest.skip(f"Threshold not defined for {platform} on {model_name}")

    assert offload_peak_memory + threshold < no_offload_peak_memory, (
        f"Offload peak memory {offload_peak_memory} MB should be less than "
        f"no offload peak memory {no_offload_peak_memory} MB by {threshold} MB"
    )
