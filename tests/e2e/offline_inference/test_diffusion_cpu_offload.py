import gc

import pytest
import torch
from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

from tests.conftest import OmniRunner
from tests.utils import DeviceMemoryMonitor, hardware_test
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform

models = ["riverclouds/qwen_image_random"]


def inference(model_name: str, offload: bool = True):
    gc.collect()
    current_omni_platform.empty_cache()
    device_index = current_omni_platform.current_device()
    current_omni_platform.reset_peak_memory_stats()
    monitor = DeviceMemoryMonitor(device_index=device_index, interval=0.02)
    monitor.start()
    with OmniRunner(
        model_name,
        # TODO: we might want to add overlapped feature e2e tests
        # cache_backend="cache_dit",
        enable_cpu_offload=offload,
    ) as runner:
        current_omni_platform.reset_peak_memory_stats()
        height = 256
        width = 256

        runner.omni.generate(
            "a photo of a cat sitting on a laptop keyboard",
            OmniDiffusionSamplingParams(
                height=height,
                width=width,
                num_inference_steps=9,
                guidance_scale=0.0,
                generator=torch.Generator(device=current_omni_platform.device_type).manual_seed(42),
            ),
        )
    peak = monitor.peak_used_mb
    monitor.stop()

    gc.collect()
    current_omni_platform.empty_cache()

    return peak


@pytest.mark.core_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4", "rocm": "MI325"})
@pytest.mark.parametrize("model_name", models)
def test_cpu_offload_diffusion_model(model_name: str):
    try:
        offload_peak_memory = inference(model_name, offload=True)
        cleanup_dist_env_and_memory()
        no_offload_peak_memory = inference(model_name, offload=False)
    except Exception:
        pytest.fail("Inference failed")
    print(f"Offload peak memory: {offload_peak_memory} MB")
    print(f"No offload peak memory: {no_offload_peak_memory} MB")
    # Set platform-specific VRAM saving thresholds to account
    # for varying runtime memory overhead and fragmentation between CUDA and ROCm.
    is_rocm = torch.version.hip is not None
    threshold = 2500 if not is_rocm else 2100
    assert offload_peak_memory + threshold < no_offload_peak_memory, (
        f"Offload peak memory {offload_peak_memory} MB should be less than "
        f"no offload peak memory {no_offload_peak_memory} MB by {threshold} MB"
    )
