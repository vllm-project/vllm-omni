import sys
from pathlib import Path

import pytest
import torch

from tests.utils import GPUMemoryMonitor
from vllm_omni.utils.platform_utils import is_npu, is_rocm

# ruff: noqa: E402
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vllm_omni import Omni

models = ["riverclouds/qwen_image_random"]


@pytest.mark.skipif(is_npu() or is_rocm(), reason="Hardware not supported")
@pytest.mark.parametrize("model_name", models)
def test_cpu_offload_diffusion_model(model_name: str):
    def inference(offload: bool = True):
        torch.cuda.empty_cache()
        device_index = torch.cuda.current_device()
        monitor = GPUMemoryMonitor(device_index=device_index, interval=0.02)
        monitor.start()
        m = Omni(model=model_name, enable_cpu_offload=offload)
        torch.cuda.reset_peak_memory_stats(device=device_index)
        height = 256
        width = 256

        m.generate(
            "a photo of a cat sitting on a laptop keyboard",
            height=height,
            width=width,
            num_inference_steps=9,
            guidance_scale=0.0,
            generator=torch.Generator("cuda").manual_seed(42),
        )
        peak = monitor.peak_used_mb
        monitor.stop()

        return peak

    offload_peak_memory = inference(offload=True)
    no_offload_peak_memory = inference(offload=False)
    print(f"Offload peak memory: {offload_peak_memory} MB")
    print(f"No offload peak memory: {no_offload_peak_memory} MB")
    assert offload_peak_memory + 2500 < no_offload_peak_memory, (
        f"Offload peak memory {offload_peak_memory} MB should be less than no offload peak memory {no_offload_peak_memory} MB"
    )
