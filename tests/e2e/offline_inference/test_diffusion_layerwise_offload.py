import sys
from pathlib import Path

import pytest
import torch
from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

from tests.utils import GPUMemoryMonitor
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform

# ruff: noqa: E402
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vllm_omni import Omni

# Models to test and expected saved memory in MB, correspondingly
MODELS_SAVED_MEMORY_MB = {"riverclouds/qwen_image_random": 4500}


def run_inference(
    model_name: str,
    layerwise_offload: bool = False,
    num_gpu_layers: int = 1,
    num_inference_steps: int = 3,
) -> float:
    # For now, only support on GPU, so apply torch.cuda operations here
    # NPU / ROCm platforms are expected to be detected and skipped this test function
    torch.cuda.empty_cache()
    device_index = torch.cuda.current_device()
    monitor = GPUMemoryMonitor(device_index=device_index, interval=0.02)
    monitor.start()

    m = Omni(
        model=model_name,
        enable_layerwise_offload=layerwise_offload,
        layerwise_num_gpu_layers=num_gpu_layers,
        boundary_ratio=0.875,
        flow_shift=5.0,
    )

    torch.cuda.reset_peak_memory_stats(device=device_index)

    # Refer to tests/e2e/offline_inference/test_t2v_model.py
    # Use minimal settings for testing
    height = 480
    width = 640
    num_frames = 5

    m.generate(
        "A cat sitting on a table",
        OmniDiffusionSamplingParams(
            height=height,
            width=width,
            generator=torch.Generator("cuda").manual_seed(42),
            guidance_scale=1.0,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
        ),
    )

    peak = monitor.peak_used_mb
    monitor.stop()

    return peak


@pytest.mark.skipif(current_omni_platform.is_npu() or current_omni_platform.is_rocm(), reason="Hardware not supported")
@pytest.mark.parametrize("model_name", MODELS_SAVED_MEMORY_MB.keys())
def test_layerwise_offload_diffusion_model(model_name: str):
    """Test that layerwise offloading reduces GPU memory usage.

    This test verifies that layerwise offloading significantly reduces peak
    GPU memory usage compared to loading the entire model on GPU. The layerwise
    offloader keeps only a single transformer block on GPU at a time, with
    prefetching for compute-memory overlap.
    """
    try:
        # Run without layerwise offloading (baseline)
        no_offload_peak_memory = run_inference(model_name, layerwise_offload=False)
        cleanup_dist_env_and_memory()

        # Run with layerwise offloading (1 layer on device)
        layerwise_offload_peak_memory = run_inference(model_name, layerwise_offload=True, num_gpu_layers=1)
        cleanup_dist_env_and_memory()

        # Run with 2 layers on device
        layerwise_offload_two_layers_peak = run_inference(model_name, layerwise_offload=True, num_gpu_layers=2)
    except Exception:
        pytest.fail("Inference failed")

    print(f"Layerwise offload peak memory (1 GPU layer): {layerwise_offload_peak_memory} MB")
    print(f"Layerwise offload peak memory (2 GPU layers): {layerwise_offload_two_layers_peak} MB")
    print(f"No offload peak memory: {no_offload_peak_memory} MB")

    # Verify that layerwise offloading significantly reduces memory usage
    # Passes only if the actual savings exceeds the expected savings
    assert layerwise_offload_peak_memory + MODELS_SAVED_MEMORY_MB[model_name] < no_offload_peak_memory, (
        f"Layerwise offload peak memory {layerwise_offload_peak_memory} MB "
        f"should be significantly less than no offload peak memory {no_offload_peak_memory} MB"
    )

    # Verify that 2 GPU layers uses more memory than 1 GPU layer
    # But not excessively more (should be a reasonable increase)
    assert layerwise_offload_peak_memory < layerwise_offload_two_layers_peak, (
        f"1 GPU layer peak {layerwise_offload_peak_memory} MB should be < "
        f"2 GPU layers peak {layerwise_offload_two_layers_peak} MB"
    )
