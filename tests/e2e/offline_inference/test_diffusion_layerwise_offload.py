import pytest
import torch
from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

from tests.e2e.offline_inference.test_diffusion_cpu_offload import check_audio_determinism
from tests.helpers.env import DeviceMemoryMonitor
from tests.helpers.runtime import OmniRunner
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.platforms import current_omni_platform

AUDIO_MODEL = {
    "stabilityai/stable-audio-open-1.0": {"cuda": 1500, "rocm": 1500},
}

IMAGE_VIDEO_MODELS = {
    "riverclouds/qwen_image_random": {"cuda": 4500, "rocm": None},
    # "Wan-AI/Wan2.2-T2V-A14B-Diffusers": {"cuda": 45000, "rocm": None},
}

MODELS = {**AUDIO_MODEL, **IMAGE_VIDEO_MODELS}

AUDIO_MODEL_PARAMS = {
    "runner_params": {},
    "sampler_params": {},
}

IMAGE_VIDEO_MODELS_PARAMS = {
    "runner_params": {"boundary_ratio": 0.875, "flow_shift": 5.0},
    "sampler_params": {"height": 480, "width": 640, "num_frames": 5},
}


def run_inference(
    model_name: str,
    layerwise_offload: bool = False,
    num_inference_steps: int = 3,
) -> float:
    current_omni_platform.empty_cache()
    device_index = current_omni_platform.current_device()
    monitor = DeviceMemoryMonitor(device_index=device_index, interval=0.02)
    monitor.start()

    if model_name in AUDIO_MODEL:
        params = AUDIO_MODEL_PARAMS
    else:
        params = IMAGE_VIDEO_MODELS_PARAMS

    with OmniRunner(
        model_name,
        enable_layerwise_offload=layerwise_offload,
        # TODO: we might want to add overlapped feature e2e tests
        # cache_backend="cache_dit",
        **params["runner_params"],
    ) as runner:
        current_omni_platform.reset_peak_memory_stats()

        # Refer to tests/e2e/offline_inference/test_t2v_model.py
        # Use minimal settings for testing
        output = runner.omni.generate(
            "A cat sitting on a table",
            OmniDiffusionSamplingParams(
                generator=torch.Generator(device=current_omni_platform.device_type).manual_seed(42),
                guidance_scale=1.0,
                num_inference_steps=num_inference_steps,
                **params["sampler_params"],
            ),
        )

    peak = monitor.peak_used_mb
    monitor.stop()

    return peak, output


@pytest.mark.parametrize("model_name", list(MODELS.keys()))
def test_layerwise_offload_diffusion_model(model_name: str):
    """Test that layerwise offloading reduces GPU memory usage.

    This test verifies that layerwise offloading significantly reduces peak
    GPU memory usage compared to loading the entire model on GPU. The layerwise
    offloader keeps only a single transformer block on GPU at a time, with
    prefetching for compute-memory overlap.
    """
    try:
        # Run without layerwise offloading (baseline)
        no_offload_peak_memory, output_no_offload = run_inference(model_name, layerwise_offload=False)
        cleanup_dist_env_and_memory()

        # Run with layerwise offloading (1 layer on device)
        layerwise_offload_peak_memory, output_offload = run_inference(model_name, layerwise_offload=True)
        cleanup_dist_env_and_memory()
    except Exception:
        pytest.fail("Inference failed")

    print(f"Layerwise offload peak memory (1 GPU layer): {layerwise_offload_peak_memory} MB")
    print(f"No offload peak memory: {no_offload_peak_memory} MB")

    if model_name == "stabilityai/stable-audio-open-1.0":
        audio_offload = output_offload[0].request_output.multimodal_output.get("audio")
        audio_no_offload = output_no_offload[0].request_output.multimodal_output.get("audio")
        check_audio_determinism(audio_offload, audio_no_offload, atol=1e-3)

    is_rocm = torch.version.hip is not None
    platform = "rocm" if is_rocm else "cuda"
    expected_saved_memory = MODELS[model_name][platform]

    if expected_saved_memory is None:
        pytest.skip(f"Threshold not defined for {platform} on {model_name}")

    # Verify that layerwise offloading significantly reduces memory usage
    # Passes only if the actual savings exceeds the expected savings
    assert layerwise_offload_peak_memory + expected_saved_memory < no_offload_peak_memory, (
        f"Layerwise offload peak memory {layerwise_offload_peak_memory} MB "
        f"should be significantly less than no offload peak memory {no_offload_peak_memory} MB"
    )
