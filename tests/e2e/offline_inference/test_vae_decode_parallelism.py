import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

# ruff: noqa: E402
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import time

from tests.conftest import OmniRunner
from vllm_omni.platforms import current_omni_platform

# os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "1"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

all_model_configs = [
    {
        "model_name": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "out_width": 1280,
        "out_height": 720,
        "out_frames": 16,
    },
    {
        "model_name": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        "out_width": 1280,
        "out_height": 720,
        "out_frames": 16,
    },
    {
        "model_name": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        "out_width": 1280,
        "out_height": 720,
        "out_frames": 16,
    },
    {
        "model_name": "Qwen/Qwen-Image",
        "out_width": 3072,
        "out_height": 3072,
        "out_frames": 1,
    },
    # {
    #     "model_name": "stepfun-ai/NextStep-1.1",
    #     "out_width": 3072,
    #     "out_height": 3072,
    #     "out_frames": 1,
    # },
]


def is_nextstep_model(model_name: str) -> bool:
    """Check if the model is a NextStep model by reading its config."""
    from vllm.transformers_utils.config import get_hf_file_to_dict

    try:
        cfg = get_hf_file_to_dict("config.json", model_name)
        if cfg and cfg.get("model_type") == "nextstep":
            return True
    except Exception:
        pass
    return False


def model_run(model_configs, tp, out_height, out_width, out_frames, using_tile, vae_patch_parallel_size=1):
    parallel_config = DiffusionParallelConfig(
        tensor_parallel_size=tp,
        vae_patch_parallel_size=vae_patch_parallel_size,
    )

    omni_kwargs = {
        "vae_use_tiling": using_tile,
        "parallel_config": parallel_config,
    }
    use_nextstep = is_nextstep_model(model_configs["model_name"])
    if use_nextstep:
        # NextStep-1.1 requires explicit pipeline class
        omni_kwargs["model_class_name"] = "NextStep11Pipeline"

    with OmniRunner(model_configs["model_name"], **omni_kwargs) as runner:
        m = runner.omni
        image = Image.new("RGB", (out_width, out_height), (0, 0, 0))
        start = time.perf_counter()
        outputs = m.generate(
            {
                "prompt": "A cat sitting on a table",
                "multi_modal_data": {"image": image},
            },
            sampling_params_list=OmniDiffusionSamplingParams(
                height=model_configs["out_height"],
                width=model_configs["out_width"],
                num_frames=model_configs["out_frames"],
                num_inference_steps=2,
                generator=torch.Generator(current_omni_platform.device_type).manual_seed(42),
            ),
        )
        end = time.perf_counter()
        first_output = outputs[0]
        req_out = first_output.request_output
        frames = req_out.images[0]
        if isinstance(frames, torch.Tensor):
            frames = frames.detach().cpu().numpy()
        elif isinstance(frames, Image.Image):
            frames = np.array(frames)
        # frames shape: (batch, num_frames, height, width, channels)
        cost = (end - start) * 1000
        return frames, cost
        cleanup_dist_env_and_memory()


@pytest.mark.parametrize("model_configs", all_model_configs)
def test_vae_parallel_model(model_configs: str):
    parallel_result, parallel_time = model_run(
        model_configs=model_configs,
        tp=2,
        out_width=1280,
        out_height=704,
        out_frames=5,
        using_tile=True,
        vae_patch_parallel_size=2,
    )
    non_parallel_result, non_parallel_time = model_run(
        model_configs=model_configs,
        tp=2,
        out_width=1280,
        out_height=704,
        out_frames=5,
        using_tile=True,
        vae_patch_parallel_size=1,
    )
    result_diff = np.abs(non_parallel_result - parallel_result)

    mean_threshold = 3e-2
    max_threshold = 3e-2  # they should be totally same
    print(
        f"{model_configs['model_name']} TP = 2 (tile + parallel vs tile): "
        f"mean_abs_diff={result_diff.mean():.6e}, max_abs_diff={result_diff.max():.6e}; "
        f"thresholds: mean<={mean_threshold:.6e}, max<={max_threshold:.6e}; "
        f"parallel generate take time: {parallel_time:.2f} ms, non-parallel take time: {non_parallel_time:.2f} ms"
    )
    # assert non_parallel_time > parallel_time
    assert result_diff.mean() < mean_threshold
    assert result_diff.max() < max_threshold
