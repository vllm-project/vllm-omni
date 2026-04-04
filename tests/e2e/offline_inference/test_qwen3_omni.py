"""
E2E offline tests for Omni model with video input and audio output.
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "0"

from pathlib import Path

import pytest

from tests.conftest import (
    generate_synthetic_video,
    modify_stage_config,
)
from tests.utils import hardware_test
from vllm_omni.platforms import current_omni_platform

models = ["Qwen/Qwen3-Omni-30B-A3B-Instruct"]


def get_cuda_graph_config():
    path = modify_stage_config(
        str(Path(__file__).parent.parent / "stage_configs" / "qwen3_omni_ci.yaml"),
        updates={
            "stage_args": {
                0: {
                    "engine_args.enforce_eager": "true",
                },
                1: {"engine_args.enforce_eager": "true"},
            },
        },
    )
    return path


# CI stage config for 2xH100-80G GPUs or AMD GPU MI325
if current_omni_platform.is_rocm():
    # ROCm stage config optimized for MI325 GPU
    stage_configs = [str(Path(__file__).parent.parent / "stage_configs" / "rocm" / "qwen3_omni_ci.yaml")]
elif current_omni_platform.is_xpu():
    stage_configs = [str(Path(__file__).parent.parent / "stage_configs" / "xpu" / "qwen3_omni_ci.yaml")]
else:
    stage_configs = [get_cuda_graph_config()]

# Create parameter combinations for model and stage config
test_params = [(model, stage_config) for model in models for stage_config in stage_configs]


def get_question(prompt_type="video"):
    prompts = {
        "video": "Describe the video briefly.",
    }
    return prompts.get(prompt_type, prompts["video"])


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_video_to_audio(omni_runner, omni_runner_handler) -> None:
    """Test processing video, generating audio output."""
    video = generate_synthetic_video(224, 224, 300)["np_array"]

    request_config = {"prompts": get_question(), "videos": video, "modalities": ["audio"]}

    # Test single completion
    omni_runner_handler.send_request(request_config)
