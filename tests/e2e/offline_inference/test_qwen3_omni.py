"""
E2E offline tests for Omni model with video input and audio output.
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "0"

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.media import generate_synthetic_video
from tests.helpers.stage_config import get_deploy_config_path, modify_stage_config
from vllm_omni.platforms import current_omni_platform

models = ["Qwen/Qwen3-Omni-30B-A3B-Instruct"]


# Single CI deploy YAML; rocm/xpu deltas are picked automatically via the
# platforms: section. Only CUDA needs an extra enforce_eager tweak.
_CI_DEPLOY = get_deploy_config_path("ci/qwen3_omni_moe.yaml")


def get_cuda_graph_config():
    return modify_stage_config(
        _CI_DEPLOY,
        updates={
            "stages": {
                0: {"enforce_eager": True},
                1: {"enforce_eager": True},
            },
        },
    )


if current_omni_platform.is_xpu():
    stage_configs = [_CI_DEPLOY]
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
    omni_runner_handler.send_omni_request(request_config)
