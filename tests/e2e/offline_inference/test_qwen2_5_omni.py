"""
E2E tests for Qwen2.5-Omni model with mixed modality inputs, audio and text output.
"""

from pathlib import Path

import pytest

from tests.conftest import (
    generate_synthetic_audio,
    generate_synthetic_image,
    generate_synthetic_video,
    modify_stage_config,
)
from tests.utils import hardware_test
from vllm_omni.platforms import current_omni_platform

models = ["Qwen/Qwen2.5-Omni-7B"]


def get_cuda_graph_config():
    path = modify_stage_config(
        str(Path(__file__).parent.parent / "stage_configs" / "qwen2_5_omni_ci.yaml"),
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


# CI stage config optimized for 24GB GPU (L4/RTX3090) or NPU
if current_omni_platform.is_npu():
    stage_config = str(Path(__file__).parent / "stage_configs" / "npu" / "qwen2_5_omni_ci.yaml")
elif current_omni_platform.is_rocm():
    # ROCm stage config optimized for MI325 GPU
    stage_config = str(Path(__file__).parent.parent / "stage_configs" / "rocm" / "qwen2_5_omni_ci.yaml")
elif current_omni_platform.is_xpu():
    # Intel XPU stage config optimized for B60 GPU
    stage_config = str(Path(__file__).parent.parent / "stage_configs" / "xpu" / "qwen2_5_omni_ci.yaml")
else:
    stage_config = get_cuda_graph_config()

# Create parameter combinations for model and stage config
test_params = [(model, stage_config) for model in models]


def get_question(prompt_type="mix"):
    prompts = {
        "mix": "What is recited in the audio? What is in this image? Describe the video briefly.",
        "text_only": "What is the capital of China?",
    }
    return prompts.get(prompt_type, prompts["mix"])


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4", "rocm": "MI325", "xpu": "B60"}, num_cards={"cuda": 4, "rocm": 2, "xpu": 3})
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_mix_to_audio(omni_runner, omni_runner_handler) -> None:
    """
    Test multi-modal input processing and text/audio output generation via OpenAI API.
    Deploy Setting: default yaml
    Input Modal: text + audio + video + image
    Output Modal: audio
    Input Setting: stream=False
    Datasets: single request
    """
    video = generate_synthetic_video(16, 16, 30)["np_array"]
    image = generate_synthetic_image(16, 16)["np_array"]
    audio = generate_synthetic_audio(1, 1, 16000)["np_array"]
    if len(audio.shape) == 2:
        audio = audio.squeeze()

    request_config = {
        "prompts": get_question(),
        "videos": video,
        "images": image,
        "audios": (audio, 16000),
        "modalities": ["audio"],
    }

    # Test single completion
    omni_runner_handler.send_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4", "rocm": "MI325", "xpu": "B60"}, num_cards={"cuda": 4, "rocm": 2, "xpu": 3})
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_text_to_text(omni_runner, omni_runner_handler) -> None:
    """
    Test text input processing and text output generation via OpenAI API.
    Deploy Setting: default yaml
    Input Modal: text
    Output Modal: text
    Input Setting: stream=False
    Datasets: single request
    """

    request_config = {"prompts": get_question("text_only"), "modalities": ["text"]}

    # Test single completion
    omni_runner_handler.send_request(request_config)
