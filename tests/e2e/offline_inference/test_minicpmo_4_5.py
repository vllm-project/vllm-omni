"""
E2E offline tests for MiniCPM-o 4.5 model with multimodal input and audio / text output.
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.media import generate_synthetic_audio, generate_synthetic_image, generate_synthetic_video
from tests.helpers.stage_config import get_deploy_config_path

models = ["openbmb/MiniCPM-o-4_5"]

_CI_DEPLOY = get_deploy_config_path("minicpmo_4_5.yaml")


test_params = [(model, None, {"deploy_config": _CI_DEPLOY, "trust_remote_code": True}) for model in models]


def get_question(prompt_type: str = "text") -> str:
    prompts = {
        "text": "What is the capital of China? Answer in 20 words.",
        "audio": "Describe the audio briefly.",
        "image": "What color are the squares in this image?",
        "video": "Describe the video briefly.",
        "mix": "Describe what is in the image and audio.",
    }
    return prompts.get(prompt_type, prompts["text"])


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "npu": "A2"}, num_cards=2)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_text_to_text(omni_runner, omni_runner_handler) -> None:
    """Test processing text, generating text output."""
    request_config = {"prompts": get_question("text"), "modalities": ["text"]}
    omni_runner_handler.send_omni_request(request_config)


@pytest.mark.full_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "npu": "A2"}, num_cards=2)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_audio_to_text(omni_runner, omni_runner_handler) -> None:
    """Test processing audio, generating text output."""
    audio = generate_synthetic_audio(1, 1, 16000)["np_array"]
    if len(audio.shape) == 2:
        audio = audio.squeeze()
    request_config = {"prompts": get_question("audio"), "audios": (audio, 16000), "modalities": ["text"]}
    omni_runner_handler.send_omni_request(request_config)


@pytest.mark.full_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "npu": "A2"}, num_cards=2)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_image_to_text(omni_runner, omni_runner_handler) -> None:
    """Test processing image, generating text output."""
    image = generate_synthetic_image(16, 16)["np_array"]
    request_config = {"prompts": get_question("image"), "images": image, "modalities": ["text"]}
    omni_runner_handler.send_omni_request(request_config)


@pytest.mark.full_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "npu": "A2"}, num_cards=2)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_video_to_text(omni_runner, omni_runner_handler) -> None:
    """Test processing video, generating text output."""
    video = generate_synthetic_video(24, 24, 20)["np_array"]
    request_config = {"prompts": get_question("video"), "videos": video, "modalities": ["text"]}
    omni_runner_handler.send_omni_request(request_config)


@pytest.mark.full_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "npu": "A2"}, num_cards=2)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_text_to_audio(omni_runner, omni_runner_handler) -> None:
    """Test processing text, generating audio output through the talker token2wav path."""
    request_config = {"prompts": get_question("text"), "modalities": ["audio"]}
    omni_runner_handler.send_omni_request(request_config)


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "npu": "A2"}, num_cards=2)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_mix_to_audio(omni_runner, omni_runner_handler) -> None:
    """Test processing mixed modalities (image + audio), generating audio output."""
    audio = generate_synthetic_audio(1, 1, 16000)["np_array"]
    if len(audio.shape) == 2:
        audio = audio.squeeze()
    image = generate_synthetic_image(16, 16)["np_array"]
    request_config = {
        "prompts": get_question("mix"),
        "audios": (audio, 16000),
        "images": image,
        "modalities": ["audio"],
    }
    omni_runner_handler.send_omni_request(request_config)


@pytest.mark.full_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "npu": "A2"}, num_cards=2)
@pytest.mark.parametrize("omni_runner", test_params, indirect=True)
def test_video_to_audio(omni_runner, omni_runner_handler) -> None:
    """Test processing video, generating audio output."""
    video = generate_synthetic_video(24, 24, 20)["np_array"]
    request_config = {"prompts": get_question("video"), "videos": video, "modalities": ["audio"]}
    omni_runner_handler.send_omni_request(request_config)
