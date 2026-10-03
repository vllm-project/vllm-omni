"""
E2E Online tests for Qwen3-Omni model thinking variant
"""

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.media import generate_synthetic_audio, generate_synthetic_image, generate_synthetic_video
from tests.helpers.runtime import OmniServerParams, dummy_messages_from_mix_data
from tests.helpers.stage_config import get_deploy_config_path

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Thinking"
STAGE_CONFIG_PATH = get_deploy_config_path("qwen3_omni_moe.yaml")

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
LARGE_IMAGE_WIDTH = 1920
LARGE_IMAGE_HEIGHT = 1080
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
NUM_VIDEO_FRAMES = 300
LONG_VIDEO_WIDTH = 224
LONG_VIDEO_HEIGHT = 224
LONG_VIDEO_NUM_FRAMES = 3600
LONG_AUDIO_DURATION = 120

test_params = [pytest.param(OmniServerParams(model=MODEL, stage_config_path=STAGE_CONFIG_PATH))]


def get_system_prompt():
    return {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": (
                    "You are Qwen, a virtual human developed by the Qwen Team, "
                    "Alibaba Group, capable of perceiving text, image, audio or video inputs, "
                    "as well as generating text and speech."
                ),
            }
        ],
    }


def get_text_only_prompt():
    return "What is the capital of China?"


def get_prompt():
    return "Analyse the audio, image or video provided"


@pytest.mark.omni
@pytest.mark.slow
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
@hardware_test(res={"cuda": "H100"}, num_cards=2)
def test_analyse_text_input(omni_server, openai_client):
    """
    Input Modal: text
    """
    messages = dummy_messages_from_mix_data(system_prompt=get_system_prompt(), content_text=get_text_only_prompt())

    request_config = {
        "model": omni_server.model,
        "messages": messages,
    }

    openai_client.send_omni_request(request_config)


@pytest.mark.omni
@pytest.mark.slow
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
@hardware_test(res={"cuda": "H100"}, num_cards=2)
def test_analyse_prompt_without_attached_media(omni_server, openai_client):
    """
    Input Modal: text asking to describe media provided but it is absent
    """
    messages = dummy_messages_from_mix_data(system_prompt=get_system_prompt(), content_text=get_prompt())

    request_config = {
        "model": omni_server.model,
        "messages": messages,
    }

    openai_client.send_omni_request(request_config)


@pytest.mark.omni
@pytest.mark.slow
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
@hardware_test(res={"cuda": "H100"}, num_cards=2)
def test_analyse_audio_input(omni_server, openai_client):
    """
    Input Modal: audio
    """
    audio_data_url = f"data:audio/wav;base64,{generate_synthetic_audio(5, 1)['base64']}"
    messages = dummy_messages_from_mix_data(
        system_prompt=get_system_prompt(), content_text=get_prompt(), audio_data_url=audio_data_url
    )

    request_config = {
        "model": omni_server.model,
        "messages": messages,
    }

    openai_client.send_omni_request(request_config)


@pytest.mark.omni
@pytest.mark.slow
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
@hardware_test(res={"cuda": "H100"}, num_cards=2)
def test_analyse_long_audio_input(omni_server, openai_client):
    """
    Input Modal: long audio
    """
    audio_data_url = f"data:audio/wav;base64,{generate_synthetic_audio(LONG_AUDIO_DURATION, 1)['base64']}"
    messages = dummy_messages_from_mix_data(
        system_prompt=get_system_prompt(), content_text=get_prompt(), audio_data_url=audio_data_url
    )

    request_config = {
        "model": omni_server.model,
        "messages": messages,
    }

    openai_client.send_omni_request(request_config)


@pytest.mark.omni
@pytest.mark.slow
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
@hardware_test(res={"cuda": "H100"}, num_cards=2)
def test_analyse_image_input(omni_server, openai_client):
    """
    Input Modal: image
    """
    image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(IMAGE_WIDTH, IMAGE_HEIGHT)['base64']}"
    messages = dummy_messages_from_mix_data(
        system_prompt=get_system_prompt(), content_text=get_prompt(), image_data_url=image_data_url
    )

    request_config = {
        "model": omni_server.model,
        "messages": messages,
    }

    openai_client.send_omni_request(request_config)


@pytest.mark.omni
@pytest.mark.slow
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
@hardware_test(res={"cuda": "H100"}, num_cards=2)
def test_analyse_large_image_input(omni_server, openai_client):
    """
    Input Modal: large image
    """
    image_data_url = (
        f"data:image/jpeg;base64,{generate_synthetic_image(LARGE_IMAGE_WIDTH, LARGE_IMAGE_HEIGHT)['base64']}"
    )
    messages = dummy_messages_from_mix_data(
        system_prompt=get_system_prompt(), content_text=get_prompt(), image_data_url=image_data_url
    )

    request_config = {
        "model": omni_server.model,
        "messages": messages,
    }

    openai_client.send_omni_request(request_config)


@pytest.mark.omni
@pytest.mark.slow
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
@hardware_test(res={"cuda": "H100"}, num_cards=2)
def test_analyse_video_input(omni_server, openai_client):
    """
    Input Modal: video
    """
    video_data_url = (
        f"data:video/mp4;base64,{generate_synthetic_video(VIDEO_WIDTH, VIDEO_HEIGHT, NUM_VIDEO_FRAMES)['base64']}"
    )
    messages = dummy_messages_from_mix_data(
        system_prompt=get_system_prompt(), content_text=get_prompt(), video_data_url=video_data_url
    )

    request_config = {
        "model": omni_server.model,
        "messages": messages,
    }

    openai_client.send_omni_request(request_config)


@pytest.mark.omni
@pytest.mark.slow
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
@hardware_test(res={"cuda": "H100"}, num_cards=2)
def test_analyse_long_video_input(omni_server, openai_client):
    """
    Input Modal: long video
    """
    video_data_url = f"data:video/mp4;base64,{generate_synthetic_video(LONG_VIDEO_WIDTH, LONG_VIDEO_HEIGHT, LONG_VIDEO_NUM_FRAMES)['base64']}"
    messages = dummy_messages_from_mix_data(
        system_prompt=get_system_prompt(), content_text=get_prompt(), video_data_url=video_data_url
    )

    request_config = {
        "model": omni_server.model,
        "messages": messages,
    }

    openai_client.send_omni_request(request_config)


@pytest.mark.omni
@pytest.mark.slow
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
@hardware_test(res={"cuda": "H100"}, num_cards=2)
def test_analyse_video_with_audio_input(omni_server, openai_client):
    """
    Input Modal: video with audio
    """
    video_data_url = f"data:video/mp4;base64,{generate_synthetic_video(VIDEO_WIDTH, VIDEO_HEIGHT, NUM_VIDEO_FRAMES, embed_audio=True)['base64']}"
    messages = dummy_messages_from_mix_data(
        system_prompt=get_system_prompt(), content_text=get_prompt(), video_data_url=video_data_url
    )

    request_config = {
        "model": omni_server.model,
        "messages": messages,
    }

    openai_client.send_omni_request(request_config)
