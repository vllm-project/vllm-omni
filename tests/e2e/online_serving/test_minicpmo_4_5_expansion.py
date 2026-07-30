"""E2E online expansion tests for MiniCPM-o 4.5.

These cover modality combinations, separate Code2Wav behavior, sequential
request isolation, longer audio output, and Chinese speech.
"""

import os

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.media import generate_synthetic_video
from tests.helpers.runtime import OmniServerParams, dummy_messages_from_mix_data
from tests.helpers.stage_config import get_deploy_config_path

pytestmark = [pytest.mark.full_model, pytest.mark.omni]

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

_MODEL = "openbmb/MiniCPM-o-4_5"
_CI_DEPLOY = get_deploy_config_path("minicpmo_4_5.yaml")

test_params = [
    pytest.param(
        OmniServerParams(
            model=_MODEL,
            stage_config_path=_CI_DEPLOY,
            use_stage_cli=False,
            server_args=[
                "--trust-remote-code",
            ],
        ),
        id="default",
    )
]


def get_system_prompt():
    return {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": (
                    "You are MiniCPM-o 4.5, a virtual human capable of perceiving "
                    "auditory and visual inputs, as well as generating text and speech."
                ),
            }
        ],
    }


def get_prompt(prompt_type: str = "text_only") -> str:
    prompts = {
        "text_only": "What is the capital of China? Answer in 20 words.",
        "mix": "What is recited in the audio? What is in this image? Describe the video briefly.",
    }
    return prompts.get(prompt_type, prompts["text_only"])


def get_max_batch_size(size_type="few"):
    batch_sizes = {"few": 5, "medium": 100, "large": 256}
    return batch_sizes.get(size_type, 5)


@hardware_test(res={"cuda": "H100", "npu": "A3"}, num_cards=1)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_text_video_to_text_001(omni_server, openai_client) -> None:
    """
    Test text + video input generating text output.
    Deploy Setting: default single GPU
    Input Modal: text + video
    Output Modal: text
    Input Setting: stream=False
    """
    video_data_url = f"data:video/mp4;base64,{generate_synthetic_video(24, 24, 20)['base64']}"
    messages = dummy_messages_from_mix_data(
        system_prompt=get_system_prompt(),
        video_data_url=video_data_url,
        content_text=get_prompt("mix"),
    )

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "stream": False,
        "modalities": ["text"],
    }

    openai_client.send_omni_request(request_config, request_num=get_max_batch_size())


@hardware_test(res={"cuda": "H100", "npu": "A3"}, num_cards=1)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_sequential_requests_independent(omni_server, openai_client) -> None:
    """
    Verify that sequential requests produce independent results and that the
    second request does not receive the first request's audio or token state.
    Deploy Setting: default single GPU
    Input Modal: text (two different prompts)
    Output Modal: text + audio (both)
    """
    messages_1 = dummy_messages_from_mix_data(
        system_prompt=get_system_prompt(),
        content_text="What is the capital of France?",
    )
    messages_2 = dummy_messages_from_mix_data(
        system_prompt=get_system_prompt(),
        content_text="What is the capital of China?",
    )

    # Send first request and ensure it completes
    openai_client.send_omni_request(
        {
            "model": omni_server.model,
            "messages": messages_1,
            "stream": True,
        },
        request_num=1,
    )

    # Send second request with different prompt; must produce its own output
    openai_client.send_omni_request(
        {
            "model": omni_server.model,
            "messages": messages_2,
            "stream": True,
            "key_words": {"text": ["Beijing"]},
        },
        request_num=1,
    )


@hardware_test(res={"cuda": "H100", "npu": "A3"}, num_cards=1)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_text_to_audio_long_output_001(omni_server, openai_client) -> None:
    """
    Test text input generating a longer audio output to exercise the
    Code2Wav stage across multiple frames.
    Deploy Setting: default single GPU
    Input Modal: text (longer prompt)
    Output Modal: text + audio
    Input Setting: stream=True
    """
    messages = dummy_messages_from_mix_data(
        system_prompt=get_system_prompt(),
        content_text="What is the capital of China? Answer in 40 words.",
    )

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "modalities": ["text", "audio"],
        "stream": True,
        # Delimit the assistant answer with the TTS template so the Talker
        # does not spend its codec-token budget speaking hidden reasoning.
        "extra_body": {
            "chat_template_kwargs": {
                "use_tts_template": True,
                "enable_thinking": False,
            }
        },
        "key_words": {"audio": ["Beijing"]},
    }

    openai_client.send_omni_request(request_config, request_num=get_max_batch_size())


@hardware_test(res={"cuda": "H100", "npu": "A3"}, num_cards=1)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_chinese_text_to_audio(omni_server, openai_client) -> None:
    """
    Test Chinese text input generating audio output.
    """
    messages = dummy_messages_from_mix_data(
        system_prompt=get_system_prompt(),
        content_text="北京，中国的首都，是一座融合了长城等历史地点与现代建筑的国际化大都市，充满了独特的文化与活力。请重复这句话。",
    )
    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "stream": True,
        "key_words": {"text": ["北京"]},
    }
    openai_client.send_omni_request(request_config)
