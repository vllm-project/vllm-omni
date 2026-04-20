# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E online serving tests for Ming-flash-omni-2.0 model (Thinker stage).
Tests multimodal understanding via OpenAI-compatible API.
"""

import os
from pathlib import Path

import pytest

from tests.conftest import (
    OmniServerParams,
    dummy_messages_from_mix_data,
    generate_synthetic_audio,
    generate_synthetic_image,
    generate_synthetic_video,
    modify_stage_config,
)
from tests.utils import hardware_test

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "0"

models = ["Jonathan1909/Ming-flash-omni-2.0"]


def get_eager_config():
    path = modify_stage_config(
        str(Path(__file__).parent.parent / "stage_configs" / "bailingmm_moe_v2_lite_ci.yaml"),
        updates={
            "stage_args": {
                0: {
                    "engine_args.enforce_eager": "true",
                },
            },
        },
    )
    return path


stage_configs = [get_eager_config()]

# Create parameter combinations for model and stage config
test_params = [
    OmniServerParams(model=model, stage_config_path=stage_config) for model in models for stage_config in stage_configs
]


def get_system_prompt():
    return {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "你是一个友好的AI助手。\n\ndetailed thinking off",
            }
        ],
    }


def get_prompt(prompt_type="text_only"):
    prompts = {
        "text_only": "What is the capital of China? Answer in 20 words.",
        "text_image": "What is in this image?",
        "text_audio": "What is in this audio?",
        "text_video": "What is in this video?",
        "mix": "What is recited in the audio? What is in this image? What is in this video?",
    }
    return prompts.get(prompt_type, prompts["text_only"])


def get_max_batch_size(size_type="few"):
    batch_sizes = {"few": 5, "medium": 100, "large": 256}
    return batch_sizes.get(size_type, 5)


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=4)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_text_to_text_001(omni_server, openai_client) -> None:
    """
    Input Modal: text
    Output Modal: text
    Input Setting: stream=False
    Datasets: single request
    """
    messages = dummy_messages_from_mix_data(
        system_prompt=get_system_prompt(),
        content_text=get_prompt("text_only"),
    )

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "stream": False,
        "modalities": ["text"],
        "key_words": {"text": ["beijing"]},
    }

    openai_client.send_omni_request(request_config)


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=4)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_text_to_text_stream_001(omni_server, openai_client) -> None:
    """
    Input Modal: text
    Output Modal: text
    Input Setting: stream=True
    Datasets: few requests
    """
    messages = dummy_messages_from_mix_data(
        system_prompt=get_system_prompt(),
        content_text=get_prompt("text_only"),
    )

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "stream": True,
        "modalities": ["text"],
        "key_words": {"text": ["beijing"]},
    }

    openai_client.send_omni_request(request_config, request_num=get_max_batch_size())


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=4)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_image_to_text_001(omni_server, openai_client) -> None:
    """
    Input Modal: image + text
    Output Modal: text
    Input Setting: stream=True
    Datasets: single request
    """
    image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(224, 224)['base64']}"
    messages = dummy_messages_from_mix_data(
        system_prompt=get_system_prompt(),
        image_data_url=image_data_url,
        content_text=get_prompt("text_image"),
    )

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "stream": True,
        "modalities": ["text"],
    }

    openai_client.send_omni_request(request_config)


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=4)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_audio_to_text_001(omni_server, openai_client) -> None:
    """
    Input Modal: audio + text
    Output Modal: text
    Input Setting: stream=True
    Datasets: single request
    """
    audio_data_url = f"data:audio/wav;base64,{generate_synthetic_audio(2, 1)['base64']}"
    messages = dummy_messages_from_mix_data(
        system_prompt=get_system_prompt(),
        audio_data_url=audio_data_url,
        content_text=get_prompt("text_audio"),
    )

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "stream": True,
        "modalities": ["text"],
    }

    openai_client.send_omni_request(request_config)


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=4)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_video_to_text_001(omni_server, openai_client) -> None:
    """
    Input Modal: video + text
    Output Modal: text
    Input Setting: stream=False
    Datasets: single request
    """
    video_data_url = f"data:video/mp4;base64,{generate_synthetic_video(224, 224, 300)['base64']}"
    messages = dummy_messages_from_mix_data(
        system_prompt=get_system_prompt(),
        video_data_url=video_data_url,
        content_text=get_prompt("text_video"),
    )

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "stream": False,
        "modalities": ["text"],
    }

    openai_client.send_omni_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=4)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_mix_to_text_001(omni_server, openai_client) -> None:
    """
    Input Modal: text + audio + image + video
    Output Modal: text
    Input Setting: stream=True
    Datasets: single request
    """
    video_data_url = f"data:video/mp4;base64,{generate_synthetic_video(224, 224, 300)['base64']}"
    image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(224, 224)['base64']}"
    audio_data_url = f"data:audio/wav;base64,{generate_synthetic_audio(2, 1)['base64']}"
    messages = dummy_messages_from_mix_data(
        system_prompt=get_system_prompt(),
        video_data_url=video_data_url,
        image_data_url=image_data_url,
        audio_data_url=audio_data_url,
        content_text=get_prompt("mix"),
    )

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "stream": True,
        "modalities": ["text"],
    }

    openai_client.send_omni_request(request_config)
