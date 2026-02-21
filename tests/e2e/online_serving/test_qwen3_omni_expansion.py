# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E Online tests for Qwen3-Omni model.
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
from pathlib import Path

import pytest

from tests.conftest import (
    dummy_messages_from_mix_data,
    generate_synthetic_audio,
    generate_synthetic_image,
    generate_synthetic_video,
    modify_stage_config,
)
from tests.utils import hardware_test

models = ["Qwen/Qwen3-Omni-30B-A3B-Instruct"]

AUDIO_KEY = ["water", "chirping", "crackling"]
IMAGE_KEY = ["square", "quadrate"]
VIDEO_KEY = ["sphere", "globe", "circle", "round", "ball"]


def get_chunk_config(default_path):
    path = modify_stage_config(
        default_path,
        updates={
            "async_chunk": True,
            "stage_args": {
                0: {
                    "engine_args.custom_process_next_stage_input_func": "vllm_omni.model_executor.stage_input_processors.qwen3_omni.thinker2talker_async_chunk"
                },
                1: {
                    "engine_args.custom_process_next_stage_input_func": "vllm_omni.model_executor.stage_input_processors.qwen3_omni.talker2code2wav_async_chunk"
                },
            },
        },
        deletes={"stage_args": {2: ["custom_process_input_func"]}},
    )
    return path


# CI stage config for 2*H100-80G GPUs
default_path = str(Path(__file__).parent.parent / "stage_configs" / "qwen3_omni_ci.yaml")
stage_configs = [default_path, get_chunk_config(default_path)]
# Create parameter combinations for model and stage config
test_params = [(model, stage_config) for model in models for stage_config in stage_configs]


def get_system_prompt():
    return {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": (
                    "You are Qwen, a virtual human developed by the Qwen Team, "
                    "Alibaba Group, capable of perceiving auditory and visual inputs, "
                    "as well as generating text and speech."
                ),
            }
        ],
    }


def get_prompt(prompt_type="text_only"):
    prompts = {
        "text_only": "What is the capital of China?",
        "mix": "What is recited in the audio? What is in this image? What is in this video?",
        "text_video": "What is in this video? ",
        "text_image": "What is in this image? ",
        "text_audio": "What is in this audio? ",
    }
    return prompts.get(prompt_type, prompts["text_only"])


def get_max_batch_size(size_type="few"):
    batch_sizes = {"few": 5, "medium": 100, "large": 256}
    return batch_sizes.get(size_type, 5)


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_text_to_audio_001(omni_server, openai_client) -> None:
    messages = dummy_messages_from_mix_data(system_prompt=get_system_prompt(), content_text=get_prompt())

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "modalities": ["audio"],
        "key_words": {"text": ["beijing"]},
    }

    openai_client.send_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_text_to_text_audio_001(omni_server, openai_client) -> None:
    messages = dummy_messages_from_mix_data(system_prompt=get_system_prompt(), content_text=get_prompt())

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "key_words": {"text": ["beijing"]},
    }

    openai_client.send_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_image_to_text_001(omni_server, openai_client) -> None:
    image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(224, 224)['base64']}"
    messages = dummy_messages_from_mix_data(image_data_url=image_data_url)

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "modalities": ["text"],
        "key_words": {"image": IMAGE_KEY},
    }

    openai_client.send_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_image_to_audio_001(omni_server, openai_client) -> None:
    image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(224, 224)['base64']}"
    messages = dummy_messages_from_mix_data(image_data_url=image_data_url)

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "modalities": ["audio"],
        "key_words": {"image": IMAGE_KEY},
    }

    openai_client.send_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_image_to_text_audio_001(omni_server, openai_client) -> None:
    image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(1280, 720)['base64']}"

    messages = dummy_messages_from_mix_data(image_data_url=image_data_url)

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "key_words": {"image": IMAGE_KEY},
    }

    openai_client.send_request(request_config, request_num=get_max_batch_size())


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_video_to_text_001(omni_server, openai_client) -> None:
    video_data_url = f"data:video/mp4;base64,{generate_synthetic_video(224, 224, 300)['base64']}"
    messages = dummy_messages_from_mix_data(video_data_url=video_data_url)

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "modalities": ["text"],
        "key_words": {"video": VIDEO_KEY},
    }

    openai_client.send_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_video_to_audio_001(omni_server, openai_client) -> None:
    video_data_url = f"data:video/mp4;base64,{generate_synthetic_video(224, 224, 300)['base64']}"
    messages = dummy_messages_from_mix_data(video_data_url=video_data_url)

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "modalities": ["audio"],
        "key_words": {"video": VIDEO_KEY},
    }

    openai_client.send_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_video_to_text_audio_001(omni_server, openai_client) -> None:
    video_data_url = f"data:video/mp4;base64,{generate_synthetic_video(224, 224, 300)['base64']}"

    messages = dummy_messages_from_mix_data(video_data_url=video_data_url)

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "key_words": {"video": VIDEO_KEY},
    }

    openai_client.send_request(request_config, request_num=get_max_batch_size())


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_text_audio_to_text_audio_001(omni_server, openai_client) -> None:
    audio_data_url = f"data:audio/wav;base64,{generate_synthetic_audio(5, 1)['base64']}"
    messages = dummy_messages_from_mix_data(
        audio_data_url=audio_data_url, system_prompt=get_system_prompt(), content_text=get_prompt("text_audio")
    )

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "modalities": ["text"],
        "key_words": {"audio": AUDIO_KEY},
    }

    openai_client.send_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_text_image_to_text_audio_001(omni_server, openai_client) -> None:
    image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(224, 224)['base64']}"

    messages = dummy_messages_from_mix_data(
        image_data_url=image_data_url, system_prompt=get_system_prompt(), content_text=get_prompt("text_image")
    )

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "modalities": ["audio"],
        "key_words": {"image": IMAGE_KEY},
    }

    openai_client.send_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_text_video_to_text_audio_001(omni_server, openai_client) -> None:
    video_data_url = f"data:video/mp4;base64,{generate_synthetic_video(1280, 720, 30)['base64']}"

    messages = dummy_messages_from_mix_data(
        video_data_url=video_data_url, system_prompt=get_system_prompt(), content_text=get_prompt("text_video")
    )

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "key_words": {"video": VIDEO_KEY},
    }

    openai_client.send_request(request_config, request_num=get_max_batch_size())


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_mix_to_text_audio_001(omni_server, openai_client) -> None:
    video_data_url = f"data:video/mp4;base64,{generate_synthetic_video(224, 224, 300)['base64']}"
    image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(224, 224)['base64']}"
    audio_data_url = f"data:audio/wav;base64,{generate_synthetic_audio(5, 1)['base64']}"
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
        "key_words": {"audio": AUDIO_KEY, "image": IMAGE_KEY, "video": VIDEO_KEY},
    }
    openai_client.send_request(request_config, request_num=get_max_batch_size())
