"""
E2E Online tests for Qwen3-Omni model with video input and audio output.
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "0"

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
from vllm_omni.platforms import current_omni_platform

models = ["Qwen/Qwen3-Omni-30B-A3B-Instruct"]


def get_chunk_config():
    path = modify_stage_config(
        str(Path(__file__).parent.parent / "stage_configs" / "qwen3_omni_ci.yaml"),
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


# CI stage config for 2xH100-80G GPUs or AMD GPU MI325
if current_omni_platform.is_rocm():
    # ROCm stage config optimized for MI325 GPU
    stage_configs = [str(Path(__file__).parent.parent / "stage_configs" / "rocm" / "qwen3_omni_ci.yaml")]
else:
    stage_configs = [get_chunk_config()]

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
        "text_only": "What is the capital of China? Answer in 20 words.",
        "mix": "What is recited in the audio? What is in this image? Describe the video briefly.",
    }
    return prompts.get(prompt_type, prompts["text_only"])


def get_max_batch_size(size_type="few"):
    batch_sizes = {"few": 5, "medium": 100, "large": 256}
    return batch_sizes.get(size_type, 5)


@pytest.mark.advanced_model
@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_mix_to_text_audio_001(omni_server, openai_client) -> None:
    """
    Test multi-modal input processing and text/audio output generation via OpenAI API.
    Deploy Setting: default yaml
    Input Modal: text + audio + video + image
    Output Modal: text + audio
    Input Setting: stream=True
    Datasets: single request
    """

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
        "key_words": {
            "audio": ["water", "chirping"],
            "image": ["square", "quadrate"],
        },
    }

    # Test single completion
    openai_client.send_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.core_model
@pytest.mark.omni
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_text_to_text_001(omni_server, openai_client) -> None:
    """
    Test text input processing and text/audio output generation via OpenAI API.
    Deploy Setting: default yaml
    Input Modal: text
    Output Modal: text + audio
    Datasets: few requests
    """
    messages = dummy_messages_from_mix_data(system_prompt=get_system_prompt(), content_text=get_prompt())

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "stream": False,
        "modalities": ["text"],
        "key_words": {"text": ["beijing"]},
    }

    openai_client.send_request(request_config, request_num=get_max_batch_size())
