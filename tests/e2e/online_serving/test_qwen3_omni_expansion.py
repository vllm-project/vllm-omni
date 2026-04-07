# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E Online tests for Qwen3-Omni model.
"""

import os

from vllm_omni.platforms import current_omni_platform

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
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

model = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

AUDIO_KEY = ["test"]
IMAGE_KEY = ["square", "quadrate", "rectangle"]
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


def get_batch_token_config(default_path):
    path = modify_stage_config(
        default_path,
        updates={
            "stage_args": {1: {"engine_args.max_num_batched_tokens": 64}},
        },
    )
    return path


# CI stage config for 2*H100-80G GPUs
default_path = str(Path(__file__).parent.parent / "stage_configs" / "qwen3_omni_ci.yaml")

if current_omni_platform.is_xpu():
    default_path = str(Path(__file__).parent.parent / "stage_configs" / "xpu" / "qwen3_omni_ci.yaml")

# Create parameter combinations for model and stage config
test_params = [
    pytest.param(OmniServerParams(model=model, stage_config_path=default_path), id="default"),
    pytest.param(OmniServerParams(model=model, stage_config_path=get_chunk_config(default_path)), id="async_chunk"),
]

test_token_params = [
    pytest.param(
        OmniServerParams(model=model, stage_config_path=get_batch_token_config(default_path)), id="batch_token_64"
    )
]


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
        "mix": "What is recited in the audio? What is in this image? What is in this video?",
        "text_video": "What is in this video? ",
        "text_image": "What is in this image? ",
        "text_audio": "What is in this audio? ",
        "text_audio_video": "First, what is in this audio? Then, what is in this video? ",
        "one_word": "What is the capital of UK? Answer in one word",
        "text_chinese": "北京，中国的首都，是一座融合了长城等历史地点与现代建筑的国际化大都市，充满了独特的文化与活力。请重复这句话。",
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
    """
    Input Modal: text
    Output Modal: audio
    Input Setting: stream=True
    Datasets: single request
    """
    messages = dummy_messages_from_mix_data(system_prompt=get_system_prompt(), content_text=get_prompt())

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "modalities": ["audio"],
        "stream": True,
        "key_words": {"text": ["beijing"]},
    }

    openai_client.send_omni_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params + test_token_params, indirect=True)
def test_text_to_text_audio_001(omni_server, openai_client) -> None:
    """
    Input Modal: text
    Output Modal: text, audio
    Input Setting: stream=False
    Datasets: few requests
    """
    messages = dummy_messages_from_mix_data(system_prompt=get_system_prompt(), content_text=get_prompt())

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "key_words": {"text": ["beijing"]},
    }

    openai_client.send_omni_request(request_config, request_num=get_max_batch_size())


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_image_to_text_001(omni_server, openai_client) -> None:
    """
    Input Modal: image
    Output Modal: text
    Input Setting: stream=True
    Datasets: single request
    """
    image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(224, 224)['base64']}"
    messages = dummy_messages_from_mix_data(image_data_url=image_data_url)

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "modalities": ["text"],
        "stream": True,
        "key_words": {"image": IMAGE_KEY},
    }

    openai_client.send_omni_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_image_to_audio_001(omni_server, openai_client) -> None:
    """
    Input Modal: image
    Output Modal: audio
    Input Setting: stream=False
    Datasets: single request
    """
    image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(224, 224)['base64']}"
    messages = dummy_messages_from_mix_data(image_data_url=image_data_url)

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "modalities": ["audio"],
        "key_words": {"image": IMAGE_KEY},
    }

    openai_client.send_omni_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_image_to_text_audio_001(omni_server, openai_client) -> None:
    """
    Input Modal: image
    Output Modal: text, audio
    Input Setting: stream=False
    Datasets: few requests
    """
    image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(1280, 720)['base64']}"

    messages = dummy_messages_from_mix_data(image_data_url=image_data_url)

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "key_words": {"image": IMAGE_KEY},
    }

    openai_client.send_omni_request(request_config, request_num=get_max_batch_size())


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_video_to_text_001(omni_server, openai_client) -> None:
    """
    Input Modal: video
    Output Modal: text
    Input Setting: stream=False
    Datasets: single request
    """
    video_data_url = f"data:video/mp4;base64,{generate_synthetic_video(224, 224, 300)['base64']}"
    messages = dummy_messages_from_mix_data(video_data_url=video_data_url)

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "modalities": ["text"],
        "key_words": {"video": VIDEO_KEY},
    }

    openai_client.send_omni_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_video_to_audio_001(omni_server, openai_client) -> None:
    """
    Input Modal: video
    Output Modal: audio
    Input Setting: stream=False
    Datasets: single request
    """
    video_data_url = f"data:video/mp4;base64,{generate_synthetic_video(224, 224, 300)['base64']}"
    messages = dummy_messages_from_mix_data(video_data_url=video_data_url)

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "modalities": ["audio"],
        "key_words": {"video": VIDEO_KEY},
    }

    openai_client.send_omni_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_video_to_text_audio_001(omni_server, openai_client) -> None:
    """
    Input Modal: video
    Output Modal: text, audio
    Input Setting: stream=False
    Datasets: few requests
    """
    video_data_url = f"data:video/mp4;base64,{generate_synthetic_video(224, 224, 300)['base64']}"

    messages = dummy_messages_from_mix_data(video_data_url=video_data_url)

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "key_words": {"video": VIDEO_KEY},
    }

    openai_client.send_omni_request(request_config, request_num=get_max_batch_size())


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params + test_token_params, indirect=True)
def test_text_audio_to_text_audio_001(omni_server, openai_client) -> None:
    """
    Input Modal: text, audio
    Output Modal: text, audio
    Input Setting: stream=False
    Datasets: single request
    """
    audio_data_url = f"data:audio/wav;base64,{generate_synthetic_audio(5, 1)['base64']}"
    messages = dummy_messages_from_mix_data(
        audio_data_url=audio_data_url, system_prompt=get_system_prompt(), content_text=get_prompt("text_audio")
    )

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "key_words": {"audio": AUDIO_KEY},
    }

    openai_client.send_omni_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params + test_token_params, indirect=True)
def test_text_image_to_text_audio_001(omni_server, openai_client) -> None:
    """
    Input Modal: text, image
    Output Modal: text, audio
    Input Setting: stream=False
    Datasets: single request
    """
    image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(224, 224)['base64']}"

    messages = dummy_messages_from_mix_data(
        image_data_url=image_data_url, system_prompt=get_system_prompt(), content_text=get_prompt("text_image")
    )

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "key_words": {"image": IMAGE_KEY},
    }

    openai_client.send_omni_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params + test_token_params, indirect=True)
def test_text_video_to_text_audio_001(omni_server, openai_client) -> None:
    """
    Input Modal: text, video
    Output Modal: text, audio
    Input Setting: stream=True
    Datasets: single requests
    """
    video_data_url = f"data:video/mp4;base64,{generate_synthetic_video(1280, 720, 30)['base64']}"

    messages = dummy_messages_from_mix_data(
        video_data_url=video_data_url, system_prompt=get_system_prompt(), content_text=get_prompt("text_video")
    )

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "stream": True,
        "key_words": {"video": VIDEO_KEY},
    }

    openai_client.send_omni_request(request_config)


@pytest.mark.skip(reason="There is a known issue with shape mismatch error.")
@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params + test_token_params, indirect=True)
def test_mix_to_text_audio_001(omni_server, openai_client) -> None:
    """
    Input Modal: text, audio, image, video
    Output Modal: text, audio
    Input Setting: stream=True
    Datasets: few requests
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
        "key_words": {"audio": AUDIO_KEY, "image": IMAGE_KEY, "video": VIDEO_KEY},
    }
    openai_client.send_omni_request(request_config, request_num=get_max_batch_size())


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_audio_in_video_001(omni_server, openai_client) -> None:
    """
    Input Modal: text + video (synthetic MP4 with embedded audio; ``use_audio_in_video`` uses audio from the video).
    Output Modal: text, audio
    Input Setting: stream=False
    Datasets: single request
    """
    video_data_url = f"data:video/mp4;base64,{generate_synthetic_video(224, 224, 300, embed_audio=True)['base64']}"
    messages = dummy_messages_from_mix_data(
        system_prompt=get_system_prompt(),
        video_data_url=video_data_url,
        content_text=get_prompt("text_audio_video"),
    )

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "stream": False,
        "use_audio_in_video": True,
        "key_words": {"video": VIDEO_KEY, "audio": AUDIO_KEY + ["beep", "electronic"]},
    }
    openai_client.send_omni_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_audio_in_video_002(omni_server, openai_client) -> None:
    """
    Input Modal: text + video (synthetic MP4 with embedded audio; ``use_audio_in_video`` uses audio from the video).
    Output Modal: text, audio
    Input Setting: stream=True
    Datasets: few requests
    """
    video_data_url = f"data:video/mp4;base64,{generate_synthetic_video(224, 224, 300, embed_audio=True)['base64']}"
    messages = dummy_messages_from_mix_data(
        system_prompt=get_system_prompt(),
        video_data_url=video_data_url,
        content_text=get_prompt("text_audio_video"),
    )

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "stream": True,
        "use_audio_in_video": True,
        "key_words": {"video": VIDEO_KEY},
    }

    openai_client.send_omni_request(request_config, request_num=get_max_batch_size())


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_one_word_prompt_001(omni_server, openai_client) -> None:
    """
    Input Modal: text only (one-word answer constraint).
    Output Modal: text, audio (default ``modalities``); ``key_words`` only assert on text.
    Input Setting: stream=True
    Datasets: single request
    """
    messages = dummy_messages_from_mix_data(
        system_prompt=get_system_prompt(),
        content_text=get_prompt("one_word"),
    )

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "stream": True,
        "key_words": {"text": ["london"]},
    }

    # Retry only when assert_omni_response fails on text/audio cosine similarity (see tests/conftest.py).
    _similarity_assert_msg = "The audio content is not same as the text"
    _max_retries = 3
    for attempt in range(_max_retries):
        try:
            openai_client.send_omni_request(request_config, request_num=get_max_batch_size())
            break
        except AssertionError as e:
            if _similarity_assert_msg not in str(e) or attempt == _max_retries - 1:
                raise
            print(f"Similarity assertion failed, retrying {attempt + 2}/{_max_retries}: {e!r}")


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_speaker_001(omni_server, openai_client) -> None:
    """
    Input Modal: text only (one-word answer constraint).
    Output Modal: text, audio (default ``modalities``); ``key_words`` only assert on text.
    Input Setting: stream=True
    Datasets: single request
    """
    messages = dummy_messages_from_mix_data(
        system_prompt=get_system_prompt(),
        content_text=get_prompt("text"),
    )

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "stream": True,
        "speaker": "Chelsie",
        "key_words": {"text": ["beijing"]},
    }

    openai_client.send_omni_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_speaker_002(omni_server, openai_client) -> None:
    """
    Input Modal: text only (one-word answer constraint).
    Output Modal: text, audio (default ``modalities``); ``key_words`` only assert on text.
    Input Setting: stream=True
    Datasets: single request
    """
    messages = dummy_messages_from_mix_data(
        system_prompt=get_system_prompt(),
        content_text=get_prompt("text"),
    )

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "stream": True,
        "speaker": "Ethan",
        "key_words": {"text": ["beijing"]},
    }

    # Retry only when assert_omni_response fails on preset voice gender (see tests/conftest.py).
    _gender_assert_substr = "estimated gender"
    _max_retries = 3
    for attempt in range(_max_retries):
        try:
            openai_client.send_omni_request(request_config, request_num=get_max_batch_size())
            break
        except AssertionError as e:
            if _gender_assert_substr not in str(e) or attempt == _max_retries - 1:
                raise
            print(f"Gender assertion failed, retrying {attempt + 2}/{_max_retries}: {e!r}")


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_speaker_003(omni_server, openai_client) -> None:
    """
    Input Modal: text only (one-word answer constraint).
    Output Modal: text, audio (default ``modalities``); ``key_words`` only assert on text.
    Input Setting: stream=True
    Datasets: single request
    """
    messages = dummy_messages_from_mix_data(
        system_prompt=get_system_prompt(),
        content_text=get_prompt("text"),
    )

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "stream": True,
        "speaker": "CHELSIE",
        "key_words": {"text": ["beijing"]},
    }

    openai_client.send_omni_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_language_001(omni_server, openai_client) -> None:
    """
    Input Modal: text only (one-word answer constraint).
    Output Modal: text, audio (default ``modalities``); ``key_words`` only assert on text.
    Input Setting: stream=True
    Datasets: single request
    """
    messages = dummy_messages_from_mix_data(
        system_prompt=get_system_prompt(),
        content_text=get_prompt("text_chinese"),
    )

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "stream": True,
        "key_words": {"text": ["北京"]},
    }

    openai_client.send_omni_request(request_config)
