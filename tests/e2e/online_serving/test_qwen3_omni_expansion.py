# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E Online tests for Qwen3-Omni model.
"""

import os

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.media import generate_synthetic_audio, generate_synthetic_image, generate_synthetic_video
from tests.helpers.runtime import OmniServerParams, dummy_messages_from_mix_data
from tests.helpers.stage_config import get_deploy_config_path, modify_stage_config

pytestmark = [pytest.mark.full_model, pytest.mark.omni]

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

model = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

AUDIO_KEY = ["test"]
IMAGE_KEY = ["square", "quadrate", "rectangle"]
VIDEO_KEY = ["sphere", "globe", "circle", "round", "ball"]

# Heavier synthetic inputs than the default expansion cases (longer timeline / more pixels).
# Long video: 120s @ 30fps => 3600 frames (generate_synthetic_video in tests/conftest.py).
# Use 224² spatial size to bound RAM (~W*H*num_frames*3) vs. 288² at this frame count.
LONG_VIDEO_WIDTH = 224
LONG_VIDEO_HEIGHT = 224
LONG_VIDEO_FRAMES = 3600
LARGE_IMAGE_WIDTH = 1920
LARGE_IMAGE_HEIGHT = 1080
LONG_AUDIO_DURATION_SEC = 120


def get_batch_token_config(default_path):
    """Override stage 1's max_num_batched_tokens to exercise small-batch paths.

    Uses the new flat-stage schema (``stages.<id>.<field>``); the legacy
    ``stage_args.<id>.engine_args.<field>`` path no longer applies because
    the deploy YAML doesn't nest engine fields under ``engine_args:``.
    """
    return modify_stage_config(
        default_path,
        updates={
            "stages": {1: {"max_num_batched_tokens": 64}},
        },
    )


def get_default_config(default_path):
    """Flip async_chunk on and bump stage 0 thinker output to 2048 tokens.

    Pipeline registry (qwen3_omni/pipeline.py) already wires
    thinker2talker_async_chunk / talker2code2wav_async_chunk on stages 0/1,
    so no per-stage processor override is needed. Using only flat-schema
    writes so _parse_stage_deploy stays in its flat branch (nested
    ``engine_args:`` would drop other overlay fields).
    """
    return modify_stage_config(
        default_path,
        updates={
            "stages": {0: {"default_sampling_params.max_tokens": 2048}},
        },
    )


# CI deploy YAML (single file; xpu deltas applied via ``platforms:`` section).
# The overlay explicitly sets ``async_chunk: False``, so ``default`` tests the
# sync path and ``async_chunk`` tests the streaming path with a longer thinker
# output — two distinct scenarios, kept as separate parametrizations.
default_path = get_deploy_config_path("ci/qwen3_omni_moe.yaml")

test_params = [
    pytest.param(
        OmniServerParams(
            model=model,
            stage_config_path=get_default_config(default_path),
            use_stage_cli=True,
            server_args=["--no-async-chunk"],
        ),
        id="default",
    ),
    pytest.param(
        OmniServerParams(
            model=model,
            stage_config_path=get_default_config(default_path),
            use_stage_cli=True,
            server_args=["--async-chunk"],
        ),
        id="async_chunk",
    ),
]

test_token_params = [
    pytest.param(
        OmniServerParams(model=model, stage_config_path=get_batch_token_config(default_path), use_stage_cli=True),
        id="batch_token_64",
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


@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_text_video_to_text_001(omni_server, openai_client) -> None:
    """
    Input Modal: long synthetic video (120s @ 30fps, LONG_VIDEO_FRAMES frames)
    Output Modal: text
    Input Setting: stream=False
    Datasets: single request
    """
    video_data_url = f"data:video/mp4;base64,{generate_synthetic_video(LONG_VIDEO_WIDTH, LONG_VIDEO_HEIGHT, LONG_VIDEO_FRAMES)['base64']}"
    messages = dummy_messages_from_mix_data(
        video_data_url=video_data_url, system_prompt=get_system_prompt(), content_text=get_prompt("text_video")
    )

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "modalities": ["text"],
        "key_words": {"video": VIDEO_KEY},
    }

    openai_client.send_omni_request(request_config, request_num=get_max_batch_size())


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


@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params + test_token_params, indirect=True)
def test_text_audio_to_text_audio_002(omni_server, openai_client) -> None:
    """
    Input Modal: text, long-duration audio (~LONG_AUDIO_DURATION_SEC s WAV)
    Output Modal: text, audio
    Input Setting: stream=False
    Datasets: single request
    """
    audio_data_url = f"data:audio/wav;base64,{generate_synthetic_audio(LONG_AUDIO_DURATION_SEC, 1)['base64']}"
    messages = dummy_messages_from_mix_data(
        audio_data_url=audio_data_url,
        system_prompt=get_system_prompt(),
        content_text=get_prompt("text_audio"),
    )

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "key_words": {"audio": AUDIO_KEY},
    }

    openai_client.send_omni_request(request_config, request_num=get_max_batch_size())


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


@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
@pytest.mark.parametrize("omni_server", test_params + test_token_params, indirect=True)
def test_large_image_to_text_audio_001(omni_server, openai_client) -> None:
    """
    Input Modal: text, high-resolution image (1080p-class JPEG)
    Output Modal: text, audio
    Input Setting: stream=False
    Datasets: single request
    """
    image_data_url = (
        f"data:image/jpeg;base64,{generate_synthetic_image(LARGE_IMAGE_WIDTH, LARGE_IMAGE_HEIGHT)['base64']}"
    )

    messages = dummy_messages_from_mix_data(
        image_data_url=image_data_url,
        system_prompt=get_system_prompt(),
        content_text=get_prompt("text_image"),
    )

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "key_words": {"image": IMAGE_KEY},
    }

    openai_client.send_omni_request(request_config, request_num=get_max_batch_size())


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
        "key_words": {"video": VIDEO_KEY},
    }
    openai_client.send_omni_request(request_config)


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

    # Retry only when assert_omni_response fails on text/audio cosine similarity (see tests/helpers/assertions.py).
    _similarity_assert_msg = "The audio content is not same as the text"
    _max_retries = 10
    for attempt in range(_max_retries):
        try:
            openai_client.send_omni_request(request_config, request_num=get_max_batch_size())
            break
        except AssertionError as e:
            if _similarity_assert_msg not in str(e) or attempt == _max_retries - 1:
                raise
            print(f"Similarity assertion failed, retrying {attempt + 2}/{_max_retries}: {e!r}")


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

    # Retry only when assert_omni_response fails on preset voice gender (see tests/helpers/assertions.py).
    _gender_assert_substr = "estimated gender"
    _max_retries = 10
    for attempt in range(_max_retries):
        try:
            openai_client.send_omni_request(request_config, request_num=get_max_batch_size())
            break
        except AssertionError as e:
            if _gender_assert_substr not in str(e) or attempt == _max_retries - 1:
                raise
            print(f"Gender assertion failed, retrying {attempt + 2}/{_max_retries}: {e!r}")


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
