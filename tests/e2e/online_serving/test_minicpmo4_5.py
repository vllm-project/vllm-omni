# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E Online tests for MiniCPM-o 4.5 with async-chunk enabled.
"""

import os
from functools import lru_cache
from pathlib import Path

import pytest

from tests.conftest import (
    OmniServerParams,
    dummy_messages_from_mix_data,
    generate_synthetic_image,
    generate_synthetic_video,
)
from tests.utils import hardware_test

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "0"

MODEL = "openbmb/MiniCPM-o-4_5"
IMAGE_KEY = ["square", "quadrate", "rectangle"]
VIDEO_KEY = ["sphere", "globe", "circle", "round", "ball"]
ARTIFACT_DIR_ENV = "MINICPMO45_E2E_OUTPUT_DIR"
CHAT_TEMPLATE_PATH = str(
    Path(__file__).parent.parent.parent.parent
    / "vllm_omni"
    / "model_executor"
    / "models"
    / "minicpmo4_5"
    / "chat_template.jinja"
)


def get_stage_config() -> str:
    return str(
        Path(__file__).parent.parent.parent.parent
        / "vllm_omni"
        / "model_executor"
        / "stage_configs"
        / "minicpmo_async_chunk.yaml"
    )


@lru_cache(maxsize=1)
def _resolve_model_path() -> str:
    if os.path.isdir(MODEL):
        return MODEL

    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=MODEL, resume_download=True)


def get_system_prompt() -> dict:
    return {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": (
                    "You are MiniCPM, a helpful multimodal assistant. "
                    "When audio output is requested, reply with speech only."
                ),
            }
        ],
    }


def get_prompt(prompt_type: str = "text") -> str:
    prompts = {
        "text": "What is the capital of China? Answer in one short spoken sentence.",
        "image": "Describe the image briefly in one short spoken sentence.",
        "video": "Describe the video briefly in one short spoken sentence.",
    }
    return prompts.get(prompt_type, prompts["text"])


def get_audio_extra_body() -> dict[str, dict[str, bool]]:
    return {
        "chat_template_kwargs": {
            "use_tts_template": True,
            "enable_thinking": False,
        }
    }


def save_audio_artifacts(case_name: str, responses) -> None:
    artifact_root = os.environ.get(ARTIFACT_DIR_ENV)
    if not artifact_root:
        return

    output_dir = Path(artifact_root) / "online_serving" / "test_minicpmo4_5"
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, response in enumerate(responses):
        if response.audio_bytes is None:
            raise AssertionError(f"{case_name}: missing audio_bytes for artifact export")

        suffix = "" if len(responses) == 1 else f"_{idx}"
        wav_path = output_dir / f"{case_name}{suffix}.wav"
        wav_path.write_bytes(response.audio_bytes)

        transcript = (response.audio_content or "").strip()
        if transcript:
            txt_path = output_dir / f"{case_name}{suffix}.txt"
            txt_path.write_text(transcript, encoding="utf-8")


def save_text_artifacts(case_name: str, responses) -> None:
    artifact_root = os.environ.get(ARTIFACT_DIR_ENV)
    if not artifact_root:
        return

    output_dir = Path(artifact_root) / "online_serving" / "test_minicpmo4_5"
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, response in enumerate(responses):
        if response.text_content is None:
            raise AssertionError(f"{case_name}: missing text_content for artifact export")

        suffix = "" if len(responses) == 1 else f"_{idx}"
        txt_path = output_dir / f"{case_name}{suffix}.txt"
        txt_path.write_text(response.text_content, encoding="utf-8")


minicpmo_server_params = [
    pytest.param(
        OmniServerParams(
            model=_resolve_model_path(),
            stage_config_path=get_stage_config(),
            server_args=[
                "--trust-remote-code",
                "--disable-log-stats",
                "--chat-template",
                CHAT_TEMPLATE_PATH,
                "--chat-template-content-format",
                "openai",
            ],
        ),
        id="async_chunk",
    )
]


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("omni_server", minicpmo_server_params, indirect=True)
def test_text_to_audio_001(omni_server, openai_client) -> None:
    """
    Input Modal: text
    Output Modal: audio
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
        "modalities": ["audio"],
        "stream": True,
        "extra_body": get_audio_extra_body(),
        "key_words": {"text": ["beijing"]},
    }

    openai_client.send_omni_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("omni_server", minicpmo_server_params, indirect=True)
def test_image_to_text_001(omni_server, openai_client) -> None:
    """
    Input Modal: image
    Output Modal: text
    Input Setting: stream=False
    Datasets: single request
    """
    image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(224, 224)['base64']}"
    messages = dummy_messages_from_mix_data(
        system_prompt=get_system_prompt(),
        image_data_url=image_data_url,
        content_text=get_prompt("image"),
    )

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "modalities": ["text"],
        "key_words": {"image": IMAGE_KEY},
    }

    responses = openai_client.send_omni_request(request_config)
    save_text_artifacts("image_to_text_001", responses)


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("omni_server", minicpmo_server_params, indirect=True)
def test_image_to_audio_001(omni_server, openai_client) -> None:
    """
    Input Modal: image
    Output Modal: audio
    Input Setting: stream=False
    Datasets: single request
    """
    image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(224, 224)['base64']}"
    messages = dummy_messages_from_mix_data(
        system_prompt=get_system_prompt(),
        image_data_url=image_data_url,
        content_text=get_prompt("image"),
    )

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "modalities": ["audio"],
        "extra_body": get_audio_extra_body(),
        "key_words": {"image": IMAGE_KEY},
    }

    responses = openai_client.send_omni_request(request_config)
    save_audio_artifacts("image_to_audio_001", responses)


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("omni_server", minicpmo_server_params, indirect=True)
def test_video_to_text_001(omni_server, openai_client) -> None:
    """
    Input Modal: video
    Output Modal: text
    Input Setting: stream=False
    Datasets: single request
    """
    video_data_url = f"data:video/mp4;base64,{generate_synthetic_video(224, 224, 300)['base64']}"
    messages = dummy_messages_from_mix_data(
        system_prompt=get_system_prompt(),
        video_data_url=video_data_url,
        content_text=get_prompt("video"),
    )

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "modalities": ["text"],
        "key_words": {"video": VIDEO_KEY},
    }

    responses = openai_client.send_omni_request(request_config)
    save_text_artifacts("video_to_text_001", responses)


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100"}, num_cards=2)
@pytest.mark.parametrize("omni_server", minicpmo_server_params, indirect=True)
def test_video_to_audio_001(omni_server, openai_client) -> None:
    """
    Input Modal: video
    Output Modal: audio
    Input Setting: stream=False
    Datasets: single request
    """
    video_data_url = f"data:video/mp4;base64,{generate_synthetic_video(224, 224, 300)['base64']}"
    messages = dummy_messages_from_mix_data(
        system_prompt=get_system_prompt(),
        video_data_url=video_data_url,
        content_text=get_prompt("video"),
    )

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "modalities": ["audio"],
        "extra_body": get_audio_extra_body(),
        "key_words": {"video": VIDEO_KEY},
    }

    responses = openai_client.send_omni_request(request_config)
    save_audio_artifacts("video_to_audio_001", responses)
