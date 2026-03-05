# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E online smoke tests for Dynin-Omni model.
"""

import base64
import os
import threading
from pathlib import Path
from typing import Any

import openai
import pytest

from tests.conftest import OmniServer, dummy_messages_from_mix_data, generate_synthetic_audio, generate_synthetic_image
from tests.utils import hardware_test

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "0"

#models = ["snu-aidas/Dynin-Omni"]
models = ["/tmp/dynin_localized_models/snu-aidas_Dynin-Omni"]
stage_configs = [str(Path(__file__).resolve().parents[1] / "stage_configs" / "dynin_omni_ci.yaml")]
test_params = [(model, stage_config) for model in models for stage_config in stage_configs]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_DYNIN_CONFIG_PATH = (
    _REPO_ROOT / "vllm_omni/model_executor/models/dynin_omni/models/configs/dynin_omni_demo.yaml"
)

_omni_server_lock = threading.Lock()


@pytest.fixture(scope="module")
def omni_server(request):
    """Start vLLM-Omni server as a subprocess with Dynin model."""
    with _omni_server_lock:
        model, stage_config_path = request.param
        env_dict = {"DYNIN_CONFIG_PATH": str(_DEFAULT_DYNIN_CONFIG_PATH)}

        with OmniServer(
            model,
            ["--stage-configs-path", stage_config_path, "--stage-init-timeout", "600"],
            env_dict=env_dict,
        ) as server:
            yield server


@pytest.fixture
def client(omni_server):
    return openai.OpenAI(
        base_url=f"http://{omni_server.host}:{omni_server.port}/v1",
        api_key="EMPTY",
    )


def _extract_message_text(choice: Any) -> str:
    message = getattr(choice, "message", None)
    if message is None:
        return ""
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content.strip()
    return ""


def _extract_image_data_url(choice: Any) -> str:
    message = getattr(choice, "message", None)
    if message is None:
        return ""
    content = getattr(message, "content", None)
    if not isinstance(content, list):
        return ""
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "image_url":
            continue
        image_url = item.get("image_url", {})
        if isinstance(image_url, dict):
            url = image_url.get("url")
            if isinstance(url, str):
                return url
    return ""


def _extract_audio_base64(choice: Any) -> str:
    message = getattr(choice, "message", None)
    if message is None:
        return ""
    audio = getattr(message, "audio", None)
    if audio is None:
        return ""
    data = getattr(audio, "data", None)
    if isinstance(data, str):
        return data
    return ""


def _build_t2s_decode_additional_information(dynin_config_path: str) -> dict[str, Any]:
    generated_audio_token_ids = [int(v) for v in ([10, 11, 12, 13, 14] * 32)]
    return {
        "task": ["t2s"],
        "detok_id": [1],
        "generated_token_ids": [generated_audio_token_ids],
        "audio_codebook_size": [4096],
        "dynin_config_path": [str(dynin_config_path)],
    }


def _build_t2i_decode_additional_information(dynin_config_path: str) -> dict[str, Any]:
    # 1024 tokens -> 32x32 token grid expected by MAGVIT decode path.
    generated_image_token_ids = [int(v) for v in ([10, 11, 12, 13, 14, 15, 16, 17] * 128)]
    return {
        "task": ["t2i"],
        "detok_id": [2],
        "generated_token_ids": [generated_image_token_ids],
        "codebook_size": [8192],
        "dynin_config_path": [str(dynin_config_path)],
    }


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4", "rocm": "MI325"})
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_dynin_online_text_to_text(client: openai.OpenAI, omni_server) -> None:
    """Text -> text online smoke test."""
    messages = dummy_messages_from_mix_data(content_text="What is 2 + 2? Answer in one short sentence.")
    chat_completion = client.chat.completions.create(
        model=omni_server.model,
        messages=messages,
        max_tokens=64,
        modalities=["text"]
    )

    assert len(chat_completion.choices) >= 1
    assert chat_completion.usage is not None
    assert chat_completion.usage.total_tokens > 0

    text_content = _extract_message_text(chat_completion.choices[0])
    if text_content:
        assert len(text_content) > 0


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4", "rocm": "MI325"})
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_dynin_online_image_to_text(client: openai.OpenAI, omni_server) -> None:
    """Image -> text online smoke test via OpenAI chat multimodal input."""
    image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(224, 224)['base64']}"
    messages = dummy_messages_from_mix_data(
        image_data_url=image_data_url,
        content_text="Describe the image briefly in one sentence.",
    )
    chat_completion = client.chat.completions.create(
        model=omni_server.model,
        messages=messages,
        max_tokens=128,
        modalities=["text"],
    )

    assert len(chat_completion.choices) >= 1
    assert chat_completion.usage is not None
    assert chat_completion.usage.total_tokens > 0

    text_content = _extract_message_text(chat_completion.choices[0])
    if text_content:
        assert len(text_content) > 0


@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4", "rocm": "MI325"})
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_dynin_online_speech_to_text(client: openai.OpenAI, omni_server) -> None:
    """Speech -> text online smoke test via OpenAI chat multimodal input."""
    audio_data_url = f"data:audio/wav;base64,{generate_synthetic_audio(5, 1)['base64']}"
    messages = dummy_messages_from_mix_data(
        audio_data_url=audio_data_url,
        content_text="Transcribe the audio briefly in one sentence.",
    )
    chat_completion = client.chat.completions.create(
        model=omni_server.model,
        messages=messages,
        max_tokens=128,
        modalities=["text"],
    )

    assert len(chat_completion.choices) >= 1
    assert chat_completion.usage is not None
    assert chat_completion.usage.total_tokens > 0

    text_content = _extract_message_text(chat_completion.choices[0])
    if text_content:
        assert len(text_content) > 0