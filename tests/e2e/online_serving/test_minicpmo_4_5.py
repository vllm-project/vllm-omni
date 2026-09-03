# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""E2E online tests for MiniCPM-o 4.5 multimodal input and audio/text output.

Exercises async chunk streaming (``--async-chunk``) across separate Thinker,
Talker, and Code2Wav stages.
"""

import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path

import pytest
from vllm.logger import DEFAULT_LOGGING_CONFIG

from tests.helpers.mark import hardware_test
from tests.helpers.media import generate_synthetic_audio, generate_synthetic_image, generate_synthetic_video
from tests.helpers.runtime import OmniServerParams, dummy_messages_from_mix_data
from tests.helpers.stage_config import get_deploy_config_path, modify_stage_config

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

_MODEL = "openbmb/MiniCPM-o-4_5"
_CI_DEPLOY = modify_stage_config(
    get_deploy_config_path("minicpmo_4_5.yaml"),
    updates={
        "stages": {
            0: {"default_sampling_params.max_tokens": 64},
            1: {
                "default_sampling_params.max_tokens": 1024,
                # Content-consistency assertions must not depend on a random
                # codec trajectory: different valid TTS samples can append an
                # audible tail even when Thinker text is identical.
                "default_sampling_params.temperature": 0.0,
            },
        },
    },
)

_PROMPT_LOG_TEMP_DIR = tempfile.TemporaryDirectory(prefix="minicpmo45-prompt-selection-")
_PROMPT_SELECTION_LOG = Path(_PROMPT_LOG_TEMP_DIR.name) / "prompt-selection.log"
_PROMPT_LOGGING_CONFIG_PATH = Path(_PROMPT_LOG_TEMP_DIR.name) / "logging.json"
_PROMPT_LOGGING_CONFIG = deepcopy(DEFAULT_LOGGING_CONFIG)
_PROMPT_LOGGING_CONFIG["handlers"]["minicpmo_prompt_selection"] = {
    "class": "logging.FileHandler",
    "filename": str(_PROMPT_SELECTION_LOG),
    "formatter": "vllm",
    "level": "DEBUG",
}
_PROMPT_LOGGING_CONFIG["loggers"]["vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_code2wav"] = {
    "handlers": ["minicpmo_prompt_selection"],
    "level": "DEBUG",
    "propagate": True,
}
_PROMPT_LOGGING_CONFIG_PATH.write_text(json.dumps(_PROMPT_LOGGING_CONFIG), encoding="utf-8")

test_params = [
    pytest.param(
        OmniServerParams(
            model=_MODEL,
            stage_config_path=_CI_DEPLOY,
            use_stage_cli=False,
            server_args=["--trust-remote-code", "--async-chunk"],
            env_dict={
                "VLLM_CONFIGURE_LOGGING": "1",
                "VLLM_LOGGING_CONFIG_PATH": str(_PROMPT_LOGGING_CONFIG_PATH),
            },
        ),
        id="async_chunk",
    ),
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
        "text_image": "What color are the squares in this image?",
    }
    return prompts.get(prompt_type, prompts["text_only"])


def get_max_batch_size(size_type="few"):
    batch_sizes = {"few": 5, "medium": 100, "large": 256}
    return batch_sizes.get(size_type, 5)


# Close <think> and emit <|tts_bos|> so Talker speaks the answer, not reasoning.
_TTS_EXTRA_BODY = {
    "chat_template_kwargs": {
        "use_tts_template": True,
        "enable_thinking": False,
    }
}


def _prompt_selection_log_offset() -> int:
    return _PROMPT_SELECTION_LOG.stat().st_size if _PROMPT_SELECTION_LOG.exists() else 0


def _read_prompt_selection_log(offset: int) -> str:
    if not _PROMPT_SELECTION_LOG.exists():
        return ""
    with open(_PROMPT_SELECTION_LOG, "rb") as log_file:
        log_file.seek(offset)
        return log_file.read().decode("utf-8")


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "npu": "A3"}, num_cards=1)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_text_to_text_001(omni_server, openai_client) -> None:
    """
    Test text-only input generating text output via OpenAI API.
    Deploy Setting: default single GPU
    Input Modal: text
    Output Modal: text
    Input Setting: stream=False
    """
    messages = dummy_messages_from_mix_data(system_prompt=get_system_prompt(), content_text=get_prompt())

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "stream": False,
        "modalities": ["text"],
        "key_words": {"text": ["Beijing"]},
    }

    openai_client.send_omni_request(request_config, request_num=get_max_batch_size())


@pytest.mark.full_model
@pytest.mark.omni
@hardware_test(res={"cuda": ["H100", "B200"], "npu": "A3"}, num_cards=1)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_text_to_audio_001(omni_server, openai_client) -> None:
    """
    Test text-only input generating text + audio output via OpenAI API.
    This exercises Talker TTS region detection and the Code2Wav stage.
    Deploy Setting: default single GPU
    Input Modal: text
    Output Modal: text + audio
    Input Setting: stream=True
    """
    messages = dummy_messages_from_mix_data(system_prompt=get_system_prompt(), content_text=get_prompt())

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "stream": True,
        "key_words": {"audio": ["Beijing"]},
        "extra_body": _TTS_EXTRA_BODY,
    }

    openai_client.send_omni_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "npu": "A3"}, num_cards=1)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_text_to_audio_with_reference_audio(omni_server, openai_client) -> None:
    request_config = {
        "model": omni_server.model,
        "messages": dummy_messages_from_mix_data(system_prompt=get_system_prompt(), content_text=get_prompt()),
        "stream": True,
        "modalities": ["text", "audio"],
        "extra_body": {
            **_TTS_EXTRA_BODY,
            "ref_audio": f"data:audio/wav;base64,{generate_synthetic_audio(5, 1)['base64']}",
        },
    }

    log_offset = _prompt_selection_log_offset()
    responses = openai_client.send_omni_request(request_config)
    prompt_log = _read_prompt_selection_log(log_offset)

    assert responses[0].success
    assert responses[0].audio_bytes
    assert "prompt_cache_id=runtime-ref-" in prompt_log
    assert "minicpmo45_ref_" in prompt_log


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "npu": "A3"}, num_cards=1)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_text_to_audio_with_default_reference(omni_server, openai_client) -> None:
    request_config = {
        "model": omni_server.model,
        "messages": dummy_messages_from_mix_data(system_prompt=get_system_prompt(), content_text=get_prompt()),
        "stream": True,
        "modalities": ["text", "audio"],
        "extra_body": _TTS_EXTRA_BODY,
    }

    responses = openai_client.send_omni_request(request_config)

    assert responses[0].success
    assert responses[0].audio_bytes


@pytest.mark.full_model
@pytest.mark.omni
@hardware_test(res={"cuda": ["H100", "B200"], "npu": "A3"}, num_cards=1)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_audio_to_text_audio_001(omni_server, openai_client) -> None:
    """
    Test audio input generating text + audio output via OpenAI API.
    Deploy Setting: default single GPU
    Input Modal: text + audio
    Output Modal: text + audio
    Input Setting: stream=True
    """
    audio_data_url = f"data:audio/wav;base64,{generate_synthetic_audio(5, 1)['base64']}"
    messages = dummy_messages_from_mix_data(
        system_prompt=get_system_prompt(),
        audio_data_url=audio_data_url,
        content_text=get_prompt(),
    )

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "stream": True,
        "key_words": {"text": ["Beijing"]},
        "modalities": ["text", "audio"],
        "extra_body": _TTS_EXTRA_BODY,
    }

    openai_client.send_omni_request(request_config, request_num=get_max_batch_size())


@pytest.mark.full_model
@pytest.mark.omni
@hardware_test(res={"cuda": ["H100", "B200"], "npu": "A3"}, num_cards=1)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_image_to_text_audio_001(omni_server, openai_client) -> None:
    """
    Test image input generating text + audio output via OpenAI API.
    Deploy Setting: default single GPU
    Input Modal: text + image
    Output Modal: text + audio
    Input Setting: stream=True
    """
    image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(24, 24)['base64']}"
    messages = dummy_messages_from_mix_data(
        system_prompt=get_system_prompt(),
        image_data_url=image_data_url,
        content_text=get_prompt("text_image"),
    )

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "stream": True,
        "extra_body": _TTS_EXTRA_BODY,
    }

    openai_client.send_omni_request(request_config, request_num=get_max_batch_size())


@pytest.mark.full_model
@pytest.mark.omni
@hardware_test(res={"cuda": ["H100", "B200"], "npu": "A3"}, num_cards=1)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_video_to_text_audio_001(omni_server, openai_client) -> None:
    """
    Test video input generating text + audio output via OpenAI API.
    Deploy Setting: default single GPU
    Input Modal: text + video
    Output Modal: text + audio
    Input Setting: stream=True
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
        "stream": True,
        "extra_body": _TTS_EXTRA_BODY,
    }

    openai_client.send_omni_request(request_config, request_num=get_max_batch_size())


@pytest.mark.core_model
@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "H100", "npu": "A3"}, num_cards=1)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_mix_to_text_audio_001(omni_server, openai_client) -> None:
    """
    Test multi-modal input (text + audio + video + image) generating text + audio output.
    Deploy Setting: default single GPU
    Input Modal: text + audio + video + image
    Output Modal: text + audio
    Input Setting: stream=True
    """
    video_data_url = f"data:video/mp4;base64,{generate_synthetic_video(24, 24, 20)['base64']}"
    image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(24, 24)['base64']}"
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
        "extra_body": _TTS_EXTRA_BODY,
    }

    openai_client.send_omni_request(request_config, request_num=get_max_batch_size())
