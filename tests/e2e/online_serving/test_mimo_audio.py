# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E Online tests for MiMo-Audio model with audio/text input and audio output.
"""

import os
from pathlib import Path

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.media import generate_synthetic_audio
from tests.helpers.runtime import OmniServerParams, dummy_messages_from_mix_data
from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.model_executor.model_loader.weight_utils import (
    download_weights_from_hf_specific,
)

MIMO_AUDIO_TOKENIZER_REPO = "XiaomiMiMo/MiMo-Audio-Tokenizer"
CHAT_TEMPLATE_PATH = str(
    Path(__file__).parent.parent.parent.parent / "examples" / "online_serving" / "mimo_audio" / "chat_template.jinja"
)
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

models = ["XiaomiMiMo/MiMo-Audio-7B-Instruct"]


def download_tokenizer():
    tokenizer_path = os.environ.get("MIMO_AUDIO_TOKENIZER_PATH", MIMO_AUDIO_TOKENIZER_REPO)
    if os.path.exists(tokenizer_path):
        return tokenizer_path
    local_path = download_weights_from_hf_specific(
        model_name_or_path=MIMO_AUDIO_TOKENIZER_REPO,
        cache_dir=None,
        allow_patterns=["*"],
        require_all=True,
    )
    return local_path


# Guard module-level setup so test collection doesn't fail in environments
# where the model cache is read-only or models aren't available.
try:
    stage_configs = [get_deploy_config_path("mimo_audio.yaml")]
    tokenizer_path = download_tokenizer()
    os.environ["MIMO_AUDIO_TOKENIZER_PATH"] = tokenizer_path

    # --load-format dummy applies to every stage pipeline-wide, avoiding a
    # per-stage yaml rewrite (the old approach wrote a tempfile + atexit-unlink
    # which raced with CI's process lifecycle).
    test_params = [
        OmniServerParams(
            model=model,
            stage_config_path=stage_config,
            server_args=["--chat-template", CHAT_TEMPLATE_PATH, "--load-format", "dummy"],
        )
        for model in models
        for stage_config in stage_configs
    ]
except Exception as exc:
    pytest.skip(
        f"MiMo-Audio online serving tests skipped: module setup failed ({type(exc).__name__}: {exc})",
        allow_module_level=True,
    )


def get_prompt(prompt_type="text_only"):
    prompts = {
        "text_only": "What is the capital of China? Answer in 20 words.",
        "audio": "What is recited in the audio?",
    }
    return prompts.get(prompt_type, prompts["text_only"])


def get_max_batch_size(size_type="few"):
    batch_sizes = {"few": 5, "medium": 100, "large": 256}
    return batch_sizes.get(size_type, 5)


@pytest.mark.advanced_model
@pytest.mark.core_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards=1)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_audio_to_text_audio_001(omni_server, openai_client) -> None:
    """
    Test audio and text input processing and text/audio output generation via OpenAI API.
    Deploy Setting: default yaml
    Input Modal: text + audio
    Output Modal: text + audio
    Input Setting: stream=True
    Datasets: single request
    """

    audio_data_url = f"data:audio/wav;base64,{generate_synthetic_audio(5, 1, sample_rate=24000)['base64']}"
    messages = dummy_messages_from_mix_data(
        audio_data_url=audio_data_url,
        content_text=get_prompt("audio"),
    )

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "stream": True,
        "key_words": {
            "audio": ["test"],
        },
    }

    # Test single completion
    openai_client.send_omni_request(request_config)


@pytest.mark.advanced_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards=1)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_text_to_text_001(omni_server, openai_client) -> None:
    """
    Test text input processing and text-only output generation via OpenAI API.
    Deploy Setting: default yaml
    Input Modal: text
    Output Modal: text
    Datasets: few requests
    """
    messages = dummy_messages_from_mix_data(content_text=get_prompt())

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "stream": False,
        "modalities": ["text"],
        "key_words": {"text": ["beijing"]},
    }

    openai_client.send_omni_request(request_config, request_num=get_max_batch_size())
