# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Offline Chatterbox audio consolidation on both model runners."""

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.stage_config import get_deploy_config_path, modify_stage_config
from vllm_omni.config.stage_config import resolve_deploy_yaml
from vllm_omni.entrypoints.openai.tts_adapters.chatterbox import CONTROLS
from vllm_omni.model_executor.models.chatterbox.conditioning import (
    VoiceConditioning,
    build_prompt,
    load_text_tokenizer,
    tokenize_text,
)
from vllm_omni.transformers_utils.configs.chatterbox import ChatterboxConfig

RUNNERS = [
    pytest.param(
        (model, modify_stage_config(get_deploy_config_path(deploy), {"model_runner": runner, "async_chunk": chunked})),
        id=f"{variant}-{runner}-{'chunked' if chunked else 'whole'}",
    )
    for variant, model, deploy in (
        ("turbo", "ResembleAI/chatterbox-turbo", "chatterbox_turbo.yaml"),
        ("original", "ResembleAI/chatterbox", "chatterbox.yaml"),
    )
    for runner in ("v1", "v2")
    for chunked in (False, True)
]


@pytest.mark.advanced_model
@pytest.mark.tts
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_runner", RUNNERS, indirect=True)
def test_text_to_audio_001(omni_runner, offline_client):
    """A batch returns full 24 kHz waveforms, including every streamed chunk."""
    model = omni_runner.model_name
    variant = "turbo" if model.endswith("-turbo") else "original"
    texts = ("The weather is nice today.", "Please leave your name and number after the tone.")
    tokenizer = load_text_tokenizer(model, variant)
    text_ids = [tokenize_text(text, tokenizer, variant) for text in texts]
    voice = VoiceConditioning.from_builtin(model)
    # Original's controls as its deploy file sets them for serving.
    additional_config = resolve_deploy_yaml(get_deploy_config_path("chatterbox.yaml"))["stages"][0]["additional_config"]
    controls = {name: additional_config[name] for name in CONTROLS} if variant == "original" else {}
    prompts = [build_prompt(ids, voice, ChatterboxConfig(variant), **controls) for ids in text_ids]
    offline_client.send_tokenized_tts_request(
        {
            "prompt_token_ids": [prompt["prompt_token_ids"] for prompt in prompts],
            "additional_information": [prompt["additional_information"] for prompt in prompts],
            "expected_sample_rate": 24000,
            "min_duration_s": 1.0,
            "assert_distinct_outputs": True,
        }
    )
