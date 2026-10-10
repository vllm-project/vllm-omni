# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Offline Chatterbox audio consolidation on both model runners."""

import pytest
import torch
from tokenizers import Tokenizer
from transformers import AutoTokenizer

from tests.helpers.mark import hardware_test
from tests.helpers.stage_config import get_deploy_config_path, modify_stage_config
from vllm_omni.model_executor.models.chatterbox.conditioning import VoiceConditioning, build_prompt, punc_norm
from vllm_omni.transformers_utils.configs.chatterbox import ChatterboxConfig
from vllm_omni.transformers_utils.repo_utils import hf_api

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
def test_text_to_audio_001(omni_runner):
    """A batch returns full 24 kHz waveforms, including every streamed chunk."""
    model = omni_runner.model_name
    variant = "turbo" if model.endswith("-turbo") else "original"
    texts = ("The weather is nice today.", "Please leave your name and number after the tone.")
    if variant == "turbo":
        tokenizer = AutoTokenizer.from_pretrained(model, tokenizer_type="gpt2")
        text_ids = [tokenizer.encode(punc_norm(text), add_special_tokens=False) for text in texts]
    else:
        tokenizer_path = hf_api().hf_hub_download(model, "tokenizer.json")
        original_tokenizer = Tokenizer.from_file(tokenizer_path)
        text_ids = [
            original_tokenizer.encode(punc_norm(text, "original").replace(" ", "[SPACE]")).ids for text in texts
        ]
    voice = VoiceConditioning.from_builtin(model)
    prompts = [build_prompt(ids, voice, ChatterboxConfig(variant)) for ids in text_ids]
    outputs = list(omni_runner.generate(prompts))
    assert len(outputs) == len(prompts)
    for output in outputs:
        audio_output = output.outputs[0].multimodal_output
        audio = audio_output["audio"]
        if isinstance(audio, list):
            audio = torch.cat([chunk.flatten() for chunk in audio])
        assert audio.numel() > 24000
        assert torch.isfinite(audio).all()
        assert audio.abs().max() > 0
