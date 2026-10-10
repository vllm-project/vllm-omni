# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Offline Chatterbox audio consolidation on both model runners."""

import pytest
import torch
from transformers import AutoTokenizer

from tests.helpers.mark import hardware_test
from tests.helpers.stage_config import get_deploy_config_path, modify_stage_config
from vllm_omni.model_executor.models.chatterbox.conditioning import VoiceConditioning, build_prompt, punc_norm
from vllm_omni.transformers_utils.configs.chatterbox import ChatterboxConfig

MODEL = "ResembleAI/chatterbox-turbo"
DEPLOY = get_deploy_config_path("chatterbox_turbo.yaml")
RUNNERS = [
    pytest.param(
        (MODEL, modify_stage_config(DEPLOY, {"model_runner": runner, "async_chunk": chunked})),
        id=f"{runner}-{'chunked' if chunked else 'whole'}",
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
    tokenizer = AutoTokenizer.from_pretrained(MODEL, tokenizer_type="gpt2")
    voice = VoiceConditioning.from_builtin(MODEL)
    prompts = [
        build_prompt(tokenizer.encode(punc_norm(text), add_special_tokens=False), voice, ChatterboxConfig())
        for text in ("The weather is nice today.", "Please leave your name and number after the tone.")
    ]
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
