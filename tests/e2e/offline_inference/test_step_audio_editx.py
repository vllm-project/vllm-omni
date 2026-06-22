# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline E2E coverage for StepAudioEditX."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from examples.offline_inference.text_to_speech.step_audio_editx import end2end

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

MODEL = "stepfun-ai/Step-Audio-EditX"
AUDIO_TOKENIZER = "stepfun-ai/Step-Audio-Tokenizer"
STAGE_CONFIG = "vllm_omni/deploy/step_audio_editx.yaml"
REF_AUDIO = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_2.wav"
REF_TEXT = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it!"


def _args(**overrides):
    base = dict(
        model=MODEL,
        audio_tokenizer=AUDIO_TOKENIZER,
        deploy_config=STAGE_CONFIG,
        edit_type="clone",
        edit_info=None,
        text="Please review the document before we begin.",
        ref_text=REF_TEXT,
        ref_audio=REF_AUDIO,
        output=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.mark.advanced_model
def test_offline_step_audio_editx_clone_smoke(tmp_path) -> None:
    """Run real offline StepAudioEditX clone inference."""
    from vllm import SamplingParams

    from vllm_omni.entrypoints.omni import Omni

    output_path = tmp_path / "step_audio_editx.wav"

    args = _args(
        model=MODEL,
        audio_tokenizer=AUDIO_TOKENIZER,
        deploy_config=STAGE_CONFIG,
        output=str(output_path),
    )

    os.environ["STEP_AUDIO_TOKENIZER_PATH"] = AUDIO_TOKENIZER
    omni = Omni(model=MODEL, deploy_config=STAGE_CONFIG, trust_remote_code=True)
    try:
        inputs = end2end._build_inputs(args)
        prompt_len = len(inputs[0]["prompt_token_ids"])
        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=max(1, min(2048, 8192 - prompt_len)),
            skip_special_tokens=False,
        )
        outputs = list(omni.generate(inputs, sampling_params_list=[sampling_params, sampling_params]))
    finally:
        omni.close()

    assert outputs


@pytest.mark.advanced_model
def test_offline_step_audio_editx_emotion_smoke(tmp_path) -> None:
    """Run real offline StepAudioEditX edit inference."""
    from vllm import SamplingParams

    from vllm_omni.entrypoints.omni import Omni

    output_path = tmp_path / "step_audio_editx_emotion.wav"
    args = _args(
        edit_type="emotion",
        edit_info="angry",
        model=MODEL,
        audio_tokenizer=AUDIO_TOKENIZER,
        deploy_config=STAGE_CONFIG,
        output=str(output_path),
    )

    os.environ["STEP_AUDIO_TOKENIZER_PATH"] = AUDIO_TOKENIZER
    omni = Omni(model=MODEL, deploy_config=STAGE_CONFIG, trust_remote_code=True)
    try:
        inputs = end2end._build_inputs(args)
        prompt_len = len(inputs[0]["prompt_token_ids"])
        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=max(1, min(2048, 8192 - prompt_len)),
            skip_special_tokens=False,
        )
        outputs = list(omni.generate(inputs, sampling_params_list=[sampling_params, sampling_params]))
    finally:
        omni.close()

    assert outputs
