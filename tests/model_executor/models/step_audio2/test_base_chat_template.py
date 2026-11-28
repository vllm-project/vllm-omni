# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU checks for Step-Audio-2-mini-Base's plain concatenation format."""

from pathlib import Path

import pytest
from transformers.utils.chat_template_utils import render_jinja_template

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

TEMPLATE = Path(__file__).resolve().parents[4] / "vllm_omni/transformers_utils/chat_templates/step_audio2_base.jinja"
DEPLOY = Path(__file__).resolve().parents[4] / "vllm_omni/deploy/step_audio_2_base.yaml"


@pytest.mark.parametrize("add_generation_prompt", [True, False])
def test_base_concatenates_content_without_turn_markers(add_generation_prompt):
    rendered = render_jinja_template(
        [
            [
                {"role": "system", "content": "请仔细聆听这段语音，然后将其内容翻译成中文并用语音播报。"},
                {"role": "user", "content": [{"type": "audio_url"}, {"type": "text", "text": "。"}]},
                {"role": "assistant", "content": "<tts_start>"},
            ]
        ],
        chat_template=TEMPLATE.read_text(),
        add_generation_prompt=add_generation_prompt,
    )[0][0]
    assert rendered == "请仔细聆听这段语音，然后将其内容翻译成中文并用语音播报。<audio_patch>。<tts_start>"


def test_base_rejects_unknown_content():
    with pytest.raises(Exception, match="Unsupported Step-Audio2 Base content type"):
        render_jinja_template(
            [[{"role": "user", "content": [{"type": "image_url"}]}]],
            chat_template=TEMPLATE.read_text(),
        )


def test_base_deploy_config_inherits_mini_and_sets_variant():
    from vllm_omni.config.stage_config import load_deploy_config

    deploy = load_deploy_config(DEPLOY)
    assert [s.stage_id for s in deploy.stages] == [0, 1]
    assert deploy.stages[0].engine_extras["hf_overrides"]["step_audio2_variant"] == "base"
    assert deploy.stages[0].default_sampling_params["repetition_penalty"] == 1.05
    assert deploy.stages[1].engine_extras["hf_overrides"]["architectures"] == [
        "StepAudio2Token2WavForConditionalGeneration"
    ]
