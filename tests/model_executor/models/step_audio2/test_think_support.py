# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU checks for Step-Audio-2-mini-Think's two-pass prompts and deploy config."""

from pathlib import Path

import pytest
from transformers.utils.chat_template_utils import render_jinja_template

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

TEMPLATE = Path(__file__).resolve().parents[4] / "vllm_omni/transformers_utils/chat_templates/step_audio2.jinja"
DEPLOY = Path(__file__).resolve().parents[4] / "vllm_omni/deploy/step_audio_2_think.yaml"


@pytest.mark.parametrize(
    ("assistant_prefix", "expected_suffix"),
    [
        ("<think>", "<|BOT|>assistant\n<think>"),
        ("\n<think>\n", "<|BOT|>assistant\n\n<think>\n"),
        (
            "<think>The speaker sounds curious.</think>\n\n<tts_start>",
            "<|BOT|>assistant\n<think>The speaker sounds curious.</think>\n\n<tts_start>",
        ),
    ],
)
def test_think_two_pass_prompts_use_shared_template(assistant_prefix, expected_suffix):
    rendered = render_jinja_template(
        [
            [
                {"role": "system", "content": "请激活深度思考模式。"},
                {"role": "user", "content": [{"type": "audio_url"}]},
                {"role": "assistant", "content": assistant_prefix},
            ]
        ],
        chat_template=TEMPLATE.read_text(),
        add_generation_prompt=False,
        continue_final_message=True,
    )[0][0]
    assert rendered == f"<|BOT|>system\n请激活深度思考模式。<|EOT|><|BOT|>human\n<audio_patch><|EOT|>{expected_suffix}"


def test_think_deploy_config_inherits_mini_and_sets_eot_stop():
    from vllm_omni.config.stage_config import load_deploy_config

    deploy = load_deploy_config(DEPLOY)
    assert [s.stage_id for s in deploy.stages] == [0, 1]
    assert deploy.stages[0].engine_extras["hf_overrides"]["step_audio2_variant"] == "think"
    assert deploy.stages[0].default_sampling_params["stop_token_ids"] == [151665]
    assert deploy.stages[0].default_sampling_params["max_tokens"] == 2048
    assert deploy.stages[0].default_sampling_params["repetition_penalty"] == 1.05
    assert deploy.stages[1].engine_extras["hf_overrides"]["architectures"] == [
        "StepAudio2Token2WavForConditionalGeneration"
    ]
