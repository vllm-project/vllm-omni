# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU checks for Step-Audio2's native chat format and speech continuation."""

from pathlib import Path

import pytest
from transformers.utils.chat_template_utils import render_jinja_template

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

TEMPLATE = Path(__file__).resolve().parents[4] / "vllm_omni/transformers_utils/chat_templates/step_audio2.jinja"


def test_user_is_rendered_as_human():
    rendered = render_jinja_template(
        [
            [
                {"role": "system", "content": "请记录下你所听到的语音内容。"},
                {"role": "user", "content": [{"type": "audio_url", "audio_url": {"url": "unused.wav"}}]},
            ]
        ],
        chat_template=TEMPLATE.read_text(),
        add_generation_prompt=True,
    )[0][0]
    assert rendered == (
        "<|BOT|>system\n请记录下你所听到的语音内容。<|EOT|><|BOT|>human\n<audio_patch><|EOT|><|BOT|>assistant\n"
    )


def test_mixed_audio_and_text_preserves_order():
    rendered = render_jinja_template(
        [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Before "},
                        {"type": "audio_url", "audio_url": {"url": "unused.wav"}},
                        {"type": "text", "text": " after.\n"},
                    ],
                }
            ]
        ],
        chat_template=TEMPLATE.read_text(),
        add_generation_prompt=False,
    )[0][0]
    assert rendered == "<|BOT|>human\nBefore <audio_patch> after.\n<|EOT|>"


def test_assistant_speech_prefix_continuation():
    rendered = render_jinja_template(
        [
            [
                {"role": "user", "content": "Hello."},
                {"role": "assistant", "content": "<tts_start>"},
            ]
        ],
        chat_template=TEMPLATE.read_text(),
        continue_final_message=True,
        add_generation_prompt=False,
    )[0][0]
    assert rendered == "<|BOT|>human\nHello.<|EOT|><|BOT|>assistant\n<tts_start>"
