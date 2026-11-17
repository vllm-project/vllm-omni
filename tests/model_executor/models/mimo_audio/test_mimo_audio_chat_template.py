# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from pathlib import Path

import pytest
from jinja2.sandbox import ImmutableSandboxedEnvironment

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_TEMPLATE = Path(__file__).resolve().parents[4] / "examples/online_serving/mimo_audio/chat_template.jinja"


def _render(messages, **kwargs):
    return (
        ImmutableSandboxedEnvironment()
        .from_string(_TEMPLATE.read_text())
        .render(messages=messages, add_generation_prompt=True, **kwargs)
    )


@pytest.mark.parametrize(
    "content",
    [
        "Please convert this text to speech: Thank you for your help today.",
        "Transcribe this audio:<|sosp|><|empty|><|eosp|>",
    ],
)
def test_no_implicit_system_message(content):
    rendered = _render([{"role": "user", "content": content}])
    assert rendered == f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n<|sostm|>"


def test_explicit_system_message_is_preserved():
    rendered = _render(
        [
            {"role": "system", "content": "Speak in a calm voice."},
            {"role": "user", "content": "Hello."},
        ]
    )
    assert rendered.startswith("<|im_start|>system\nSpeak in a calm voice.<|im_end|>\n")
    assert rendered.count("<|im_start|>system") == 1
    assert rendered.endswith("<|im_start|>assistant\n<|sostm|>")


def test_tool_instructions_keep_their_system_message():
    rendered = _render([{"role": "user", "content": "Hello."}], tools=[{"name": "test_tool"}])
    assert rendered.startswith("<|im_start|>system\nYou are a helpful assistant.")
    assert "<tools>" in rendered
