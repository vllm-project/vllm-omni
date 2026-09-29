# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MiniCPM structured-content rendering must preserve native part boundaries."""

from pathlib import Path

import pytest
from jinja2 import TemplateError
from transformers.utils.chat_template_utils import render_jinja_template

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

TEMPLATE = Path(__file__).resolve().parents[4] / "vllm_omni/transformers_utils/chat_templates/minicpmo45_native.jinja"


@pytest.mark.parametrize(
    "parts,expected",
    [
        ([{"type": "image"}, {"type": "text", "text": "Question?"}], "(<image>./</image>)Question?"),
        ([{"type": "text", "text": "Question?"}, {"type": "image"}], "Question?(<image>./</image>)"),
        ([{"type": "image"}, {"type": "text", "text": "\n\nQuestion?"}], "(<image>./</image>)\n\nQuestion?"),
        ([{"type": "text", "text": " A "}, {"type": "text", "text": " B "}], " A  B "),
        ([{"type": "audio"}, {"type": "text", "text": "Transcribe."}], "(<audio>./</audio>)Transcribe."),
        ([{"type": "video"}, {"type": "image"}], "(<video>./</video>)(<image>./</image>)"),
        ("\nKeep this whitespace.\n", "\nKeep this whitespace.\n"),
    ],
)
def test_native_content_concatenation(parts, expected):
    # render_jinja_template takes a batch of conversations and returns
    # (rendered_conversations, continuation_chunks), so [0][0] is the string.
    rendered = render_jinja_template(
        [[{"role": "user", "content": parts}]],
        chat_template=TEMPLATE.read_text(),
        add_generation_prompt=True,
        enable_thinking=False,
        use_tts_template=True,
    )[0][0]
    assert rendered == (
        f"<|im_start|>user\n{expected}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n<|tts_bos|>"
    )


def test_unsupported_part_is_not_silently_discarded():
    with pytest.raises(TemplateError, match="Unsupported MiniCPM-o content part"):
        render_jinja_template(
            [[{"role": "user", "content": [{"type": "unknown"}]}]],
            chat_template=TEMPLATE.read_text(),
            add_generation_prompt=True,
        )
