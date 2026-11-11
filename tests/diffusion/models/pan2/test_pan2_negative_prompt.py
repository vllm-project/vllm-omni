# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU checks of PAN 2's default negative prompt."""

import json

import pytest

from vllm_omni.diffusion.models.pan2.pipeline_pan2 import PAN2_DEFAULT_NEGATIVE_PROMPT, pan2_default_negative_prompt

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _time_ranges(node):
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "time_range_sec":
                yield value
            else:
                yield from _time_ranges(value)
    elif isinstance(node, list):
        for item in node:
            yield from _time_ranges(item)


@pytest.mark.parametrize(
    ("num_frames", "fps", "span"),
    [(241, 24, "0 to 10"), (313, 24, "0 to 13"), (121, 24, "0 to 5"), (481, 24, "0 to 20"), (85, 24, "0 to 3.5")],
)
def test_default_negative_prompt_spans_the_clip(num_frames, fps, span):
    text = pan2_default_negative_prompt(num_frames, fps)

    assert text == json.dumps(json.loads(text), ensure_ascii=False, separators=(",", ":"))
    ranges = list(_time_ranges(json.loads(text)))
    assert len(ranges) == 3
    assert set(ranges) == {span}


def test_default_negative_prompt_only_changes_time_ranges():
    rendered = json.loads(pan2_default_negative_prompt(241, 24))
    # The shipped template spans 10 s, so a 241-frame clip at 24 fps reproduces it exactly.
    assert rendered == PAN2_DEFAULT_NEGATIVE_PROMPT
