# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""One prompt encoder serves both the T2I and Edit branches.

The two branches were merged into a single implementation. These tests pin the
behaviour each one must keep: the exact conditioning slice its template implies,
and the vision-key filtering that keeps Edit-only inputs off the T2I path.
"""

from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from vllm_omni.diffusion.models.mage_flow import prompt_utils

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _FakeConditioningModel:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        sequence_length = kwargs["input_ids"].shape[1]
        hidden_states = torch.arange(sequence_length * 2, dtype=torch.float32).reshape(1, sequence_length, 2)
        return SimpleNamespace(last_hidden_state=hidden_states)


class _FakePromptProcessor:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        sequence_length = 70
        inputs = {
            "input_ids": torch.arange(sequence_length).reshape(1, -1),
            "attention_mask": torch.tensor([[1] * 68 + [0, 0]]),
            "position_ids": torch.arange(sequence_length).reshape(1, -1),
            "ignored_processor_value": torch.ones(1),
        }
        if "images" in kwargs:
            inputs.update(
                {
                    "pixel_values": torch.ones(1, 3, 2, 2),
                    "image_grid_thw": torch.ones(1, 3, dtype=torch.long),
                    "mm_token_type_ids": torch.ones(1, sequence_length, dtype=torch.long),
                }
            )
        return inputs


def test_prompt_encoder_preserves_exact_slices_for_both_branches():
    """One encoder serves T2I and Edit; each keeps its own template and slice."""
    model = _FakeConditioningModel()
    text_encoder = SimpleNamespace(model=model)
    processor = _FakePromptProcessor()
    image = Image.new("RGB", (4, 4))

    text_result = prompt_utils.encode_mage_flow_prompt(
        text_encoder,
        processor,
        "a landscape",
        device=torch.device("cpu"),
    )
    edit_result = prompt_utils.encode_mage_flow_prompt(
        text_encoder,
        processor,
        "make it brighter",
        device=torch.device("cpu"),
        reference_images=[image],
    )

    expected = torch.arange(140, dtype=torch.float32).reshape(1, 70, 2)
    # Slice runs from the template's prefix length to the last valid token, so
    # padding never reaches the conditioning tensor.
    torch.testing.assert_close(
        text_result,
        expected[:, prompt_utils.MAGE_FLOW_PROMPT_START_INDEX : 68],
    )
    torch.testing.assert_close(
        edit_result,
        expected[:, prompt_utils.MAGE_FLOW_EDIT_PROMPT_START_INDEX : 68],
    )
    # Vision keys reach the model only on the Edit branch, and processor keys
    # the model does not accept reach it on neither.
    assert "pixel_values" not in model.calls[0]
    assert "pixel_values" in model.calls[1]
    assert "ignored_processor_value" not in model.calls[0]
    assert "ignored_processor_value" not in model.calls[1]
    assert processor.calls[0]["text"] == [prompt_utils.format_mage_flow_prompt("a landscape")]
    assert processor.calls[1]["text"] == [prompt_utils.format_mage_flow_edit_prompt("make it brighter", 1)]
