# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni import (
    Qwen3OmniMoeForConditionalGeneration,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_non_thinker_mrope_positions_do_not_require_multimodal_kwargs():
    model = object.__new__(Qwen3OmniMoeForConditionalGeneration)
    model.model_stage = "talker"

    positions, delta = model.get_mrope_input_positions([1, 2, 3])

    assert delta == 0
    assert torch.equal(
        positions,
        torch.tensor(
            [
                [0, 1, 2],
                [0, 1, 2],
                [0, 1, 2],
            ]
        ),
    )
