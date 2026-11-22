# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""``StagePool`` tells a non-empty output from an empty one for array and tensor payloads."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_omni.engine.stage_pool import StagePool

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _pool() -> StagePool:
    return StagePool(0, [SimpleNamespace(stage_type="diffusion", final_output_type="video")])  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("payload", "non_empty"),
    [
        (np.zeros((2, 4, 4, 3)), True),
        (torch.zeros(2, 4, 4, 3), True),
        (np.zeros((0, 4, 4, 3)), False),
    ],
)
def test_array_and_tensor_payloads_have_an_unambiguous_answer(payload: object, non_empty: bool) -> None:
    output = SimpleNamespace(request_id="r", final_output_type="video", outputs=[], video=payload)

    assert _pool().has_non_empty_output(output) is non_empty
