# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU lock: adjacent condition images stay separate token blocks.

Two condition images with no text between them are one run of True in
``image_pad_mask``. Block ids come from ``img_shapes`` token counts, so they
must not collapse into one bidirectional attention span.
"""

import pytest
import torch

from vllm_omni.diffusion.models.qwen_image_21.qwen_image_21_transformer import (
    QwenImage21Transformer2DModel,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_adjacent_condition_images_stay_separate_blocks() -> None:
    # text(2) | cond0(2) | cond1(3) | text(1) | target(2)
    # cond0 and cond1 are one contiguous True run.
    image_pad_mask = torch.tensor([False, False, True, True, True, True, True, False, True, True])
    img_shapes = [(1, 1, 2), (1, 1, 3), (1, 1, 2)]

    image_ids, target_token_mask = QwenImage21Transformer2DModel.build_token_metadata(image_pad_mask, img_shapes)

    assert image_ids.tolist() == [-1, -1, 0, 0, 1, 1, 1, -1, 2, 2]
    assert target_token_mask.tolist() == [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
    ]
    assert QwenImage21Transformer2DModel._build_full_attn_spans(image_ids) == [
        (2, 4),
        (4, 7),
        (8, 10),
    ]
