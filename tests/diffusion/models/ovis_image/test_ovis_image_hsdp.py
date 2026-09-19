# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch.nn as nn

from vllm_omni.diffusion.models.ovis_image.ovis_image_transformer import (
    OvisImageSingleTransformerBlock,
    OvisImageTransformer2DModel,
    OvisImageTransformerBlock,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_hsdp_selects_both_block_types_without_selecting_children():
    model = OvisImageTransformer2DModel.__new__(OvisImageTransformer2DModel)
    nn.Module.__init__(model)
    for attribute, block_type in (
        ("transformer_blocks", OvisImageTransformerBlock),
        ("single_transformer_blocks", OvisImageSingleTransformerBlock),
    ):
        block = block_type.__new__(block_type)
        nn.Module.__init__(block)
        block.proj = nn.Linear(4, 4)
        model.add_module(attribute, nn.ModuleList([block, nn.Linear(4, 4)]))
    model.proj_out = nn.Linear(4, 4)

    selected = [
        name
        for name, module in model.named_modules()
        if any(condition(name, module) for condition in model._hsdp_shard_conditions)
    ]

    assert selected == ["transformer_blocks.0", "single_transformer_blocks.0"]
