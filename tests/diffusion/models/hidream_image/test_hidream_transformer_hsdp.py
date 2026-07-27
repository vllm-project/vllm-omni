# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from torch import nn

from vllm_omni.diffusion.models.hidream_image.hidream_image_transformer import (
    HiDreamImageTransformer2DModel,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_hidream_exposes_hsdp_shard_conditions_for_stream_blocks():
    model = object.__new__(HiDreamImageTransformer2DModel)
    nn.Module.__init__(model)
    model.double_stream_blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(2)])
    model.single_stream_blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])

    conditions = getattr(model, "_hsdp_shard_conditions", None)

    assert conditions is not None
    assert len(conditions) == 1

    matched = []
    for name, module in model.named_modules():
        if any(cond(name, module) for cond in conditions):
            matched.append(name)

    assert matched == [
        "double_stream_blocks.0",
        "double_stream_blocks.1",
        "single_stream_blocks.0",
        "single_stream_blocks.1",
        "single_stream_blocks.2",
    ]


def test_hidream_hsdp_shard_condition_does_not_match_non_block_modules():
    model = object.__new__(HiDreamImageTransformer2DModel)
    nn.Module.__init__(model)
    model.double_stream_blocks = nn.ModuleList([nn.Linear(4, 4)])
    model.single_stream_blocks = nn.ModuleList([nn.Linear(4, 4)])
    model.t_embedder = nn.Linear(4, 4)
    model.x_embedder = nn.Linear(4, 4)
    model.final_layer = nn.Linear(4, 4)

    conditions = model._hsdp_shard_conditions
    non_block_matched = []
    for name, module in model.named_modules():
        if "double_stream_blocks" not in name and "single_stream_blocks" not in name:
            if any(cond(name, module) for cond in conditions):
                non_block_matched.append(name)

    assert non_block_matched == []


class _HiDreamLikeBlock(nn.Module):
    """Minimal block tree with numeric nested modules like HiDream adaLN/MoE."""

    def __init__(self) -> None:
        super().__init__()
        self.block = nn.Module()
        self.block.adaLN_modulation = nn.Sequential(nn.ReLU(), nn.Linear(4, 4))
        self.block.ff_i = nn.Module()
        self.block.ff_i.experts = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])


def test_hidream_hsdp_shard_condition_does_not_match_nested_block_children():
    model = object.__new__(HiDreamImageTransformer2DModel)
    nn.Module.__init__(model)
    model.double_stream_blocks = nn.ModuleList([_HiDreamLikeBlock()])
    model.single_stream_blocks = nn.ModuleList([_HiDreamLikeBlock()])

    conditions = model._hsdp_shard_conditions
    matched = []
    for name, module in model.named_modules():
        if any(cond(name, module) for cond in conditions):
            matched.append(name)

    assert matched == [
        "double_stream_blocks.0",
        "single_stream_blocks.0",
    ]
