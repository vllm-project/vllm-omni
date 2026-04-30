from collections.abc import Callable
from typing import cast

import pytest
import torch.nn as nn

from vllm_omni.diffusion.models.longcat_image.longcat_image_transformer import (
    LongCatImageTransformer2DModel,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_longcat_image_exposes_hsdp_shard_conditions_for_both_block_lists():
    model = object.__new__(LongCatImageTransformer2DModel)
    nn.Module.__init__(model)
    model.transformer_blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(2)])
    model.single_transformer_blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])
    model.proj_out = nn.Linear(4, 4)

    conditions = cast(
        list[Callable[[str, nn.Module], bool]],
        getattr(model, "_hsdp_shard_conditions"),
    )

    assert len(conditions) == 1

    condition = conditions[0]

    assert condition("transformer_blocks.0", model.transformer_blocks[0])
    assert condition("transformer_blocks.1", model.transformer_blocks[1])
    assert condition("single_transformer_blocks.0", model.single_transformer_blocks[0])
    assert condition("single_transformer_blocks.1", model.single_transformer_blocks[1])
    assert condition("single_transformer_blocks.2", model.single_transformer_blocks[2])
    assert not condition("transformer_blocks", model.transformer_blocks)
    assert not condition("single_transformer_blocks", model.single_transformer_blocks)
    assert not condition("proj_out", model.proj_out)
