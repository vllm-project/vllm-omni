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

    conditions = getattr(model, "_hsdp_shard_conditions", None)

    assert conditions is not None
    assert len(conditions) == 1

    matched: list[str] = []
    for name, module in model.named_modules():
        if any(cond(name, module) for cond in conditions):
            matched.append(name)

    assert matched == [
        "transformer_blocks.0",
        "transformer_blocks.1",
        "single_transformer_blocks.0",
        "single_transformer_blocks.1",
        "single_transformer_blocks.2",
    ]
