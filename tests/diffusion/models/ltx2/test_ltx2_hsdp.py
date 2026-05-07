import pytest
import torch.nn as nn

from vllm_omni.diffusion.models.ltx2.ltx2_transformer import LTX2VideoTransformer3DModel

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_ltx2_exposes_hsdp_shard_conditions_for_transformer_blocks():
    model = object.__new__(LTX2VideoTransformer3DModel)
    nn.Module.__init__(model)
    model.transformer_blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(2)])
    model.norm_out = nn.LayerNorm(4)

    conditions = getattr(model, "_hsdp_shard_conditions", None)

    assert conditions is not None
    assert len(conditions) == 1

    matched = []
    for name, module in model.named_modules():
        if any(cond(name, module) for cond in conditions):
            matched.append(name)

    assert matched == ["transformer_blocks.0", "transformer_blocks.1"]
