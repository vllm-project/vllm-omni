import pytest
import torch.nn as nn

pytest.importorskip("dreamid_omni.modules.model")

from vllm_omni.diffusion.models.dreamid_omni.fusion import FusionModel

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_dreamid_omni_exposes_hsdp_shard_conditions_for_fused_blocks():
    model = object.__new__(FusionModel)
    nn.Module.__init__(model)
    model.fused_blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(2)])
    model.attn = nn.Identity()

    conditions = getattr(model, "_hsdp_shard_conditions", None)

    assert conditions is not None
    assert len(conditions) == 1

    matched = []
    for name, module in model.named_modules():
        if any(cond(name, module) for cond in conditions):
            matched.append(name)

    assert matched == ["fused_blocks.0", "fused_blocks.1"]
