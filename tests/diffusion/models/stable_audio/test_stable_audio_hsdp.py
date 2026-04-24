import pytest
import torch.nn as nn

from vllm_omni.diffusion.models.stable_audio.stable_audio_transformer import StableAudioDiTModel

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_stable_audio_exposes_hsdp_shard_conditions_for_transformer_blocks():
    model = object.__new__(StableAudioDiTModel)
    nn.Module.__init__(model)
    model.transformer_blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(2)])
    model.proj_out = nn.Linear(4, 4)

    conditions = getattr(model, "_hsdp_shard_conditions", None)

    assert conditions is not None
    assert len(conditions) == 1

    matched = []
    for name, module in model.named_modules():
        if any(cond(name, module) for cond in conditions):
            matched.append(name)

    assert matched == ["transformer_blocks.0", "transformer_blocks.1"]
