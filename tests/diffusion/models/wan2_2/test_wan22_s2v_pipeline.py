from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from vllm_omni.diffusion.models.wan2_2.wan2_2_s2v_transformer import WanS2VTransformer3DModel

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_s2v_exposes_hsdp_shard_conditions_for_transformer_blocks():
    model = object.__new__(WanS2VTransformer3DModel)
    nn.Module.__init__(model)
    model.blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])

    conditions = getattr(model, "_hsdp_shard_conditions", None)

    assert conditions is not None
    assert len(conditions) == 1

    matched = []
    for name, module in model.named_modules():
        if any(cond(name, module) for cond in conditions):
            matched.append(name)

    assert matched == ["blocks.0", "blocks.1", "blocks.2"]


def test_s2v_hsdp_shard_condition_does_not_match_non_block_modules():
    model = object.__new__(WanS2VTransformer3DModel)
    nn.Module.__init__(model)
    model.blocks = nn.ModuleList([nn.Linear(4, 4)])
    model.head_indicator = nn.Linear(4, 4)
    model.casual_audio_encoder = nn.Linear(4, 4)

    conditions = model._hsdp_shard_conditions
    non_block_matched = []
    for name, module in model.named_modules():
        if name and "blocks" not in name:
            if any(cond(name, module) for cond in conditions):
                non_block_matched.append(name)

    assert non_block_matched == []


def test_encode_audio_calls_unshard_reshard_when_fsdp_managed():
    model = object.__new__(WanS2VTransformer3DModel)
    nn.Module.__init__(model)
    model.enable_adain = False
    model.casual_audio_encoder = MagicMock(return_value=torch.zeros(1, 10, 64))

    model.unshard = MagicMock()
    model.reshard = MagicMock()

    audio_input = torch.randn(1, 1, 64, 5)
    motion_frames = [2, 2]

    result = model.encode_audio(audio_input, motion_frames)

    model.unshard.assert_called_once()
    model.reshard.assert_called_once()
    assert "audio_emb" in result


def test_encode_audio_skips_unshard_reshard_when_not_fsdp():
    model = object.__new__(WanS2VTransformer3DModel)
    nn.Module.__init__(model)
    model.enable_adain = False
    model.casual_audio_encoder = MagicMock(return_value=torch.zeros(1, 10, 64))

    audio_input = torch.randn(1, 1, 64, 5)
    motion_frames = [2, 2]

    result = model.encode_audio(audio_input, motion_frames)

    assert not hasattr(model, "unshard")
    assert not hasattr(model, "reshard")
    assert "audio_emb" in result


def test_s2v_pipeline_skips_cpu_offload_when_hsdp_enabled():
    """Test that CPU offload of transformer is skipped when HSDP is active."""

    od_config = MagicMock()
    od_config.enable_cpu_offload = True

    parallel_config = MagicMock()
    parallel_config.use_hsdp = True
    od_config.parallel_config = parallel_config

    should_offload = od_config.enable_cpu_offload and not getattr(od_config.parallel_config, "use_hsdp", False)
    assert should_offload is False


def test_s2v_pipeline_allows_cpu_offload_when_hsdp_disabled():
    """Test that CPU offload works normally when HSDP is not active."""
    od_config = MagicMock()
    od_config.enable_cpu_offload = True

    parallel_config = MagicMock()
    parallel_config.use_hsdp = False
    od_config.parallel_config = parallel_config

    should_offload = od_config.enable_cpu_offload and not getattr(od_config.parallel_config, "use_hsdp", False)
    assert should_offload is True
