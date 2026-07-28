# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from torch import nn

from vllm_omni.diffusion.cache.cachedit import CacheDiTAdapterConfig
from vllm_omni.diffusion.models.ltx2.ltx2_transformer import LTX2VideoTransformer3DModel, _make_rms_norm

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_ltx_rms_norm_no_affine_matches_official_torch_module():
    norm = _make_rms_norm(8, eps=1e-6, elementwise_affine=False)

    assert "weight" not in dict(norm.named_parameters())
    assert norm.weight is None
    assert "weight" not in dict(norm.named_buffers())
    assert "weight" not in norm.state_dict()


def test_ltx_rms_norm_affine_weight_remains_parameter():
    norm = _make_rms_norm(8, eps=1e-6, elementwise_affine=True)

    assert isinstance(dict(norm.named_parameters())["weight"], nn.Parameter)
    assert "weight" not in dict(norm.named_buffers())
    assert "weight" in norm.state_dict()


def test_ltx_transformer_has_separate_cfg_cache_dit_config():
    adapter_config = getattr(LTX2VideoTransformer3DModel, "_cache_dit_adapter_config")

    assert isinstance(adapter_config, CacheDiTAdapterConfig)
    assert adapter_config.has_separate_cfg


def test_ltx_transformer_exposes_hsdp_shard_conditions_for_blocks():
    model = object.__new__(LTX2VideoTransformer3DModel)
    nn.Module.__init__(model)
    model.transformer_blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(2)])
    model.norm_out = nn.LayerNorm(4)

    conditions = getattr(model, "_hsdp_shard_conditions", None)

    assert conditions is not None
    assert len(conditions) == 1

    matched = []
    for name, module in model.named_modules():
        if any(condition(name, module) for condition in conditions):
            matched.append(name)

    assert matched == ["transformer_blocks.0", "transformer_blocks.1"]


@pytest.mark.parametrize("rope_type", ["interleaved", "split"])
def test_ltx_sp_plan_shards_video_and_audio_timesteps_together(rope_type):
    root_plan = LTX2VideoTransformer3DModel._build_sp_plan(rope_type)[""]

    video_timestep = root_plan["timestep"]
    audio_timestep = root_plan["audio_timestep"]

    assert video_timestep.split_dim == audio_timestep.split_dim == 1
    assert video_timestep.expected_dims == audio_timestep.expected_dims == 2
    assert not video_timestep.split_output
    assert not audio_timestep.split_output
