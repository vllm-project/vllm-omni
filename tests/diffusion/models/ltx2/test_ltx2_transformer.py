# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from torch import nn

from vllm_omni.diffusion.cache.cache_dit_backend import CacheDiTAdapterConfig
from vllm_omni.diffusion.models.ltx2.ltx2_transformer import LTX2VideoTransformer3DModel, _make_rms_norm

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_ltx_rms_norm_no_affine_identity_weight_is_non_persistent_buffer():
    norm = _make_rms_norm(8, eps=1e-6, elementwise_affine=False)

    assert "weight" not in dict(norm.named_parameters())
    assert "weight" in dict(norm.named_buffers())
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
