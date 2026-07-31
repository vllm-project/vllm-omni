# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.cache.cachedit import CacheDiTAdapterConfig
from vllm_omni.diffusion.models.ltx2.ltx2_transformer import (
    LTX2AudioVideoAttnProcessor,
    LTX2VideoTransformer3DModel,
    _LTX2ParallelAttention,
    _make_rms_norm,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


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


def test_ltx_sp_keeps_cross_attention_key_padding_mask(monkeypatch):
    monkeypatch.setattr(LTX2AudioVideoAttnProcessor, "_is_sp_enabled", staticmethod(lambda: True))
    processor = LTX2AudioVideoAttnProcessor()
    hidden_states = torch.zeros(1, 3, 4)
    encoder_hidden_states = torch.zeros(1, 4, 4)
    key_padding_mask = torch.tensor([[True, True, True, False]])

    actual = processor._prepare_attention_mask(
        SimpleNamespace(),
        hidden_states,
        encoder_hidden_states,
        key_padding_mask,
        batch_size=1,
        sequence_length=4,
    )

    torch.testing.assert_close(actual, key_padding_mask[:, None, None, :])


def _fake_ltx_parallel_attention():
    calls = []

    class Backend:
        @staticmethod
        def get_name():
            return "FLASH_ATTN"

        @staticmethod
        def supports_attention_mask():
            return True

    def run(name):
        def forward(query, _key, _value, _metadata):
            calls.append(name)
            return torch.zeros_like(query)

        return forward

    layer = object.__new__(_LTX2ParallelAttention)
    nn.Module.__init__(layer)
    layer.attn_backend = Backend
    layer.attention = SimpleNamespace(forward=run("native"))
    layer.sdpa_fallback = SimpleNamespace(forward=run("sdpa"))
    layer.backend_pref = "FLASH_ATTN"
    return layer, calls


@pytest.mark.parametrize(("query_length", "key_length"), [(4, 4), (3, 4)])
def test_ltx_masked_cross_attention_uses_sdpa(query_length, key_length):
    layer, calls = _fake_ltx_parallel_attention()
    query = torch.zeros(1, query_length, 2, 4, dtype=torch.bfloat16)
    key = value = torch.zeros(1, key_length, 2, 4, dtype=torch.bfloat16)
    key_padding_mask = torch.tensor([[True] * (key_length - 1) + [False]])
    metadata = AttentionMetadata(attn_mask=key_padding_mask[:, None, None, :])

    layer._run_local_attention(query, key, value, metadata)

    assert calls == ["sdpa"]


def test_ltx_masked_self_attention_uses_sdpa():
    layer, calls = _fake_ltx_parallel_attention()
    query = key = value = torch.zeros(1, 4, 2, 4, dtype=torch.bfloat16)
    key_padding_mask = torch.tensor([[True, True, True, False]])
    metadata = AttentionMetadata(attn_mask=key_padding_mask[:, None, None, :])

    layer._run_local_attention(query, key, value, metadata)

    assert calls == ["sdpa"]


def test_ltx_unmasked_cross_attention_keeps_native_backend():
    layer, calls = _fake_ltx_parallel_attention()
    query = key = value = torch.zeros(1, 4, 2, 4, dtype=torch.bfloat16)

    layer._run_local_attention(query, key, value, None)

    assert calls == ["native"]
