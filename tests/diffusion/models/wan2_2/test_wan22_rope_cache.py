# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Exercise RoPE cache reuse and invalidation through both transformer forwards."""

from types import SimpleNamespace

import pytest
import torch
import vllm.distributed.parallel_state as vllm_parallel_state

import vllm_omni.diffusion.distributed.parallel_state as omni_parallel_state
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.diffusion.models.wan2_2.wan2_2_transformer import WanTransformer3DModel
from vllm_omni.diffusion.models.wan2_2.wan2_2_vace_transformer import WanVACETransformer3DModel

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture(params=[WanTransformer3DModel, WanVACETransformer3DModel], ids=["wan", "vace"])
def transformer(request, monkeypatch):
    pp = SimpleNamespace(world_size=1, rank_in_group=0, is_first_rank=True, is_last_rank=True)
    monkeypatch.setattr(vllm_parallel_state, "_PP", pp)
    monkeypatch.setattr(omni_parallel_state, "_PP", pp)
    # No attention blocks or pretrained weights are needed to exercise the real RoPE,
    # patch embedding, conditioning, and output paths on CPU.
    model = request.param(
        num_attention_heads=1,
        attention_head_dim=12,
        in_channels=4,
        out_channels=4,
        text_dim=8,
        freq_dim=8,
        ffn_dim=24,
        num_layers=0,
        rope_max_seq_len=8,
    ).eval()
    # vLLM's Conv3dLayer leaves both parameters empty until checkpoint loading.
    torch.nn.init.normal_(model.patch_embedding.weight, std=0.02)
    torch.nn.init.zeros_(model.patch_embedding.bias)
    with set_forward_context(omni_diffusion_config=OmniDiffusionConfig(model="")):
        yield model


@pytest.mark.parametrize("change", ["resolution", "dtype"])
@torch.no_grad()
def test_rope_cache_reuses_and_invalidates(transformer, change, mocker):
    rope = mocker.spy(transformer.rope, "forward")
    hidden_states = torch.randn(1, 4, 2, 4, 4)
    encoder_hidden_states = torch.randn(1, 3, 8)
    timestep = torch.tensor([1])

    first = transformer(hidden_states, timestep, encoder_hidden_states, return_dict=False)[0]
    cached = transformer._cached_rope_emb
    repeated = transformer(hidden_states, timestep, encoder_hidden_states, return_dict=False)[0]
    assert rope.call_count == 1
    assert transformer._cached_rope_emb is cached
    torch.testing.assert_close(repeated, first, rtol=0, atol=0)

    if change == "resolution":
        hidden_states = torch.randn(1, 4, 2, 4, 6)
    else:
        transformer.to(torch.float64)
        hidden_states = hidden_states.to(torch.float64)
        encoder_hidden_states = encoder_hidden_states.to(torch.float64)

    changed = transformer(hidden_states, timestep, encoder_hidden_states, return_dict=False)[0]
    rebuilt = transformer._cached_rope_emb
    assert rope.call_count == 2
    assert rebuilt is not cached
    patch_size = transformer.config.patch_size
    tokens = (hidden_states.shape[2] // patch_size[0]) * (hidden_states.shape[3] // patch_size[1])
    tokens *= hidden_states.shape[4] // patch_size[2]
    for frequency in rebuilt:
        assert frequency.shape == (1, tokens, 1, transformer.config.attention_head_dim // 2)
        assert frequency.dtype == hidden_states.dtype

    repeated = transformer(hidden_states, timestep, encoder_hidden_states, return_dict=False)[0]
    assert rope.call_count == 2
    assert transformer._cached_rope_emb is rebuilt
    torch.testing.assert_close(repeated, changed, rtol=0, atol=0)
