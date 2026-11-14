# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Projection packing contracts, independent of padding or audio equivalence."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import AttentionConfig
from vllm_omni.diffusion.models.cosyvoice3_audio.cosyvoice3_dit import DiTAttention

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def make_attention():
    config = SimpleNamespace(
        diffusion_attention_config=AttentionConfig(default="TORCH_SDPA"),
        parallel_config=SimpleNamespace(ring_degree=1),
    )
    with set_current_diffusion_config(config):
        return DiTAttention(64, heads=2, dim_head=32)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_pack_preserves_loaded_weights_projections_and_rng(dtype):
    attention = make_attention().to(dtype=dtype).eval()
    checkpoint = attention.state_dict()
    restored = make_attention().to(dtype=dtype).eval()
    restored.load_state_dict(checkpoint, strict=True)
    x = torch.randn(2, 5, 64, dtype=dtype)
    expected = tuple(projection(x) for projection in (restored.to_q, restored.to_k, restored.to_v))
    rng = torch.get_rng_state()
    restored.fuse_qkv_projections()
    assert torch.equal(rng, torch.get_rng_state())
    for name in ("weight", "bias"):
        torch.testing.assert_close(
            getattr(restored.to_qkv, name),
            torch.cat([checkpoint[f"to_{part}.{name}"] for part in ("q", "k", "v")]),
            rtol=0,
            atol=0,
        )
    tolerance = 1e-5 if dtype == torch.float32 else 2e-2
    for actual, original in zip(restored.to_qkv(x).chunk(3, dim=-1), expected, strict=True):
        torch.testing.assert_close(actual, original, rtol=tolerance, atol=tolerance)
    packed = restored.to_qkv
    restored.fuse_qkv_projections()
    assert restored.to_qkv is packed and not packed.weight.requires_grad


def test_pack_rejects_training():
    with pytest.raises(ValueError, match="inference mode"):
        make_attention().fuse_qkv_projections()
