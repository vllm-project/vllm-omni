# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compare the MOSS launch policy and residual norm with upstream operators."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.v1.attention.backends import triton_attn

from vllm_omni.model_executor.models.moss_tts.backbone_attention import MossTiledAttentionImpl, install
from vllm_omni.model_executor.models.moss_tts.residual_norm import norm
from vllm_omni.platforms import current_omni_platform

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.cuda,
    pytest.mark.skipif(not current_omni_platform.is_available(), reason="GPU required"),
]


def make_impl(cls):
    return cls(
        num_heads=32,
        num_kv_heads=8,
        head_size=128,
        scale=128**-0.5,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
    )


@pytest.mark.parametrize("batch,length", [(1, 63), (32, 63), (64, 257)])
def test_tiled_attention_matches_upstream(batch, length):
    torch.manual_seed(81)
    device = "cuda"
    blocks = (length + 15) // 16
    cache = torch.randn(batch * blocks, 8, 16, 256, device=device, dtype=torch.bfloat16)
    q = torch.randn(batch, 32, 128, device=device, dtype=torch.bfloat16)
    metadata = SimpleNamespace(
        num_actual_tokens=batch,
        causal=True,
        use_cascade=False,
        mm_prefix_range_tensor=None,
        rswa_prefix_lens=None,
        rswa_window=None,
        query_start_loc=torch.arange(batch + 1, device=device, dtype=torch.int32),
        seq_lens=torch.full((batch,), length, device=device, dtype=torch.int32),
        max_query_len=1,
        max_seq_len=length,
        block_table=torch.arange(batch * blocks, device=device, dtype=torch.int32).view(batch, blocks),
        seq_threshold_3D=None,
        num_par_softmax_segments=None,
        softmax_segm_output=None,
        softmax_segm_max=None,
        softmax_segm_expsum=None,
    )
    layer = SimpleNamespace(
        _q_scale=torch.ones((), device=device),
        _k_scale=torch.ones((), device=device),
        _v_scale=torch.ones((), device=device),
    )
    expected, actual = torch.empty_like(q), torch.empty_like(q)
    make_impl(triton_attn.TritonAttentionImpl).forward(layer, q, q, q, cache, metadata, expected)
    make_impl(MossTiledAttentionImpl).forward(layer, q, q, q, cache, metadata, actual)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.005)


def test_install_keeps_other_models_and_upstream_function_unchanged():
    model, other = nn.Module(), nn.Module()
    model.impl, other.impl = make_impl(triton_attn.TritonAttentionImpl), make_impl(triton_attn.TritonAttentionImpl)
    original = triton_attn.unified_attention
    assert install(model) == 1
    assert isinstance(model.impl, MossTiledAttentionImpl)
    assert type(other.impl) is triton_attn.TritonAttentionImpl
    assert triton_attn.unified_attention is original
    assert install(model) == 0


@pytest.mark.parametrize("batch", [1, 64])
def test_residual_norm_matches_native_fp32_accumulation(batch):
    torch.manual_seed(73)
    module = RMSNorm(2560, eps=1e-6).to(device="cuda", dtype=torch.bfloat16)
    x = torch.randn(batch, 2560, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    expected = module.forward_native(x, residual)
    actual = norm(x, residual, module.weight, module.variance_epsilon)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, rtol=0.008, atol=0.016)
