# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Unit tests for Flux2KleinKV pipeline.

These tests focus on the core KV cache behaviors:
- Token layout building
- SP gather/restore
- Flux2Attention KV cache mode (extract/cached)
"""

import pytest
import torch

import vllm_omni.diffusion.models.flux2_klein.kv_cache as kv_cache_module
from vllm_omni.diffusion.models.flux2_klein.flux2_klein_transformer import Flux2Attention

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _DummyParallelConfig:
    sequence_parallel_size = 1


class _DummyKVCache:
    def __init__(self):
        self.stored_key = None
        self.stored_value = None

    def store(self, key, value):
        self.stored_key = key
        self.stored_value = value

    def get(self):
        return self.stored_key, self.stored_value


class _DummyAttention:
    def __call__(self, query, key, value, attn_metadata=None):
        return query


class _DummyLinear:
    def __init__(self, output):
        self.output = output

    def __call__(self, *args, **kwargs):
        return self.output


class _DummyRope:
    def __call__(self, x, *args, **kwargs):
        return x


def test_gather_full_reference_kv_restores_reference_tokens_from_sp_shards():
    """Test that KV cache gather correctly restores reference tokens from SP shards."""
    key = torch.tensor(
        [
            [
                [[10.0]],
                [[11.0]],
                [[12.0]],
            ]
        ]
    )
    value = key + 100.0

    def _fake_sp_gather(tensor, dim):
        assert dim == 1
        offset = 100.0 if tensor.flatten()[0].item() > 100.0 else 0.0
        return torch.tensor(
            [
                [
                    [[11.0 + offset]],
                    [[12.0 + offset]],
                    [[21.0 + offset]],
                    [[22.0 + offset]],
                ]
            ],
            dtype=tensor.dtype,
        )

    full_ref_key, full_ref_value = kv_cache_module.gather_full_reference_kv(
        key,
        value,
        num_txt_tokens=1,
        num_ref_tokens=2,
        total_nontext_tokens=4,
        sp_world_size=2,
        sp_gather_fn=_fake_sp_gather,
    )

    assert torch.equal(full_ref_key, torch.tensor([[[[11.0]], [[12.0]]]]))
    assert torch.equal(full_ref_value, torch.tensor([[[[111.0]], [[112.0]]]]))


def test_build_token_layout_treats_full_nontext_input_as_non_sharded():
    """Test token layout building for non-sharded case."""
    layout = kv_cache_module.build_flux2_kv_token_layout(
        num_txt_tokens=512,
        num_ref_tokens_global=4352,
        total_nontext_tokens_global=8448,
        local_nontext_tokens=8448,
        sp_rank=1,
        sp_world_size=2,
    )

    assert layout.sp_world_size == 1
    assert layout.sp_rank == 0
    assert layout.local_ref_tokens == 4352
    assert layout.local_img_tokens == 4096


def test_build_token_layout_handles_single_rank():
    """Test token layout with single SP rank (no sharding)."""
    layout = kv_cache_module.build_flux2_kv_token_layout(
        num_txt_tokens=256,
        num_ref_tokens_global=2048,
        total_nontext_tokens_global=4096,
        local_nontext_tokens=4096,
        sp_rank=0,
        sp_world_size=1,
    )

    assert layout.sp_world_size == 1
    assert layout.sp_rank == 0
    assert layout.local_ref_tokens == 2048
    assert layout.local_img_tokens == 2048


def test_flux2_attention_kv_extract_handles_encoder_tokens():
    """Test Flux2Attention in extract mode handles encoder tokens correctly."""
    attn = object.__new__(Flux2Attention)
    attn.parallel_config = _DummyParallelConfig()
    attn.added_kv_proj_dim = 8
    attn.query_num_heads = 2
    attn.kv_num_heads = 2
    attn.head_dim = 4
    attn.add_query_num_heads = 2
    attn.add_kv_num_heads = 2

    qkv_output = torch.arange(72, dtype=torch.float32).reshape(1, 3, 24)
    attn.to_qkv = _DummyLinear((qkv_output, None))

    add_kv_output = torch.arange(48, dtype=torch.float32).reshape(1, 2, 24)
    attn.add_kv_proj = _DummyLinear((add_kv_output, None))

    attn.norm_q = lambda x: x
    attn.norm_k = lambda x: x
    attn.norm_added_q = lambda x: x
    attn.norm_added_k = lambda x: x
    attn.rope = _DummyRope()
    attn.attn = _DummyAttention()
    attn.to_add_out = lambda x: x
    attn.to_out = [lambda x: x, lambda x: x]

    query = torch.zeros(1, 3, 8)
    encoder_hidden_states = torch.zeros(1, 2, 8)
    kv_cache = _DummyKVCache()

    output = attn.forward(
        hidden_states=query,
        encoder_hidden_states=encoder_hidden_states,
        kv_cache=kv_cache,
        kv_cache_mode="extract",
        num_ref_tokens=1,
    )

    assert isinstance(output, tuple)
    assert isinstance(output[0], torch.Tensor)
    assert output[0].shape[0] == 1


def test_flux2_attention_kv_extract_mode_caches_kv():
    """Test Flux2Attention in extract mode caches key/value."""
    attn = object.__new__(Flux2Attention)
    attn.parallel_config = _DummyParallelConfig()
    attn.added_kv_proj_dim = None
    attn.query_num_heads = 2
    attn.kv_num_heads = 2
    attn.head_dim = 4

    qkv_output = torch.arange(72, dtype=torch.float32).reshape(1, 3, 24)
    attn.to_qkv = _DummyLinear((qkv_output, None))

    attn.norm_q = lambda x: x
    attn.norm_k = lambda x: x
    attn.rope = _DummyRope()
    attn.attn = _DummyAttention()
    attn.to_out = [lambda x: x, lambda x: x]

    query = torch.zeros(1, 4, 8)
    kv_cache = _DummyKVCache()

    attn.forward(
        hidden_states=query,
        kv_cache=kv_cache,
        kv_cache_mode="extract",
        num_ref_tokens=2,
        kv_total_nontext_tokens=4,
    )

    assert kv_cache.stored_key is not None
    assert kv_cache.stored_value is not None


def test_flux2_attention_kv_cached_mode_uses_stored_kv():
    """Test Flux2Attention in cached mode uses stored key/value."""
    attn = object.__new__(Flux2Attention)
    attn.parallel_config = _DummyParallelConfig()
    attn.added_kv_proj_dim = None
    attn.query_num_heads = 2
    attn.kv_num_heads = 2
    attn.head_dim = 4

    qkv_output = torch.arange(72, dtype=torch.float32).reshape(1, 3, 24)
    attn.to_qkv = _DummyLinear((qkv_output, None))

    attn.norm_q = lambda x: x
    attn.norm_k = lambda x: x
    attn.rope = _DummyRope()
    attn.attn = _DummyAttention()
    attn.to_out = [lambda x: x, lambda x: x]

    query = torch.zeros(1, 4, 8)
    stored_key = torch.zeros(1, 2, 2, 4)
    stored_value = torch.zeros(1, 2, 2, 4)
    kv_cache = _DummyKVCache()
    kv_cache.stored_key = stored_key
    kv_cache.stored_value = stored_value

    output = attn.forward(
        hidden_states=query,
        kv_cache=kv_cache,
        kv_cache_mode="cached",
        num_ref_tokens=2,
    )

    assert isinstance(output, torch.Tensor)
    assert output.shape[0] == 1
