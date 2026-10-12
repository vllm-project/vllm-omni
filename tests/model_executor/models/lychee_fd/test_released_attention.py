# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Scoped dispatch, native cache views and fallback ownership."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl

from vllm_omni.model_executor.models.lychee_fd import released_attention as module

pytestmark = [pytest.mark.cpu, pytest.mark.core_model]


def configured():
    impl = object.__new__(module.LycheeFlashAttentionImpl)
    impl.__dict__.update(
        num_heads=28,
        num_kv_heads=4,
        head_size=128,
        scale=128**-0.5,
        vllm_flash_attn_version=2,
        attn_type=AttentionType.DECODER,
        kv_cache_dtype="auto",
        dcp_world_size=1,
        batch_invariant_enabled=False,
        alibi_slopes=None,
        sliding_window=(-1, -1),
        logits_soft_cap=0,
        sinks=None,
        kv_sharing_target_layer_name=None,
    )
    return impl


def inputs(requests=1, context=65):
    query = SimpleNamespace(is_cuda=True, dtype=torch.bfloat16, ndim=3, shape=(requests, 28, 128))
    cache = SimpleNamespace(dtype=torch.bfloat16, ndim=4, shape=(10, 4, 16, 256))
    metadata = SimpleNamespace(
        seq_lens=torch.ones(requests, dtype=torch.int32),
        num_actual_tokens=requests,
        query_start_loc=torch.arange(requests + 1, dtype=torch.int32),
        block_table=torch.zeros(requests, 10, dtype=torch.int32),
        max_query_len=1,
        max_seq_len=context,
        causal=True,
        use_cascade=False,
        max_num_splits=0,
        scheduler_metadata=None,
        mm_prefix_query_range_tensor=None,
        rswa_prefix_lens=None,
    )
    return query, cache, metadata


@pytest.mark.parametrize("requests", [1, 2, 3, 4])
@pytest.mark.parametrize("context", [1, 63, 64, 65, 127, 128])
def test_scoped_single_row_decode(requests, context):
    assert configured()._use_released_split(*inputs(requests, context), None, None)


@pytest.mark.parametrize(
    "target, name, value",
    [
        ("query", "is_cuda", False),
        ("query", "dtype", torch.float16),
        ("cache", "dtype", torch.float16),
        ("cache", "shape", (10, 4, 32, 256)),
        ("impl", "vllm_flash_attn_version", 3),
        ("impl", "num_heads", 14),
        ("impl", "kv_cache_dtype", "fp8"),
        ("impl", "dcp_world_size", 2),
        ("impl", "batch_invariant_enabled", True),
        ("impl", "alibi_slopes", object()),
        ("impl", "sliding_window", (127, 0)),
        ("impl", "logits_soft_cap", 1),
        ("impl", "sinks", object()),
        ("impl", "kv_sharing_target_layer_name", "shared"),
        ("meta", "max_seq_len", 129),
        ("meta", "max_seq_len", 0),
        ("meta", "max_query_len", 2),
        ("meta", "num_actual_tokens", 2),
        ("meta", "causal", torch.tensor(True)),
        ("meta", "use_cascade", True),
        ("meta", "max_num_splits", 1),
        ("meta", "scheduler_metadata", object()),
        ("meta", "mm_prefix_query_range_tensor", object()),
        ("meta", "rswa_prefix_lens", object()),
    ],
)
def test_unsupported_configuration_delegates(target, name, value):
    impl = configured()
    query, cache, metadata = inputs()
    setattr({"impl": impl, "query": query, "cache": cache, "meta": metadata}[target], name, value)
    assert not impl._use_released_split(query, cache, metadata, None, None)


def test_profile_output_quantization_and_oversized_batch_delegate():
    impl = configured()
    q, cache, meta = inputs()
    assert not impl._use_released_split(q, cache, None, None, None)
    assert not impl._use_released_split(q, cache, meta, object(), None)
    assert not impl._use_released_split(q, cache, meta, None, object())
    assert not impl._use_released_split(*inputs(5), None, None)


def test_raw_dispatch_writes_native_output_and_passes_paged_views(monkeypatch):
    impl = configured()
    impl._use_released_split = lambda *args: True
    _, _, meta = inputs(2)
    q = torch.zeros(3, 28, 128, dtype=torch.bfloat16)
    cache = torch.zeros(10, 4, 16, 256, dtype=q.dtype)
    output = torch.full_like(q, 99)
    calls = []

    def raw(*args):
        calls.append(args)
        args[3].fill_(7)
        return args[3], None

    monkeypatch.setattr(torch.ops._vllm_fa2_C, "varlen_fwd", raw)
    assert impl.forward(None, q, None, None, cache, meta, output) is output
    args = calls[0]
    assert args[1].untyped_storage().data_ptr() == cache.untyped_storage().data_ptr()
    assert args[2].untyped_storage().data_ptr() == cache.untyped_storage().data_ptr()
    assert args[1].shape == (10, 16, 4, 128) and args[1].stride(-1) == 1
    assert args[4] is meta.query_start_loc and args[5] is meta.query_start_loc
    assert args[6] is meta.seq_lens and args[8] is meta.block_table
    assert args[20] == 2 and args[11] == 65
    assert (output[:2] == 7).all() and (output[2] == 99).all()


def test_fallback_forwards_every_native_argument(monkeypatch):
    impl = configured()
    impl._use_released_split = lambda *args: False
    base = Mock(return_value=object())
    monkeypatch.setattr(FlashAttentionImpl, "forward", base)
    values = [object() for _ in range(9)]
    result = impl.forward(*values)
    assert result is base.return_value
    base.assert_called_once_with(*values[:7], output_scale=values[7], output_block_scale=values[8])


def test_adaptation_preserves_configured_native_instance_and_other_backends(monkeypatch):
    base = object.__new__(FlashAttentionImpl)
    owner = object()
    base.__dict__.update(config_owner=owner, scale=0.2)
    monkeypatch.setattr(module.current_platform, "is_device_capability", lambda capability: capability == 80)
    adapted = module.adapt_released_attention(base)
    assert type(adapted) is module.LycheeFlashAttentionImpl
    assert adapted.config_owner is owner and adapted.scale == 0.2
    assert type(base) is FlashAttentionImpl
    assert module.adapt_released_attention(adapted) is adapted
    alternate = object()
    assert module.adapt_released_attention(alternate) is alternate
    monkeypatch.setattr(module.current_platform, "is_device_capability", lambda _: False)
    assert module.adapt_released_attention(base) is base
