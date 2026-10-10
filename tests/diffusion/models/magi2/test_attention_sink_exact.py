# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The cached, token-major sink correction must reproduce the uncached one bit for bit."""

import gc
import weakref
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.models.magi2 import attention
from vllm_omni.diffusion.models.magi2.attention import VarlenHandler
from vllm_omni.diffusion.models.magi2.parallel import Magi2ParallelGroup

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model]

_BITS = {torch.float32: torch.int32, torch.bfloat16: torch.int16, torch.float16: torch.int16}
_SINK_KINDS = ("normal", "large", "neg_inf", "pos_inf", "all_neg_inf")


def _reference_correct(out, lse, sink):
    """Uncached correction: per-call sink LSE and an isfinite/where guard on the head-major delta."""
    if sink is None or sink.numel() == 0:
        return out, lse
    old_lse = lse.float().transpose(0, 1)
    sink_lse = torch.logsumexp(sink.float(), dim=0).unsqueeze(0)
    new_lse = torch.logaddexp(old_lse, sink_lse)
    delta = old_lse - new_lse
    delta = torch.where(torch.isfinite(delta), delta, torch.full_like(delta, -torch.inf))
    corrected = out * torch.exp(delta).unsqueeze(-1).to(out.dtype)
    return corrected, new_lse.transpose(0, 1).contiguous()


def _reference_rank_sink(sink, rank, world_size):
    if world_size == 1:
        return sink
    return torch.chunk(sink, world_size, dim=-1)[rank].contiguous()


def _reference_musa_fa3_tail(out, lse, sink, dtype):
    """The MUSA FlashAttention-3 branch of packed attention after the provider call."""
    if sink is not None:
        out = out.float()
    return _reference_correct(out, lse, sink)[0].to(dtype)


def _layout(tensor):
    # Strides of size-0/1 dimensions carry no layout information.
    return [stride for size, stride in zip(tensor.shape, tensor.stride()) if size > 1]


def _assert_bitwise_equal(actual, expected):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert _layout(actual) == _layout(expected)
    assert torch.equal(actual.view(_BITS[actual.dtype]), expected.view(_BITS[expected.dtype]))


def _lse(heads, tokens, generator):
    lse = torch.randn(heads, tokens, generator=generator) * 4
    specials = [-torch.inf, torch.inf, torch.nan, -0.0, 0.0, 1e-45, -1e-45, 3.0e38, -3.0e38, 88.0, -104.0, 1e4]
    flat = lse.view(-1)
    count = min(len(specials), flat.numel())
    # Spread the special values over heads and tokens instead of filling the first row.
    positions = torch.randperm(flat.numel(), generator=generator)[:count]
    flat[positions] = torch.tensor(specials[:count])
    return lse


def _sink(kind, num_sink, heads, generator):
    sink = torch.randn(num_sink, heads, generator=generator) * 2
    if kind == "large":
        sink = sink * 40
    elif kind == "neg_inf":
        sink[0, 0] = -torch.inf
    elif kind == "pos_inf":
        sink[-1, -1] = torch.inf
    elif kind == "all_neg_inf":
        sink.fill_(-torch.inf)
    return sink


def _out(tokens, heads, head_dim, dtype, generator):
    out = torch.randn(tokens, heads, head_dim, generator=generator)
    flat = out.view(-1)
    # Non-finite outputs only on the FP32 tail: the CPU BF16 casts encode NaN differently per loop kind.
    specials = [-0.0, 1e-40, -1e-40, 6.0e4]
    if dtype == torch.float32:
        specials += [torch.inf, -torch.inf, torch.nan]
    count = min(len(specials), flat.numel())
    flat[torch.randperm(flat.numel(), generator=generator)[:count]] = torch.tensor(specials[:count])
    return out.to(dtype)


@pytest.mark.cpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("tokens", [0, 1, 9, 64])
@pytest.mark.parametrize("heads", [1, 3])
@pytest.mark.parametrize("head_dim", [1, 16])
def test_correction_matches_uncached_reference_bitwise(dtype, tokens, heads, head_dim):
    generator = torch.Generator().manual_seed(tokens * 100 + heads * 10 + head_dim)
    for num_sink in (1, 2):
        for kind in _SINK_KINDS:
            out = _out(tokens, heads, head_dim, dtype, generator)
            lse = _lse(heads, tokens, generator)
            sink = _sink(kind, num_sink, heads, generator)
            expected_out, expected_lse = _reference_correct(out, lse, sink)
            for sink_lse in (None, attention._sink_lse(sink)):
                actual_out, actual_lse = attention.correct_out_lse_with_sink(out, lse, sink, sink_lse=sink_lse)
                _assert_bitwise_equal(actual_out, expected_out)
                _assert_bitwise_equal(actual_lse, expected_lse)


@pytest.mark.cpu
@pytest.mark.parametrize("sink", [None, torch.empty(0, 3)])
def test_absent_sink_returns_inputs(sink):
    out, lse = torch.randn(4, 3, 8), torch.randn(3, 4)
    actual_out, actual_lse = attention.correct_out_lse_with_sink(out, lse, sink, sink_lse=torch.zeros(1, 3))
    assert actual_out is out and actual_lse is lse


@pytest.mark.cpu
def test_precomputed_sink_lse_keeps_head_count_check():
    with pytest.raises(ValueError, match="head counts differ"):
        attention.correct_out_lse_with_sink(
            torch.randn(4, 3, 8), torch.randn(3, 4), torch.randn(1, 3), sink_lse=torch.zeros(1, 2)
        )


@pytest.mark.cpu
def test_scale_is_token_major_so_the_multiply_needs_no_transposed_copy(monkeypatch):
    seen = []
    exp = torch.exp

    def recording_exp(tensor, *args, **kwargs):
        seen.append((tuple(tensor.shape), tensor.is_contiguous()))
        return exp(tensor, *args, **kwargs)

    monkeypatch.setattr(torch, "exp", recording_exp)
    generator = torch.Generator().manual_seed(5)
    out, lse, sink = torch.randn(9, 3, 8), _lse(3, 9, generator), torch.randn(1, 3)
    with torch.inference_mode():
        attention.correct_out_lse_with_sink(out, lse, sink)
    assert seen == [((9, 3), True)]


@pytest.mark.cpu
def test_autograd_keeps_reference_values_and_gradients():
    generator = torch.Generator().manual_seed(11)
    out = torch.randn(9, 3, 8, generator=generator)
    lse = _lse(3, 9, generator)
    sink = torch.randn(2, 3, generator=generator)
    grads = []
    for correct in (attention.correct_out_lse_with_sink, _reference_correct):
        leaves = [tensor.clone().requires_grad_() for tensor in (out, lse, sink)]
        corrected, new_lse = correct(*leaves)
        finite = torch.isfinite(corrected)
        (corrected.masked_fill(~finite, 0).sum() + new_lse.nan_to_num(0, 0, 0).sum()).backward()
        grads.append((corrected.detach(), new_lse.detach(), *(leaf.grad for leaf in leaves)))
    for actual, expected in zip(*grads):
        _assert_bitwise_equal(actual, expected)


def _group(world_size=1, rank=0):
    return Magi2ParallelGroup(None, world_size, rank)


@pytest.mark.cpu
def test_sink_lse_cache_reuses_until_the_sink_changes(monkeypatch):
    compute = Mock(wraps=attention._sink_lse)
    monkeypatch.setattr(attention, "_sink_lse", compute)
    cache = attention._SinkLseCache()
    sink = nn.Parameter(torch.randn(2, 4))

    def check(group=_group(), expected_calls=None):
        with torch.inference_mode():
            first = cache.get(sink, group)
            assert cache.get(sink, group) is first
        local_sink = _reference_rank_sink(sink.detach(), group.rank, group.world_size)
        _assert_bitwise_equal(first, torch.logsumexp(local_sink.float(), dim=0).unsqueeze(0))
        assert compute.call_count == expected_calls
        return first

    check(expected_calls=1)
    # Reloading through the parameter bumps its version counter.
    with torch.no_grad():
        sink.copy_(torch.randn(2, 4))
    check(expected_calls=2)
    # Rebinding the parameter to new storage changes its data pointer.
    sink.data = torch.randn(2, 4)
    check(expected_calls=3)
    # Each Ulysses rank reads its own head chunk.
    check(_group(2, 1), expected_calls=4)
    check(_group(2, 0), expected_calls=5)
    check(_group(), expected_calls=6)
    # Same values in a different tensor are a different sink.
    sink = nn.Parameter(sink.detach().clone())
    check(expected_calls=7)
    # A dtype change is part of the key.
    sink.data = sink.data.double()
    check(expected_calls=8)
    # A different tensor over the same storage and version counter is still a different sink.
    alias = sink.detach()
    sink.data.copy_(torch.randn(2, 4))
    with torch.inference_mode():
        value = cache.get(alias, _group())
    _assert_bitwise_equal(value, torch.logsumexp(alias.float(), dim=0).unsqueeze(0))
    assert compute.call_count == 9


@pytest.mark.cpu
def test_sink_lse_cache_steps_aside_when_it_cannot_prove_reuse(monkeypatch):
    cache = attention._SinkLseCache()
    group = _group()
    assert cache.get(None, group) is None
    assert cache.get(torch.empty(0, 4), group) is None
    # Autograd must see the sink in every call that records it, and an
    # inference-mode cached LSE cannot be saved for backward.
    assert cache.get(nn.Parameter(torch.randn(1, 4)), group) is None
    assert cache.get(torch.randn(1, 4), group) is None

    class Subclass(torch.Tensor):
        pass

    with torch.no_grad():
        assert cache.get(torch.randn(1, 4).as_subclass(Subclass), group) is None
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
        assert cache.get(torch.randn(1, 4), group) is None


@pytest.mark.cpu
def test_sink_lse_cache_keys_inference_tensors_on_their_storage(monkeypatch):
    compute = Mock(wraps=attention._sink_lse)
    monkeypatch.setattr(attention, "_sink_lse", compute)
    cache = attention._SinkLseCache()
    with torch.inference_mode():
        sink = torch.randn(2, 4)
        first = cache.get(sink, _group())
        assert cache.get(sink, _group()) is first
        _assert_bitwise_equal(first, torch.logsumexp(sink.float(), dim=0).unsqueeze(0))
        assert compute.call_count == 1
        # A copy of the same values in other storage is a different sink.
        copy = sink.clone()
        _assert_bitwise_equal(cache.get(copy, _group()), first)
        assert compute.call_count == 2


@pytest.mark.cpu
def test_sink_lse_cache_does_not_keep_the_sink_alive():
    cache = attention._SinkLseCache()
    sink = torch.randn(1, 4)
    sink_ref = weakref.ref(sink)
    with torch.no_grad():
        cached = cache.get(sink, _group())
    del sink
    gc.collect()
    assert sink_ref() is None
    assert cached.grad_fn is None


def _musa_platform(monkeypatch):
    monkeypatch.setattr(
        attention,
        "current_omni_platform",
        SimpleNamespace(is_cuda=lambda: False, is_musa=lambda: True, device_type="cpu"),
    )
    monkeypatch.setattr(attention, "_MUSA_FA3_MIN_TOKENS", 0)
    monkeypatch.setattr(attention, "flash_attn_3_varlen_unsupported", lambda softcap, return_softmax_lse: None)
    monkeypatch.setattr(
        attention, "torch_varlen_attention_with_sink", Mock(side_effect=AssertionError("fallback used"))
    )


@pytest.mark.cpu
@pytest.mark.parametrize("world_size,rank", [(1, 0), (2, 1), (4, 2)])
def test_kernel_matches_uncached_path_across_steps(monkeypatch, world_size, rank):
    _musa_platform(monkeypatch)
    group = _group(world_size, rank)
    monkeypatch.setattr(attention, "get_magi2_ulysses_group", lambda: group)
    # One rank's view of the head exchange: keep every token, select this rank's heads.
    monkeypatch.setattr(
        attention,
        "scatter_heads_gather_seqlen",
        lambda tensors, split_sizes, group: tuple(torch.chunk(t, group.world_size, dim=1)[group.rank] for t in tensors),
    )
    monkeypatch.setattr(attention, "scatter_seqlen_gather_heads", lambda output, split_sizes, group: output)
    compute = Mock(wraps=attention._sink_lse)
    monkeypatch.setattr(attention, "_sink_lse", compute)

    generator = torch.Generator().manual_seed(29 + world_size)
    tokens, heads, kv_heads, head_dim = 40, 8, 4, 64
    local_heads = heads // world_size
    q = torch.randn(tokens, heads, head_dim, generator=generator).to(torch.bfloat16)
    k = torch.randn(tokens, kv_heads, head_dim, generator=generator).to(torch.bfloat16)
    v = torch.randn(tokens, kv_heads, head_dim, generator=generator).to(torch.bfloat16)
    cu = torch.tensor([0, 1, 17, 40], dtype=torch.int32)
    varlen = VarlenHandler(cu, cu, 23, 23)
    sink = nn.Parameter(torch.randn(1, heads, generator=generator))
    kernel = attention.Magi2PackedAttentionKernel(softcap=-1.0)
    metadata = AttentionMetadata(
        extra={"magi2_varlen": varlen, "magi2_split_sizes": [tokens], "magi2_sink": sink},
    )

    provider_outputs = []

    def provider(q, k, v, **kwargs):
        out = torch.randn(q.shape, generator=generator).to(q.dtype)
        lse = _lse(local_heads, q.shape[0], generator)
        provider_outputs.append((out, lse))
        return out, lse

    monkeypatch.setattr(attention, "flash_attn_3_varlen", provider)

    def step():
        with torch.inference_mode():
            actual = kernel(q, k, v, metadata)
        out, lse = provider_outputs[-1]
        local_sink = _reference_rank_sink(sink.detach(), rank, world_size)
        _assert_bitwise_equal(actual, _reference_musa_fa3_tail(out, lse, local_sink, q.dtype))

    for _ in range(3):
        step()
    assert compute.call_count == 1
    with torch.no_grad():
        sink.mul_(-1.5)
    step()
    step()
    assert compute.call_count == 2


@pytest.mark.cpu
def test_kernels_cache_their_own_layer_sinks(monkeypatch):
    _musa_platform(monkeypatch)
    monkeypatch.setattr(attention, "get_magi2_ulysses_group", lambda: _group())
    generator = torch.Generator().manual_seed(31)
    q = torch.randn(12, 2, 64, generator=generator).to(torch.bfloat16)
    out = torch.randn(q.shape, generator=generator).to(q.dtype)
    lse = _lse(2, 12, generator)
    monkeypatch.setattr(attention, "flash_attn_3_varlen", lambda q, k, v, **kwargs: (out, lse))
    compute = Mock(wraps=attention._sink_lse)
    monkeypatch.setattr(attention, "_sink_lse", compute)
    cu = torch.tensor([0, 12], dtype=torch.int32)
    varlen = VarlenHandler(cu, cu, 12, 12)
    layers = [
        (attention.Magi2PackedAttentionKernel(-1.0), nn.Parameter(torch.randn(1, 2, generator=generator)))
        for _ in range(3)
    ]
    for _ in range(2):
        for kernel, sink in layers:
            metadata = AttentionMetadata(extra={"magi2_varlen": varlen, "magi2_split_sizes": [12], "magi2_sink": sink})
            with torch.inference_mode():
                actual = kernel(q, q, q, metadata)
            _assert_bitwise_equal(actual, _reference_musa_fa3_tail(out, lse, sink.detach(), q.dtype))
    assert compute.call_count == len(layers)


class _SinkOwner(nn.Module):
    def __init__(self, sink: torch.Tensor) -> None:
        super().__init__()
        self.sinks = nn.Parameter(sink)
        self.kernel = attention.Magi2PackedAttentionKernel(-1.0)


def _single_rank_fa3(monkeypatch, seed):
    _musa_platform(monkeypatch)
    monkeypatch.setattr(attention, "get_magi2_ulysses_group", lambda: _group())
    generator = torch.Generator().manual_seed(seed)
    q = torch.randn(12, 2, 64, generator=generator).to(torch.bfloat16)
    out = torch.randn(q.shape, generator=generator).to(q.dtype)
    lse = _lse(2, 12, generator)
    cu = torch.tensor([0, 12], dtype=torch.int32)
    return q, out, lse, VarlenHandler(cu, cu, 12, 12), generator


@pytest.mark.cpu
def test_kernel_cache_survives_staging_the_module_in_inference_mode(monkeypatch):
    q, out, lse, varlen, generator = _single_rank_fa3(monkeypatch, 41)
    monkeypatch.setattr(attention, "flash_attn_3_varlen", lambda q, k, v, **kwargs: (out, lse))
    compute = Mock(wraps=attention._sink_lse)
    monkeypatch.setattr(attention, "_sink_lse", compute)
    owner = _SinkOwner(torch.randn(1, 2, generator=generator))

    def step():
        metadata = AttentionMetadata(
            extra={"magi2_varlen": varlen, "magi2_split_sizes": [12], "magi2_sink": owner.sinks}
        )
        with torch.inference_mode():
            actual = owner.kernel(q, q, q, metadata)
        _assert_bitwise_equal(actual, _reference_musa_fa3_tail(out, lse, owner.sinks.detach(), q.dtype))

    step()
    assert compute.call_count == 1
    # The pipeline stages the transformer to the host and back inside inference
    # mode, which leaves the parameter an inference tensor in new storage.
    with torch.inference_mode():
        owner.to(torch.float64)
        owner.to(torch.float32)
    assert owner.sinks.is_inference()
    for _ in range(3):
        step()
    assert compute.call_count == 2


@pytest.mark.cpu
def test_grad_mode_does_not_reuse_an_inference_mode_sink_lse(monkeypatch):
    q, out, lse, varlen, generator = _single_rank_fa3(monkeypatch, 43)
    sink = nn.Parameter(torch.randn(1, 2, generator=generator), requires_grad=False)
    kernel = attention.Magi2PackedAttentionKernel(-1.0)
    metadata = AttentionMetadata(extra={"magi2_varlen": varlen, "magi2_split_sizes": [12], "magi2_sink": sink})
    monkeypatch.setattr(attention, "flash_attn_3_varlen", lambda q, k, v, **kwargs: (out, lse))
    with torch.inference_mode():
        kernel(q, q, q, metadata)
    tracked_lse = lse.clone().requires_grad_()
    monkeypatch.setattr(attention, "flash_attn_3_varlen", lambda q, k, v, **kwargs: (out, tracked_lse))
    actual = kernel(q, q, q, metadata)
    assert actual.requires_grad
    _assert_bitwise_equal(actual.detach(), _reference_musa_fa3_tail(out, lse, sink.detach(), q.dtype))


@pytest.mark.cpu
@pytest.mark.parametrize("cached", [False, True])
def test_uneven_sink_heads_still_raise(monkeypatch, cached):
    _musa_platform(monkeypatch)
    group = _group(2, 0)
    monkeypatch.setattr(attention, "get_magi2_ulysses_group", lambda: group)
    monkeypatch.setattr(attention, "scatter_heads_gather_seqlen", lambda tensors, split_sizes, group: tensors)
    monkeypatch.setattr(attention, "flash_attn_3_varlen", Mock(side_effect=AssertionError("attention ran")))
    q = torch.randn(4, 2, 64).to(torch.bfloat16)
    cu = torch.tensor([0, 4], dtype=torch.int32)
    varlen = VarlenHandler(cu, cu, 4, 4)
    sink = torch.randn(1, 3)
    metadata = AttentionMetadata(extra={"magi2_varlen": varlen, "magi2_split_sizes": [4], "magi2_sink": sink})
    with torch.no_grad(), pytest.raises(ValueError, match="divide across Ulysses ranks"):
        if cached:
            attention.Magi2PackedAttentionKernel(-1.0)(q, q, q, metadata)
        else:
            attention.ulysses_packed_attention_with_sink(q, q, q, varlen, [4], sink=sink, group=group)


def _musa_available():
    return hasattr(torch, "musa") and torch.musa.is_available()


@pytest.mark.musa
@pytest.mark.parametrize("tokens,heads,num_sink", [(29616, 3, 1), (14601, 6, 1), (4099, 3, 2), (1, 3, 1)])
def test_real_musa_correction_is_bitwise_unchanged(tokens, heads, num_sink):
    if not _musa_available():
        pytest.skip("requires a MUSA device")
    generator = torch.Generator().manual_seed(tokens + heads)
    out = torch.randn(tokens, heads, 128, generator=generator).to(torch.bfloat16)
    lse = _lse(heads, tokens, generator)
    sink = torch.randn(num_sink, heads, generator=generator)
    out, lse, sink = (tensor.to("musa") for tensor in (out, lse, sink))
    cache = attention._SinkLseCache()
    with torch.inference_mode():
        expected = _reference_musa_fa3_tail(out, lse, sink, torch.bfloat16)
        _, expected_lse = _reference_correct(out.float(), lse, sink)
        for _ in range(2):
            sink_lse = cache.get(sink, _group())
            actual_out, actual_lse = attention.correct_out_lse_with_sink(out.float(), lse, sink, sink_lse=sink_lse)
            _assert_bitwise_equal(actual_out.to(torch.bfloat16).cpu(), expected.cpu())
            _assert_bitwise_equal(actual_lse.cpu(), expected_lse.cpu())


@pytest.mark.musa
def test_real_musa_fa3_output_correction_is_bitwise_unchanged():
    if not _musa_available():
        pytest.skip("requires a MUSA device")
    if attention.flash_attn_3_varlen_unsupported(False, True) is not None:
        pytest.skip("standalone FA3 provider with LSE is not installed")
    generator = torch.Generator().manual_seed(37)
    q = torch.randn(4096, 3, 128, generator=generator).to(device="musa", dtype=torch.bfloat16)
    k = torch.randn(4096, 3, 128, generator=generator).to(device="musa", dtype=torch.bfloat16)
    v = torch.randn(4096, 3, 128, generator=generator).to(device="musa", dtype=torch.bfloat16)
    cu = torch.tensor([0, 1000, 4096], dtype=torch.int32, device="musa")
    sink = nn.Parameter(torch.randn(1, 3, generator=generator).to("musa"))
    with torch.inference_mode():
        out, lse = attention.flash_attn_3_varlen(
            q,
            k,
            v,
            cu_seqlens_q=cu,
            cu_seqlens_k=cu,
            max_seqlen_q=3096,
            max_seqlen_k=3096,
            softmax_scale=128**-0.5,
            softcap=-1.0,
            return_softmax_lse=True,
        )
        expected = _reference_musa_fa3_tail(out, lse, sink.detach(), q.dtype)
        cache = attention._SinkLseCache()
        for _ in range(2):
            corrected = attention.correct_out_lse_with_sink(out.float(), lse, sink, sink_lse=cache.get(sink, _group()))[
                0
            ]
            _assert_bitwise_equal(corrected.to(q.dtype).cpu(), expected.cpu())


@pytest.mark.musa
def test_real_musa_cache_follows_the_sink_device():
    if not _musa_available():
        pytest.skip("requires a MUSA device")
    cache = attention._SinkLseCache()
    sink = nn.Parameter(torch.randn(1, 3))
    with torch.no_grad():
        host = cache.get(sink, _group())
        sink.data = sink.data.to("musa")
        device = cache.get(sink, _group())
        expected = torch.logsumexp(sink.detach().float(), dim=0).unsqueeze(0)
    assert host.device.type == "cpu" and device.device.type == "musa"
    _assert_bitwise_equal(device.cpu(), expected.cpu())
