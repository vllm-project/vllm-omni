# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for bounded per-convolution fast path verdict caches."""

from __future__ import annotations

import gc
import weakref

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.distributed.autoencoders.wan_vae_fastpath import forwards as fp

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.fixture(params=["_SPATIAL_PAD_VERDICTS", "_CONV_OUT_LAYOUT_VERDICTS"])
def verdict_cache(request, monkeypatch):
    cache = weakref.WeakKeyDictionary()
    monkeypatch.setattr(fp, request.param, cache)
    return request.param, cache


@pytest.fixture(autouse=True)
def integer_probe_operands(monkeypatch):
    # CPU convolution formulations are only bitwise interchangeable on integer
    # data; the Gaussian probe is meant to tell cuDNN kernels apart.
    monkeypatch.setattr(
        fp,
        "_random_like",
        lambda t, generator, std=1.0: torch.empty_like(t).random_(-2, 3, generator=generator),
    )


@pytest.fixture
def cpu_assembly(monkeypatch):
    """Run the verdict paths on CPU using PyTorch in place of Triton assembly."""
    monkeypatch.setattr(fp, "_kernels_allowed", lambda x: True)

    def cat_time(x, cache_x, pad_front, keep_cache_frames=0):
        history = 0 if cache_x is None else cache_x.shape[2]
        zeros = x.new_zeros((*x.shape[:2], pad_front - history, *x.shape[3:]))
        assembled = torch.cat([zeros, *([] if cache_x is None else [cache_x]), x], dim=2)
        return (assembled, assembled[:, :, -keep_cache_frames:].clone()) if keep_cache_frames else assembled

    def cat_pad(x, cache_x, padding, keep_cache_frames=0, channels_last_output=False):
        in_time = cat_time(x, cache_x, padding[4])
        assembled = torch.nn.functional.pad(in_time, (*padding[:4], 0, 0))
        if channels_last_output:
            assembled = assembled.contiguous(memory_format=torch.channels_last_3d)
        return (assembled, in_time[:, :, -keep_cache_frames:].clone()) if keep_cache_frames else assembled

    monkeypatch.setattr(fp.dm, "cat_time_5d", cat_time)
    monkeypatch.setattr(fp.dm, "cat_pad_5d", cat_pad)


def _is_candidate_call(spatial_padding: bool, x: torch.Tensor, padding) -> bool:
    """Whether a ``F.conv3d`` call is the fast formulation under verification."""
    return padding == (0, 1, 1) if spatial_padding else x.is_contiguous(memory_format=torch.channels_last_3d)


def test_verdict_cache_bounds_each_convolution_and_releases_modules(verdict_cache) -> None:
    _, cache = verdict_cache
    conv = nn.Conv3d(2, 2, 1)
    other_conv = nn.Conv3d(2, 2, 1)
    verdicts = cache.setdefault(conv, {})
    other_verdicts = cache.setdefault(other_conv, {})
    fp._record_verdict(other_verdicts, (0,), False)

    limit = fp._MAX_VERDICTS_PER_CONV
    for width in range(limit + 1):
        fp._record_verdict(verdicts, (width,), width % 2 == 0)
        assert len(verdicts) <= limit
    assert list(verdicts) == [(width,) for width in range(1, limit + 1)]
    assert set(verdicts.values()) == {True, False}
    assert other_verdicts == {(0,): False}

    # Updating an existing entry at capacity must not evict another shape.
    keys = list(verdicts)
    fp._record_verdict(verdicts, (1,), True)
    assert list(verdicts) == keys
    assert verdicts[(1,)] is True

    conv_ref = weakref.ref(conv)
    del conv
    gc.collect()
    assert conv_ref() is None
    assert list(cache) == [other_conv]


@torch.no_grad()
@pytest.mark.parametrize("matches", [True, False])
def test_evicted_verdict_is_revalidated(verdict_cache, cpu_assembly, monkeypatch, matches: bool) -> None:
    cache_name, cache = verdict_cache
    spatial_padding = cache_name == "_SPATIAL_PAD_VERDICTS"
    monkeypatch.setattr(fp, "_MAX_VERDICTS_PER_CONV", 2)
    conv = fp.WanCausalConv3d(2, 2, 3, padding=1)
    conv.weight.fill_(1)
    conv.bias.zero_()
    conv3d = torch.nn.functional.conv3d

    def checked_conv3d(x, weight, bias, stride, padding, dilation, groups):
        output = conv3d(x, weight, bias, stride, padding, dilation, groups)
        is_fast = _is_candidate_call(spatial_padding, x, padding)
        # Force a mismatch on the candidate path to verify reference selection.
        return output + 1 if is_fast and not matches else output

    monkeypatch.setattr(fp.F, "conv3d", checked_conv3d)
    comparisons = []
    bitwise_equal = fp._bitwise_equal

    def compare(a, b):
        result = bitwise_equal(a, b)
        comparisons.append(result)
        return result

    monkeypatch.setattr(fp, "_bitwise_equal", compare)

    def run(width):
        # Integer-valued inputs/weights keep both CPU convolution layouts exact.
        x = torch.full((1, 2, 1, 3, width), float(width))
        expected = conv3d(torch.nn.functional.pad(x, conv._padding), conv.weight, conv.bias)
        if spatial_padding:
            output = fp._run_cached_causal_conv(conv, x, [None], 0)
        else:
            output = fp._run_conv_out_channels_last(conv, x, [None], 0)
        if output is None:
            assert not spatial_padding and not matches  # Cached rejection declines.
        else:
            assert torch.equal(output, expected)
        assert len(cache[conv]) <= 2

    # A match on the real operands is confirmed on random ones; a mismatch is final.
    check = [True, True] if matches else [False]
    run(4)
    run(5)
    assert comparisons == check * 2
    run(4)
    assert comparisons == check * 2  # Both positive and negative hits are reused.
    run(6)
    assert {key[0][-1] for key in cache[conv]} == {5, 6}
    run(4)
    assert comparisons == check * 4  # The evicted shape is checked again.
    assert {key[0][-1] for key in cache[conv]} == {4, 6}
    assert all(value is matches for value in cache[conv].values())


@torch.no_grad()
@pytest.mark.parametrize("degenerate", ["input", "weight"])
def test_verdict_is_not_taken_from_degenerate_operands(
    verdict_cache, cpu_assembly, monkeypatch, degenerate: str
) -> None:
    """A formulation that only differs on non-zero products must be rejected on its first call.

    All-zero activations (e.g. black frames) or weights make any two kernels
    agree, so a verdict taken from them alone would later return different
    bytes for real inputs, or after an in-place weight update.
    """
    cache_name, cache = verdict_cache
    spatial_padding = cache_name == "_SPATIAL_PAD_VERDICTS"
    conv3d = torch.nn.functional.conv3d

    def other_kernel(x, weight, bias, stride, padding, dilation, groups):
        output = conv3d(x, weight, bias, stride, padding, dilation, groups)
        if _is_candidate_call(spatial_padding, x, padding):
            # Differs from the reference wherever the convolution itself is non-zero.
            output = output + conv3d(x, weight, None, stride, padding, dilation, groups)
        return output

    monkeypatch.setattr(fp.F, "conv3d", other_kernel)
    generator = torch.Generator().manual_seed(0)
    conv = fp.WanCausalConv3d(2, 2, 3, padding=1)
    conv.weight.copy_(torch.randint(-2, 3, conv.weight.shape, generator=generator))
    conv.bias.copy_(torch.randint(-2, 3, conv.bias.shape, generator=generator))
    shape = (1, 2, 1, 3, 4)
    real = torch.randint(-2, 3, shape, generator=generator).float()
    trained = conv.weight.clone()
    if degenerate == "weight":
        conv.weight.zero_()
    first = torch.zeros(shape) if degenerate == "input" else real

    def run(x):
        # Integer-valued operands keep both CPU convolution layouts exact.
        expected = conv3d(torch.nn.functional.pad(x, conv._padding), conv.weight, conv.bias)
        if spatial_padding:
            output = fp._run_cached_causal_conv(conv, x, [None], 0)
        else:
            # Mirror decoder_forward: a declined conv_out takes the standard path.
            output = fp._run_conv_out_channels_last(conv, x, [None], 0)
            if output is None:
                output = fp._run_cached_causal_conv(conv, x, [None], 0)
        assert torch.equal(output, expected)

    run(first)
    assert list(cache[conv].values()) == [False]
    conv.weight.copy_(trained)  # An in-place update keeps the verdict key.
    run(real)
    assert list(cache[conv].values()) == [False]


@torch.no_grad()
@pytest.mark.parametrize("matches", [True, False])
@pytest.mark.parametrize("history", [None, 1, 2])
def test_causal_conv_forward_shares_spatial_padding_verdicts(cpu_assembly, monkeypatch, matches: bool, history) -> None:
    """A normally called (e.g. hooked) convolution keeps the verified cuDNN padding path."""
    cache = weakref.WeakKeyDictionary()
    monkeypatch.setattr(fp, "_SPATIAL_PAD_VERDICTS", cache)
    generator = torch.Generator().manual_seed(0)
    conv = fp.WanCausalConv3d(2, 2, 3, padding=1)
    conv.weight.copy_(torch.randint(-2, 3, conv.weight.shape, generator=generator))
    conv.bias.copy_(torch.randint(-2, 3, conv.bias.shape, generator=generator))
    x = torch.randint(-2, 3, (1, 2, 2, 3, 4), generator=generator).float()
    cache_x = None if history is None else torch.randint(-2, 3, (1, 2, history, 3, 4), generator=generator).float()
    # Integer-valued operands keep both CPU convolution formulations exact.
    expected = fp.WanCausalConv3d.forward(conv, x, cache_x)
    conv3d = torch.nn.functional.conv3d

    def checked_conv3d(x, weight, bias, stride, padding, dilation, groups):
        output = conv3d(x, weight, bias, stride, padding, dilation, groups)
        # Force a mismatch on the candidate path to verify reference selection.
        return output + 1 if _is_candidate_call(True, x, padding) and not matches else output

    monkeypatch.setattr(fp.F, "conv3d", checked_conv3d)
    comparisons = []
    bitwise_equal = fp._bitwise_equal

    def compare(a, b):
        comparisons.append(bitwise_equal(a, b))
        return comparisons[-1]

    monkeypatch.setattr(fp, "_bitwise_equal", compare)
    for _ in range(2):
        assert torch.equal(fp.causal_conv_forward(conv, x, cache_x), expected)
    check = [True, True] if matches else [False]
    assert comparisons == check  # Verified once, then reused.
    assert list(cache[conv].values()) == [matches]
    # The cached call site shares the verdict instead of probing again.
    assert torch.equal(fp._run_cached_causal_conv(conv, x, [cache_x], 0), expected)
    assert comparisons == check
