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
def test_evicted_verdict_is_revalidated(verdict_cache, monkeypatch, matches: bool) -> None:
    cache_name, cache = verdict_cache
    spatial_padding = cache_name == "_SPATIAL_PAD_VERDICTS"
    monkeypatch.setattr(fp, "_MAX_VERDICTS_PER_CONV", 2)
    monkeypatch.setattr(fp, "_kernels_allowed", lambda x: True)

    # Run the verdict paths on CPU using PyTorch in place of Triton assembly.
    def cat_time(x, cache_x, pad_front, keep_cache_frames):
        assert cache_x is None
        assembled = torch.nn.functional.pad(x, (0, 0, 0, 0, pad_front, 0))
        return assembled, assembled[:, :, -keep_cache_frames:].clone()

    def cat_pad(x, cache_x, padding, keep_cache_frames, channels_last_output=False):
        assembled, keep = cat_time(x, cache_x, padding[4], keep_cache_frames)
        assembled = torch.nn.functional.pad(assembled, (*padding[:4], 0, 0))
        if channels_last_output:
            assembled = assembled.contiguous(memory_format=torch.channels_last_3d)
        return assembled, keep

    monkeypatch.setattr(fp.dm, "cat_time_5d", cat_time)
    monkeypatch.setattr(fp.dm, "cat_pad_5d", cat_pad)
    conv = fp.WanCausalConv3d(2, 2, 3, padding=1)
    conv.weight.fill_(1)
    conv.bias.zero_()
    conv3d = torch.nn.functional.conv3d

    def checked_conv3d(x, weight, bias, stride, padding, dilation, groups):
        output = conv3d(x, weight, bias, stride, padding, dilation, groups)
        is_fast = padding == (0, 1, 1) if spatial_padding else x.is_contiguous(memory_format=torch.channels_last_3d)
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

    run(4)
    run(5)
    assert comparisons == [matches, matches]
    run(4)
    assert comparisons == [matches, matches]  # Both positive and negative hits are reused.
    run(6)
    assert {key[0][-1] for key in cache[conv]} == {5, 6}
    run(4)
    assert comparisons == [matches] * 4  # The evicted shape is checked again.
    assert {key[0][-1] for key in cache[conv]} == {4, 6}
    assert all(value is matches for value in cache[conv].values())
