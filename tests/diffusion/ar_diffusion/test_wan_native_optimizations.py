# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request caches must expire, and scoped adapters must restore native modules."""

import pytest
import torch
from torch import nn

from benchmarks.ar_diffusion.native_optimizations import CachedProjection, optimized

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class CountingProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, value):
        self.calls += 1
        return value * 2


class AdapterStage(nn.Module):
    """Minimal module graph exercising adapter installation, not model math."""

    def __init__(self):
        super().__init__()
        self.start_layer, self.end_layer = 0, 1
        self.wan = nn.Module()
        self.wan.condition_embedder = nn.Identity()
        block = nn.Module()
        block.attn2 = nn.Module()
        for name in ("to_k", "to_v", "norm_k"):
            setattr(block.attn2, name, CountingProjection())
        self.wan.blocks = nn.ModuleList([block])


def test_projection_cache_uses_identity_and_expires_at_request_boundary():
    base = CountingProjection()
    cached = CachedProjection(base)
    value = torch.tensor([1.0])
    first = cached(value)
    assert cached(value) is first
    assert base.calls == 1
    equal_but_distinct = value.clone()
    assert torch.equal(cached(equal_but_distinct), first)
    assert base.calls == 2
    cached.clear()
    value.fill_(3)
    assert torch.equal(cached(value), torch.tensor([6.0]))
    assert base.calls == 3


@pytest.mark.parametrize("variant", ["cached", "fused"])
@pytest.mark.parametrize("abort", [False, True])
def test_scoped_optimizations_restore_original_modules_on_exit(variant, abort):
    stage = AdapterStage()
    condition = stage.wan.condition_embedder
    block = stage.wan.blocks[0]
    originals = {name: getattr(block.attn2, name) for name in ("to_k", "to_v", "norm_k")}

    def run():
        with optimized(stage, variant) as caches:
            assert stage.wan.condition_embedder is condition
            assert len(caches) == 3
            for name in originals:
                assert isinstance(getattr(block.attn2, name), CachedProjection)
            if abort:
                raise RuntimeError("request failed")

    if abort:
        with pytest.raises(RuntimeError, match="request failed"):
            run()
    else:
        run()
    assert stage.wan.condition_embedder is condition
    assert stage.wan.blocks[0] is block
    for name, original in originals.items():
        assert getattr(block.attn2, name) is original
