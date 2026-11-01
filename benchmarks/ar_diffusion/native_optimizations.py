# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request-local projections and layout changes for fixed-stage Wan benchmarks."""

from contextlib import contextmanager

import torch
from torch import nn


class CachedProjection(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.values = {}

    def forward(self, value):
        key = id(value)
        if key not in self.values:
            self.values[key] = (value, self.base(value))
        return self.values[key][1]

    def clear(self):
        self.values.clear()


class FusedNativeBlock(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base

    @property
    def attn1(self):
        return self.base.attn1

    def forward(self, hidden, text, temb, rotary, mask=None, vsa_shape=None, preserve_vsa=False):
        if temb.ndim != 3 or mask is not None or vsa_shape is not None or preserve_vsa:
            raise ValueError("pointwise benchmark fusion supports native Wan T2V only")
        from .native_pointwise import residual

        block = self.base
        shift, scale, gate, ff_shift, ff_scale, ff_gate = (block.scale_shift_table + temb).chunk(6, dim=1)

        def add(value, delta, gate):
            if value.dtype == delta.dtype == gate.dtype == torch.bfloat16:
                return residual(value, delta, gate)
            return (value + delta * gate).type_as(value)

        normalized = block.norm1(hidden, scale, shift).type_as(hidden)
        # StageWanTransformer's paged self-attention hook ignores metadata.
        hidden = add(hidden, block.attn1(normalized, rotary, None), gate)
        hidden = hidden + block.attn2(block.norm2(hidden).type_as(hidden), text, None)
        normalized = block.norm3(hidden, ff_scale, ff_shift).type_as(hidden)
        return add(hidden, block.ffn(normalized), ff_gate)


@contextmanager
def optimized(model, variant):
    if variant == "baseline":
        yield []
        return
    if variant not in ("cached", "fused"):
        raise ValueError("unknown native optimization")
    wan = model.wan
    modules = []
    blocks = []
    for index in range(model.start_layer, model.end_layer):
        block = wan.blocks[index]
        projections = {name: getattr(block.attn2, name) for name in ("to_k", "to_v", "norm_k")}
        blocks.append((index, block, projections))
    try:
        for index, block, projections in blocks:
            for name, projection in projections.items():
                wrapper = CachedProjection(projection)
                setattr(block.attn2, name, wrapper)
                modules.append(wrapper)
            if variant == "fused":
                wan.blocks[index] = FusedNativeBlock(block)
        yield modules
    finally:
        for index, block, projections in blocks:
            wan.blocks[index] = block
            for name, projection in projections.items():
                setattr(block.attn2, name, projection)
