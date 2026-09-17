# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""H3 row-chunk producer using the shared BF16 indexed gate operator.

Each chunk duplicates the owner's existing compact coarse tail. This keeps
the current native collective and its completion/abort protocol unchanged.
The additional bytes are measured and reported by the benchmark.
"""

import torch
from vllm.triton_utils import tl, triton

from vllm_omni.diffusion.layers.indexed_modulation import bf16_indexed_gate_add_


@triton.jit
def _produce(
    fine,
    coarse,
    owners,
    out,
    coarse_row_stride,
    coarse_head_stride,
    ROWS: tl.constexpr,  # noqa: N803
    OFFSET: tl.constexpr,  # noqa: N803
    COUNT: tl.constexpr,  # noqa: N803
    TAIL: tl.constexpr,  # noqa: N803
    WIDTH: tl.constexpr,  # noqa: N803
    BLOCK: tl.constexpr,  # noqa: N803
):
    row = tl.program_id(0)
    owner = row // (COUNT + TAIL)
    within = row % (COUNT + TAIL)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    fine_row = owner * ROWS + OFFSET + within
    fine_value = tl.load(fine + fine_row * WIDTH + col, (within < COUNT) & (col < WIDTH), other=0)
    tile = tl.load(owners + owner * TAIL + (within - COUNT), within >= COUNT, other=-1)
    coarse_offset = tile * coarse_row_stride + (col // 128) * coarse_head_stride + col % 128
    coarse_value = tl.load(coarse + coarse_offset, (within >= COUNT) & (tile >= 0) & (col < WIDTH), other=0)
    value = tl.where(within < COUNT, fine_value, coarse_value)
    tl.store(out + row * WIDTH + col, value, col < WIDTH)


def produce(fine, coarse, owner_tiles, landing, start, count):
    assert fine.shape == (1, 95936, 7, 128) and fine.is_contiguous()
    assert coarse.shape == (1, 1648, 7, 128) and coarse.stride(-1) == 1
    assert 0 <= start < start + count <= 11992
    assert landing.shape == (1, 8 * (count + 270), 7, 128) and landing.is_contiguous()
    _produce[(8 * (count + 270), triton.cdiv(896, 1024))](
        fine,
        coarse,
        owner_tiles,
        landing,
        coarse.stride(1),
        coarse.stride(2),
        ROWS=11992,
        OFFSET=start,
        COUNT=count,
        TAIL=270,
        WIDTH=896,
        BLOCK=1024,
        num_warps=8,
    )


def gate_chunk(bundle, gate, row_map, count):
    """The source kernel's unchanged BF16 multiply then BF16 add, on a row slice."""
    assert bundle.shape == (1, count + 270, 56, 128) and bundle.is_contiguous()
    assert gate.shape == (1, count, 56, 128) and gate.is_contiguous()
    assert row_map.shape == (count,) and row_map.dtype == torch.int32
    return bf16_indexed_gate_add_(bundle, gate, row_map)
