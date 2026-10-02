# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Inference-only, inward-shifted 2D neighborhood attention for the LiDAR VAE.

Only geometry and compiled code are shared across requests. Spatial padding is
owned by the caller; masks describe the actual (possibly padded) input grid.
"""

from __future__ import annotations

import math
from functools import lru_cache
from types import FunctionType

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask, flex_attention

_BLOCK_SIZE = 64
_QUERY_CHUNK_SIZE = 4096
_CACHE_SIZE = 32
_KERNEL_OPTIONS = {
    "BLOCK_M": 64,
    "BLOCK_N": 64,
    "num_warps": 4,
    "num_stages": 1,
    "USE_TMA": False,
    "FORCE_USE_FLEX_ATTENTION": True,
    # Triton's dot precision is local to this operator, including the P @ V
    # multiplication. Do not change the surrounding model's TF32 policy.
    "FLOAT32_PRECISION": "'ieee'",
}


def _pair(value, name: str) -> tuple[int, int]:
    pair = tuple(value) if isinstance(value, tuple | list) else (value, value)
    if len(pair) != 2 or any(type(v) is not int or v < 1 for v in pair):
        raise ValueError(f"LiDAR attention {name} must contain two positive integers, got {value!r}.")
    return pair


def _axis_start(q, length: int, kernel: int, dilation: int):
    residue = q % dilation
    group_length = (length - 1 - residue) // dilation + 1
    return torch.minimum((q // dilation - kernel // 2).clamp(min=0), group_length - kernel)


def _mask_predicate(height: int, width: int, kernel: tuple[int, int], dilation: tuple[int, int]):
    def mask_mod(batch, head, q, p):
        qh, qw = q // width, q % width
        ph, pw = p // width, p % width
        sh = _axis_start(qh, height, kernel[0], dilation[0])
        sw = _axis_start(qw, width, kernel[1], dilation[1])
        return (
            (q < height * width)
            & (p < height * width)
            & (qh % dilation[0] == ph % dilation[0])
            & (qw % dilation[1] == pw % dilation[1])
            & (ph // dilation[0] >= sh)
            & (ph // dilation[0] < sh + kernel[0])
            & (pw // dilation[1] >= sw)
            & (pw // dilation[1] < sw + kernel[1])
        )

    return mask_mod


@lru_cache(maxsize=_CACHE_SIZE)
def _get_block_mask(
    device: torch.device, height: int, width: int, kernel: tuple[int, int], dilation: tuple[int, int]
) -> BlockMask:
    for length, k, d in zip((height, width), kernel, dilation):
        # Every residue group must contain the entire window, including the
        # shorter groups when the axis is not divisible by dilation.
        if length < k * d:
            raise ValueError(f"LiDAR attention axis length {length} is smaller than kernel {k} * dilation {d}.")
    sequence = height * width
    blocks = math.ceil(sequence / _BLOCK_SIZE)
    counts = torch.empty((blocks,), dtype=torch.int32, device=device)
    indices = torch.empty((blocks, blocks), dtype=torch.int32, device=device)
    dh = torch.arange(kernel[0], device=device)
    dw = torch.arange(kernel[1], device=device)
    for first in range(0, sequence, _QUERY_CHUNK_SIZE):
        q = torch.arange(first, min(first + _QUERY_CHUNK_SIZE, sequence), device=device)
        qh, qw = q // width, q % width
        ph = (_axis_start(qh, height, kernel[0], dilation[0])[:, None] + dh) * dilation[0]
        ph = ph + (qh % dilation[0])[:, None]
        pw = (_axis_start(qw, width, kernel[1], dilation[1])[:, None] + dw) * dilation[1]
        pw = pw + (qw % dilation[1])[:, None]
        key_blocks = ((ph[:, :, None] * width + pw[:, None, :]) // _BLOCK_SIZE).flatten(1)
        row = (q - first) // _BLOCK_SIZE
        rows = math.ceil(q.numel() / _BLOCK_SIZE)
        present = torch.zeros((rows, blocks), dtype=torch.bool, device=device)
        present.view(-1).scatter_(0, (row[:, None] * blocks + key_blocks).flatten(), True)
        start = first // _BLOCK_SIZE
        counts[start : start + rows] = present.sum(-1, dtype=torch.int32)
        # Full-width contiguous int32 metadata matches FlexAttention's native
        # layout. Only the first `counts[row]` entries are visited by the kernel.
        indices[start : start + rows] = present.to(torch.int8).argsort(descending=True, stable=True).to(torch.int32)
    return BlockMask.from_kv_blocks(
        counts[None, None],
        indices[None, None],
        BLOCK_SIZE=(_BLOCK_SIZE, _BLOCK_SIZE),
        mask_mod=_mask_predicate(height, width, kernel, dilation),
        seq_lengths=(sequence, sequence),
        compute_q_blocks=False,
    )


def _run_attention(query, key, value, block_mask, scale):
    head_dim = query.shape[-1]
    # CUDA FlexAttention requires head dimensions >= 16 in some PyTorch
    # versions. Zero padding preserves Q @ K and P @ V with an explicit scale.
    if head_dim < 16:
        query, key, value = (F.pad(t, (0, 16 - head_dim)) for t in (query, key, value))
    output = flex_attention(query, key, value, block_mask=block_mask, scale=scale, kernel_options=_KERNEL_OPTIONS)
    return output[..., :head_dim] if head_dim < 16 else output


@lru_cache(maxsize=_CACHE_SIZE)
def _get_compiled_runner(device, shape, kernel, dilation, scale):
    # Dynamo caches by code object. A new wrapper around the same code would
    # still hit its recompile limit across decoder levels and partial chunks.
    # Give each static specialization its own code object, without capturing
    # tensors or a mask (which would keep evicted device allocations alive).
    runner = FunctionType(_run_attention.__code__.replace(), globals(), _run_attention.__name__)
    return torch.compile(runner, fullgraph=True, dynamic=False)


def neighborhood_attention_2d(query, key, value, *, kernel_size, dilation, scale):
    """Attend to a fixed 2D window on FP32 [B,H,W,heads,head_dim] tensors.

    CUDA always uses compiled sparse attention and propagates failures. Eager
    CPU FlexAttention is available only for small correctness fixtures, since
    PyTorch's eager implementation materializes token-level scores.
    """
    if query.ndim != 5 or any(t.shape != query.shape for t in (key, value)) or min(query.shape) < 1:
        raise ValueError("LiDAR attention requires matching nonempty [B,H,W,heads,head_dim] tensors.")
    if any(t.dtype != torch.float32 or t.device != query.device for t in (query, key, value)):
        raise ValueError("LiDAR attention requires FP32 query, key, and value on the same device.")
    if query.device.type not in {"cpu", "cuda"}:
        raise RuntimeError(f"LiDAR FlexAttention requires CUDA (or CPU correctness fixtures), got {query.device}.")
    kernel, dilation = _pair(kernel_size, "kernel_size"), _pair(dilation, "dilation")
    batch, height, width, heads, dim = query.shape
    if query.device.type == "cpu" and height * width > 4096:
        raise RuntimeError("Eager CPU LiDAR FlexAttention is limited to 4096 tokens; use CUDA for production grids.")
    # Tensor.device resolves an unindexed 'cuda' to the actual device, so an
    # offloaded/reloaded module never reuses another GPU's mask or runner.
    try:
        mask = _get_block_mask(query.device, height, width, kernel, dilation)
        q, k, v = (
            t.permute(0, 3, 1, 2, 4).reshape(batch, heads, height * width, dim).contiguous()
            for t in (query, key, value)
        )
        runner = (
            _get_compiled_runner(query.device, tuple(query.shape), kernel, dilation, scale)
            if query.is_cuda
            else _run_attention
        )
        with torch.autocast(device_type=query.device.type, enabled=False):
            output = runner(q, k, v, mask, scale)
    except ValueError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"Cosmos3 LiDAR FlexAttention failed on {query.device} for shape {tuple(query.shape)}, "
            f"kernel={kernel}, dilation={dilation}; compiled sparse attention is required on CUDA."
        ) from exc
    return output.transpose(1, 2).reshape(batch, height, width, heads, dim).contiguous()
