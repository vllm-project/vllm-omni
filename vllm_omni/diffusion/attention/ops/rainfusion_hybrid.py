# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""RainFusion block selection with independent Q/KV layouts and dense guards.

Context and first-frame query blocks already attend to every key in rf_v2.
Compute those rows with Flash and use the native rectangular RainFusion op for
the remaining video queries. No reference queries are manufactured on reuse.
"""

from __future__ import annotations

import atexit
import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

import torch

SpanSpec = tuple[int, tuple[int, int, int]]


def _tile_indices(start: int, grid: tuple[int, int, int]) -> torch.Tensor:
    """CPU equivalent of rf_v2's 8x8 tiling, including spatial remainders."""
    frames, height, width = grid
    ids = torch.arange(start, start + math.prod(grid), dtype=torch.long).reshape(frames, height, width)
    h_main, w_main = height // 8 * 8, width // 8 * 8
    if h_main == height and w_main == width:
        return ids.reshape(frames, height // 8, 8, width // 8, 8).permute(0, 1, 3, 2, 4).reshape(-1)
    rest = ids[1:]
    main = rest[:, :h_main, :w_main]
    main = main.reshape(frames - 1, h_main // 8, 8, w_main // 8, 8).permute(0, 1, 3, 2, 4)
    parts = [main.reshape(frames - 1, h_main * w_main)]
    if height != h_main:
        parts.append(rest[:, h_main:, :].reshape(frames - 1, (height - h_main) * width))
    if width != w_main:
        parts.append(rest[:, :h_main, w_main:].reshape(frames - 1, h_main * (width - w_main)))
    return torch.cat((ids[0].reshape(-1), torch.cat(parts, dim=1).reshape(-1)))


def _packing_cpu(length: int, spans: tuple[SpanSpec, ...], block_size: int):
    """Make a complete permutation and protected blocks without device sync."""
    previous_end = 0
    dense_parts = []
    for start, grid in spans:
        if len(grid) != 3 or any(dim <= 0 for dim in grid):
            raise ValueError("RainFusion video grids must contain three positive dimensions")
        end = start + math.prod(grid)
        if start < previous_end or end > length:
            raise ValueError("RainFusion spans overlap or exceed the valid sequence")
        if start > previous_end:
            dense_parts.append(torch.arange(previous_end, start, dtype=torch.long))
        previous_end = end
    if previous_end < length:
        dense_parts.append(torch.arange(previous_end, length, dtype=torch.long))
    dense = torch.cat(dense_parts) if dense_parts else torch.empty(0, dtype=torch.long)
    parts = []
    protected = set()
    cursor = position = 0
    for index, (start, grid) in enumerate(spans):
        if position % block_size:
            raise ValueError("RainFusion clip boundaries must start on a block boundary")
        protected.update(
            range(position // block_size, position // block_size + math.ceil(grid[1] * grid[2] / block_size))
        )
        parts.append(_tile_indices(start, grid))
        position += math.prod(grid)
        if index < len(spans) - 1 and position % block_size:
            needed = block_size - position % block_size
            if cursor + needed > dense.numel():
                raise ValueError("RainFusion layout lacks dense rows for clip boundary isolation")
            protected.add(position // block_size)
            parts.append(dense[cursor : cursor + needed])
            cursor += needed
            position += needed
    if cursor < dense.numel():
        end = position + dense.numel() - cursor
        protected.update(range(position // block_size, math.ceil(end / block_size)))
        parts.append(dense[cursor:])
        position = end
    if not parts or position != length:
        raise ValueError("RainFusion packing must cover the entire valid sequence")
    return torch.cat(parts), protected


@dataclass(frozen=True)
class HybridPlan:
    sparse_queries: torch.Tensor
    dense_queries: torch.Tensor
    key_permutation: torch.Tensor
    protected_keys: torch.Tensor
    key_block_ids: torch.Tensor
    dense_cu_q: torch.Tensor
    dense_cu_k: torch.Tensor
    sparse_query_rows: int
    dense_query_rows: int
    key_rows: int
    key_blocks: int


@lru_cache(maxsize=32)
def get_hybrid_plan(
    q_length: int,
    kv_length: int,
    q_spans: tuple[SpanSpec, ...],
    kv_spans: tuple[SpanSpec, ...],
    device: str,
    block_size: int = 128,
) -> HybridPlan:
    """Share small static index tensors across layers; never cache activations."""
    q_perm, q_protected = _packing_cpu(q_length, q_spans, block_size)
    k_perm, k_protected = _packing_cpu(kv_length, kv_spans, block_size)
    dense_mask = torch.zeros(q_length, dtype=torch.bool)
    for block in q_protected:
        dense_mask[block * block_size : (block + 1) * block_size] = True
    sparse = q_perm[~dense_mask]
    dense = q_perm[dense_mask].sort().values
    key_blocks = math.ceil(kv_length / block_size)

    def upload(tensor):
        # One-time metadata upload; complete it before releasing CPU indices.
        return tensor.to(device=device)

    return HybridPlan(
        sparse_queries=upload(sparse),
        dense_queries=upload(dense),
        key_permutation=upload(k_perm),
        protected_keys=upload(torch.tensor(sorted(k_protected), dtype=torch.long)),
        key_block_ids=upload(torch.arange(key_blocks, dtype=torch.float32)),
        dense_cu_q=upload(torch.tensor([0, dense.numel()], dtype=torch.int32)),
        dense_cu_k=upload(torch.tensor([0, kv_length], dtype=torch.int32)),
        sparse_query_rows=sparse.numel(),
        dense_query_rows=dense.numel(),
        key_rows=kv_length,
        key_blocks=key_blocks,
    )


# Release cached device metadata before torch_npu tears down its context.
atexit.register(get_hybrid_plan.cache_clear)


def _pool(tensor: torch.Tensor, block_size: int) -> torch.Tensor:
    batch, length, heads, dim = tensor.shape
    full, tail = divmod(length, block_size)
    parts = []
    if full:
        parts.append(tensor[:, : full * block_size].reshape(batch, full, block_size, heads, dim).mean(dim=2))
    if tail:
        parts.append(tensor[:, full * block_size :].mean(dim=1, keepdim=True))
    return torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]


def block_mask(
    query: torch.Tensor,
    key: torch.Tensor,
    plan: HybridPlan,
    sparsity: float,
    scale: float,
    block_size: int = 128,
):
    """Preserve rf_v2's BF16 softmax/topk/tie policy and all protected keys."""
    q_pool, k_pool = _pool(query, block_size), _pool(key, block_size)
    scores = torch.einsum("blnd,bsnd->bnls", q_pool, k_pool) * scale
    scores = torch.nn.functional.softmax(scores, dim=-1)
    keep = max(1, math.ceil(plan.key_blocks * (1 - sparsity)))
    threshold = torch.topk(scores, k=keep, dim=-1).values[..., -1:]
    mask = scores >= threshold
    if plan.protected_keys.numel():
        mask[:, :, :, plan.protected_keys] = True
    return mask


def select_blocks(query, key, plan, sparsity, scale, block_size=128):
    mask = block_mask(query, key, plan, sparsity, scale, block_size)
    count = mask.sum(dim=-1)
    indices = plan.key_block_ids.reshape(1, 1, 1, -1).expand_as(mask)
    sorted_ids = torch.where(mask, indices, 1e9).sort(dim=-1).values
    indices = torch.where(indices < count.unsqueeze(-1), sorted_ids, -1).to(torch.int64)
    return indices[0].transpose(0, 1).contiguous(), count[0].transpose(0, 1).contiguous()


def _native_forward(*args, **kwargs):
    from mindiesd.layers.flash_attn.sparse_flash_attn_rf_v2 import rain_fusion_attention

    return rain_fusion_attention(*args, **kwargs)


def _native_mask_forward(query, key, value, mask, scale, inner_precise):
    from mindiesd.layers import _custom_ops  # noqa: F401

    out, _ = torch.ops.mindiesd.block_sparse_attention(
        query=query.transpose(1, 2).contiguous(),
        key=key.transpose(1, 2).contiguous(),
        value=value.transpose(1, 2).contiguous(),
        block_sparse_mask=mask.to(torch.int8).contiguous(),
        block_shape=[128, 128],
        q_input_layout="BNSD",
        kv_input_layout="BNSD",
        num_key_value_heads=query.shape[2],
        scale_value=scale,
        inner_precise=inner_precise,
        actual_seq_lengths=[query.shape[1]],
        actual_seq_lengths_kv=[key.shape[1]],
        softmax_lse_flag=0,
    )
    return out.transpose(1, 2)


def hybrid_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    plan: HybridPlan,
    dense_forward: Callable,
    *,
    sparsity: float,
    scale: float,
    inner_precise: int = 0,
    kernel_dtype: str = "fp16",
    input_scale: float = 256.0,
    kernel: str = "bsa",
) -> torch.Tensor:
    """BSND, one packed document; output contains only the caller's Q rows."""
    if query.shape[0] != 1 or key.shape != value.shape or query.shape[2:] != key.shape[2:]:
        raise ValueError("hybrid RainFusion requires one BSND document and matching KV heads")
    if query.shape[1] != plan.sparse_query_rows + plan.dense_query_rows or key.shape[1] != plan.key_rows:
        raise ValueError("hybrid RainFusion plan does not match valid Q/KV lengths")
    out = torch.empty_like(query)
    if plan.dense_query_rows:
        dense_q = query.index_select(1, plan.dense_queries)
        dense_out = dense_forward(dense_q, key, value, plan)
        out.index_copy_(1, plan.dense_queries, dense_out)
    if plan.sparse_query_rows:
        sparse_q = query.index_select(1, plan.sparse_queries)
        sparse_k = key.index_select(1, plan.key_permutation)
        if kernel == "bsa":
            mask = block_mask(sparse_q, sparse_k, plan, sparsity, scale)
        elif kernel == "rf2":
            select_idx, select_count = select_blocks(sparse_q, sparse_k, plan, sparsity, scale)
        else:
            raise ValueError("hybrid RF kernel must be bsa or rf2")
        sparse_v = value.index_select(1, plan.key_permutation)
        heads, dim = query.shape[2:]
        output_dtype = query.dtype
        restore_scale = 1.0
        kernel_scale = scale
        if kernel_dtype == "fp16":
            # 910B's BF16 RF kernel requires inner_precise=0. FP16 permits its
            # faster mode 1. Select blocks in the original dtype, then apply
            # exact power-of-two range protection to the attention inputs.
            if input_scale < 1 or not math.isfinite(input_scale) or math.log2(input_scale) % 1:
                raise ValueError("RF input_scale must be a finite power of two >= 1")
            sparse_q = (sparse_q / input_scale).to(torch.float16)
            sparse_k = (sparse_k / input_scale).to(torch.float16)
            sparse_v = (sparse_v / input_scale).to(torch.float16)
            kernel_scale *= input_scale**2
            restore_scale = input_scale
            inner_precise = 1
        elif kernel_dtype != "bf16":
            raise ValueError("hybrid RF kernel_dtype must be fp16 or bf16")
        if kernel == "bsa":
            sparse_out = _native_mask_forward(sparse_q, sparse_k, sparse_v, mask, kernel_scale, inner_precise)
        else:
            sparse_out = _native_forward(
                sparse_q.reshape(-1, heads, dim),
                sparse_k.reshape(-1, heads, dim),
                sparse_v.reshape(-1, heads, dim),
                scale=kernel_scale,
                head_num=heads,
                input_layout="TND",
                select_idx=select_idx,
                select_num_idx=select_count,
                blockshape=[128, 128],
                actual_seq_lengths=[plan.sparse_query_rows],
                actual_seq_lengths_kv=[plan.key_rows],
                inner_precise=inner_precise,
            ).reshape(1, plan.sparse_query_rows, heads, dim)
        sparse_out = sparse_out.to(output_dtype) * restore_scale
        out.index_copy_(1, plan.sparse_queries, sparse_out)
    return out
