# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""A5 rectangular attention with full-KV packing and protected query rows."""

import atexit
import math
from dataclasses import dataclass
from functools import lru_cache

import torch
from mindiesd.layers.flash_attn.sparse_flash_attn_rf_v2 import _multi_span_permutation, _normalize_video_spans, avgpool
from mindiesd.layers.flash_attn.sparse_flash_attn_rf_v3 import rain_fusion_attention_v3

from vllm_omni.diffusion.attention.backends.abstract import ReferenceImageLayout


@dataclass
class Plan:
    key_indices: torch.Tensor
    sparse_indices: torch.Tensor
    dense_indices: torch.Tensor
    protected_keys: torch.Tensor
    image_groups: tuple[torch.Tensor, ...]
    image_query_blocks: torch.Tensor
    sparse_rows: int
    dense_rows: int
    key_rows: int


@lru_cache(maxsize=32)
def get_plan(
    kv_rows: int,
    reference_rows: int,
    spans: tuple[tuple[int, tuple[int, ...]], ...],
    device: str,
    *,
    image_spans: tuple[tuple[int, tuple[int, ...]], ...] = (),
    image_layout: ReferenceImageLayout | None = None,
) -> Plan:
    block_size = 128
    descriptors = [{"start": start, "latent_shape": list(grid)} for start, grid in spans]
    normalized = _normalize_video_spans(descriptors, kv_rows)
    image_groups: tuple[tuple[int, ...], ...] = ()
    eligible_blocks: tuple[int, ...] = ()
    if image_spans:
        if image_layout is None:
            raise ValueError("rectangular image sparsity requires the original reference layout")
        from mindiesd.layers.flash_attn.sparse_flash_attn_rf_v2 import _normalize_image_spans

        images = _normalize_image_spans(
            [{"start": start, "latent_shape": list(grid)} for start, grid in image_spans], kv_rows
        )
        perm, _, dense_blocks, first, (image_groups, eligible_blocks) = _multi_span_permutation(
            normalized, kv_rows, block_size, "BSND", "cpu", image_spans=images, return_reference_plan=True
        )
        positions = image_layout.reference_positions
        if (
            len(positions) != reference_rows
            or tuple(sorted(set(positions))) != positions
            or (positions and (positions[0] < 0 or positions[-1] >= kv_rows))
        ):
            raise ValueError("reference positions must describe the valid, sorted cached KV rows")
        reference_idx = torch.tensor(positions, dtype=torch.long, device="cpu")
        is_active = torch.ones(kv_rows, dtype=torch.bool, device="cpu")
        is_active[reference_idx] = False
        # Gather directly from [cached reference KV | active KV] into RF order.
        original_to_kv = torch.empty(kv_rows, dtype=torch.long, device="cpu")
        original_to_kv[reference_idx] = torch.arange(reference_rows, device="cpu")
        original_to_kv[is_active] = torch.arange(reference_rows, kv_rows, device="cpu")
        perm = original_to_kv[perm]
    else:
        perm, _, dense_blocks, first = _multi_span_permutation(normalized, kv_rows, block_size, "BSND", "cpu")

    protected = dense_blocks | first
    eligible = set(eligible_blocks)
    sparse: list[int] = []
    dense: list[int] = []
    image_query_blocks: list[int] = []
    for block in range(math.ceil(kv_rows / block_size)):
        ids = perm[block * block_size : (block + 1) * block_size]
        kept = ids[ids >= reference_rows]
        if not kept.numel():
            continue
        partial = kept.numel() != ids.numel()
        if block in protected or (image_spans and partial):
            dense.extend((kept - reference_rows).tolist())
        else:
            if partial:
                raise ValueError("unprotected block crosses removed reference Q boundary")
            if block in eligible:
                image_query_blocks.append(len(sparse) // block_size)
            # Kept sparse blocks retain their RF boundaries and order.
            sparse.extend((kept - reference_rows).tolist())
    if sorted(sparse + dense) != list(range(kv_rows - reference_rows)):
        raise ValueError("rectangular query plan must cover every surviving row once")

    return Plan(
        key_indices=perm.to(device),
        sparse_indices=torch.tensor(sparse, dtype=torch.long, device=device),
        dense_indices=torch.tensor(dense, dtype=torch.long, device=device),
        protected_keys=torch.tensor(sorted(protected), dtype=torch.long, device=device),
        image_groups=tuple(torch.tensor(group, dtype=torch.long, device=device) for group in image_groups if group),
        image_query_blocks=torch.tensor(image_query_blocks, dtype=torch.long, device=device),
        sparse_rows=len(sparse),
        dense_rows=len(dense),
        key_rows=kv_rows,
    )


atexit.register(get_plan.cache_clear)


def native_bf16(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor, scale: float) -> torch.Tensor:
    return rain_fusion_attention_v3(
        q,
        k,
        v,
        mask,
        scale=scale,
        head_num=q.shape[2],
        num_key_value_heads=q.shape[2],
        input_layout="BSND",
        actual_seq_lengths=[q.shape[1]],
        actual_seq_lengths_kv=[k.shape[1]],
        block_size_q=128,
        block_size_kv=128,
        inner_precise=4,
    )


def native_mix(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor, scale: float) -> torch.Tensor:
    from mindiesd.layers.flash_attn.sparse_flash_attn_rf_v3 import _eagle_qbsa_quant_qkv
    from torch_npu import float8_e4m3fn

    if q.shape[-1] != 128:
        raise ValueError("rectangular mix validated for head_dim128 only")
    qq, kk, vv, qs, ks, vs = _eagle_qbsa_quant_qkv(q, k, v, block_size_q=64, layout="BSND")
    out, _ = torch.ops.mindiesd.eagle_quant_block_sparse_attention(
        query=qq,
        key=kk,
        value=vv.view(torch.int8),
        block_sparse_mask=mask,
        block_shape=[128, 128],
        q_input_layout="BNSD",
        kv_input_layout="BNSD",
        num_key_value_heads=q.shape[2],
        scale_value=scale,
        inner_precise=4,
        softmax_lse_flag=0,
        actual_seq_lengths=[q.shape[1]],
        actual_seq_lengths_kv=[k.shape[1]],
        query_scale=qs,
        key_scale=ks,
        value_scale=vs,
        query_dtype=torch.int8,
        key_dtype=torch.int8,
        value_dtype=float8_e4m3fn,
        output_dtype=torch.bfloat16,
    )
    return out.transpose(1, 2).contiguous()


def run(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    plan: Plan,
    sparsity: float,
    scale: float,
    *,
    precision: str = "bf16",
    image_sparsity: float = 0.0,
) -> torch.Tensor:
    if precision not in ("bf16", "mix"):
        raise ValueError("unsupported rectangular precision")

    def native(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor, scale: float) -> torch.Tensor:
        return native_mix(q, k, v, mask, scale) if precision == "mix" else native_bf16(q, k, v, mask, scale)

    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("BF16 inputs required")
    if (
        q.shape[0] != 1
        or k.shape != v.shape
        or q.shape[1] != plan.sparse_rows + plan.dense_rows
        or k.shape[1] != plan.key_rows
    ):
        raise ValueError("layout mismatch")
    kr = k.index_select(1, plan.key_indices)
    vr = v.index_select(1, plan.key_indices)
    out = torch.empty_like(q)
    if plan.sparse_rows:
        qs = q.index_select(1, plan.sparse_indices)
        qp = avgpool(qs, 128, "BSND")
        kp = avgpool(kr, 128, "BSND")
        scores = torch.einsum("blnd,bsnd->bnls", qp, kp) * scale
        probs = torch.nn.functional.softmax(scores, dim=-1)
        keep = max(1, math.ceil(probs.shape[-1] * (1 - sparsity)))
        threshold = torch.topk(probs, k=keep, dim=-1).values[..., -1:]
        mask = probs >= threshold
        if plan.protected_keys.numel():
            mask[:, :, :, plan.protected_keys] = True
        # Match RF-v3: overwrite just the complete image columns on eligible
        # query blocks, ranking by the original Q/K scores before softmax.
        rows = plan.image_query_blocks
        if image_sparsity > 0 and rows.numel():
            for columns in plan.image_groups:
                values = scores[..., rows[:, None], columns[None, :]]
                keep = max(1, math.ceil(columns.numel() * (1 - image_sparsity)))
                selected = values.topk(keep, dim=-1).indices
                mask[..., rows[:, None], columns[None, :]] = torch.zeros_like(values, dtype=torch.bool).scatter_(
                    -1, selected, True
                )
        ys = native(qs, kr, vr, mask.to(torch.int8), scale)
        out.index_copy_(1, plan.sparse_indices, ys)
    if plan.dense_rows:
        qd = q.index_select(1, plan.dense_indices)
        dense_mask = torch.ones(
            1,
            q.shape[2],
            math.ceil(plan.dense_rows / 128),
            math.ceil(plan.key_rows / 128),
            device=q.device,
            dtype=torch.int8,
        )
        yd = native(qd, kr, vr, dense_mask, scale)
        out.index_copy_(1, plan.dense_indices, yd)
    return out
