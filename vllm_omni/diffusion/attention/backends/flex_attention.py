# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Range-mask construction adapted from Tencent-Hunyuan/Hy-Nomas (Copyright 2026 Tencent).

from collections import OrderedDict
from functools import cache

import numpy as np
import torch
from torch.nn.attention.flex_attention import BlockMask, flex_attention

from vllm_omni.diffusion.attention.backends.abstract import AttentionBackend, AttentionImpl, AttentionMetadata


@cache
def compiled_flex():
    return torch.compile(flex_attention, dynamic=True, fullgraph=True)


def build_range_mask(q_length, k_length, ranges, device, query_mask, key_mask):
    ranges = np.asarray(ranges, dtype=np.int64).reshape(-1, 4)
    for index, (qs, qe, ks, ke) in enumerate(ranges):
        if not (0 <= qs <= qe <= q_length and 0 <= ks <= ke <= k_length):
            raise ValueError("Attention range exceeds its sequence.")
        for a, b, c, d in ranges[:index]:
            if max(qs, a) < min(qe, b) and max(ks, c) < min(ke, d):
                raise ValueError("Attention rectangles must not overlap.")
    edges = np.unique(np.r_[0, ranges[:, :2].ravel(), q_length])
    rows: list[list[list[int]]] = []
    for start in edges[:-1]:
        intervals = sorted((int(ks), int(ke)) for qs, qe, ks, ke in ranges if qs <= start < qe and ks < ke)
        merged: list[list[int]] = []
        for left, right in intervals:
            if merged and left <= merged[-1][1]:
                merged[-1][1] = max(right, merged[-1][1])
            else:
                merged.append([left, right])
        rows.append(merged)
    width = max(1, max(map(len, rows), default=0))
    starts = np.zeros((len(rows) + 1, width), dtype=np.int64)
    ends = np.zeros_like(starts)
    row_ids = np.full(q_length + 1, len(rows), dtype=np.int64)
    for index, row_intervals in enumerate(rows):
        row_ids[edges[index] : edges[index + 1]] = index
        for column, (left, right) in enumerate(row_intervals):
            starts[index, column], ends[index, column] = left, right
    lookup = torch.from_numpy(row_ids).to(device)
    lo = torch.from_numpy(starts.ravel()).to(device)
    hi = torch.from_numpy(ends.ravel()).to(device)
    q_visible = torch.from_numpy(np.r_[query_mask, False]).to(device)
    k_visible = torch.from_numpy(np.r_[key_mask, False]).to(device)

    def mask_mod(batch, head, query, key):
        row = lookup[query.clamp(0, q_length)]
        allowed = (key >= lo[row * width]) & (key < hi[row * width])
        for column in range(1, width):
            allowed = allowed | ((key >= lo[row * width + column]) & (key < hi[row * width + column]))
        return (
            allowed
            & q_visible[query.clamp(0, q_length)]
            & k_visible[key.clamp(0, k_length)]
            & (query >= 0)
            & (query < q_length)
            & (key >= 0)
            & (key < k_length)
        )

    # Build only block indices, without a dense token-by-token history mask.
    q_blocks, k_blocks = (q_length + 127) // 128, (k_length + 127) // 128
    occupancy = np.zeros((q_blocks, k_blocks), dtype=np.uint8)
    full = np.zeros_like(occupancy, dtype=bool)
    for qs, qe, ks, ke in ranges:
        if qs != qe and ks != ke:
            occupancy[qs // 128 : (qe + 127) // 128, ks // 128 : (ke + 127) // 128] = 1
            full[(qs + 127) // 128 : qe // 128, (ks + 127) // 128 : ke // 128] = True
    qt = np.pad(query_mask, (0, q_blocks * 128 - q_length)).reshape(q_blocks, 128)
    kt = np.pad(key_mask, (0, k_blocks * 128 - k_length)).reshape(k_blocks, 128)
    occupancy &= qt.any(1)[:, None] & kt.any(1)[None, :]
    full &= qt.all(1)[:, None] & kt.all(1)[None, :]
    occupancy[full] = 2
    fields = {}
    for prefix, selected in (("kv", occupancy == 1), ("full_kv", occupancy == 2)):
        indices = np.zeros((q_blocks, max(k_blocks, 1)), dtype=np.int32)
        for row in range(q_blocks):
            columns = np.flatnonzero(selected[row]).astype(np.int32)
            indices[row, : len(columns)] = columns
        fields[prefix + "_num_blocks"] = torch.from_numpy(selected.sum(-1, dtype=np.int32)[None, None]).to(device)
        fields[prefix + "_indices"] = torch.from_numpy(indices[None, None]).to(device)
    return BlockMask.from_kv_blocks(
        **fields, BLOCK_SIZE=(128, 128), mask_mod=mask_mod, seq_lengths=(q_length, k_length)
    )


class FlexAttentionImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        qkv_layout: str | None = None,
        backend_kwargs: dict | None = None,
        **extra_impl_args,
    ) -> None:
        if causal:
            raise ValueError("FLEX_ATTENTION uses explicit ranges rather than an implicit causal mask")
        if backend_kwargs:
            raise ValueError("FLEX_ATTENTION does not accept backend kwargs")
        self.softmax_scale = softmax_scale
        self.masks: OrderedDict[tuple, BlockMask] = OrderedDict()

    def clear(self) -> None:
        self.masks.clear()

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_metadata: AttentionMetadata | None = None
    ) -> torch.Tensor:
        restriction = None if attn_metadata is None else attn_metadata.range_mask
        ranges = None if restriction is None else restriction.rectangles
        query_mask = None if restriction is None else restriction.query_mask
        key_mask = None if restriction is None else restriction.key_mask
        if attn_metadata is not None and attn_metadata.attn_mask is not None:
            if restriction is not None or attn_metadata.attn_mask.ndim != 2:
                raise ValueError("FLEX_ATTENTION accepts a 2D padding mask or explicit range_mask")
            key_mask = attn_metadata.attn_mask.bool()
            query_mask = key_mask if q.shape[1] == k.shape[1] else None
        if q.ndim != 4 or k.ndim != 4 or k.shape != v.shape or q.shape[0] != k.shape[0]:
            raise ValueError("FLEX_ATTENTION requires matching BSND query/key/value tensors.")
        if (
            q.shape[-1] != k.shape[-1]
            or k.shape[2] <= 0
            or q.shape[2] % k.shape[2]
            or q.device != k.device
            or q.device != v.device
            or q.dtype != k.dtype
            or q.dtype != v.dtype
        ):
            raise ValueError("FLEX_ATTENTION requires matching heads, dtype and device.")
        for visibility, tensor in ((query_mask, q), (key_mask, k)):
            if visibility is not None and (
                visibility.shape != tensor.shape[:2] or visibility.dtype != torch.bool or visibility.device != q.device
            ):
                raise ValueError("FLEX_ATTENTION requires Boolean visibility masks matching Q/K.")
        if q.device.type not in ("cuda", "cpu"):
            raise NotImplementedError("FLEX_ATTENTION currently supports CPU and CUDA tensors")
        if q.device.type == "cuda" and q.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("FLEX_ATTENTION CUDA execution requires FP16 or BF16")
        if torch.is_grad_enabled():
            raise RuntimeError("FLEX_ATTENTION requires no-grad inference.")
        block_mask = None if attn_metadata is None else attn_metadata.extra.get("block_mask")
        if block_mask is not None:
            if (
                not isinstance(block_mask, BlockMask)
                or ranges is not None
                or query_mask is not None
                or key_mask is not None
            ):
                raise ValueError("Prebuilt Flex BlockMask cannot be combined with rectangle/token masks")
            kernel = compiled_flex() if q.device.type == "cuda" else flex_attention
            return kernel(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                block_mask=block_mask,
                scale=self.softmax_scale,
                enable_gqa=q.shape[2] != k.shape[2],
            ).transpose(1, 2)
        output = torch.empty_like(q)
        for batch in range(q.shape[0]):
            qr = np.ones(q.shape[1], dtype=bool) if query_mask is None else query_mask[batch].detach().cpu().numpy()
            kr = np.ones(k.shape[1], dtype=bool) if key_mask is None else key_mask[batch].detach().cpu().numpy()
            rectangles = [(0, q.shape[1], 0, k.shape[1])] if ranges is None else ranges[batch]
            if not q.shape[1] or not k.shape[1]:
                output[batch].zero_()
                continue
            cache_key = (q.shape[1], k.shape[1], str(q.device), tuple(rectangles), qr.tobytes(), kr.tobytes())
            if cache_key not in self.masks:
                self.masks[cache_key] = build_range_mask(q.shape[1], k.shape[1], rectangles, q.device, qr, kr)
                while len(self.masks) > 32:
                    self.masks.popitem(last=False)
            self.masks.move_to_end(cache_key)
            mask = self.masks[cache_key]
            kernel = compiled_flex() if q.device.type == "cuda" else flex_attention
            result = kernel(
                q[batch].transpose(0, 1)[None],
                k[batch].transpose(0, 1)[None],
                v[batch].transpose(0, 1)[None],
                block_mask=mask,
                scale=self.softmax_scale,
                enable_gqa=q.shape[2] != k.shape[2],
                kernel_options={"ROWS_GUARANTEED_SAFE": False, "BLOCKS_ARE_CONTIGUOUS": False},
            )
            output[batch].copy_(result[0].transpose(0, 1))
        return output


class FlexAttentionBackend(AttentionBackend):
    supports_range_mask = True
    supported_platforms = ("cuda", "cpu")

    @classmethod
    def supports_attention_mask(cls, attention_spec=None):
        return True

    @classmethod
    def supports_cross_head_geometry(cls, q_heads, kv_heads):
        return kv_heads > 0 and q_heads % kv_heads == 0

    @staticmethod
    def get_name():
        return "FLEX_ATTENTION"

    @staticmethod
    def get_impl_cls():
        return FlexAttentionImpl

    @staticmethod
    def get_supported_head_sizes():
        return [8, 16, 32, 64, 128, 256]
