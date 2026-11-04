# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from collections import OrderedDict
from functools import cache

import numpy as np
import torch
from torch import nn
from torch.nn.attention.flex_attention import BlockMask, flex_attention

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention


@cache
def compiled_flex():
    return torch.compile(flex_attention, dynamic=True, fullgraph=True)


def build_range_mask(q_length, k_length, ranges, device, query_mask, key_mask):
    ranges = np.asarray(ranges, dtype=np.int64).reshape(-1, 4)
    for index, (qs, qe, ks, ke) in enumerate(ranges):
        if not (0 <= qs <= qe <= q_length and 0 <= ks <= ke <= k_length):
            raise ValueError("Hy-Nomas attention range exceeds its sequence.")
        for a, b, c, d in ranges[:index]:
            if max(qs, a) < min(qe, b) and max(ks, c) < min(ke, d):
                raise ValueError("Hy-Nomas attention rectangles must not overlap.")
    edges = np.unique(np.r_[0, ranges[:, :2].ravel(), q_length])
    rows = []
    for start in edges[:-1]:
        intervals = sorted((int(ks), int(ke)) for qs, qe, ks, ke in ranges if qs <= start < qe and ks < ke)
        merged = []
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
    for index, intervals in enumerate(rows):
        row_ids[edges[index] : edges[index + 1]] = index
        for column, (left, right) in enumerate(intervals):
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


class RangeAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.masks = OrderedDict()

    def clear(self):
        self.masks.clear()

    def forward(self, q, k, v, attn_metadata=None):
        extra = {} if attn_metadata is None else attn_metadata.extra
        ranges = extra.get("ranges")
        query_mask, key_mask = extra.get("query_mask"), extra.get("key_mask")
        scale = extra.get("scale")
        if q.ndim != 4 or k.ndim != 4 or k.shape != v.shape or q.shape[0] != k.shape[0]:
            raise ValueError("Hy-Nomas attention requires matching BSND query/key/value tensors.")
        if (
            q.shape[2:] != k.shape[2:]
            or q.device != k.device
            or q.device != v.device
            or q.dtype != k.dtype
            or q.dtype != v.dtype
        ):
            raise ValueError("Hy-Nomas range attention requires matching heads, dtype and device.")
        for mask, tensor in ((query_mask, q), (key_mask, k)):
            if mask is not None and (
                mask.shape != tensor.shape[:2] or mask.dtype != torch.bool or mask.device != q.device
            ):
                raise ValueError("Hy-Nomas range attention requires boolean visibility masks matching Q/K.")
        if q.device.type == "cuda" and (q.dtype != torch.bfloat16 or q.shape[-1] != 128):
            raise ValueError("Hy-Nomas CUDA range attention requires BF16 MHA with head dimension 128.")
        if torch.is_grad_enabled():
            raise RuntimeError("Hy-Nomas range attention requires no-grad inference.")
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
                scale=scale,
                kernel_options={"ROWS_GUARANTEED_SAFE": False, "BLOCKS_ARE_CONTIGUOUS": False},
            )
            output[batch].copy_(result[0].transpose(0, 1))
        return output


class HyNomasAttention(Attention):
    def __init__(self, heads, head_dim, *, role="hy_nomas.cached"):
        super().__init__(
            num_heads=heads,
            head_size=head_dim,
            causal=False,
            softmax_scale=head_dim**-0.5,
            role=role,
            skip_sequence_parallel=True,
            custom_attention=RangeAttention(),
        )

    def forward(self, q, k, v, *, ranges=None, query_mask=None, key_mask=None, scale=None):
        metadata = AttentionMetadata(
            extra={"ranges": ranges, "query_mask": query_mask, "key_mask": key_mask, "scale": scale}
        )
        return super().forward(q, k, v, metadata)

    def clear(self):
        self.attention.clear()


def action_ranges(history_mask, text_mask, image_tokens, offsets, windows, tokens_per_frame):
    result = []
    history_length = history_mask.shape[1]
    for batch in range(text_mask.shape[0]):
        visible_history = history_mask[batch].nonzero().flatten().tolist()
        visible_text = text_mask[batch].nonzero().flatten().tolist()
        positions_q = list(range(image_tokens)) + [image_tokens + index for index in visible_text]
        positions_k = visible_history + [history_length + index for index in positions_q]
        off = offsets[batch] if batch < len(offsets) else None
        if off is None:
            result.append(
                [(0, image_tokens + text_mask.shape[1], 0, history_length + image_tokens + text_mask.shape[1])]
            )
            continue
        off = [min(int(index), len(visible_text)) for index in off]
        rectangles = []

        def emit(start, end, own_action):
            if end <= start:
                return
            key_ranges = [(0, len(visible_history)), (len(visible_history), len(visible_history) + image_tokens)]
            text_start = len(visible_history) + image_tokens
            key_ranges.append((text_start, text_start + off[0]))
            if own_action is not None and 0 <= own_action < len(off) - 1:
                key_ranges.append((text_start + off[own_action], text_start + off[own_action + 1]))
            for left, right in key_ranges:
                if right > left:
                    rectangles.append(
                        (positions_q[start], positions_q[end - 1] + 1, positions_k[left], positions_k[right - 1] + 1)
                    )

        for start, end, action in windows:
            emit(
                max(0, min(int(start) * tokens_per_frame, image_tokens)),
                max(0, min(int(end) * tokens_per_frame, image_tokens)),
                int(action),
            )
        emit(image_tokens, image_tokens + off[0], None)
        for index in range(len(off) - 1):
            emit(image_tokens + off[index], image_tokens + off[index + 1], index)
        result.append(rectangles)
    return result
