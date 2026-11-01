# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Copy independently allocated paged KV layers into a model-owned snapshot."""

from collections.abc import Sequence

import torch
from vllm.triton_utils import HAS_TRITON, tl, triton

if HAS_TRITON:

    @triton.jit
    def _copy_paged_kv_kernel(
        sources,
        block_table,
        dst_keys,
        dst_values,
        request_row,
        length,
        table_stride: tl.constexpr,
        page_size: tl.constexpr,
        heads: tl.constexpr,
        head_dim: tl.constexpr,
        dst_k_strides: tl.constexpr,
        dst_v_strides: tl.constexpr,
        tile: tl.constexpr,
    ):
        layer = tl.program_id(0)
        which = tl.program_id(1)
        offsets = tl.program_id(2) * tile + tl.arange(0, tile)
        dim = offsets % head_dim
        head = offsets // head_dim % heads
        token = offsets // (heads * head_dim)
        mask = token < length
        block = tl.load(block_table + request_row * table_stride + token // page_size, mask=mask, other=0)
        source_addr = tl.load(sources + layer * 10 + which)
        strides = sources + layer * 10 + 2 + which * 4
        s_block = tl.load(strides)
        s_token = tl.load(strides + 1)
        s_head = tl.load(strides + 2)
        s_dim = tl.load(strides + 3)
        source = tl.cast(source_addr, tl.pointer_type(dst_keys.dtype.element_ty))
        src_offset = block * s_block + token % page_size * s_token + head * s_head + dim * s_dim
        data = tl.load(source + src_offset, mask=mask, other=0)
        if which == 0:
            dst_offset = (
                layer * dst_k_strides[0] + token * dst_k_strides[1] + head * dst_k_strides[2] + dim * dst_k_strides[3]
            )
            tl.store(dst_keys + dst_offset, data, mask=mask)
        else:
            dst_offset = (
                layer * dst_v_strides[0] + token * dst_v_strides[1] + head * dst_v_strides[2] + dim * dst_v_strides[3]
            )
            tl.store(dst_values + dst_offset, data, mask=mask)


class PagedKVCopy:
    """Cache source addresses/strides once; read the live GPU page table per copy.

    Sources are normalized [pages, page_size, heads, head_dim] views. Keeping
    the views alive also keeps their allocations valid. Destination strides
    allow a reserved NAR suffix without an intermediate contiguous tensor.
    """

    def __init__(self, pairs: Sequence[tuple[torch.Tensor, torch.Tensor]]) -> None:
        if not pairs:
            raise ValueError("KV snapshot requires at least one source layer")
        self.pairs = tuple(pairs)
        self.shape = pairs[0][0].shape
        self.device = pairs[0][0].device
        self.dtype = pairs[0][0].dtype
        for pair in pairs:
            for tensor in pair:
                if tensor.ndim != 4 or tensor.shape != self.shape:
                    raise ValueError("KV snapshot source layers must have matching four-dimensional shapes")
                if tensor.device != self.device or tensor.dtype != self.dtype:
                    raise ValueError("KV snapshot source layers must have matching dtype/device")
        # This upload happens once per source allocation, not per request.
        metadata = [[k.data_ptr(), v.data_ptr(), *k.stride(), *v.stride()] for k, v in pairs]
        self._host_sources = torch.tensor(metadata, dtype=torch.int64, pin_memory=self.device.type == "cuda")
        self.sources = self._host_sources.to(self.device, non_blocking=True)

    def copy(
        self,
        block_table: torch.Tensor,
        request_row: int,
        length: int,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> bool:
        if not HAS_TRITON or self.device.type != "cuda":
            return False
        layers, capacity, heads, head_dim = keys.shape
        if (
            values.shape != keys.shape
            or layers != len(self.pairs)
            or length < 1
            or length > capacity
            or (heads, head_dim) != tuple(self.shape[2:])
            or keys.dtype != self.dtype
            or values.dtype != self.dtype
            or keys.device != self.device
            or values.device != self.device
            or block_table.device != self.device
            or block_table.dtype != torch.int32
            or block_table.ndim != 2
            or block_table.stride(1) != 1
            or not 0 <= request_row < block_table.shape[0]
            or length > block_table.shape[1] * self.shape[1]
        ):
            raise ValueError("KV snapshot destination or block table has incompatible geometry")
        tile = 1024
        _copy_paged_kv_kernel[(layers, 2, triton.cdiv(length * heads * head_dim, tile))](
            self.sources,
            block_table,
            keys,
            values,
            request_row,
            length,
            table_stride=block_table.stride(0),
            page_size=self.shape[1],
            heads=heads,
            head_dim=head_dim,
            dst_k_strides=keys.stride(),
            dst_v_strides=values.stride(),
            tile=tile,
        )
        return True
