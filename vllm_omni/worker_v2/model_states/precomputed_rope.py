# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MRv2 position lookup for models that precompute a spatial decode grid."""

from typing import cast

import torch
from vllm.model_executor.models.interfaces import SupportsMRoPE
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.buffer_utils import UvaBackedTensor
from vllm.v1.worker.gpu.mm.rope import RopeState


class PrecomputedDecodeRopeState(RopeState):
    def __init__(
        self, num_dims: int, max_num_reqs: int, max_num_tokens: int, max_model_len: int, device: torch.device
    ) -> None:
        super().__init__(num_dims, max_num_reqs, max_num_tokens, max_model_len, device)
        self.table_lens = UvaBackedTensor(max_num_reqs, dtype=torch.int32)

    def init_prefill_positions(
        self, req_idx: int, model: torch.nn.Module, prefill_token_ids: list[int], mm_features: list
    ) -> None:
        positions, delta = cast(SupportsMRoPE, model).get_mrope_input_positions(prefill_token_ids, mm_features)
        positions = positions[:, : self.max_model_len].cpu().numpy()
        self.table_lens.np[req_idx] = positions.shape[1]
        self.prefill_delta.np[req_idx] = delta
        for dim, row in enumerate(positions):
            self.prefill_positions.stage_write(self.num_dims * req_idx + dim, 0, row)

    def apply_staged_writes(self) -> None:
        super().apply_staged_writes()
        self.table_lens.copy_to_uva()

    def prepare_positions(
        self,
        idx_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
        prefill_lens: torch.Tensor,
        num_computed_tokens: torch.Tensor,
    ) -> None:
        # The lookup boundary is the end of the model's table, independent
        # of the scheduler's prefill/decode boundary.
        _gather_positions[(idx_mapping.shape[0],)](
            self.positions,
            self.positions.stride(0),
            self.prefill_positions.gpu,
            self.max_model_len,
            self.table_lens.gpu,
            self.prefill_delta.gpu,
            idx_mapping,
            query_start_loc,
            num_computed_tokens,
            self.num_dims,
            256,
        )


@triton.jit
def _gather_positions(
    out,
    out_stride,
    table,
    row_stride,
    lengths,
    deltas,
    slots,
    starts,
    computed,
    num_dims: tl.constexpr,
    block_size: tl.constexpr,
):
    batch = tl.program_id(0)
    slot = tl.load(slots + batch).to(tl.int64)
    start = tl.load(starts + batch)
    size = tl.load(starts + batch + 1) - start
    offset = tl.load(computed + slot)
    length = tl.load(lengths + slot)
    delta = tl.load(deltas + slot)
    for block in range(0, size, block_size):
        i = block + tl.arange(0, block_size)
        pos = offset + i
        for dim in tl.static_range(num_dims):
            addr = table + (slot * num_dims + dim) * row_stride + pos
            # A multi-token query can straddle the final valid grid column.
            value = tl.load(addr, mask=(i < size) & (pos < length), other=pos + delta)
            tl.store(out + dim * out_stride + start + i, value, mask=i < size)
