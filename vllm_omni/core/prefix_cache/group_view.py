# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""KV-cache group access for the omni prefix cache.

The sole path through which the prefix cache touches vLLM scheduler
internals (block tables, slot mappings). Group-spec rejection happens
at kv-cache init via ``check_prefix_cache_kv_groups``; the factory only
returns None when the input batch has no block table.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm_omni.core.prefix_cache.interface import OmniPrefixCacheUnmatchError

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_input_batch import InputBatch


class FullAttentionGroupView:
    """View over the first (full-attention) KV-cache group.

    Step slots come from the CPU block table (`step_slots_cpu`), not the
    device slot_mapping.
    """

    def __init__(self, input_batch: InputBatch, block_size: int):
        self._input_batch = input_batch
        self.block_size = block_size

    def _block_table_cpu(self) -> torch.Tensor:
        return self._input_batch.block_table[0].block_table.cpu

    def batch_req_ids(self) -> list[str]:
        return list(self._input_batch.req_ids)

    def step_slots_cpu(self, req_ids: list[str], num_scheduled: dict[str, int]) -> torch.Tensor:
        """This step's slot mapping, computed on CPU from the block table.

        The device slot_mapping would need a stream sync to read back, which
        stalls the whole forward; the CPU block table carries the same
        information (positions are num_computed .. +num_scheduled per request).
        """
        block_table = self._block_table_cpu()
        bs = self.block_size
        max_blocks = int(block_table.shape[1])
        computed = self._input_batch.num_computed_tokens_cpu
        parts: list[torch.Tensor] = []
        for req_id in req_ids:
            n = int(num_scheduled.get(req_id, 0))
            if n <= 0:
                continue
            req_idx = self._input_batch.req_id_to_index[req_id]
            start = int(computed[req_idx])
            pos = torch.arange(start, start + n, dtype=torch.long)
            offs = pos // bs
            if int(offs[-1]) >= max_blocks:
                keep = offs < max_blocks
                pos, offs = pos[keep], offs[keep]
                if pos.numel() == 0:
                    continue
            parts.append(block_table[req_idx, offs].to(torch.long) * bs + (pos % bs))
        return torch.cat(parts) if parts else torch.empty((0,), dtype=torch.long)


def check_prefix_cache_kv_groups(kv_cache_groups: object) -> None:
    """Reject hybrid / multi-group models at kv-cache init, not first step.

    Only needs ``kv_cache_config.kv_cache_groups``. ``FullAttentionSpec``
    is imported here so ``tests/core`` can load this module without vllm.
    """
    groups = list(kv_cache_groups or ())
    if len(groups) != 1:
        raise OmniPrefixCacheUnmatchError(
            "omni prefix caching requires a single full-attention kv-cache group; "
            f"found {len(groups)}. disable enable_prefix_caching for this model"
        )
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    spec = getattr(groups[0], "kv_cache_spec", None)
    if not isinstance(spec, FullAttentionSpec):
        raise OmniPrefixCacheUnmatchError(
            "omni prefix caching requires a single full-attention kv-cache group; "
            f"found {type(spec).__name__}. disable enable_prefix_caching for this model"
        )


def get_prefix_cache_group_view(
    input_batch: InputBatch,
    block_size: int,
    kv_cache_groups: object = None,
) -> FullAttentionGroupView | None:
    """Build the group view; None only if the batch has no block table.

    Group spec is checked first (raises). Selection is by spec, not by
    counting block tables: a hybrid model's group 0 is not necessarily
    full attention, and a narrower per-group table would make
    step_slots_cpu silently clamp.
    """
    check_prefix_cache_kv_groups(kv_cache_groups)
    block_tables = getattr(input_batch.block_table, "block_tables", None)
    if not block_tables:
        return None
    return FullAttentionGroupView(input_batch, block_size)
