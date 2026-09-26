# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""KV-cache group access for the omni prefix cache.

The sole path through which the prefix cache touches vLLM scheduler
internals (block tables, slot mappings). Group-spec rejection happens
at kv-cache init via ``check_prefix_cache_kv_groups``; the factory only
returns None when the input batch has no block table.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from vllm_omni.core.prefix_cache.interface import OmniPrefixCacheUnmatchError, PrefixCacheConfig

if TYPE_CHECKING:
    from vllm.config.cache import CacheConfig
    from vllm.config.kv_transfer import KVTransferConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec
    from vllm.v1.worker.block_table import BlockTable
    from vllm.v1.worker.gpu_input_batch import InputBatch


class FullAttentionGroupView:
    """View over the dense full-attention KV-cache group.

    Step slots come from the CPU block table (`step_slots_cpu`), not the
    device slot_mapping.
    """

    def __init__(self, input_batch: InputBatch, block_size: int, group_id: int = 0, dcp_world_size: int = 1):
        self._input_batch = input_batch
        self.block_size = block_size
        self.group_id = group_id
        table = input_batch.block_table[group_id]
        check_prefix_cache_block_layout(table, block_size, dcp_world_size)
        self.kernel_block_size = table.block_size
        self.dcp_world_size = dcp_world_size
        self.blocks_per_kv_block = table.blocks_per_kv_block

    def _block_table_cpu(self) -> torch.Tensor:
        return self._input_batch.block_table[self.group_id].block_table.cpu

    def batch_req_ids(self) -> list[str]:
        return list(self._input_batch.req_ids)

    def step_slots_cpu(self, req_ids: list[str], num_scheduled: dict[str, int]) -> torch.Tensor:
        """This step's slot mapping, computed on CPU from the block table.

        The device slot_mapping would need a stream sync to read back, which
        stalls the whole forward; the CPU block table carries the same
        information (positions are num_computed .. +num_scheduled per request).
        """
        block_table = self._block_table_cpu()
        # BlockTable expands allocator IDs into kernel-block IDs. Both
        # geometries address the same flat token storage, including tails.
        bs = self.block_size if self.dcp_world_size > 1 else self.kernel_block_size
        split = self.blocks_per_kv_block if self.dcp_world_size > 1 else 1
        max_blocks = int(block_table.shape[1]) // split
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
            block_ids = block_table[req_idx, offs * split].to(torch.long)
            if self.dcp_world_size > 1:
                block_ids //= split
            parts.append(block_ids * bs + (pos % bs))
        return torch.cat(parts) if parts else torch.empty((0,), dtype=torch.long)


def check_prefix_cache_block_layout(block_table: BlockTable, block_size: int, dcp_world_size: int = 1) -> None:
    """Reject block-table layouts ``step_slots_cpu`` cannot mirror.

    DCP stripes physical KV slots across ranks. Stage outputs instead use
    the allocator ID and its full virtual token span on every rank.
    """
    if block_table.dcp_world_size != dcp_world_size:
        raise OmniPrefixCacheUnmatchError(
            "omni prefix caching DCP world size differs from the block table "
            f"({dcp_world_size} != {block_table.dcp_world_size})"
        )
    kv_bs = block_table.kv_cache_block_size
    if int(kv_bs) * dcp_world_size != int(block_size):
        raise OmniPrefixCacheUnmatchError(
            f"omni prefix caching output block_size {block_size} does not match "
            f"the DCP KV block span ({kv_bs} * {dcp_world_size})"
        )


def check_prefix_cache_kv_groups(kv_cache_groups: Sequence[KVCacheGroupSpec] | None, dcp_world_size: int = 1) -> int:
    """Pick a non-recycling full-attention group for output storage.

    Only needs ``kv_cache_config.kv_cache_groups``. ``FullAttentionSpec``
    is imported here so ``tests/core`` can load this module without vllm.
    """
    from vllm.v1.kv_cache_interface import FullAttentionSpec, RSWASpec, SlidingWindowSpec

    groups = tuple(kv_cache_groups or ())
    specs = tuple(group.kv_cache_spec for group in groups)
    if not specs or any(
        not isinstance(spec, (FullAttentionSpec, SlidingWindowSpec)) or isinstance(spec, RSWASpec) for spec in specs
    ):
        raise OmniPrefixCacheUnmatchError(
            "omni prefix caching supports full-attention and sliding-window KV groups only; "
            f"found {[type(spec).__name__ for spec in specs]}. disable enable_prefix_caching for this model"
        )
    if dcp_world_size > 1 and any(isinstance(spec, SlidingWindowSpec) for spec in specs):
        raise OmniPrefixCacheUnmatchError("vLLM does not support DCP with sliding-window KV groups")
    for group_id, spec in enumerate(specs):
        if isinstance(spec, FullAttentionSpec):
            return group_id
    raise OmniPrefixCacheUnmatchError(
        "omni prefix caching requires a full-attention KV group for output storage; sliding-window blocks are recycled"
    )


def check_prefix_cache_kv_transfer(kv_transfer_config: KVTransferConfig | None) -> None:
    """Reject kv_consumer / kv_both stages.

    KV loaded from a producer also shows up as ``num_computed_tokens``; the
    manager would read it as a local hit with no rows behind it.
    """
    if kv_transfer_config is not None and kv_transfer_config.is_kv_consumer:
        raise OmniPrefixCacheUnmatchError(
            "omni prefix caching cannot tell locally cached tokens from KV received "
            "through a KV connector; disable enable_prefix_caching on kv_consumer / "
            "kv_both stages"
        )


def check_prefix_cache_token_accounting(cache_config: CacheConfig, speculative_config: object) -> None:
    """Reject configs where ``num_computed_tokens`` stops meaning "first
    scheduled position, block aligned".

    Speculative decoding: with async scheduling vLLM keeps
    ``num_computed_tokens_cpu`` optimistic (all drafts accepted) during the
    forward and corrects it after; ``step_slots_cpu`` would read the
    uncorrected value and mirror rows at the wrong slots. Refused as a
    whole until that path is verified. vLLM's single-group coordinator
    requires the prefix hash unit to equal the allocator block size.
    """
    if speculative_config is not None:
        raise OmniPrefixCacheUnmatchError(
            "omni prefix caching is not supported together with speculative decoding; "
            "disable enable_prefix_caching on this stage"
        )
    unit = cache_config.prefix_match_unit
    block_size = cache_config.block_size
    if unit is not None and block_size is not None and int(unit) != int(block_size):
        raise OmniPrefixCacheUnmatchError(
            f"omni prefix caching requires block-aligned hits; prefix_match_unit={unit} with block_size={block_size} "
            "enables sub-block hits. Unset prefix_match_unit or disable enable_prefix_caching"
        )


def stage_prefix_cache_config(
    *,
    kv_cache_config: KVCacheConfig,
    cache_config: CacheConfig,
    kv_transfer_config: KVTransferConfig | None,
    scheduler_config: object,
    model_config: object,
    is_pooling_model: bool,
    speculative_config: object = None,
    dcp_world_size: int = 1,
    allow_dcp: bool = True,
) -> PrefixCacheConfig | None:
    """Runner-side gate shared by the GPU and NPU model runners.

    Returns None when the stage does not run an omni prefix cache
    (``enable_prefix_caching`` off, or a pooling stage that never saves).
    Otherwise refuses kv_consumer / kv_both, speculative decoding,
    sub-block matching and unsupported kv groups, then sizes the config from
    the scheduler. One place so a platform runner cannot silently skip a
    refusal the other one has.
    """
    if not cache_config.enable_prefix_caching or is_pooling_model:
        return None
    if dcp_world_size > 1 and not allow_dcp:
        raise OmniPrefixCacheUnmatchError("omni prefix caching DCP output storage is supported on CUDA only")
    check_prefix_cache_kv_transfer(kv_transfer_config)
    check_prefix_cache_token_accounting(cache_config, speculative_config)
    output_group_id = check_prefix_cache_kv_groups(kv_cache_config.kv_cache_groups, dcp_world_size)
    output_spec = kv_cache_config.kv_cache_groups[output_group_id].kv_cache_spec
    return PrefixCacheConfig.from_vllm_config(
        num_blocks=kv_cache_config.num_blocks,
        block_size=output_spec.block_size * dcp_world_size,
        output_group_id=output_group_id,
        dcp_world_size=dcp_world_size,
        scheduler_config=scheduler_config,
        model_config=model_config,
    )


def get_prefix_cache_group_view(
    input_batch: InputBatch,
    block_size: int,
    kv_cache_groups: Sequence[KVCacheGroupSpec] | None = None,
    dcp_world_size: int = 1,
) -> FullAttentionGroupView | None:
    """Build the group view; None only if the batch has no block table.

    Group spec is checked first (raises). Sliding-window groups can recycle
    blocks, so output rows follow a full-attention group even when it is not
    group 0.
    """
    group_id = check_prefix_cache_kv_groups(kv_cache_groups, dcp_world_size)
    block_tables = input_batch.block_table.block_tables
    if not block_tables:
        return None
    return FullAttentionGroupView(input_batch, block_size, group_id, dcp_world_size)
