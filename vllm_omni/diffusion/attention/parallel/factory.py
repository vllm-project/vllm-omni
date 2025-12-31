# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from vllm_omni.diffusion.attention.parallel.base import NoParallelAttention, ParallelAttentionStrategy
from vllm_omni.diffusion.attention.parallel.ring import RingParallelAttention
from vllm_omni.diffusion.attention.parallel.ulysses import UlyssesParallelAttention
from vllm_omni.diffusion.data import get_current_omni_diffusion_config
from vllm_omni.diffusion.distributed.parallel_state import get_sequence_parallel_world_size, get_sp_group


def build_parallel_attention_strategy(
    *,
    scatter_idx: int,
    gather_idx: int,
    use_sync: bool,
) -> ParallelAttentionStrategy:
    """Select a parallel attention strategy based on current diffusion config.

    Design principle:
    - Attention kernel backend selection remains in `attention/selector.py`.
    - Parallel attention selection is handled here, based on distributed config
      and initialized process groups.
    """
    try:
        cfg = get_current_omni_diffusion_config()
        p = cfg.parallel_config
    except Exception:
        return NoParallelAttention()

    ulysses_degree = getattr(p, "ulysses_degree", 1)
    ring_degree = getattr(p, "ring_degree", 1)

    try:
        sp_group = get_sp_group()
        # Ensure SP group is initialized and world size > 1
        if get_sequence_parallel_world_size() <= 1:
            return NoParallelAttention()
    except Exception:
        return NoParallelAttention()

    # Ulysses (or Hybrid Ulysses+Ring)
    if ulysses_degree > 1:
        return UlyssesParallelAttention(
            sp_group=sp_group,
            scatter_idx=scatter_idx,
            gather_idx=gather_idx,
            use_sync=use_sync,
        )

    # Pure Ring Attention
    if ring_degree > 1:
        return RingParallelAttention(
            sp_group=sp_group,
        )

    return NoParallelAttention()
