# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from vllm.logger import init_logger

from vllm_omni.diffusion.attention.parallel.base import NoParallelAttention, ParallelAttentionStrategy
from vllm_omni.diffusion.attention.parallel.ring import RingParallelAttention
from vllm_omni.diffusion.attention.parallel.ulysses import UlyssesParallelAttention
from vllm_omni.diffusion.distributed.parallel_state import get_sequence_parallel_world_size, get_sp_group
from vllm_omni.diffusion.forward_context import get_forward_context

logger = init_logger(__name__)


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
        cfg = get_forward_context().omni_diffusion_config
        p = cfg.parallel_config
    except Exception as e:
        logger.debug(f"No forward context available for parallel attention strategy: {e}")
        return NoParallelAttention()

    ulysses_degree = getattr(p, "ulysses_degree", 1)
    ring_degree = getattr(p, "ring_degree", 1)

    try:
        sp_group = get_sp_group()
        # Ensure SP group is initialized and world size > 1
        if get_sequence_parallel_world_size() <= 1:
            return NoParallelAttention()
    except Exception as e:
        # Log warning if SP is configured but group is not available
        if ulysses_degree > 1 or ring_degree > 1:
            logger.warning(
                f"SP configured (ulysses={ulysses_degree}, ring={ring_degree}) but SP group not available: {e}. "
                f"Falling back to NoParallelAttention. This may cause incorrect results."
            )
        return NoParallelAttention()

    # Ulysses (or Hybrid Ulysses+Ring)
    if ulysses_degree > 1:
        logger.debug(f"Using UlyssesParallelAttention (ulysses_degree={ulysses_degree})")
        return UlyssesParallelAttention(
            sp_group=sp_group,
            scatter_idx=scatter_idx,
            gather_idx=gather_idx,
            use_sync=use_sync,
        )

    # Pure Ring Attention
    if ring_degree > 1:
        logger.debug(f"Using RingParallelAttention (ring_degree={ring_degree})")
        return RingParallelAttention(
            sp_group=sp_group,
        )

    return NoParallelAttention()
