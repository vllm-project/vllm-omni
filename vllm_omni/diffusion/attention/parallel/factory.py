# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from vllm.logger import init_logger

from vllm_omni.diffusion.attention.parallel.allgather_kv import (
    AllGatherKVParallelAttention,
)
from vllm_omni.diffusion.attention.parallel.base import (
    NoParallelAttention,
    ParallelAttentionStrategy,
)
from vllm_omni.diffusion.attention.parallel.ring import RingParallelAttention
from vllm_omni.diffusion.attention.parallel.ulysses import UlyssesParallelAttention
from vllm_omni.diffusion.distributed.parallel_state import (
    get_sequence_parallel_world_size,
    get_sp_group,
)
from vllm_omni.diffusion.forward_context import get_forward_context, is_forward_context_available

logger = init_logger(__name__)


def build_parallel_attention_strategy(
    *,
    scatter_idx: int,
    gather_idx: int,
    use_sync: bool,
    causal: bool = False,
) -> ParallelAttentionStrategy:
    """Select a parallel attention strategy based on current diffusion config.

    Design principle:
    - Attention kernel backend selection remains in `attention/selector.py`.
    - Parallel attention selection is handled here, based on distributed config
      and initialized process groups.
    """
    if not is_forward_context_available():
        return NoParallelAttention()
    cfg = get_forward_context().omni_diffusion_config
    p = cfg.parallel_config

    ulysses_degree = getattr(p, "ulysses_degree", 1)
    ring_degree = getattr(p, "ring_degree", 1)
    allgather_degree = getattr(p, "allgather_degree", 1)
    ulysses_a2a_permute = getattr(p, "ulysses_a2a_permute", False)

    sp_configured = ulysses_degree > 1 or ring_degree > 1 or allgather_degree > 1
    if not sp_configured:
        return NoParallelAttention()

    try:
        sp_group = get_sp_group()
    except Exception as e:
        raise RuntimeError(
            f"SP is configured (ulysses={ulysses_degree}, ring={ring_degree}, "
            f"allgather={allgather_degree}), but the SP group is unavailable."
        ) from e
    if get_sequence_parallel_world_size() <= 1:
        raise RuntimeError(
            f"SP is configured (ulysses={ulysses_degree}, ring={ring_degree}, "
            f"allgather={allgather_degree}), but the initialized SP world size is not greater than one."
        )

    if allgather_degree > 1:
        if causal:
            raise ValueError("AllGather-KV SP only supports non-causal attention.")
        if ulysses_degree > 1 or ring_degree > 1:
            raise ValueError(
                f"AllGather-KV SP is mutually exclusive with Ulysses/Ring in v1 "
                f"(got ulysses_degree={ulysses_degree}, ring_degree={ring_degree}, "
                f"allgather_degree={allgather_degree})."
            )
        logger.debug(f"Using AllGatherKVParallelAttention (allgather_degree={allgather_degree})")
        return AllGatherKVParallelAttention(sp_group=sp_group)

    # Ulysses (or Hybrid Ulysses+Ring)
    if ulysses_degree > 1:
        logger.debug(f"Using UlyssesParallelAttention (ulysses_degree={ulysses_degree})")
        return UlyssesParallelAttention(
            sp_group=sp_group,
            scatter_idx=scatter_idx,
            gather_idx=gather_idx,
            use_sync=use_sync,
            ulysses_a2a_permute=ulysses_a2a_permute,
        )

    # Pure Ring Attention
    if ring_degree > 1:
        logger.debug(f"Using RingParallelAttention (ring_degree={ring_degree})")
        return RingParallelAttention(
            sp_group=sp_group,
        )

    return NoParallelAttention()
