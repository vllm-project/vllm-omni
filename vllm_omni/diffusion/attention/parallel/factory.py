# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from vllm.logger import init_logger

from vllm_omni.diffusion.attention.capabilities import CrossAttentionParallelLayout
from vllm_omni.diffusion.attention.parallel.allgather_kv import (
    AllGatherKVParallelAttention,
)
from vllm_omni.diffusion.attention.parallel.base import (
    NoParallelAttention,
    ParallelAttentionStrategy,
)
from vllm_omni.diffusion.attention.parallel.cross import ReplicatedQShardedKVParallelAttention
from vllm_omni.diffusion.attention.parallel.ring import RingParallelAttention
from vllm_omni.diffusion.attention.parallel.ulysses import UlyssesParallelAttention
from vllm_omni.diffusion.attention.parallel.ulysses_allgather import (
    UlyssesAllGatherKVParallelAttention,
)
from vllm_omni.diffusion.distributed.parallel_state import (
    get_sequence_parallel_world_size,
    get_sp_group,
)
from vllm_omni.diffusion.distributed.ulysses_transport import resolve_ulysses_transports
from vllm_omni.diffusion.forward_context import get_forward_context, is_forward_context_available

logger = init_logger(__name__)


def build_parallel_attention_strategy(
    *,
    scatter_idx: int,
    gather_idx: int,
    use_sync: bool,
    causal: bool = False,
    cross_parallel_layout: CrossAttentionParallelLayout | None = None,
    num_heads: int | None = None,
    attention_backend: str | None = None,
    attention_impl: str | None = None,
) -> ParallelAttentionStrategy:
    """Select a parallel attention strategy based on current diffusion config.

    Design principle:
    - Attention kernel backend selection remains in `attention/selector.py`.
    - Parallel attention selection is handled here, based on distributed config
      and initialized process groups.
    """
    if cross_parallel_layout is CrossAttentionParallelLayout.LOCAL_Q_REPLICATED_KV:
        return NoParallelAttention()
    if not is_forward_context_available():
        return NoParallelAttention()
    cfg = get_forward_context().omni_diffusion_config
    if cfg is None:
        return NoParallelAttention()
    p = cfg.parallel_config

    ulysses_degree = getattr(p, "ulysses_degree", 1)
    ring_degree = getattr(p, "ring_degree", 1)
    allgather_degree = getattr(p, "allgather_degree", 1)
    ulysses_overlap_chunks = int(getattr(p, "ulysses_overlap_chunks", 0) or 0)
    ulysses_a2a_backend = getattr(p, "ulysses_a2a_backend", "auto")
    if getattr(p, "ulysses_a2a_permute", None) and ulysses_a2a_backend == "auto":  # deprecated alias
        ulysses_a2a_backend = "copy_engine_overlap" if ulysses_overlap_chunks else "copy_engine"
    if ulysses_a2a_backend == "auto" and getattr(cfg, "diffusion_use_graph", False):
        # Symmetric-memory workspaces are bound to one stream and the overlap's flags to host epochs, so neither
        # survives CUDA-graph capture and replay; graphed denoising keeps NCCL.
        ulysses_a2a_backend = "nccl"
    if ulysses_a2a_backend not in ("auto", "nccl"):
        from vllm_omni.diffusion.distributed.a2a_permute import validate_a2a_permute_config

        validate_a2a_permute_config(
            enforce_eager=getattr(cfg, "enforce_eager", False),
            ulysses_mode=getattr(p, "ulysses_mode", "strict"),
            ulysses_degree=ulysses_degree,
            ring_degree=ring_degree,
            allgather_degree=allgather_degree,
            scatter_idx=scatter_idx,
            gather_idx=gather_idx,
        )

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

    if cross_parallel_layout is not None:
        return ReplicatedQShardedKVParallelAttention(sp_group)

    # AllGather-KV, optionally composed with Ulysses as an orthogonal 2D
    # sequence-parallel topology. K/V are gathered over the AllGather group
    # *after* any Ulysses reshard, so the attention backend always runs a
    # local-Q/global-KV problem.
    if allgather_degree > 1:
        if causal:
            raise ValueError("AllGather-KV SP only supports non-causal attention.")
        if ring_degree > 1:
            raise ValueError(
                "AllGather-KV SP cannot be composed with Ring "
                f"(got ulysses_degree={ulysses_degree}, ring_degree={ring_degree}, "
                f"allgather_degree={allgather_degree})."
            )
        if ulysses_degree > 1:
            logger.debug(
                f"Using UlyssesAllGatherKVParallelAttention "
                f"(ulysses_degree={ulysses_degree}, allgather_degree={allgather_degree})"
            )
            return UlyssesAllGatherKVParallelAttention(
                sp_group=sp_group,
                scatter_idx=scatter_idx,
                gather_idx=gather_idx,
                use_sync=use_sync,
                # Ulysses x AllGather-KV keeps NCCL under "auto".
                ulysses_a2a_permute=ulysses_a2a_backend == "copy_engine",
            )
        logger.debug(f"Using AllGatherKVParallelAttention (allgather_degree={allgather_degree})")
        return AllGatherKVParallelAttention(sp_group=sp_group)

    # Ulysses (or Hybrid Ulysses+Ring)
    if ulysses_degree > 1:
        logger.debug(f"Using UlyssesParallelAttention (ulysses_degree={ulysses_degree})")
        transports = resolve_ulysses_transports(
            requested=ulysses_a2a_backend,
            group=getattr(sp_group, "ulysses_group", None),
            world_size=getattr(sp_group, "ulysses_world_size", ulysses_degree),
            pure_strict=(
                ring_degree == 1
                and getattr(p, "ulysses_mode", "strict") == "strict"
                and scatter_idx == 2
                and gather_idx == 1
                and not causal
            ),
            num_heads=num_heads,
            attention_backend=attention_backend,
            attention_impl=attention_impl,
            overlap_chunks=ulysses_overlap_chunks,
        )
        return UlyssesParallelAttention(
            sp_group=sp_group,
            scatter_idx=scatter_idx,
            gather_idx=gather_idx,
            use_sync=use_sync,
            transports=transports,
        )

    # Pure Ring Attention
    if ring_degree > 1:
        logger.debug(f"Using RingParallelAttention (ring_degree={ring_degree})")
        return RingParallelAttention(
            sp_group=sp_group,
        )

    return NoParallelAttention()
