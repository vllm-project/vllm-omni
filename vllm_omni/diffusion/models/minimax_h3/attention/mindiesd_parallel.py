# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MiniMax-H3 admission and ownership for MindIE-SD distributed attention.

The SDK owns both A2A directions and attention. It is a complete executor,
not a scatter/gather transport. Declined calls have submitted no collectives.
"""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from vllm.logger import init_logger

from vllm_omni.diffusion.data import MindIEParallelConfig
from vllm_omni.diffusion.forward_context import get_ulysses_mode
from vllm_omni.platforms import current_omni_platform

if TYPE_CHECKING:
    from mindiesd.parallel import ParallelState

    from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
    from vllm_omni.diffusion.attention.layer import Attention

logger = init_logger(__name__)


class MindIEParallelResource:
    """Own a caller-supplied SDK state on one live communicator/device.

    The pinned SDK has no close method. Omni drains work and drops its state
    reference at coordinator teardown; cyclic SDK chains are collected there.
    SDK internals and its process-group ownership are never modified.
    """

    def __init__(self, group: dist.ProcessGroup, device: torch.device, config: MindIEParallelConfig):
        from mindiesd.parallel import ParallelState

        self.group = group
        self.device = device
        self.config = config
        self.state: ParallelState | None = ParallelState(
            group=group,
            overlap_comm=config.overlap_comm,
            optimize_attention_pipeline=True,
            bsa_backend="cann",
        )
        self.failed = False
        self.active = False
        self.completed_calls = 0

    def execute(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        spans: list,
        *,
        used_len: int,
        scale: float,
        sparsity: float,
        pre_rearranged: bool,
    ) -> torch.Tensor:
        if self.state is None or self.failed:
            raise RuntimeError("MindIE parallel resource is closed or failed; recreate the worker")
        if self.active:
            raise RuntimeError("concurrent calls cannot share a MindIE parallel resource")
        from mindiesd.parallel import distributed_sparse_attention

        self.active = True
        try:
            output = distributed_sparse_attention(
                q,
                k,
                v,
                spans,
                group=self.group,
                state=self.state,
                sparse_type="rf_v3",
                used_len=used_len,
                precision="bf16",
                scale=scale,
                sparsity=sparsity,
                pre_rearranged=pre_rearranged,
            )
            self.completed_calls += 1
            return output
        except Exception:
            # No alternate transport or replay after a partial collective.
            self.failed = True
            raise
        finally:
            self.active = False

    def close(self) -> None:
        if self.state is None:
            return
        if self.active:
            raise RuntimeError("cannot close active MindIE parallel resource")
        # Every worker owns one current device. Drain both SDK streams before
        # releasing their buffers; failure leaves state and storage intact.
        with torch.npu.device(self.device):
            current_omni_platform.synchronize()
        logger.info("MiniMax-H3 MindIE-SD pipeline completed_calls=%d", self.completed_calls)
        self.state = None
        gc.collect()


def supports_mindiesd_ulysses(strategy) -> bool:
    """One topology contract for request reorder and per-layer SDK admission."""
    return (
        strategy.name == "ulysses"
        and get_ulysses_mode(default="strict") == "strict"
        and strategy._ulysses_pg is not None
        and strategy._sp_group.ring_world_size == 1
        and strategy._sp_group.allgather_world_size == 1
        and strategy._scatter_idx == 2
        and strategy._gather_idx == 1
        and not strategy._ulysses_a2a_permute
    )


def try_distributed_sparse_attention(
    attention: Attention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    metadata: AttentionMetadata,
) -> torch.Tensor | None:
    """Run the full executor, or decline before state allocation/communication."""
    impl, backend, _ = attention.effective_attention()
    if not current_omni_platform.is_npu() or backend.get_name() != "RAINFUSION_ATTN":
        return None
    config = impl.rainfusion.parallel
    if config is None:
        return None
    if impl.rainfusion.precision != "bf16" or impl.rainfusion.sparse_type != "rf_v3":
        raise ValueError("MindIE parallel requires block_sparse precision='bf16' and sparse_type='rf_v3'")
    if (
        attention.causal
        or attention.qkv_layout != "BSND"
        or attention._has_custom_attention
        or attention.is_paged_kv_active()
        or attention.paged_kv_cache_role is not None
        or query.ndim != 4
        or query.shape[0] != 1
        or query.shape != key.shape
        or query.shape != value.shape
        or any(t.dtype != torch.bfloat16 or t.device != query.device for t in (query, key, value))
        or metadata.full_attn_spans is not None
        or metadata.attn_mask is not None
    ):
        return None
    if any(
        t is not None
        for t in (metadata.joint_query, metadata.joint_key, metadata.joint_value, metadata.joint_attn_mask)
    ):
        return None
    if any(
        metadata.extra.get(name) is not None
        for name in (
            "minimax_h3_reference_kv_post_parallel",
            "minimax_h3_compact_reference_rows",
            "gate_compress",
            "vdn_window",
        )
    ):
        return None
    strategy = attention._get_active_parallel_strategy()
    if not supports_mindiesd_ulysses(strategy) or not dist.is_initialized():
        return None
    coordinator = strategy._sp_group
    group = strategy._ulysses_pg
    world = dist.get_world_size(group)
    if world <= 1 or query.shape[2] % world:
        return None
    effective_metadata = attention._with_kv_cache_dtype(metadata)
    plan = impl._resolve_plan(effective_metadata)
    if plan is None or not 0 < plan.used_len <= world * query.shape[1]:
        return None
    precision = effective_metadata.extra.get("kv_cache_dtype", impl.rainfusion.precision)
    if precision not in ("float", "bf16"):
        return None
    pre_rearranged = bool(getattr(metadata.video_layout, "pre_rearranged", False))
    spans = plan.video_spans or [{"start": plan.prefix_len, "latent_shape": list(plan.latent_shape)}]
    resource = coordinator.resources.get_or_create(
        ("mindiesd", group, query.device),
        lambda: MindIEParallelResource(group, query.device, config),
    )
    if resource.config.overlap_comm != config.overlap_comm:
        raise RuntimeError("MindIE parallel configuration must agree across layers sharing a communicator/device")
    return resource.execute(
        query,
        key,
        value,
        spans,
        used_len=plan.used_len,
        scale=impl.softmax_scale,
        sparsity=impl.rainfusion.sparsity,
        pre_rearranged=pre_rearranged,
    )
