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

from vllm_omni.diffusion.data import MindIEParallelConfig, resolve_mindie_sparse_backend
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

    def __init__(
        self,
        group: dist.ProcessGroup,
        device: torch.device,
        config: MindIEParallelConfig,
        *,
        cp_group: dist.ProcessGroup | None = None,
        precision: str = "bf16",
        sparse_type: str = "rf_v3",
        bsa_backend: str | None = None,
    ):
        self.device = device
        self.config = config
        self.precision = precision
        self.sparse_type = sparse_type
        self.bsa_backend = resolve_mindie_sparse_backend(sparse_type, precision, bsa_backend)
        from mindiesd.parallel import ParallelState

        self.state: ParallelState | None = ParallelState(
            group=group,
            cp_group=cp_group,
            overlap_comm=config.overlap_comm,
            optimize_attention_pipeline=config.optimize_attention_pipeline,
            bsa_backend=self.bsa_backend,
            quantized_comm=config.quantized_comm,
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
                state=self.state,
                sparse_type=self.sparse_type,
                used_len=used_len,
                precision=self.precision,
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
    if strategy.name not in ("ulysses", "ulysses_allgather_kv"):
        return False
    ulysses = strategy._ulysses if strategy.name == "ulysses_allgather_kv" else strategy
    coordinator = ulysses._sp_group
    if strategy.name == "ulysses_allgather_kv" and (
        strategy._sp_group is not coordinator or coordinator.ulysses_group is not ulysses._ulysses_pg
    ):
        return False
    return (
        get_ulysses_mode(default="strict") == "strict"
        and ulysses._ulysses_pg is not None
        and coordinator.ring_world_size == 1
        and (
            (strategy.name == "ulysses" and coordinator.allgather_world_size == 1)
            or (
                strategy.name == "ulysses_allgather_kv"
                and coordinator.allgather_world_size in (2, 4)
                and coordinator.ulysses_world_size in (4, 8)
                and coordinator.allgather_group is not None
            )
        )
        and ulysses._scatter_idx == 2
        and ulysses._gather_idx == 1
        and not ulysses._ulysses_a2a_permute
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
    precision = impl.rainfusion.precision
    sparse_type = impl.rainfusion.sparse_type
    backend = resolve_mindie_sparse_backend(sparse_type, precision, getattr(impl.rainfusion, "bsa_backend", None))
    allowed_dtypes = (torch.bfloat16, torch.float16) if precision == "fp16" else (torch.bfloat16,)
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
        or any(
            t.dtype not in allowed_dtypes or t.dtype != query.dtype or t.device != query.device
            for t in (query, key, value)
        )
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
    ulysses = strategy._ulysses if strategy.name == "ulysses_allgather_kv" else strategy
    coordinator = ulysses._sp_group
    group = ulysses._ulysses_pg
    cp_group = coordinator.allgather_group if coordinator.allgather_world_size > 1 else None
    world = dist.get_world_size(group)
    if world <= 1 or query.shape[2] % world:
        return None
    effective_metadata = attention._with_kv_cache_dtype(metadata)
    plan = impl._resolve_plan(effective_metadata)
    cp_size = dist.get_world_size(cp_group) if cp_group is not None else 1
    if plan is None or not 0 < plan.used_len <= world * cp_size * query.shape[1]:
        return None
    cache_precision = effective_metadata.extra.get("kv_cache_dtype", "float")
    if cache_precision not in ("float", "bf16", precision):
        raise ValueError(
            "MindIE parallel precision must be selected by block_sparse.precision without a conflicting KV dtype"
        )
    pre_rearranged = bool(getattr(metadata.video_layout, "pre_rearranged", False))
    if cp_group is not None:
        if not pre_rearranged:
            raise ValueError("CP x Ulysses requires globally pre_rearranged=True input shards")
        if query.shape[-1] != 128 or dist.get_world_size() != world * cp_size or world * cp_size not in (8, 16, 32):
            raise ValueError("MindIE CP x Ulysses requires D128 and one complete 8/16/32-rank sequence world")
    if precision == "mxfp8" and query.shape[-1] != 128:
        raise ValueError("native MXFP8 BSA requires head_dim=128")
    spans = plan.video_spans or [{"start": plan.prefix_len, "latent_shape": list(plan.latent_shape)}]
    resource = coordinator.resources.get_or_create(
        ("mindiesd", group, cp_group, query.device, sparse_type, precision, backend),
        lambda: MindIEParallelResource(
            group,
            query.device,
            config,
            cp_group=cp_group,
            precision=precision,
            sparse_type=sparse_type,
            bsa_backend=backend,
        ),
    )
    if resource.config != config:
        raise RuntimeError("MindIE parallel configuration must agree across layers sharing a communicator/device")
    tensors = (query, key, value)
    if precision == "fp16" and query.dtype != torch.float16:
        # H3's BF16 producer stays unchanged; this explicit mode selects FP16
        # attention operands and restores the producer dtype on return.
        tensors = (query.to(torch.float16), key.to(torch.float16), value.to(torch.float16))
    return resource.execute(
        *tensors,
        spans,
        used_len=plan.used_len,
        scale=impl.softmax_scale,
        sparsity=impl.rainfusion.sparsity,
        pre_rearranged=pre_rearranged,
    ).to(query.dtype)
