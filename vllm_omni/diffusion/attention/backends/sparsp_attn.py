# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in CUDA SparSP: a backend-owned complete sparse Ring execution path."""

from dataclasses import replace
from typing import Any

import torch

from vllm_omni.diffusion.attention.backends.abstract import AttentionBackend, AttentionImpl, AttentionMetadata
from vllm_omni.diffusion.attention.capabilities import (
    AttentionComputation,
    CapabilityResult,
    CompilationMode,
    ExecutionContext,
    ExecutionPathResult,
    ParallelStrategy,
)
from vllm_omni.diffusion.attention.sparsp.config import SparSPSpec
from vllm_omni.diffusion.attention.sparsp.executor import SparseRingExecutor
from vllm_omni.diffusion.attention.sparsp.provider import SpargeProvider
from vllm_omni.diffusion.distributed.parallel_state import get_sp_group
from vllm_omni.diffusion.forward_context import get_forward_context, is_forward_context_available
from vllm_omni.platforms import current_omni_platform


class SparSPAttentionBackend(AttentionBackend):
    supported_platforms = ("cuda",)
    owns_sequence_parallel = True

    @staticmethod
    def get_name() -> str:
        return "SPARSP_ATTN"

    @staticmethod
    def get_impl_cls() -> type["SparSPAttentionImpl"]:
        return SparSPAttentionImpl

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 128]

    @classmethod
    def validate_available(cls) -> None:
        capability = current_omni_platform.get_device_capability()
        if capability is None or capability.major != 8:
            raise ValueError("SPARSP_ATTN's Q128/K64 provider currently requires CUDA SM80/86/89")
        SpargeProvider.validate_available()


class SparSPAttentionImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        qkv_layout: str | None = None,
        backend_kwargs: dict[str, Any] | None = None,
        **extra_impl_args,
    ) -> None:
        del prefix, qkv_layout
        if causal or head_size not in (64, 128) or num_kv_heads not in (None, num_heads):
            raise ValueError("SPARSP_ATTN supports noncausal MHA with head dimension 64 or 128")
        if extra_impl_args.get("computation", AttentionComputation.SELF) is not AttentionComputation.SELF:
            raise ValueError("SPARSP_ATTN is for video self-attention; configure cross attention separately")
        kwargs = backend_kwargs or {}
        if set(kwargs) - {"sparsp"}:
            raise ValueError("SPARSP_ATTN accepts only the typed sparsp configuration block")
        self.config = SparSPSpec(**kwargs.get("sparsp", {}))
        self.num_heads, self.head_size, self.scale = num_heads, head_size, softmax_scale
        self._key = (num_heads, head_size, self.config)

    def resolve_execution_path(
        self,
        context: ExecutionContext,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
    ) -> ExecutionPathResult:
        result = ExecutionPathResult.unmigrated("SPARSP_ATTN", context, path="sparsp_cuda_ring")
        if context.platform != "cuda" or context.parallel_strategy is not ParallelStrategy.RING:
            return replace(result, support=CapabilityResult.unsupported("SPARSP_ATTN requires pure CUDA Ring SP"))
        return replace(result, support=CapabilityResult.supported(), compilation_mode=CompilationMode.EAGER_ONLY)

    def _get_executor(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> SparseRingExecutor:
        if not is_forward_context_available():
            raise ValueError("SPARSP_ATTN requires Omni's eager diffusion forward context and Ring groups")
        ctx = get_forward_context()
        config = ctx.omni_diffusion_config
        if config is None or not config.enforce_eager:
            raise ValueError("SPARSP_ATTN requires enforce_eager=True")
        if config.diffusion_attention_schedule is not None:
            raise ValueError("SPARSP_ATTN does not support attention scheduling")
        parallel = config.parallel_config
        if (
            parallel.ring_degree < 2
            or parallel.ulysses_degree != 1
            or parallel.allgather_degree != 1
            or parallel.tensor_parallel_size != 1
        ):
            raise ValueError("SPARSP_ATTN requires pure Ring SP: ring>=2, ulysses=allgather=TP=1")
        if self.config.routing == "selective_pair_relay" and parallel.ring_degree != 4:
            raise ValueError("SparSP pair relay requires four Ring ranks")
        if query.ndim != 4 or query.shape != key.shape or query.shape != value.shape:
            raise ValueError("SPARSP_ATTN requires equal [B,S,H,D] Q/K/V")
        if (
            query.shape[0] < 1
            or query.shape[1] < 64
            or query.shape[1] % 64
            or query.shape[2:] != (self.num_heads, self.head_size)
        ):
            raise ValueError("SPARSP_ATTN requires a nonempty batch and complete native 64-token KV blocks per rank")
        if query.dtype not in (torch.bfloat16, torch.float16) or key.dtype != query.dtype or value.dtype != query.dtype:
            raise ValueError("SPARSP_ATTN requires matching BF16 or FP16 Q/K/V")
        if torch.is_grad_enabled() and any(t.requires_grad for t in (query, key, value)):
            raise ValueError("SPARSP_ATTN is inference-only and does not implement autograd")
        if any(t.device != query.device or not t.is_cuda for t in (query, key, value)):
            raise ValueError("SPARSP_ATTN requires Q/K/V on one CUDA device")
        if attn_metadata is not None:
            if any(
                getattr(attn_metadata, name) is not None
                for name in (
                    "attn_mask",
                    "joint_attn_mask",
                    "joint_query",
                    "joint_key",
                    "joint_value",
                    "packed_padding",
                    "full_attn_spans",
                    "query_ranges",
                    "cross_query_mask",
                    "video_layout",
                )
            ):
                raise ValueError("SPARSP_ATTN does not support joint, packed, padded or masked attention")
            extra = attn_metadata.extra
            if any(name in extra for name in ("cu_seqlens_q", "cu_seqlens_k", "valid_kv_length")):
                raise ValueError("SPARSP_ATTN requires complete unpadded native blocks")
            if extra.get("kv_cache_dtype") not in (None, "auto", "float"):
                raise ValueError("SPARSP_ATTN does not consume quantized KV caches")
        layout = ctx.sparsp_video_layout
        if self.config.placement == "interval_periodic" and layout is None:
            raise ValueError("SparSP interval placement requires a model adapter for tokens and RoPE")
        if layout is not None and (
            layout.degree != parallel.ring_degree
            or layout.permutation.numel() != query.shape[1] * parallel.ring_degree
            or layout.permutation.device != query.device
        ):
            raise ValueError("SparSP placement does not match the device, Ring degree or complete query grid")
        owner = get_sp_group()
        if owner.ring_world_size != parallel.ring_degree:
            raise ValueError("SparSP configured Ring degree does not match the process-group owner")
        executor = owner.sparsp_executors.get(self._key)
        if executor is None:
            executor = SparseRingExecutor(self.config, owner.ring_group, self.num_heads, self.head_size, query.device)
            owner.sparsp_executors[self._key] = executor
        elif executor.device != query.device:
            raise ValueError("SparSP executor cannot change its owning CUDA device")
        return executor

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        executor = self._get_executor(query, key, value, attn_metadata)
        if query.shape[0] == 1:
            return executor(query, key, value, self.scale)
        # The packet kernels operate on one sample. Keep the model's batching
        # and serialize only this attention op's independent batch elements.
        return torch.cat(
            [
                executor(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), self.scale)
                for q, k, v in zip(query, key, value, strict=True)
            ]
        )
