# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in Sol-Attn for a single noncausal BF16 self-attention sequence."""

from __future__ import annotations

import inspect
from dataclasses import replace

import torch
from vllm.model_executor.models.utils import extract_layer_index

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    ConditioningBlockMask,
)
from vllm_omni.diffusion.attention.backends.cudnn_attn import CuDNNAttentionImpl
from vllm_omni.diffusion.attention.backends.sdpa import SDPAImpl
from vllm_omni.diffusion.attention.capabilities import (
    AttentionComputation,
    CapabilityResult,
    CompilationMode,
    ExecutionContext,
    ExecutionPathResult,
    MaskMode,
    PackingMode,
    ParallelStrategy,
)
from vllm_omni.diffusion.attention.ops import sol_attn as sol_ops
from vllm_omni.diffusion.attention.sol_config import parse_sol_block_ranges, parse_sol_tau_profile, sol_block_selected
from vllm_omni.diffusion.config import get_current_diffusion_config_or_none
from vllm_omni.diffusion.data import SolAttnSpec
from vllm_omni.platforms import current_omni_platform


class SolAttnBackend(AttentionBackend):
    accept_output_buffer = True
    supports_prefix_kv_slicing = True
    supported_platforms = ("cuda",)

    @classmethod
    def supports_packed_mask_free(cls) -> bool:
        return True

    @classmethod
    def validate_available(cls) -> None:
        try:
            from sol_attn import sol_attn
        except ImportError as error:
            raise ImportError(
                "SOL_ATTN requires the upstream sol_attn package. Install it with "
                "`pip install 'git+https://github.com/lishunyang12/Sana.git@d9ef6321230cff13398cfc86aa107f97f093e925"
                "#subdirectory=techniques/sparse_backends'`."
            ) from error
        if not callable(sol_attn):
            raise ImportError("SOL_ATTN requires a sol_attn package exporting the callable sol_attn kernel.")
        if "exact_block_mask" not in inspect.signature(sol_attn).parameters:
            raise ImportError(
                "SOL_ATTN requires upstream sol_attn with exact_block_mask support; "
                "reinstall the pinned version in the SOL attention guide."
            )

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [128]

    @staticmethod
    def get_name() -> str:
        return "SOL_ATTN"

    @staticmethod
    def get_impl_cls() -> type[SolAttnImpl]:
        return SolAttnImpl


class SolAttnImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        qkv_layout: str | None = None,
        backend_kwargs: dict | None = None,
        computation: AttentionComputation | str = AttentionComputation.SELF,
        layer_count: int | None = None,
        requires_conditioning_mask: bool = False,
        **extra_impl_args,
    ) -> None:
        if head_size != 128:
            raise ValueError("SOL_ATTN requires head_size=128.")
        if causal or (num_kv_heads is not None and num_kv_heads != num_heads):
            raise ValueError("SOL_ATTN requires noncausal attention with equal Q/KV heads; GQA/MQA is unsupported.")
        if AttentionComputation(computation) is not AttentionComputation.SELF:
            raise ValueError("SOL_ATTN supports self-attention only.")
        if qkv_layout not in (None, "BSND"):
            raise ValueError("SOL_ATTN requires qkv_layout='BSND'.")
        options = backend_kwargs or {}
        if set(options) - {"sol_attn"}:
            raise ValueError(f"Unknown SOL_ATTN backend options: {sorted(set(options) - {'sol_attn'})}")
        settings = options.get("sol_attn")
        self.config = SolAttnSpec(**({} if settings is None else settings))
        self.requires_conditioning_mask = requires_conditioning_mask
        self.softmax_scale = softmax_scale
        try:
            self.layer_idx = extract_layer_index(prefix) if prefix else None
        except (AssertionError, ValueError):
            self.layer_idx = None
        if layer_count is not None and (type(layer_count) is not int or layer_count <= 0):
            raise ValueError("SOL_ATTN layer_count must be a positive model-owned integer.")
        self.tau = self.config.tau
        self.dense_block = False
        tau_rules = parse_sol_tau_profile(self.config.tau_profile)
        if parse_sol_block_ranges(self.config.dense_blocks) or tau_rules:
            if self.layer_idx is None or (layer_count is not None and self.layer_idx >= layer_count):
                raise ValueError("SOL_ATTN dense_blocks/tau_profile require a valid model-owned block index.")
            self.dense_block = sol_block_selected(self.config.dense_blocks, self.layer_idx, layer_count)
            for blocks, tau in tau_rules:
                if sol_block_selected(blocks, self.layer_idx, layer_count):
                    self.tau = tau
        diffusion_config = get_current_diffusion_config_or_none()
        parallel = getattr(diffusion_config, "parallel_config", None)
        if getattr(parallel, "ring_degree", 1) > 1 or getattr(parallel, "allgather_degree", 1) > 1:
            raise ValueError("SOL_ATTN needs full Q/K/V sequences; use local attention or Ulysses with ring_degree=1.")
        capability = current_omni_platform.get_device_capability()
        self.kv_splits = 1
        if capability is not None and tuple(capability) == (9, 0):
            try:
                import cuda.bindings.driver  # noqa: F401
                import cutlass.cute  # noqa: F401
            except ImportError:
                # The upstream Triton path uses one split when CuTe is absent.
                pass
            else:
                self.kv_splits = 4
        dense_impl = CuDNNAttentionImpl if capability is not None and capability.major == 12 else SDPAImpl
        self.dense_attention = dense_impl(num_heads, head_size, softmax_scale, num_kv_heads=num_kv_heads)

    def _should_use_dense(self, used: int) -> bool:
        return used < self.config.min_tokens or self.dense_block

    @staticmethod
    def _used_length(
        query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, metadata: AttentionMetadata | None
    ) -> int:
        if query.ndim != 4 or query.shape != key.shape or query.shape != value.shape:
            raise ValueError("SOL_ATTN requires matching Q/K/V shapes [1, S, H, 128].")
        if query.shape[0] != 1 or query.shape[-1] != 128 or any(size == 0 for size in query.shape):
            raise ValueError("SOL_ATTN requires a single nonempty sequence with head_dim=128.")
        if any(tensor.dtype != torch.bfloat16 for tensor in (query, key, value)):
            raise TypeError("SOL_ATTN requires BF16 Q/K/V, including dense guard steps.")
        if query.device != key.device or query.device != value.device:
            raise ValueError("SOL_ATTN requires Q/K/V on the same device.")
        if metadata is None:
            return query.shape[1]
        if any(
            item is not None
            for item in (
                metadata.attn_mask,
                metadata.joint_attn_mask,
                metadata.joint_query,
                metadata.joint_key,
                metadata.joint_value,
                metadata.full_attn_spans,
                metadata.query_ranges,
                metadata.cross_query_mask,
            )
        ):
            raise ValueError("SOL_ATTN does not support masks, joint attention, or piecewise attention.")
        if metadata.extra.get("kv_cache_dtype") not in (None, "auto", "float"):
            raise ValueError("SOL_ATTN does not support quantized KV caches.")
        packed = metadata.packed_padding
        used = metadata.extra.get("valid_kv_length", query.shape[1])
        if packed is not None:
            if packed.q_length != packed.kv_length:
                raise ValueError("SOL_ATTN requires equal packed Q/KV lengths; compact reference KV is unsupported.")
            if "valid_kv_length" in metadata.extra and used != packed.kv_length:
                raise ValueError("SOL_ATTN packed_padding and valid_kv_length disagree.")
            used = packed.q_length
        elif "cu_seqlens_q" in metadata.extra or "cu_seqlens_k" in metadata.extra:
            raise ValueError(
                "SOL_ATTN requires producer-owned packed_padding; multi-document attention is unsupported."
            )
        if type(used) is not int or not 0 < used <= query.shape[1]:
            raise ValueError("SOL_ATTN valid length must be a positive host integer within the sequence.")
        return used

    def _sink_block_mask(
        self, metadata: AttentionMetadata | None, used: int, device: torch.device
    ) -> torch.Tensor | None:
        if self.config.sink_conditioning == "off":
            return None
        protection = metadata.conditioning_block_mask if metadata is not None else None
        if protection is None and not self.requires_conditioning_mask:
            return None
        if not isinstance(protection, ConditioningBlockMask):
            raise ValueError(
                "SOL_ATTN requires producer-owned conditioning_block_mask metadata for this attention stack."
            )
        if (
            type(protection.used_len) is not int
            or protection.used_len != used
            or type(protection.block_size) is not int
            or protection.block_size != 64
        ):
            raise ValueError(
                "SOL_ATTN conditioning protection requires block64 metadata matching the valid sequence length."
            )
        mask = protection.blocks
        if (
            not isinstance(mask, torch.Tensor)
            or mask.dtype != torch.bool
            or mask.device != device
            or mask.shape != ((used + 63) // 64,)
            or not mask.is_contiguous()
        ):
            raise ValueError(
                "SOL_ATTN conditioning protection requires a contiguous bool block mask on the Q/K/V device."
            )
        return mask

    def resolve_execution_path(
        self,
        context: ExecutionContext,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
    ) -> ExecutionPathResult:
        result = ExecutionPathResult(
            backend="SOL_ATTN",
            path="sol_sparse",
            support=CapabilityResult.supported(),
            compilation_mode=CompilationMode.CUSTOM_OP,
            platform=context.platform,
            kernel_variant=None,
            parallel_strategy=context.parallel_strategy,
        )
        if context.platform != "cuda":
            reason = "SOL_ATTN runs on CUDA only."
        elif context.parallel_strategy not in (ParallelStrategy.NONE, ParallelStrategy.ULYSSES):
            reason = "SOL_ATTN supports local attention and pure Ulysses only."
        elif (
            context.causal
            or context.computation is not AttentionComputation.SELF
            or context.mask_mode is not MaskMode.NONE
            or context.packing_mode is PackingMode.MULTI_DOCUMENT
            or context.piecewise
            or context.paged_kv
            or context.cross_parallel_layout is not None
            or context.kv_cache_dtype not in (None, "auto", "float")
        ):
            reason = "SOL_ATTN requires mask-free noncausal self attention without paged/quantized KV."
        else:
            try:
                used = self._used_length(query, key, value, attn_metadata)
                self._sink_block_mask(attn_metadata, used, query.device)
                if self._should_use_dense(used):
                    result = replace(result, path="dense_guard", kernel_variant=None)
            except (TypeError, ValueError) as error:
                reason = str(error)
            else:
                if context.outer_boundaries:
                    return replace(
                        result,
                        support=CapabilityResult.unmigrated("SOL_ATTN outer compilation boundaries are unverified."),
                        compilation_mode=CompilationMode.EAGER_ONLY,
                    )
                return result
        return replace(result, support=CapabilityResult.unsupported(reason))

    def _resolve_kv_splits(self, used: int) -> int:
        return 1 if used < 65536 else self.kv_splits

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        used = self._used_length(query, key, value, attn_metadata)
        sink_block_mask = self._sink_block_mask(attn_metadata, used, query.device)
        query_used, key_used, value_used = (tensor[:, :used].contiguous() for tensor in (query, key, value))
        if self._should_use_dense(used):
            output = self.dense_attention.forward_cuda(query_used, key_used, value_used)
        else:
            kv_splits = self._resolve_kv_splits(used)
            output = sol_ops.sol_attention(
                query_used,
                key_used,
                value_used,
                self.softmax_scale,
                self.tau,
                kv_splits,
                sink_block_mask,
            )
        if used == query.shape[1]:
            return output
        return torch.cat((output, torch.zeros_like(query[:, used:])), dim=1)
