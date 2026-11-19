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
    TextAudioBlockMask,
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
from vllm_omni.diffusion.data import SolAttnSpec, parse_kv_cache_skip_selector
from vllm_omni.diffusion.forward_context import get_forward_context, is_forward_context_available
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
        self.softmax_scale = softmax_scale
        selector = self.config.dense_layers
        self.dense_layers = parse_kv_cache_skip_selector([selector] if isinstance(selector, int) else selector) or set()
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
        self.kv_splits = 1 if isinstance(self.config.kv_splits, str) else self.config.kv_splits
        if self.config.kv_splits != 1:
            if capability is None or tuple(capability) != (9, 0):
                if self.config.kv_splits != "auto":
                    raise ValueError("SOL_ATTN kv_splits=2/4 requires SM90; use kv_splits=1 or 'auto'.")
            else:
                try:
                    import cuda.bindings.driver  # noqa: F401
                    import cutlass.cute  # noqa: F401
                except ImportError as error:
                    if self.config.kv_splits != "auto":
                        raise ImportError(
                            "SOL_ATTN kv_splits=2/4 requires the CuTe DSL runtime; use kv_splits=1."
                        ) from error
                else:
                    if self.config.kv_splits == "auto":
                        self.kv_splits = 4
        dense_impl = CuDNNAttentionImpl if capability is not None and capability.major == 12 else SDPAImpl
        self.dense_attention = dense_impl(num_heads, head_size, softmax_scale, num_kv_heads=num_kv_heads)

    def _should_use_dense(self, used: int) -> bool:
        if used < self.config.min_tokens or self.dense_block:
            return True
        if self.layer_idx is None and self.dense_layers:
            return True
        if self.layer_idx in self.dense_layers:
            return True
        step = get_forward_context().denoise_step_idx if is_forward_context_available() else None
        return self.config.dense_steps > 0 and (step is None or step < self.config.dense_steps)

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

    def _sink_range(self, metadata: AttentionMetadata | None, used: int) -> tuple[int, int]:
        if self.config.sink_mode == "text_audio":
            return 0, 0
        tokens = min(self.config.sink_tokens, used)
        start = used - tokens if self.config.sink_start is None else min(self.config.sink_start, used)
        end = min(start + tokens, used)
        layout = metadata.video_layout if metadata is not None else None
        if self.config.prefix_sink and layout is not None:
            if layout.used_len is not None and layout.used_len != used:
                raise ValueError("SOL_ATTN video layout and valid sequence length disagree.")
            prefix = layout.prefix_len
            if layout.video_spans:
                targets = [span for span in layout.video_spans if span.role == "target"]
                if len(targets) != 1:
                    raise ValueError("SOL_ATTN prefix_sink requires exactly one target-video span.")
                prefix = targets[0].start
            if type(prefix) is not int or not 0 <= prefix <= used:
                raise ValueError("SOL_ATTN prefix_sink requires a valid video prefix length.")
            if prefix:
                end = max(end if end > start else 0, prefix)
                start = 0
        return start, end - start

    def _sink_block_mask(
        self, metadata: AttentionMetadata | None, used: int, device: torch.device
    ) -> torch.Tensor | None:
        if self.config.sink_mode != "text_audio":
            return None
        protection = metadata.text_audio_block_mask if metadata is not None else None
        if not isinstance(protection, TextAudioBlockMask):
            raise ValueError("SOL_ATTN text_audio requires producer-owned text_audio_block_mask metadata.")
        if (
            type(protection.used_len) is not int
            or protection.used_len != used
            or type(protection.block_size) is not int
            or protection.block_size != 64
        ):
            raise ValueError("SOL_ATTN text_audio requires block64 metadata matching the valid sequence length.")
        mask = protection.blocks
        if (
            not isinstance(mask, torch.Tensor)
            or mask.dtype != torch.bool
            or mask.device != device
            or mask.shape != ((used + 63) // 64,)
            or not mask.is_contiguous()
        ):
            raise ValueError("SOL_ATTN text_audio requires a contiguous bool block mask on the Q/K/V device.")
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
                _, sink_tokens = self._sink_range(attn_metadata, used)
                self._sink_block_mask(attn_metadata, used, query.device)
                if self._should_use_dense(used) or sink_tokens == used:
                    result = replace(result, path="dense_guard", kernel_variant=None)
                if result.path == "sol_sparse":
                    self._resolve_kv_splits(used)
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
        splits = 1 if self.config.kv_splits == "auto" and used < 65536 else self.kv_splits
        if splits > (used + 4095) // 4096:
            raise ValueError("SOL_ATTN needs at least one 4096-token route group per KV split; use kv_splits=1.")
        return splits

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        used = self._used_length(query, key, value, attn_metadata)
        sink_start, sink_tokens = self._sink_range(attn_metadata, used)
        sink_block_mask = self._sink_block_mask(attn_metadata, used, query.device)
        query_used, key_used, value_used = (tensor[:, :used].contiguous() for tensor in (query, key, value))
        if self._should_use_dense(used) or sink_tokens == used:
            output = self.dense_attention.forward_cuda(query_used, key_used, value_used)
        else:
            kv_splits = self._resolve_kv_splits(used)
            output = sol_ops.sol_attention(
                query_used,
                key_used,
                value_used,
                self.softmax_scale,
                self.tau,
                self.config.thresh_type,
                kv_splits,
                sink_start,
                sink_tokens,
                sink_block_mask,
            )
            if sink_tokens:
                output[:, sink_start : sink_start + sink_tokens] = self.dense_attention.forward_cuda(
                    query_used[:, sink_start : sink_start + sink_tokens], key_used, value_used
                )
        if used == query.shape[1]:
            return output
        return torch.cat((output, torch.zeros_like(query[:, used:])), dim=1)
