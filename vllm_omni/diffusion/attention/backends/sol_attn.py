# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in Sol-Attn for a single noncausal BF16 self-attention sequence."""

from __future__ import annotations

from dataclasses import replace

import torch
from vllm.model_executor.models.utils import extract_layer_index

from vllm_omni.diffusion.attention.backends.abstract import AttentionBackend, AttentionImpl, AttentionMetadata
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
from vllm_omni.diffusion.config import get_current_diffusion_config_or_none
from vllm_omni.diffusion.data import SolAttnSpec, parse_kv_cache_skip_selector
from vllm_omni.diffusion.forward_context import get_forward_context, is_forward_context_available
from vllm_omni.platforms import current_omni_platform

if not hasattr(torch.ops.vllm_omni, "sol_attention"):

    @torch.library.custom_op("vllm_omni::sol_attention", mutates_args=())
    def _sol_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale: float,
        tau: float,
        thresh_type: str,
        kv_splits: int,
        sink_start: int,
        sink_tokens: int,
    ) -> torch.Tensor:
        from sol_attn import sol_attn

        return sol_attn(
            query,
            key,
            value,
            scale=scale,
            tau=tau,
            thresh_type=thresh_type,
            kv_splits=kv_splits,
            sink_start=sink_start,
            sink_tokens=sink_tokens,
        )

    @_sol_attention.register_fake
    def _sol_attention_fake(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale: float,
        tau: float,
        thresh_type: str,
        kv_splits: int,
        sink_start: int,
        sink_tokens: int,
    ) -> torch.Tensor:
        return torch.empty_like(query)


_sol_attention_op = torch.ops.vllm_omni.sol_attention


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
                "`pip install 'git+https://github.com/NVlabs/Sana.git@sol-engine"
                "#subdirectory=techniques/sparse_backends'`."
            ) from error
        if not callable(sol_attn):
            raise ImportError("SOL_ATTN requires a sol_attn package exporting the callable sol_attn kernel.")

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
        diffusion_config = get_current_diffusion_config_or_none()
        parallel = getattr(diffusion_config, "parallel_config", None)
        if getattr(parallel, "ring_degree", 1) > 1 or getattr(parallel, "allgather_degree", 1) > 1:
            raise ValueError("SOL_ATTN needs full Q/K/V sequences; use local attention or Ulysses with ring_degree=1.")
        capability = current_omni_platform.get_device_capability()
        self.kv_splits = 1 if isinstance(self.config.kv_splits, str) else self.config.kv_splits
        if self.kv_splits != 1:
            if capability is None or tuple(capability) != (9, 0):
                raise ValueError("SOL_ATTN kv_splits=2/4 requires SM90; use kv_splits=1 or 'auto'.")
            try:
                import cuda.bindings.driver  # noqa: F401
                import cutlass.cute  # noqa: F401
            except ImportError as error:
                raise ImportError("SOL_ATTN kv_splits=2/4 requires the CuTe DSL runtime; use kv_splits=1.") from error
        dense_impl = CuDNNAttentionImpl if capability is not None and capability.major == 12 else SDPAImpl
        self.dense_attention = dense_impl(num_heads, head_size, softmax_scale, num_kv_heads=num_kv_heads)

    def _should_use_dense(self) -> bool:
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
                end = max(end if tokens else 0, prefix)
                start = 0
        return start, end - start

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
            path="dense_guard" if self._should_use_dense() else "sol_sparse",
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
                if sink_tokens == used:
                    result = replace(result, path="dense_guard")
                if result.path == "sol_sparse":
                    self._validate_split_length(used)
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

    def _validate_split_length(self, used: int) -> None:
        if self.kv_splits > (used + 4095) // 4096:
            raise ValueError("SOL_ATTN needs at least one 4096-token route group per KV split; use kv_splits=1.")

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        used = self._used_length(query, key, value, attn_metadata)
        sink_start, sink_tokens = self._sink_range(attn_metadata, used)
        query_used, key_used, value_used = (tensor[:, :used].contiguous() for tensor in (query, key, value))
        if self._should_use_dense() or sink_tokens == used:
            output = self.dense_attention.forward_cuda(query_used, key_used, value_used)
        else:
            self._validate_split_length(used)
            output = _sol_attention_op(
                query_used,
                key_used,
                value_used,
                self.softmax_scale,
                self.config.tau,
                self.config.thresh_type,
                self.kv_splits,
                sink_start,
                sink_tokens,
            )
            if sink_tokens:
                output[:, sink_start : sink_start + sink_tokens] = self.dense_attention.forward_cuda(
                    query_used[:, sink_start : sink_start + sink_tokens], key_used, value_used
                )
        if used == query.shape[1]:
            return output
        return torch.cat((output, torch.zeros_like(query[:, used:])), dim=1)
