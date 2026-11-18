# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in Sol-Attn for a single noncausal BF16 self-attention sequence."""

from __future__ import annotations

import math
from collections.abc import Callable
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
from vllm_omni.diffusion.attention.ops.block_sparse import (
    block_sparse_attn_bshd,
    mean_pool_tiles,
    resolve_block_sparse_provider,
    validate_softmax_scale,
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


def _require_sol_bsa(
    provider: str, scale: float, device: torch.device
) -> Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    validate_softmax_scale(scale)
    if provider not in ("cudnn", "triton"):
        raise ValueError("SOL BSA requires an explicit cudnn or triton provider.")
    if provider == "triton" and not math.isclose(scale, 128**-0.5, rel_tol=0, abs_tol=1e-6):
        raise ValueError("SOL BSA Triton requires head_size**-0.5 scaling.")
    resolve_block_sparse_provider(provider, "bf16", device, torch.bfloat16, 128, 64)
    try:
        from sol_attn.triton_ref.preprocess import prepare
    except ImportError as error:
        raise ImportError("SOL BSA requires the pinned sol_attn package with triton_ref.preprocess.prepare.") from error
    if not callable(prepare):
        raise ImportError("SOL BSA requires callable triton_ref.preprocess.prepare.")
    return prepare


def _sol_bsa_block_map(
    query: torch.Tensor,
    key_centroids: torch.Tensor,
    threshold: torch.Tensor,
    sizes: torch.Tensor,
    scale: float,
    sink_start: int,
    sink_tokens: int,
    sink_block_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply SOL thresholds, neighboring tiles and dense query/key sink tiles."""
    pooled_query = mean_pool_tiles(query, sizes, 64)
    scores = pooled_query @ key_centroids.float().transpose(1, 2).transpose(-1, -2)
    selected = scores * (scale * math.log2(math.e)) > threshold.transpose(1, 2).unsqueeze(-1)
    positions = torch.arange(sizes.numel(), device=query.device)
    selected |= (positions[:, None] - positions[None, :]).abs() <= 1
    if sink_tokens:
        first, last = sink_start // 64, (sink_start + sink_tokens + 63) // 64
        selected[..., first:last, :] = True
        selected[..., first:last] = True
    if sink_block_mask is not None:
        selected |= sink_block_mask[:, None] | sink_block_mask[None, :]
    return selected


def _merge_sol_bsa_residual(
    query: torch.Tensor,
    key_centroids: torch.Tensor,
    value_sums: torch.Tensor,
    selected: torch.Tensor,
    sizes: torch.Tensor,
    exact_output: torch.Tensor,
    exact_lse: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Reference SOL merge with bounded query chunks and natural-log normalization.

    Upstream summarizes K by its mean and V by its sum. Block multiplicity
    belongs only in the residual denominator. This correctness-first path
    deliberately does not claim the speed of a fused SOL residual kernel.
    """
    output = torch.empty_like(query)
    keys = key_centroids.float().transpose(1, 2).transpose(-1, -2)
    values = value_sums.float().transpose(1, 2)
    log_sizes = sizes.float().log()
    for start in range(0, query.shape[1], 256):
        stop = min(start + 256, query.shape[1])
        scores = query[:, start:stop].float().transpose(1, 2) @ keys * scale
        exact_mask = selected[:, :, start // 64 : (stop + 63) // 64].repeat_interleave(64, -2)
        scores.masked_fill_(exact_mask[:, :, : stop - start], -torch.inf)
        residual_lse = (scores + log_sizes).logsumexp(-1)
        exact_lse_chunk = exact_lse[:, :, start:stop]
        joint_lse = torch.logaddexp(exact_lse_chunk, residual_lse)
        residual = (scores - joint_lse.unsqueeze(-1)).exp() @ values
        exact_weight = (exact_lse_chunk - joint_lse).exp().unsqueeze(-1)
        merged = exact_output[:, start:stop].float().transpose(1, 2) * exact_weight + residual
        output[:, start:stop] = merged.transpose(1, 2).to(query.dtype)
    return output


if not hasattr(torch.ops.vllm_omni, "sol_bsa_attention"):

    @torch.library.custom_op("vllm_omni::sol_bsa_attention", mutates_args=())
    def _sol_bsa_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale: float,
        tau: float,
        thresh_type: str,
        sink_start: int,
        sink_tokens: int,
        provider: str,
        sink_block_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        prepare = _require_sol_bsa(provider, scale, query.device)
        tokens = query.shape[1]
        blocks = (tokens + 63) // 64
        key_centroids, value_sums, threshold = prepare(
            query, key, value, tau=tau, scale=scale, thresh_type=thresh_type, tokens=tokens
        )
        key_centroids, value_sums = key_centroids[:, :blocks], value_sums[:, :blocks]
        sizes = (tokens - torch.arange(blocks, device=query.device, dtype=torch.int32) * 64).clamp(max=64)
        padding = blocks * 64 - tokens
        query_padded, key_padded, value_padded = (
            torch.nn.functional.pad(tensor, (0, 0, 0, 0, 0, padding)) if padding else tensor
            for tensor in (query, key, value)
        )
        selected = _sol_bsa_block_map(
            query_padded, key_centroids, threshold, sizes, scale, sink_start, sink_tokens, sink_block_mask
        )
        exact_output, exact_lse = block_sparse_attn_bshd(
            query_padded, key_padded, value_padded, selected, sizes, scale, provider=provider, return_lse=True
        )
        return _merge_sol_bsa_residual(
            query, key_centroids, value_sums, selected, sizes, exact_output, exact_lse, scale
        )

    @_sol_bsa_attention.register_fake
    def _sol_bsa_attention_fake(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale: float,
        tau: float,
        thresh_type: str,
        sink_start: int,
        sink_tokens: int,
        provider: str,
        sink_block_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return torch.empty_like(query)


_sol_bsa_attention_op = torch.ops.vllm_omni.sol_bsa_attention


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
                "`pip install 'git+https://github.com/NVlabs/Sana.git@f4035e0a976d828ed41c7b0b799e9712b47cd60d"
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
        if self.config.provider != "native":
            _require_sol_bsa(self.config.provider, softmax_scale, torch.device("cuda"))
        elif self.config.kv_splits != 1:
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
        sparse_path = "sol_sparse" if self.config.provider == "native" else "sol_bsa"
        path = "dense_guard" if self._should_use_dense() else sparse_path
        result = ExecutionPathResult(
            backend="SOL_ATTN",
            path=path,
            support=CapabilityResult.supported(),
            compilation_mode=CompilationMode.CUSTOM_OP,
            platform=context.platform,
            kernel_variant=f"bsa_{self.config.provider}_bf16" if path == "sol_bsa" else None,
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
                if sink_tokens == used:
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
        if self.config.provider != "native":
            return 1
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
        if self._should_use_dense() or sink_tokens == used:
            output = self.dense_attention.forward_cuda(query_used, key_used, value_used)
        elif self.config.provider != "native":
            output = _sol_bsa_attention_op(
                query_used,
                key_used,
                value_used,
                self.softmax_scale,
                self.config.tau,
                self.config.thresh_type,
                sink_start,
                sink_tokens,
                self.config.provider,
                sink_block_mask,
            )
        else:
            kv_splits = self._resolve_kv_splits(used)
            output = _sol_attention_op(
                query_used,
                key_used,
                value_used,
                self.softmax_scale,
                self.config.tau,
                self.config.thresh_type,
                kv_splits,
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
