# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) Microsoft Corporation and Jiarui Fang
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team & Jiarui Fang
# Adapted from
# https://github.com/feifeibear/long-context-attention/blob/main/yunchang/attention/layer.py


from dataclasses import replace

import torch
import torch.nn as nn
from vllm.logger import init_logger
from vllm.model_executor.models.utils import extract_layer_index

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.sdpa import SDPABackend
from vllm_omni.diffusion.attention.parallel import build_parallel_attention_strategy
from vllm_omni.diffusion.attention.parallel.base import NoParallelAttention
from vllm_omni.diffusion.attention.parallel.ring import RingParallelAttention
from vllm_omni.diffusion.attention.selector import get_attn_backend_for_role
from vllm_omni.diffusion.config import get_current_diffusion_config_or_none
from vllm_omni.diffusion.distributed.parallel_state import get_sp_group
from vllm_omni.diffusion.forward_context import get_forward_context, is_forward_context_available
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)


def _try_extract_layer_index(prefix: str) -> int | None:
    if not prefix:
        return None
    try:
        return extract_layer_index(prefix)
    except (AssertionError, ValueError):
        return None


class Attention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: int | None = None,
        prefix: str = "",
        # Per-role backend selection (RFC: per-role attention backend)
        role: str = "self",
        role_category: str | None = None,
        # Model-defined Q/K/V tensor layout hint for backend execution.
        qkv_layout: str | None = None,
        # ulysses attention
        scatter_idx: int = 2,
        gather_idx: int = 1,
        use_sync: bool = False,
        skip_sequence_parallel: bool = False,
        # Opt-out for KV-cache quantization at this specific attention layer.
        # Set by the model author when quant is known to degrade quality or
        # perf for this layer (e.g. Wan2.2 cross-attn has short sequences and
        # block-FP8 quant offers no win). Default False = follow global config.
        disable_kv_quant: bool = False,
    ):
        super().__init__()

        self.role = role
        self.role_category = role_category
        self.qkv_layout = qkv_layout

        # Resolve backend via role-aware config.
        # The global diffusion config is set during model init via
        # set_current_diffusion_config(); no env-var re-parsing needed here.
        backend_kwargs: dict | None = None
        self.backend_pref = None

        config = get_current_diffusion_config_or_none()
        attention_config = config.diffusion_attention_config if config is not None else None

        attn_backend_cls, spec = get_attn_backend_for_role(
            role=role,
            head_size=head_size,
            attention_config=attention_config,
            role_category=role_category,
        )
        if spec is not None:
            backend_kwargs = spec.extra or None
            self.backend_pref = spec.backend
            logger.debug("Attention(role=%s) → backend=%s", role, spec.backend)
        else:
            logger.debug("Attention(role=%s) → platform default", role)

        self.attn_backend = attn_backend_cls
        self.attn_impl_cls = self.attn_backend.get_impl_cls()
        self.attention = self.attn_impl_cls(
            num_heads=num_heads,
            head_size=head_size,
            softmax_scale=softmax_scale,
            causal=causal,
            num_kv_heads=num_kv_heads,
            qkv_layout=qkv_layout,
            backend_kwargs=backend_kwargs,
        )
        # Instantiate fallback backend for float32 support
        self.sdpa_fallback = SDPABackend.get_impl_cls()(
            num_heads=num_heads,
            head_size=head_size,
            softmax_scale=softmax_scale,
            causal=causal,
            num_kv_heads=num_kv_heads,
            qkv_layout=qkv_layout,
        )

        self.softmax_scale = softmax_scale
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx
        self.use_sync = use_sync
        self.causal = causal
        self.skip_sequence_parallel = skip_sequence_parallel

        self.use_ring = False
        self.ring_pg = None
        self.ring_runner = None

        if config is not None:
            if config.parallel_config.ring_degree > 1:
                self.use_ring = True
                try:
                    sp_group = get_sp_group()
                    self.ring_pg = sp_group.ring_group
                    self.ring_runner = RingParallelAttention(
                        sp_group,
                        attn_backend_pref=self.backend_pref,
                    )
                except Exception:
                    self.use_ring = False
                    self.ring_runner = None

        self.parallel_strategy = build_parallel_attention_strategy(
            scatter_idx=scatter_idx,
            gather_idx=gather_idx,
            use_sync=use_sync,
        )
        # Fallback strategy when SP is not active (outside sharded regions)
        self._no_parallel_strategy = NoParallelAttention()

        self.layer_idx: int | None = _try_extract_layer_index(prefix)

        self._kv_cache_dtype: str | None = None
        self._kv_cache_skip_steps: set[int] | None = None
        self._kv_cache_skip_layers: set[int] | None = None
        # Per-layer opt-out from KV-cache quantization (set by model author).
        self._disable_kv_quant: bool = disable_kv_quant
        self._init_kv_cache_quantization(config)

    def _get_active_parallel_strategy(self):
        """Get the parallel strategy based on current SP active state.

        Returns NoParallelAttention if we're outside an SP sharded region
        (e.g., in noise_refiner/context_refiner before unified_prepare in Z-Image).
        This avoids unnecessary SP communication for layers not covered by _sp_plan.
        """
        if self.skip_sequence_parallel:
            return self._no_parallel_strategy
        if is_forward_context_available():
            ctx = get_forward_context()
            if not ctx.sp_active:
                return self._no_parallel_strategy
        return self.parallel_strategy

    def _init_kv_cache_quantization(self, config) -> None:
        if config is None:
            return
        dtype = config.kv_cache_dtype
        if dtype:
            if config.parallel_config.ring_degree > 1:
                raise ValueError(
                    "KV quantization is not compatible with ring attention "
                    "(ring_degree > 1). Ring kernels do not propagate quantization descale "
                    "factors. Use Ulysses SP instead."
                )
            platform_key = current_omni_platform.device_name
            if not self.attention.supports_kv_cache_dtype(dtype, platform_key):
                logger.warning_once(
                    "Attention backend %s does not support kv_cache_dtype='%s' on %s. "
                    "KV quantization will be disabled.",
                    self.attn_backend.get_name(),
                    dtype,
                    platform_key,
                )
                dtype = None
        self._kv_cache_dtype = dtype
        self._kv_cache_skip_steps = config.kv_cache_skip_step_indices
        self._kv_cache_skip_layers = config.kv_cache_skip_layer_indices

    def _should_apply_kv_cache_quant(self) -> bool:
        skip_steps = self._kv_cache_skip_steps
        skip_layers = self._kv_cache_skip_layers
        if skip_steps is not None:
            step_idx = get_forward_context().denoise_step_idx if is_forward_context_available() else None
            if step_idx is not None and step_idx in skip_steps:
                return False
        if skip_layers is not None:
            if self.layer_idx is not None and self.layer_idx in skip_layers:
                return False
        return True

    def _with_kv_cache_dtype(self, attn_metadata: AttentionMetadata | None) -> AttentionMetadata | None:
        kv_cache_dtype = self._kv_cache_dtype
        if kv_cache_dtype is None or self._disable_kv_quant or not self._should_apply_kv_cache_quant():
            if attn_metadata is None or "kv_cache_dtype" not in attn_metadata.extra:
                return attn_metadata
            extra = dict(attn_metadata.extra)
            extra.pop("kv_cache_dtype", None)
            return replace(attn_metadata, extra=extra)

        if attn_metadata is None:
            return AttentionMetadata(extra={"kv_cache_dtype": kv_cache_dtype})
        extra = dict(attn_metadata.extra)
        extra["kv_cache_dtype"] = kv_cache_dtype
        return replace(attn_metadata, extra=extra)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        # Get the appropriate parallel strategy based on SP active state
        strategy = self._get_active_parallel_strategy()

        # 1. Prepare inputs (Communication / Resharding)
        # For Ulysses: AllToAll Q/K/V; Slicing joint_q/k/v
        # For Ring: Concat joint_q
        query, key, value, attn_metadata, ctx = strategy.pre_attention(query, key, value, attn_metadata)

        attn_metadata = self._with_kv_cache_dtype(attn_metadata)

        # 2. Kernel Execution (Computation)
        if self.use_ring and strategy is not self._no_parallel_strategy:
            out = self._run_ring_attention(query, key, value, attn_metadata)
        else:
            out = self._run_local_attention(query, key, value, attn_metadata)

        # 3. Post-processing (Reverse Communication)
        # For Ulysses: AllToAll Output, and AllGather Joint Output
        out = strategy.post_attention(out, ctx)

        return out

    def _run_local_attention(self, query, key, value, attn_metadata):
        if query.dtype == torch.float32:
            logger.warning_once(
                f"Only SDPA supports float32. Overriding user config {type(self.attention)} "
                f"attention_backend='{self.backend_pref}' to 'sdpa' for dtype={query.dtype}."
            )
            return self.sdpa_fallback.forward(query, key, value, attn_metadata)

        # Fallback to standard attention
        return self.attention.forward(query, key, value, attn_metadata)

    def _run_ring_attention(self, query, key, value, attn_metadata):
        # Delegate to RingParallelAttention strategy if available
        if self.ring_runner is not None:
            return self.ring_runner.run_attention(
                query, key, value, attn_metadata, softmax_scale=self.softmax_scale, causal=self.causal
            )

        raise RuntimeError("Ring attention is enabled but strategy is not RingParallelAttention")
