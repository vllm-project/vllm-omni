# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Copyright (c) Microsoft Corporation and Jiarui Fang
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team & Jiarui Fang
# Adapted from
# https://github.com/feifeibear/long-context-attention/blob/main/yunchang/attention/layer.py


from dataclasses import replace

import torch
import torch.nn as nn
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheSpec

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.sdpa import SDPABackend
from vllm_omni.diffusion.attention.parallel import build_parallel_attention_strategy
from vllm_omni.diffusion.attention.parallel.base import NoParallelAttention
from vllm_omni.diffusion.attention.parallel.ring import RingParallelAttention
from vllm_omni.diffusion.attention.selector import get_attn_backend_for_role
from vllm_omni.diffusion.config import get_current_diffusion_config_or_none
from vllm_omni.diffusion.diffusion_kv.config import DiffusionKVCacheMode
from vllm_omni.diffusion.distributed.parallel_state import get_sp_group
from vllm_omni.diffusion.forward_context import (
    get_forward_context,
    get_ulysses_mode,
    is_forward_context_available,
)
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
    _scheduler_paged_kv = False

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
        # Opt-in marker for Scheduler-managed paged KV. Unmarked diffusion
        # attention remains dense and contributes no native KVCacheSpec.
        paged_kv_cache_role: str | None = None,
        paged_kv_cache_dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.role = role
        self.role_category = role_category
        self.qkv_layout = qkv_layout
        # ``prefix`` is also the stable layer identity used by vLLM's native
        # KV-cache metadata.  Keep it on the Omni layer so the active paged
        # adapter can dispatch the already-resharded Q/K/V to the matching
        # native cache tensor without replacing this Omni execution path.
        self.prefix = prefix
        if paged_kv_cache_role == "":
            raise ValueError("paged_kv_cache_role must be non-empty when provided")
        self.paged_kv_cache_role = paged_kv_cache_role
        self.paged_kv_cache_dtype = paged_kv_cache_dtype
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_size = head_size

        # Resolve backend via role-aware config.
        # The global diffusion config is set during model init via
        # set_current_diffusion_config(); no env-var re-parsing needed here.
        backend_kwargs: dict | None = None
        self.backend_pref = None
        self.backend_explicit = False

        config = get_current_diffusion_config_or_none()
        attention_config = config.diffusion_attention_config if config is not None else None

        from vllm_omni.diffusion.model_metadata import get_diffusion_model_metadata

        model_class_name = getattr(config, "model_class_name", None) if config is not None else None
        allow_trtllm_default = get_diffusion_model_metadata(model_class_name).attention_mask_free

        attn_backend_cls, spec = get_attn_backend_for_role(
            role=role,
            head_size=head_size,
            attention_config=attention_config,
            role_category=role_category,
            allow_trtllm_default=allow_trtllm_default,
        )
        scheduler_paged_kv = (
            config is not None
            and getattr(config, "diffusion_kv_mode", DiffusionKVCacheMode.DENSE_LEGACY)
            is DiffusionKVCacheMode.PAGED_SCHEDULER
        )
        self._scheduler_paged_kv = scheduler_paged_kv
        if (
            scheduler_paged_kv
            and paged_kv_cache_role is not None
            and spec is None
            and not attn_backend_cls.supports_paged_kv
        ):
            # FLASH_ATTN is an Omni selector, not a device-specific kernel.
            # vLLM resolves it to CUDA FlashAttention on GPU and Ascend native
            # attention on NPU. A platform may legitimately default dense
            # attention to CUDNN/SDPA (for example when optional dense FA
            # extras are absent), but a paged layer must advertise the
            # capability so the Worker can register its native cache view.
            # Load that selector from the registry: this is not a user-explicit
            # dense FLASH_ATTN request, so skip platform explicit-backend
            # validation (Blackwell would otherwise require CuTe FA4). Formal
            # paged execution still delegates to vLLM's native paged backend.
            from vllm_omni.diffusion.attention.backends.registry import DiffusionAttentionBackendEnum

            dense_backend_name = attn_backend_cls.get_name()
            attn_backend_cls = DiffusionAttentionBackendEnum.FLASH_ATTN.get_class()
            logger.info(
                "Resolved marked paged diffusion attention role=%r to %r because the platform default %r "
                "does not support Scheduler-owned KV",
                role,
                attn_backend_cls.get_name(),
                dense_backend_name,
            )
        parallel_config = getattr(config, "parallel_config", None)
        allgather_degree = getattr(parallel_config, "allgather_degree", 1)
        # TODO: Move AllGather-KV compatibility into an AttentionBackend capability
        # so validation does not depend on backend names.
        if not skip_sequence_parallel and allgather_degree > 1 and attn_backend_cls.get_name() == "TRTLLM_ATTN":
            raise ValueError(
                "TRTLLM_ATTN does not support AllGather-KV sequence parallelism. "
                "Set --allgather-degree 1 or select another diffusion attention backend."
            )
        self.attn_spec = spec
        if spec is not None:
            backend_kwargs = spec.backend_kwargs()
            self.backend_pref = spec.backend
            self.backend_explicit = True
            logger.debug("Attention(role=%s) → backend=%s", role, spec.backend)
        else:
            # Propagate the resolved platform default so Ring Attention can
            # make a compatible automatic selection on the current GPU.
            self.backend_pref = attn_backend_cls.get_name()
            logger.debug("Attention(role=%s) → platform default (%s)", role, self.backend_pref)

        self.attn_backend = attn_backend_cls
        self.attn_impl_cls = self.attn_backend.get_impl_cls()
        self.attention = self.attn_impl_cls(
            num_heads=num_heads,
            head_size=head_size,
            softmax_scale=softmax_scale,
            causal=causal,
            num_kv_heads=num_kv_heads,
            qkv_layout=qkv_layout,
            prefix=prefix,
            backend_kwargs=backend_kwargs,
            role=role,
            backend_explicit=self.backend_explicit,
        )
        # Some model-specific paths call this helper directly. The main backend
        # dispatch below never switches to it implicitly.
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
                sp_group = get_sp_group()
                self.ring_pg = sp_group.ring_group
                self.ring_runner = RingParallelAttention(
                    sp_group,
                    attn_backend_pref=self.backend_pref,
                    attn_backend_explicit=self.backend_explicit,
                )

        self.parallel_strategy = build_parallel_attention_strategy(
            scatter_idx=scatter_idx,
            gather_idx=gather_idx,
            use_sync=use_sync,
            causal=causal,
        )
        # Local strategy when SP is intentionally inactive outside sharded regions.
        self._no_parallel_strategy = NoParallelAttention()

        self.layer_idx: int | None = _try_extract_layer_index(prefix)

        self._kv_cache_dtype: str | None = None
        self._kv_cache_skip_steps: set[int] | None = None
        self._kv_cache_skip_layers: set[int] | None = None
        # Per-layer opt-out from KV-cache quantization (set by model author).
        self._disable_kv_quant: bool = disable_kv_quant
        self._init_kv_cache_quantization(config)

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        """Return native rank-local geometry for an opted-in paged cache."""

        if self.paged_kv_cache_role is None:
            return None
        dtype = self.paged_kv_cache_dtype or vllm_config.model_config.dtype
        # Keep backend layout discovery under the same config context used by
        # upstream vLLM's attention-spec collector.
        with set_current_vllm_config(vllm_config):
            indexes_kv_by_block_stride = self.attn_backend.indexes_kv_by_block_stride()
        return FullAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            dtype=dtype,
            indexes_kv_by_block_stride=indexes_kv_by_block_stride,
            non_causal=not self.causal,
        )

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
        dtype = getattr(config, "diffusion_kv_cache_dtype", None)
        if dtype == "auto":
            dtype = None
        parallel_config = getattr(config, "parallel_config", None)
        ring_degree = getattr(parallel_config, "ring_degree", 1)
        if dtype:
            if ring_degree > 1:
                raise ValueError(
                    "KV quantization is not compatible with ring attention "
                    "(ring_degree > 1). Ring kernels do not propagate quantization descale "
                    "factors. Use Ulysses SP instead."
                )
            platform_key = current_omni_platform.device_name
            if not self.attention.supports_kv_cache_dtype(dtype, platform_key):
                raise ValueError(
                    f"Attention backend {self.attn_backend.get_name()} does not support "
                    f"kv_cache_dtype={dtype!r} on {platform_key}. Select a compatible "
                    "backend or set diffusion_kv_cache_dtype='auto'."
                )
        self._kv_cache_dtype = dtype
        self._kv_cache_skip_steps = getattr(config, "diffusion_kv_cache_skip_step_indices", None)
        self._kv_cache_skip_layers = getattr(config, "diffusion_kv_cache_skip_layer_indices", None)

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
        if torch.compiler.is_compiling() and is_forward_context_available():
            od_config = get_forward_context().omni_diffusion_config
            parallel_config = getattr(od_config, "parallel_config", None)
            if getattr(parallel_config, "use_hsdp", False):
                # Keep HSDP/FSDP2 parameter all-gather outside Inductor's
                # attention graph; otherwise scheduler dependency analysis can
                # fail on the fused attention region.
                return self._forward_hsdp_compile_boundary(query, key, value, attn_metadata)

        return self._forward_impl(query, key, value, attn_metadata)

    @torch.compiler.disable
    def _forward_hsdp_compile_boundary(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        return self._forward_impl(query, key, value, attn_metadata)

    def _forward_impl(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        # Get the appropriate parallel strategy based on SP active state
        strategy = self._get_active_parallel_strategy()
        paged_adapter = self._active_paged_kv_adapter()
        in_kv_memory_profile = is_forward_context_available() and get_forward_context().in_diffusion_kv_memory_profile
        if (
            self._scheduler_paged_kv
            and self.paged_kv_cache_role is not None
            and paged_adapter is None
            and not in_kv_memory_profile
        ):
            raise RuntimeError(
                "Scheduler-paged diffusion attention reached model forward without an active Worker adapter. "
                "Only the startup KV memory profile may execute before paged KV initialization."
            )
        use_paged_attention = paged_adapter is not None and self.paged_kv_cache_role is not None
        if use_paged_attention and not getattr(self.attn_backend, "supports_paged_kv", False):
            raise NotImplementedError(
                f"Diffusion paged KV requires an Omni backend with paged support; "
                f"selected {self.attn_backend.get_name()}"
            )
        if use_paged_attention and strategy is not self._no_parallel_strategy:
            strategy_name = strategy.name
            if self.use_ring or strategy_name == "ring":
                raise NotImplementedError(
                    "paged Scheduler KV is not supported with Ring attention; use strict Ulysses or no SP"
                )
            if strategy_name == "allgather_kv":
                raise NotImplementedError("paged Scheduler KV is not supported with AllGather-KV sequence parallelism")
            if strategy_name == "ulysses" and get_ulysses_mode(default="strict") != "strict":
                raise NotImplementedError("paged Scheduler KV currently supports only strict Ulysses")

        # 1. Prepare inputs (Communication / Resharding)
        # For Ulysses: AllToAll Q/K/V; Slicing joint_q/k/v
        # For Ring: Concat joint_q
        query, key, value, attn_metadata, ctx = strategy.pre_attention(query, key, value, attn_metadata)

        # Scheduler rows describe the logical sequence, while strict Ulysses
        # may append synthetic tokens solely to make the image shard divisible.
        # Remove those tokens after the all-to-all and put zero placeholders
        # back before the reverse all-to-all.
        paged_sp_padding = 0
        paged_sp_padding_offset = query.shape[1]
        if use_paged_attention and strategy is not self._no_parallel_strategy and strategy.name == "ulysses":
            forward_ctx = get_forward_context() if is_forward_context_available() else None
            paged_sp_padding = int(getattr(forward_ctx, "sp_padding_size", 0))
            if paged_sp_padding:
                joint_len = int(getattr(ctx, "joint_len", 0))
                joint_strategy = str(getattr(ctx, "joint_strategy", "front"))
                paged_sp_padding_offset = query.shape[1] - (joint_len if joint_strategy == "rear" else 0)
                padding_start = paged_sp_padding_offset - paged_sp_padding
                if padding_start < 0:
                    raise ValueError(
                        "Paged Ulysses padding exceeds the post-all-to-all sequence: "
                        f"padding={paged_sp_padding}, sequence={query.shape[1]}"
                    )

                def _remove_paged_sp_padding(tensor: torch.Tensor) -> torch.Tensor:
                    return torch.cat(
                        (tensor[:, :padding_start], tensor[:, paged_sp_padding_offset:]),
                        dim=1,
                    ).contiguous()

                query = _remove_paged_sp_padding(query)
                key = _remove_paged_sp_padding(key)
                value = _remove_paged_sp_padding(value)

        # 2. This is the shared GPU/NPU boundary. The Worker adapter prepares
        # the native page-table context after SP has produced rank-local Q/K/V;
        # backend resolution below selects CUDA or Ascend execution.
        if use_paged_attention:
            assert paged_adapter is not None
            paged_kv_context = paged_adapter.prepare_layer_context(
                self.prefix,
                query,
                key,
                value,
                omni_attn_metadata=attn_metadata,
            )
            out = self.attention.forward_paged(paged_kv_context)
        else:
            attn_metadata = self._with_kv_cache_dtype(attn_metadata)
            if self.use_ring and strategy is not self._no_parallel_strategy:
                out = self._run_ring_attention(query, key, value, attn_metadata)
            else:
                out = self._run_local_attention(query, key, value, attn_metadata)

        if paged_sp_padding:
            padding_start = paged_sp_padding_offset - paged_sp_padding
            output_padding = out.new_zeros((out.shape[0], paged_sp_padding, *out.shape[2:]))
            out = torch.cat((out[:, :padding_start], output_padding, out[:, padding_start:]), dim=1)

        # 3. Post-processing (Reverse Communication)
        # For Ulysses: AllToAll Output, and AllGather Joint Output
        out = strategy.post_attention(out, ctx)

        return out

    @staticmethod
    def _active_paged_kv_adapter():
        """Return the Worker adapter selected by Runner-owned metadata."""

        if not is_forward_context_available():
            return None
        return getattr(get_forward_context(), "paged_kv_adapter", None)

    def is_paged_kv_active(self) -> bool:
        """Return whether this layer will use Scheduler-managed paged KV."""

        return self.paged_kv_cache_role is not None and self._active_paged_kv_adapter() is not None

    def _run_local_attention(self, query, key, value, attn_metadata):
        self._assert_metadata_compatible(attn_metadata)

        in_kv_memory_profile = is_forward_context_available() and get_forward_context().in_diffusion_kv_memory_profile
        # The startup KV-capacity profile needs tensor shapes, not a paged
        # attention result. If dense FLASH_ATTN deps are absent (NPU MindIE-SD
        # or CUDA CuTe FA4), SDPA provides that profile forward. Formal paged
        # requests never use this branch because their Worker adapter is active.
        # Do not silently remap float32 (or other dtypes) to SDPA: a
        # user-explicit backend must run or raise.
        if (
            self._scheduler_paged_kv
            and self.paged_kv_cache_role is not None
            and in_kv_memory_profile
            and self.attn_backend.get_name() == "FLASH_ATTN"
            and not current_omni_platform.supports_diffusion_dense_flash_attention()
        ):
            logger.warning_once(
                "The startup KV memory profile is using SDPA because dense FLASH_ATTN is unavailable. "
                "Formal paged requests still use the platform-native paged attention backend."
            )
            return self.sdpa_fallback.forward(query, key, value, attn_metadata)

        return self.attention.forward(query, key, value, attn_metadata)

    def _assert_metadata_compatible(self, attn_metadata: AttentionMetadata | None) -> None:
        if attn_metadata is None:
            return
        backend_name = self.attn_backend.get_name()
        if attn_metadata.attn_mask is not None and not self.attn_backend.supports_attention_mask(
            getattr(self, "attn_spec", None)
        ):
            raise ValueError(
                f"Attention backend '{backend_name}' does not support attn_mask. Select a mask-capable backend."
            )
        if attn_metadata.full_attn_spans is None:
            return
        if attn_metadata.attn_mask is not None and attn_metadata.attn_mask.ndim == 4:
            return
        if not self.attn_backend.supports_piecewise_spans:
            raise ValueError(
                f"Attention backend '{backend_name}' does not support "
                f"piecewise attention (full_attn_spans without a 4D attn_mask). "
                f"Use a Flash backend (FLASH_ATTN / FLASH_ATTN_HUB / FLASH_ATTN_3_HUB), "
                f"or provide a 4D attn_mask that encodes the mixed causal/full pattern."
            )

    def _run_ring_attention(self, query, key, value, attn_metadata):
        if attn_metadata is not None and attn_metadata.attn_mask is not None:
            raise ValueError("Ring attention does not support attn_mask; use Ulysses SP or disable mask_sp_padding.")
        skip = getattr(self.attention, "skip", None)
        if skip is not None and getattr(skip, "configured", False):
            raise NotImplementedError(
                "Skip-Softmax (TRTLLM_ATTN) is not supported with ring sequence parallelism: "
                "the ring path bypasses the backend, so the skip config would be silently ignored. "
                "Use Ulysses SP instead, or remove the skip_softmax config."
            )
        # Delegate to RingParallelAttention strategy if available
        if self.ring_runner is not None:
            return self.ring_runner.run_attention(
                query, key, value, attn_metadata, softmax_scale=self.softmax_scale, causal=self.causal
            )

        raise RuntimeError("Ring attention is enabled but strategy is not RingParallelAttention")
