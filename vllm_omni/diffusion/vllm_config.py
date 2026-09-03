# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Build native vLLM configuration views for the Omni diffusion runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from vllm.config import CompilationConfig, DeviceConfig, VllmConfig
from vllm.transformers_utils.config import get_hf_text_config

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.diffusion_kv.config import DiffusionKVCacheMode
from vllm_omni.platforms import current_omni_platform


def resolve_diffusion_max_model_len(od_config: OmniDiffusionConfig) -> int:
    """Resolve the native KV admission bound for a diffusion stage."""

    configured = getattr(od_config, "max_model_len", None)
    if configured is not None:
        configured = int(configured)
        if configured == -1:
            configured = None
        elif configured <= 0:
            raise ValueError(f"max_model_len must be positive for Scheduler-managed Diffusion KV, got {configured}")
        else:
            return configured

    # ``max_model_len`` keeps the native vLLM meaning: the per-sequence KV
    # admission ceiling consumed by ``KVCacheManager``. HunyuanImage3 exposes
    # that ceiling directly as ``max_position_embeddings`` because its text,
    # reference-image, and generated-image tokens share one Transformer
    # sequence. Other diffusion models commonly express their serving limits
    # through resolution, duration, or frame-count constraints instead. When
    # paged KV support expands to those models, the model integration must
    # derive ``max_model_len`` from its serving limits, while model-owned
    # preprocessing validates each request and derives its matching
    # ``seq_len``. Text-encoder length fields must not be mistaken for the DiT
    # KV limit.
    hf_config = getattr(od_config, "tf_model_config", None)
    for field_name in ("max_position_embeddings", "max_sequence_length", "seq_length"):
        value = getattr(hf_config, field_name, None)
        if value is None and isinstance(hf_config, dict):
            value = hf_config.get(field_name)
        if value is not None and int(value) > 0:
            return int(value)

    mode = getattr(getattr(od_config, "diffusion_kv_mode", None), "value", None)
    if mode == "paged_scheduler":
        raise ValueError(
            "paged_scheduler Diffusion KV requires max_model_len or a model config "
            "with max_position_embeddings/max_sequence_length/seq_length"
        )
    # Dense execution does not consume the native KV admission bound, but the
    # shared Worker VllmConfig still requires a positive placeholder.
    return 1


@dataclass
class _DiffusionVllmModelConfig:
    """Private adapter for the native ``VllmConfig.model_config`` surface."""

    model: str | None
    dtype: torch.dtype
    max_model_len: int
    original_max_model_len: int | None
    runner_type: str = "generate"
    quantization: str | None = None
    quantization_config: Any | None = None
    hf_config: Any | None = None
    hf_text_config: Any | None = None
    multimodal_config: Any | None = None
    enforce_eager: bool = False
    disable_cascade_attn: bool = False
    enable_return_routed_experts: bool = False
    enable_prompt_embeds: bool = False
    logits_processors: tuple[str, ...] = ()
    use_mla: bool = False
    is_moe: bool = False
    # Native vLLM attention layers/builders read these fields directly.  The
    # diffusion model does not have a full vLLM ModelConfig, so keep the
    # narrow compatibility surface here instead of manufacturing a second
    # model configuration object in the Worker backend.
    is_mm_prefix_lm: bool = False
    rswa_window: int | None = None
    _attention_num_heads: int | None = field(default=None, init=False, repr=False)
    _attention_num_kv_heads: int | None = field(default=None, init=False, repr=False)
    _attention_head_size: int | None = field(default=None, init=False, repr=False)

    # Needed for models that bundle things like LogitsProcessors, e.g., SenseNova
    head_dtype: torch.dtype | None = None

    @property
    def is_quantized(self) -> bool:
        return self.quantization is not None

    def is_nvfp4_quantized(self) -> bool:
        return self.quantization == "modelopt_fp4"

    def set_attention_geometry(
        self,
        *,
        num_heads: int,
        num_kv_heads: int,
        head_size: int,
    ) -> None:
        """Record rank-local geometry needed by native metadata builders.

        Diffusion attention modules are loaded outside vLLM's model loader, so
        their dimensions are not available when this lightweight config is
        created.  The paged Worker backend fills them in after model loading
        and before calling ``init_attn_backend``.
        """

        values = {
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "head_size": head_size,
        }
        invalid = {name: value for name, value in values.items() if type(value) is not int or value <= 0}
        if invalid:
            raise ValueError(f"Diffusion attention geometry must contain positive integers: {invalid!r}")
        self._attention_num_heads = num_heads
        self._attention_num_kv_heads = num_kv_heads
        self._attention_head_size = head_size

    def _require_attention_geometry(self) -> tuple[int, int, int]:
        geometry = (
            self._attention_num_heads,
            self._attention_num_kv_heads,
            self._attention_head_size,
        )
        if any(value is None for value in geometry):
            raise RuntimeError("Diffusion attention geometry is unavailable before paged cache layer registration")
        num_heads, num_kv_heads, head_size = geometry
        assert num_heads is not None and num_kv_heads is not None and head_size is not None
        return num_heads, num_kv_heads, head_size

    def get_num_attention_heads(self, parallel_config: Any) -> int:
        del parallel_config
        return self._require_attention_geometry()[0]

    def get_num_kv_heads(self, parallel_config: Any) -> int:
        del parallel_config
        return self._require_attention_geometry()[1]

    def get_head_size(self) -> int:
        return self._require_attention_geometry()[2]

    @property
    def is_diffusion(self) -> bool:
        # This config is also installed as the current VllmConfig for the
        # existing Omni diffusion runtime, which does not use vLLM's native
        # diffusion runner. Preserve that established backend-selection flag.
        return False


def _make_diffusion_vllm_model_config(od_config: OmniDiffusionConfig) -> _DiffusionVllmModelConfig:
    quant_config = getattr(od_config, "quantization_config", None)
    quantization = quant_config.get_name() if quant_config is not None and hasattr(quant_config, "get_name") else None
    hf_config = getattr(od_config, "tf_model_config", None)
    hf_text_config = get_hf_text_config(hf_config) if hasattr(hf_config, "get_text_config") else hf_config
    configured_max_model_len = getattr(od_config, "max_model_len", None)
    max_model_len = resolve_diffusion_max_model_len(od_config)
    paged_scheduler = (
        getattr(od_config, "diffusion_kv_mode", DiffusionKVCacheMode.DENSE_LEGACY)
        is DiffusionKVCacheMode.PAGED_SCHEDULER
    )
    return _DiffusionVllmModelConfig(
        model=od_config.model,
        dtype=od_config.dtype,
        max_model_len=max_model_len,
        # Preserve the user's -1 sentinel only for the native paged-cache
        # sizing flow. Dense keeps its established resolved model-config view.
        original_max_model_len=configured_max_model_len if paged_scheduler else max_model_len,
        quantization=quantization,
        quantization_config=quant_config,
        hf_config=hf_config,
        hf_text_config=hf_text_config,
        enforce_eager=getattr(od_config, "enforce_eager", False),
        is_moe=bool(getattr(od_config, "is_moe", False)),
    )


def create_base_diffusion_vllm_config(
    device: torch.device,
    od_config: OmniDiffusionConfig,
) -> VllmConfig:
    """Create the native vLLM 0.27 config used by diffusion."""

    return VllmConfig(
        compilation_config=CompilationConfig(),
        device_config=DeviceConfig(device=device),
        additional_config=od_config.additional_config,
    )


def configure_diffusion_vllm_config(vllm_config: VllmConfig, od_config: OmniDiffusionConfig) -> VllmConfig:
    """Populate native model, parallel, cache, and Scheduler configuration."""

    parallel_config = od_config.parallel_config
    vllm_config.parallel_config.tensor_parallel_size = parallel_config.tensor_parallel_size
    vllm_config.parallel_config.data_parallel_size = parallel_config.data_parallel_size
    if parallel_config.enable_expert_parallel and od_config.is_moe:
        vllm_config.parallel_config.data_parallel_size = (
            parallel_config.data_parallel_size * parallel_config.cfg_parallel_size
        )
        vllm_config.parallel_config.prefill_context_parallel_size = parallel_config.sequence_parallel_size
    vllm_config.parallel_config.enable_expert_parallel = parallel_config.enable_expert_parallel

    vllm_config.model_config = _make_diffusion_vllm_model_config(od_config)  # type: ignore[assignment]
    vllm_config.quant_config = od_config.quantization_config
    vllm_config.profiler_config = od_config.profiler_config
    if (
        getattr(od_config, "diffusion_kv_mode", DiffusionKVCacheMode.DENSE_LEGACY)
        is DiffusionKVCacheMode.PAGED_SCHEDULER
    ):
        vllm_config.parallel_config.pipeline_parallel_size = parallel_config.pipeline_parallel_size
        vllm_config.cache_config.gpu_memory_utilization = float(getattr(od_config, "gpu_memory_utilization", 0.9))
        vllm_config.cache_config.kv_cache_memory_bytes = getattr(od_config, "kv_cache_memory_bytes", None)
        # Prefix hashes/publication are intentionally deferred to the next PR.
        vllm_config.cache_config.enable_prefix_caching = False
        vllm_config.scheduler_config.max_num_seqs = int(od_config.max_num_seqs)
        max_num_batched_tokens = getattr(od_config, "max_num_batched_tokens", None)
        if max_num_batched_tokens is not None:
            vllm_config.scheduler_config.max_num_batched_tokens = int(max_num_batched_tokens)
        current_omni_platform.configure_diffusion_vllm_config(vllm_config, od_config)
    return vllm_config


def create_diffusion_vllm_config(
    device: torch.device,
    od_config: OmniDiffusionConfig,
) -> VllmConfig:
    """Build the same fully populated native config in Engine and Workers."""

    vllm_config = create_base_diffusion_vllm_config(
        device,
        od_config,
    )
    return configure_diffusion_vllm_config(vllm_config, od_config)
