# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import dataclasses
import math
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


def reject_outer_full_cudagraph(vllm_config: Any) -> None:
    """Refuse vLLM's outer full CUDA graph capture on the MRV1 runner.

    The talker walks its pending requests and keeps its own CUDA graphs. A full
    outer graph replays the model over a padded batch, which breaks that
    per-request bookkeeping, so the engine dies on the first request. The
    shipped eager default and PIECEWISE are unaffected. The check covers only
    the default MRV1 runner, where the failure was reported.
    """
    if getattr(vllm_config.model_config, "use_v2_model_runner", False):
        return
    compilation_config = getattr(vllm_config, "compilation_config", None)
    mode = getattr(compilation_config, "cudagraph_mode", None)
    if mode is None or not mode.has_full_cudagraphs():
        return
    raise ValueError(
        f"VoxCPM2 does not support cudagraph_mode={mode.name}. The talker owns its own CUDA graphs "
        "and cannot also run inside vLLM's outer full-graph capture. Use "
        '--compilation-config \'{"cudagraph_mode": "PIECEWISE"}\', or drop --no-enforce-eager to use '
        "the shipped default, which is the fastest configuration for this model."
    )


@dataclasses.dataclass(frozen=True)
class _VoxCPM2RuntimeConfig:
    startup_lora_path: str | None = None
    enable_profiling: bool = False
    enable_nvtx_profile: bool = False
    enable_loc_dit_layer_nvtx: bool = False
    enable_loc_dit_fused_qkv: bool = True
    enable_loc_dit_fused_mlp: bool = True
    enable_loc_dit_zero_dt_cache: bool = True
    enable_loc_dit_fast_rope: bool = False
    enable_loc_dit_skip_qkv_contig: bool = True
    enable_loc_dit_reduce_overhead_no_cg: bool = False
    enable_loc_dit_fullgraph_no_cg: bool = False
    inference_timesteps: int = 10
    cfg_value: float = 2.0
    cfg_cutoff_ratio: float = 1.0
    decode_graph_capture_policy: str = "all"
    enable_vae_cuda_graph: bool = False
    enable_incremental_vae: bool = False
    enable_exact_patch_batch: bool = False
    mixed_graph_cache_limit: int = 0  # Zero means unlimited cached mixed graph buckets.
    enable_cfm_cuda_graph: bool = False
    enable_cfm_prealloc_output: bool = False
    enable_batched_cfm: bool = True
    deterministic_cfm_noise: bool = False
    deterministic_cfm_seed: int = 20260601
    audio_emit_every: int = 1
    vae_decode_every: int = 3
    enable_delayed_audio_copy: bool = False
    delayed_audio_copy_use_events: bool = False
    coalesce_audio_d2h: bool = True
    enable_batched_vae_decode: bool = True
    enable_batched_fsq_fusion: bool = True
    batched_fsq_fusion_max_batch: int = 32
    enable_batched_prefill_tail: bool = True
    enable_unified_decode_graph: bool = True
    unified_decode_graph_max_batch_size: int = 64

    @classmethod
    def from_vllm_config(cls, vllm_config: Any) -> _VoxCPM2RuntimeConfig:
        model_config = vllm_config.model_config
        values: dict[str, Any] = {}
        if getattr(model_config, "use_v2_model_runner", False):
            values.update(enable_incremental_vae=True, vae_decode_every=1, audio_emit_every=3)
        raw = getattr(model_config, "voxcpm2_runtime_config", None)
        if raw is None:
            raw = getattr(getattr(model_config, "hf_config", None), "voxcpm2_runtime_config", None)
        if raw is None:
            return cls(**values)._normalized()
        if hasattr(raw, "to_dict"):
            raw = raw.to_dict()
        elif not isinstance(raw, dict) and hasattr(raw, "__dict__"):
            raw = vars(raw)
        if not isinstance(raw, dict):
            logger.warning("Ignoring invalid voxcpm2_runtime_config=%r; expected a dict.", raw)
            return cls(**values)._normalized()

        fields = {field.name: field for field in dataclasses.fields(cls)}
        for key, value in raw.items():
            if key not in fields:
                logger.warning("Ignoring unknown VoxCPM2 runtime config key: %s", key)
                continue
            default = getattr(cls(), key)
            values[key] = cls._coerce_value(key, value, default)
        return cls(**values)._normalized()

    def _normalized(self) -> _VoxCPM2RuntimeConfig:
        cfg = self
        if cfg.enable_exact_patch_batch and (
            not cfg.enable_incremental_vae or cfg.vae_decode_every != 1 or not cfg.enable_unified_decode_graph
        ):
            raise ValueError("Exact patch batches require incremental per-patch unified graphs")
        if cfg.enable_batched_cfm and cfg.enable_cfm_cuda_graph:
            logger.warning(
                "VoxCPM2 batched CFM and CFM CUDA Graph are mutually exclusive; "
                "disabling CFM CUDA Graph and keeping batched CFM."
            )
            cfg = dataclasses.replace(cfg, enable_cfm_cuda_graph=False)
        if cfg.enable_cfm_prealloc_output and not cfg.enable_cfm_cuda_graph:
            logger.warning("VoxCPM2 CFM preallocated output requires CFM CUDA Graph; disabling it.")
            cfg = dataclasses.replace(cfg, enable_cfm_prealloc_output=False)
        if cfg.delayed_audio_copy_use_events and not cfg.enable_delayed_audio_copy:
            logger.warning("VoxCPM2 delayed audio copy events require delayed audio copy; disabling event polling.")
            cfg = dataclasses.replace(cfg, delayed_audio_copy_use_events=False)
        if cfg.enable_incremental_vae and cfg.enable_batched_vae_decode:
            # Incremental unified graphs batch VAE themselves; this flag only
            # controls the legacy latent-cohort collection path.
            cfg = dataclasses.replace(cfg, enable_batched_vae_decode=False)
        if cfg.enable_batched_vae_decode and (
            cfg.enable_delayed_audio_copy or cfg.audio_emit_every > 1 or cfg.enable_vae_cuda_graph
        ):
            logger.warning(
                "VoxCPM2 batched VAE decode is incompatible with delayed audio copy, "
                "audio_emit_every > 1, and VAE CUDA Graph; disabling batched VAE decode."
            )
            cfg = dataclasses.replace(cfg, enable_batched_vae_decode=False)
        if cfg.enable_unified_decode_graph:
            if cfg.enable_cfm_cuda_graph:
                logger.info("VoxCPM2 unified decode graph supersedes independent CFM CUDA Graph; disabling it.")
                cfg = dataclasses.replace(cfg, enable_cfm_cuda_graph=False, enable_cfm_prealloc_output=False)
            if not cfg.enable_batched_cfm:
                logger.info("VoxCPM2 unified decode graph requires batched CFM; enabling it.")
                cfg = dataclasses.replace(cfg, enable_batched_cfm=True)
        return cfg

    def unified_decode_graph_available(self, *, use_cuda_graph: bool) -> bool:
        return bool(use_cuda_graph and self.enable_unified_decode_graph and not self.deterministic_cfm_noise)

    @staticmethod
    def _coerce_value(key: str, value: Any, default: Any) -> Any:
        if key == "startup_lora_path":
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError("VoxCPM2 startup_lora_path must be a non-empty local directory path or null")
            return value
        if isinstance(default, bool):
            if isinstance(value, str):
                return value.strip().lower() in ("1", "true", "yes", "on")
            return bool(value)
        if isinstance(default, int) and not isinstance(default, bool):
            value = int(value)
            if key == "mixed_graph_cache_limit" and value < 0:
                raise ValueError("VoxCPM2 mixed_graph_cache_limit must be non-negative")
            if key == "inference_timesteps":
                return max(2, value)
            if key in {"audio_emit_every", "vae_decode_every", "batched_fsq_fusion_max_batch"}:
                return max(1, value)
            return value
        if isinstance(default, float):
            if key == "cfg_value" and not math.isfinite(float(value)):
                raise ValueError("VoxCPM2 cfg_value must be finite.")
            if key == "cfg_cutoff_ratio":
                return min(1.0, max(0.0, float(value)))
            return float(value)
        if isinstance(default, str):
            return str(value)
        return value
