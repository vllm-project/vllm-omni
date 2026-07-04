# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import dataclasses
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclasses.dataclass(frozen=True)
class _VoxCPM2RuntimeConfig:
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
    cfg_cutoff_ratio: float = 1.0
    decode_graph_capture_policy: str = "all"
    enable_vae_cuda_graph: bool = False
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
    enable_batched_fsq_fusion: bool = False
    batched_fsq_fusion_max_batch: int = 32
    enable_batched_prefill_tail: bool = False
    enable_unified_decode_graph: bool = True
    unified_decode_graph_max_batch_size: int = 64

    @classmethod
    def from_vllm_config(cls, vllm_config: Any) -> _VoxCPM2RuntimeConfig:
        model_config = vllm_config.model_config
        raw = getattr(model_config, "voxcpm2_runtime_config", None)
        if raw is None:
            raw = getattr(getattr(model_config, "hf_config", None), "voxcpm2_runtime_config", None)
        if raw is None:
            return cls()
        if hasattr(raw, "to_dict"):
            raw = raw.to_dict()
        elif not isinstance(raw, dict) and hasattr(raw, "__dict__"):
            raw = vars(raw)
        if not isinstance(raw, dict):
            logger.warning("Ignoring invalid voxcpm2_runtime_config=%r; expected a dict.", raw)
            return cls()

        fields = {field.name: field for field in dataclasses.fields(cls)}
        values: dict[str, Any] = {}
        for key, value in raw.items():
            if key not in fields:
                logger.warning("Ignoring unknown VoxCPM2 runtime config key: %s", key)
                continue
            default = getattr(cls(), key)
            values[key] = cls._coerce_value(key, value, default)
        return cls(**values)._normalized()

    def _normalized(self) -> _VoxCPM2RuntimeConfig:
        cfg = self
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
        if isinstance(default, bool):
            if isinstance(value, str):
                return value.strip().lower() in ("1", "true", "yes", "on")
            return bool(value)
        if isinstance(default, int) and not isinstance(default, bool):
            value = int(value)
            if key in {"audio_emit_every", "vae_decode_every", "batched_fsq_fusion_max_batch"}:
                return max(1, value)
            return value
        if isinstance(default, float):
            if key == "cfg_cutoff_ratio":
                return min(1.0, max(0.0, float(value)))
            return float(value)
        if isinstance(default, str):
            return str(value)
        return value
