# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Sampling contract for FastVideo FastH3 full checkpoints."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vllm_omni.diffusion.sched.sigma_schedule import DMD2SigmaSchedule
from vllm_omni.errors import OmniClientError
from vllm_omni.model_executor.model_loader.weight_utils import download_weights_from_hf_specific

from .fasth3 import FASTH3_BASE_MODEL, _resolve_dit_attention_backend

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

FASTH3_V2_MODEL_ID = "FastVideo/FastVideo-FastH3-8-Step-V2"
FASTH3_V2_BASE_SCHEDULE = DMD2SigmaSchedule.from_positions((0.999, 0.874, 0.749, 0.624, 0.5, 0.375, 0.25, 0.125, 0.0))


@dataclass(frozen=True)
class FastH3CheckpointSpec:
    """The full V2 release has its own schedule and trained attention policy.

    This is independent of ``FastH3WeightFusion``: the released weights already
    contain the entire student, including its learned compression gates.
    """

    vsa_sparsity: float = 0.8

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, object]) -> FastH3CheckpointSpec:
        """Validate the release's own fastvideo_inference.json."""
        required = {
            "model_id": FASTH3_V2_MODEL_ID,
            "schema_version": "fasth3-inference-contract-v1",
            "dmd_denoising_steps": [999, 874, 749, 624, 500, 375, 250, 125],
            "transformer_forwards": 8,
            "video_scheduler_shift": 10.0,
            "audio_scheduler_shift": 3.0,
            "guidance_scale": 1.0,
            "attention_backend": "VIDEO_SPARSE_ATTN_H3",
            "vsa_sparsity": 0.8,
            "vsa_tile_size": 64,
            "task": "t2av",
        }
        for key, expected in required.items():
            if metadata.get(key) != expected:
                raise ValueError(f"unsupported FastH3 contract: {key}={metadata.get(key)!r}, expected {expected!r}")
        return cls()

    def release_metadata(self) -> dict[str, Any]:
        """Express the sampling policy in the existing H3 pipeline schema."""
        return {
            "partition": "fl2va",
            "tasks": ["t2va"],
            "sigma_shift_scales": {"video": 10.0, "audio": 3.0},
            "base_schedule": list(FASTH3_V2_BASE_SCHEDULE.base_schedule),
        }

    def resolve_native_vaes(self, model_root: Path) -> Path:
        """Reuse the frozen base VAEs with Omni's native tiled/parallel runtime.

        Only VAE components are fetched; the student and text encoder come
        directly from the FastVideo release. The release pins the base revision.
        """
        provenance = json.loads((model_root / "provenance.json").read_text(encoding="utf-8"))
        base = provenance.get("base_model", "")
        prefix = f"hf://{FASTH3_BASE_MODEL}@"
        if not isinstance(base, str) or not base.startswith(prefix) or not base[len(prefix) :]:
            raise ValueError("FastH3 V2 provenance must pin a MiniMaxAI/MiniMax-H3 base revision")
        return (
            Path(
                download_weights_from_hf_specific(
                    model_name_or_path=FASTH3_BASE_MODEL,
                    cache_dir=None,
                    allow_patterns=["FL2VA/video_vae/**", "FL2VA/audio_vae/**"],
                    revision=base[len(prefix) :],
                    require_all=True,
                )
            )
            / "FL2VA"
        )

    def check_serving_contract(self, *, partition: str, od_config: OmniDiffusionConfig) -> None:
        if partition != "fl2va":
            raise ValueError("FastH3 V2 requires --task-type fl2va (T2VA requests only)")
        if getattr(od_config, "lora_path", None):
            raise ValueError("FastH3 V2 is a full checkpoint; additional LoRA adapters are unsupported")
        if _resolve_dit_attention_backend(od_config) != "FASTVIDEO_VSA":
            raise ValueError("FastH3 V2 requires --diffusion-attention-backend FASTVIDEO_VSA")
        attention_config = getattr(od_config, "diffusion_attention_config", None)
        per_role = getattr(attention_config, "per_role", None) or {}
        spec = per_role.get("self") or getattr(attention_config, "default", None)
        if getattr(spec, "fastvideo_vsa_topk", None) is not None:
            raise ValueError("FastH3 V2 pins VSA sparsity=0.8; remove the fixed fastvideo_vsa_topk override")
        parallel = getattr(od_config, "parallel_config", None)
        if any(int(getattr(parallel, key, 1) or 1) != 1 for key in ("ring_degree", "allgather_degree")):
            raise ValueError("FastH3 V2 supports local attention or pure Ulysses sequence parallelism")

    def check_request(self, sampling: OmniDiffusionSamplingParams, *, step_execution: bool = False) -> None:
        if step_execution:
            # StepScheduler admits requests before the pipeline hook runs and
            # derives its lifetime from these fields.  The pinned schedule must
            # therefore be explicit and cannot be overridden by custom arrays.
            if sampling.num_inference_steps != FASTH3_V2_BASE_SCHEDULE.num_inference_steps:
                raise OmniClientError("FastH3 V2 step execution requires num_inference_steps=8")
            if sampling.timesteps is not None or sampling.sigmas is not None:
                raise OmniClientError("FastH3 V2 step execution does not support custom timesteps or sigmas")
        if sampling.lora_request is not None:
            raise OmniClientError("FastH3 V2 does not support per-request LoRA adapters")
        steps = sampling.num_inference_steps
        if steps is not None and steps != FASTH3_V2_BASE_SCHEDULE.num_inference_steps:
            raise OmniClientError("FastH3 V2 requires num_inference_steps=8 (nine sigma points), or omitted")
        extra = sampling.extra_args or {}
        for key, expected in (("flow_shift", 10.0), ("audio_flow_shift", 3.0)):
            try:
                value = float(extra.get(key, expected))
            except (TypeError, ValueError) as exc:
                raise OmniClientError(f"FastH3 V2 requires {key}={expected:g}") from exc
            if not math.isclose(value, expected):
                raise OmniClientError(f"FastH3 V2 requires {key}={expected:g}, got {value:g}")
        if sampling.guidance_scale is not None and sampling.guidance_scale != 1.0:
            raise OmniClientError("FastH3 V2 requires guidance_scale=1")
