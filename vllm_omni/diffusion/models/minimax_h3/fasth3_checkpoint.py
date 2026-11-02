# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Sampling contract for FastVideo FastH3 full checkpoints."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from vllm_omni.diffusion.sched.sigma_schedule import DMD2SigmaSchedule
from vllm_omni.errors import OmniClientError
from vllm_omni.model_executor.model_loader.weight_utils import download_weights_from_hf_specific

from .fasth3 import FASTH3_BASE_MODEL

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

FASTH3_V2_MODEL_ID = "FastVideo/FastVideo-FastH3-8-Step-V2"
FASTH3_V2_BASE_SCHEDULE = DMD2SigmaSchedule.from_positions((0.999, 0.874, 0.749, 0.624, 0.5, 0.375, 0.25, 0.125, 0.0))
# The public V2 export omits provenance.json. This is the Omni adapter's native
# VAE compatibility pin, not a claim about the release's training provenance.
FASTH3_V2_NATIVE_VAE_REVISION = "9bfb6693f2cf6de171db46d1aa586f67d773a1da"


@dataclass(frozen=True)
class FastH3CheckpointSpec:
    """The full V2 release has its own schedule and trained attention policy.

    This is independent of ``FastH3WeightFusion``: the released weights already
    contain the entire student, including its learned compression gates.
    """

    name: ClassVar[str] = "FastH3 V2"
    partition: ClassVar[str] = "fl2va"
    transformer_component: ClassVar[str] = "transformer"
    video_shift: ClassVar[float] = 10.0
    vsa_tile_size: ClassVar[int] = 64
    vsa_ref_policy: ClassVar[str | None] = None
    vsa_ref_keep_rate: ClassVar[float | None] = None
    vsa_sparsity: float = 0.8

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, object]) -> FastH3CheckpointSpec:
        """Validate the release's own fastvideo_inference.json."""
        if metadata.get("model_type") == "ref2va":
            return FastH3OmniCheckpointSpec.from_metadata(metadata)
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

    def release_metadata(self) -> dict[str, object]:
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
        directly from the FastVideo release. Honor an explicit provenance pin;
        public V2 exports without that file use the adapter's compatibility pin.
        """
        revision = FASTH3_V2_NATIVE_VAE_REVISION
        provenance_path = model_root / "provenance.json"
        if provenance_path.exists():
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            base = provenance.get("base_model", "")
            prefix = f"hf://{FASTH3_BASE_MODEL}@"
            if not isinstance(base, str) or not base.startswith(prefix) or not base[len(prefix) :]:
                raise ValueError(f"{self.name} provenance must pin a MiniMaxAI/MiniMax-H3 base revision")
            revision = base[len(prefix) :]
        return (
            Path(
                download_weights_from_hf_specific(
                    model_name_or_path=FASTH3_BASE_MODEL,
                    cache_dir=None,
                    allow_patterns=["FL2VA/video_vae/**", "FL2VA/audio_vae/**"],
                    revision=revision,
                    require_all=True,
                )
            )
            / "FL2VA"
        )

    def check_serving_contract(self, *, partition: str, od_config: OmniDiffusionConfig) -> None:
        if partition != self.partition:
            raise ValueError(f"{self.name} requires --task-type {self.partition}")
        if od_config.lora_path:
            raise ValueError(f"{self.name} is a full checkpoint; additional LoRA adapters are unsupported")
        attention, _ = od_config.diffusion_attention_config.resolve_with_source("self")
        if attention is None or attention.backend.upper() != "FASTVIDEO_VSA":
            raise ValueError(f"{self.name} requires --diffusion-attention-backend FASTVIDEO_VSA")
        if attention.fastvideo_vsa_topk is not None:
            raise ValueError(
                f"{self.name} pins VSA sparsity={self.vsa_sparsity}; remove the fixed fastvideo_vsa_topk override"
            )
        parallel = od_config.parallel_config
        if parallel.ring_degree != 1 or parallel.allgather_degree != 1:
            raise ValueError(f"{self.name} supports local attention or pure Ulysses sequence parallelism")

    def check_request(self, sampling: OmniDiffusionSamplingParams, *, step_execution: bool = False) -> None:
        if step_execution:
            # StepScheduler admits requests before the pipeline hook runs and
            # derives its lifetime from these fields.  The pinned schedule must
            # therefore be explicit and cannot be overridden by custom arrays.
            if sampling.num_inference_steps != FASTH3_V2_BASE_SCHEDULE.num_inference_steps:
                raise OmniClientError(f"{self.name} step execution requires num_inference_steps=8")
            if sampling.timesteps is not None or sampling.sigmas is not None:
                raise OmniClientError(f"{self.name} step execution does not support custom timesteps or sigmas")
        if sampling.timesteps is not None or sampling.sigmas is not None:
            raise OmniClientError(f"{self.name} does not support custom timesteps or sigmas")
        if sampling.lora_request is not None:
            raise OmniClientError(f"{self.name} does not support per-request LoRA adapters")
        steps = sampling.num_inference_steps
        if steps is not None and steps != FASTH3_V2_BASE_SCHEDULE.num_inference_steps:
            raise OmniClientError(f"{self.name} requires num_inference_steps=8 (nine sigma points), or omitted")
        extra = sampling.extra_args or {}
        for key, expected in (("flow_shift", self.video_shift), ("audio_flow_shift", 3.0)):
            try:
                value = float(extra.get(key, expected))
            except (TypeError, ValueError) as exc:
                raise OmniClientError(f"{self.name} requires {key}={expected:g}") from exc
            if not math.isclose(value, expected):
                raise OmniClientError(f"{self.name} requires {key}={expected:g}, got {value:g}")
        if sampling.guidance_scale is not None and sampling.guidance_scale != 1.0:
            raise OmniClientError(f"{self.name} requires guidance_scale=1")


@dataclass(frozen=True)
class FastH3OmniCheckpointSpec(FastH3CheckpointSpec):
    """Ref2VA PDD release, with frozen components from a pinned native base."""

    name: ClassVar[str] = "FastH3 Omni"
    partition: ClassVar[str] = "ref2va"
    transformer_component: ClassVar[str] = "transformer_ref"
    video_shift: ClassVar[float] = 12.0
    vsa_tile_size: ClassVar[int] = 128
    vsa_ref_policy: ClassVar[str] = "p2_multi_region"
    vsa_ref_keep_rate: ClassVar[float] = 0.1
    vsa_sparsity: float = 0.9
    base_revision: str = ""

    def check_serving_contract(self, *, partition: str, od_config: OmniDiffusionConfig) -> None:
        super().check_serving_contract(partition=partition, od_config=od_config)
        from vllm_omni.diffusion.attention.ops.block_sparse import resolve_block_sparse_provider
        from vllm_omni.diffusion.distributed.utils import get_local_device

        attention, _ = od_config.diffusion_attention_config.resolve_with_source("self")
        assert attention is not None  # The base contract requires an explicit VSA spec.
        resolve_block_sparse_provider(
            attention.fastvideo_vsa_provider,
            attention.fastvideo_vsa_precision,
            get_local_device(),
            od_config.dtype,
            128,
            block_size=self.vsa_tile_size,
        )

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, object]) -> FastH3OmniCheckpointSpec:
        from vllm_omni.model_executor.models.minimax_h3.checkpoint import fasth3_base_revision

        required = {
            "schema_version": "fasth3-inference-contract-v1",
            "model_type": "ref2va",
            "transformer_component": "transformer_ref",
            "conditioning": "fixed_ordered_references_target_only_flow",
            "pdd_steps": 32,
            "pdd_step_indices": list(range(0, 33, 4)),
            "grid_max_t": 0.999,
            "num_inference_steps": 8,
            "transformer_forwards": 8,
            "video_scheduler_shift": 12.0,
            "audio_scheduler_shift": 3.0,
            "guidance_scale": 1.0,
            "attention_backend": "VIDEO_SPARSE_ATTN_H3",
            "vsa_sparsity": 0.9,
            "vsa_tile_size": 128,
            "vsa_ref_keep_rate": 0.1,
            "vsa_ref_policy": "p2_multi_region",
        }
        for key, expected in required.items():
            if metadata.get(key) != expected:
                raise ValueError(
                    f"unsupported FastH3 Omni contract: {key}={metadata.get(key)!r}, expected {expected!r}"
                )
        return cls(base_revision=fasth3_base_revision(metadata))

    def release_metadata(self) -> dict[str, object]:
        # Used for admission/step counts only; the PDD sampler applies its own
        # 0.999-fixed-point shift, rather than DMD2SigmaSchedule.shifted_sigmas.
        return {
            "partition": "ref2va",
            "tasks": ["ref2va"],
            "sigma_shift_scales": {"video": 12.0, "audio": 3.0},
            "base_schedule": [0.999 * (1 - i / 32) for i in range(0, 33, 4)],
        }

    def resolve_base_components(self, *, load_text_encoder: bool) -> Path:
        patterns = ["Ref2VA/video_vae/**", "Ref2VA/audio_vae/**"]
        if load_text_encoder:
            patterns += ["Ref2VA/text_encoder/**", "Ref2VA/tokenizer/**", "Ref2VA/processor/**"]
        return (
            Path(
                download_weights_from_hf_specific(
                    model_name_or_path=FASTH3_BASE_MODEL,
                    cache_dir=None,
                    allow_patterns=patterns,
                    revision=self.base_revision,
                    require_all=True,
                )
            )
            / "Ref2VA"
        )

    def resolve_native_vaes(self, model_root: Path) -> Path:
        return self.resolve_base_components(load_text_encoder=False)

    def check_request(self, sampling: OmniDiffusionSamplingParams, *, step_execution: bool = False) -> None:
        super().check_request(sampling, step_execution=step_execution)
        extra = sampling.extra_args or {}
        if extra.get("task", "ref2va") != "ref2va":
            raise OmniClientError("FastH3 Omni supports Ref2VA requests only")
        if any(extra.get(key) is not None for key in ("pdd_step_indices", "pdd_steps", "grid_max_t")):
            raise OmniClientError("FastH3 Omni pins its PDD grid in the checkpoint contract")
        if extra.get("latent_refine") not in (None, False):
            raise OmniClientError("FastH3 Omni does not support partial-schedule latent refinement")
