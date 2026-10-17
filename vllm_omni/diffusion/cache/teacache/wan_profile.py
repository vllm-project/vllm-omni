# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Explicit Wan calibration profiles. No production-qualified default yet."""

import math
from pathlib import Path
from typing import Any

from vllm_omni.diffusion.cache.teacache.config import TeaCacheConfig

WAN_MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
WAN_MODEL_REVISION = "0fad780a534b6463e45facd96134c9f345acfa5b"


def stage_config(
    profile: dict[str, Any], *, model: str, revision: str | None, pp_size: int, cfg_size: int, pp_rank: int
) -> TeaCacheConfig:
    """Resolve a topology-local profile, refusing unknown model provenance."""
    if (profile.get("model_id"), profile.get("revision"), profile.get("dtype")) != (
        WAN_MODEL_ID,
        WAN_MODEL_REVISION,
        "bfloat16",
    ):
        raise ValueError("Wan TeaCache profile must identify the pinned Wan2.1 1.3B BF16 model")
    if profile.get("pp_size") != pp_size or profile.get("cfg_size") != cfg_size:
        raise ValueError("Wan TeaCache profile topology does not match the pipeline")
    if pp_size not in (1, 2) or cfg_size not in (1, 2) or pp_rank not in range(pp_size):
        raise ValueError("Unsupported Wan TeaCache stage topology")
    path = Path(model)
    if path.is_dir():
        resolved = path.resolve()
        if resolved.parent.name == "snapshots":
            actual_revision = resolved.name
        else:
            metadata = path / ".cache/huggingface/download/model_index.json.metadata"
            metadata_lines = metadata.read_text().splitlines() if metadata.is_file() else []
            actual_revision = metadata_lines[0] if metadata_lines else None
        if actual_revision != WAN_MODEL_REVISION:
            raise ValueError("Local Wan model must retain its pinned Hugging Face snapshot/download metadata")
    elif model != WAN_MODEL_ID or revision != WAN_MODEL_REVISION:
        raise ValueError("Remote Wan model requires the exact model ID and revision from its profile")
    stages = profile.get("stages", {})
    if set(stages) != {str(rank) for rank in range(pp_size)}:
        raise ValueError("Wan TeaCache profile must contain exactly one entry per PP stage")
    configs = []
    for rank in range(pp_size):
        stage = stages[str(rank)]
        coefficients = stage.get("coefficients")
        if (
            not isinstance(coefficients, list)
            or len(coefficients) != 5
            or not all(
                isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in coefficients
            )
        ):
            raise ValueError("Wan stage coefficients must contain five finite numbers")
        if stage.get("max_input_distance") is None or stage.get("threshold") is None:
            raise ValueError("Wan stage requires a calibrated input bound and threshold")
        configs.append(
            TeaCacheConfig(
                transformer_type="WanTransformer3DModel",
                coefficients=coefficients,
                rel_l1_thresh=stage["threshold"],
                max_input_distance=stage["max_input_distance"],
                warmup_steps=stage.get("warmup_steps", 0),
                clamp_negative_prediction=True,
            )
        )
    return configs[pp_rank]
