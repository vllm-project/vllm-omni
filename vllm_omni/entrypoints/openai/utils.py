# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Shared OpenAI serving utilities.

PUT HERE:
  - Cross-cutting stage-config / LoRA / pipeline helpers reused by multiple
    OpenAI serving paths (not owned by a single endpoint family).

DO NOT PUT HERE:
  - Modality endpoint request/job helpers — those go in package ``helpers.py``
    (e.g. ``images.helpers``, ``video.generation.helpers``).
  - Pure image/video/audio media encode/decode — those stay in the matching
    root ``*_utils*`` / mixin until family PRs absorb them.

LONGEVITY:
  - TODO(#5227, Phase 1): revisit and tidy this file during later OpenAI
    entrypoint refactor stages; prefer growing endpoint-family helpers over this
    root grab-bag when ownership is clear.
"""

from __future__ import annotations

from typing import Any

from vllm_omni.lora.request import LoRARequest
from vllm_omni.lora.utils import stable_lora_int_id


def get_stage_type(stage_cfg: Any) -> str:
    """Best-effort stage type resolver across dict/omegaconf/object configs."""
    if isinstance(stage_cfg, dict):
        return stage_cfg.get("stage_type", "llm")
    if hasattr(stage_cfg, "get"):
        try:
            return stage_cfg.get("stage_type", "llm")
        except Exception:
            pass
    return getattr(stage_cfg, "stage_type", "llm")


def is_image_generation_stage(stage_cfg: Any) -> bool:
    """Return whether a stage produces the pipeline's final image output.

    Covers both classical diffusion stages (``stage_type == "diffusion"``) and
    generation-LLM stages that emit images (``stage_type == "llm"`` with
    ``final_output_type == "image"`` — the legacy topology MammothModa2 used
    before #7134 migrated it to the shared diffusion runtime). Serving
    endpoints use this so image routes accept either topology.
    """
    if get_stage_type(stage_cfg) == "diffusion":
        return True
    final_output_type = None
    if isinstance(stage_cfg, dict):
        final_output = stage_cfg.get("final_output", False)
        final_output_type = stage_cfg.get("final_output_type")
    elif hasattr(stage_cfg, "get"):
        try:
            final_output = stage_cfg.get("final_output", False)
            final_output_type = stage_cfg.get("final_output_type")
        except Exception:
            final_output = getattr(stage_cfg, "final_output", False)
            final_output_type = getattr(stage_cfg, "final_output_type", None)
    else:
        final_output = getattr(stage_cfg, "final_output", False)
        final_output_type = getattr(stage_cfg, "final_output_type", None)
    return bool(final_output) and final_output_type in {"image", "images"}


def is_video_generation_pipeline(stage_configs: list[Any] | None) -> bool:
    """Return whether a pipeline declares a final video output stage."""
    for stage in stage_configs or ():
        if isinstance(stage, dict):
            final_output = stage.get("final_output", False)
            final_output_type = stage.get("final_output_type")
        elif hasattr(stage, "get"):
            try:
                final_output = stage.get("final_output", False)
                final_output_type = stage.get("final_output_type")
            except Exception:
                final_output = getattr(stage, "final_output", False)
                final_output_type = getattr(stage, "final_output_type", None)
        else:
            final_output = getattr(stage, "final_output", False)
            final_output_type = getattr(stage, "final_output_type", None)
        if final_output and final_output_type in {"video", "videos"}:
            return True
    return False


def parse_lora_request(lora_body: Any) -> tuple[LoRARequest | None, float | None]:
    """Parse a request-level LoRA object into a LoRARequest and optional scale.

    Raises:
        ValueError: If the object shape is invalid or required fields are missing.
    """
    if lora_body is None:
        return None, None

    if not isinstance(lora_body, dict):
        raise ValueError("Invalid lora field: expected an object.")

    lora_name = lora_body.get("name") or lora_body.get("lora_name") or lora_body.get("adapter")
    lora_path = (
        lora_body.get("local_path")
        or lora_body.get("path")
        or lora_body.get("lora_path")
        or lora_body.get("lora_local_path")
    )
    lora_scale = lora_body.get("scale")
    if lora_scale is None:
        lora_scale = lora_body.get("lora_scale")
    lora_int_id = lora_body.get("int_id")
    if lora_int_id is None:
        lora_int_id = lora_body.get("lora_int_id")
    if lora_int_id is None and lora_path:
        lora_int_id = stable_lora_int_id(str(lora_path))

    if not lora_name or not lora_path:
        raise ValueError("Invalid lora object: both name and path are required.")

    scale = float(lora_scale) if lora_scale is not None else None
    return LoRARequest(str(lora_name), int(lora_int_id), str(lora_path)), scale


def parse_lora_requests(
    lora_body: Any,
) -> tuple[LoRARequest | list[LoRARequest] | None, float | list[float] | None]:
    """Parse a request-level LoRA object or array of objects.

    Accepts the legacy single-object shape (see ``parse_lora_request``) or a
    non-empty list of such objects to stack multiple adapters on one request.
    A single object (or a one-element array) keeps the legacy scalar
    ``(LoRARequest, float)`` shape; only a genuine stack of 2+ adapters
    produces lists. Per-adapter scales default to 1.0 when omitted.

    Raises:
        ValueError: If any object shape is invalid or required fields are missing.
    """
    if lora_body is None:
        return None, None

    items = lora_body if isinstance(lora_body, list) else [lora_body]
    if not items:
        raise ValueError("Invalid lora field: expected a non-empty array of objects.")

    lora_requests: list[LoRARequest] = []
    lora_scales: list[float] = []
    for item in items:
        lora_request, lora_scale = parse_lora_request(item)
        if lora_request is None:
            raise ValueError("Invalid lora field: array items must be objects.")
        lora_requests.append(lora_request)
        lora_scales.append(lora_scale if lora_scale is not None else 1.0)
    if len(lora_requests) == 1:
        return lora_requests[0], lora_scales[0]
    return lora_requests, lora_scales


def resolve_diffusion_od_config(engine_client: Any, diffusion_engine: Any = None) -> Any:
    """Resolve the OmniDiffusionConfig from the engine or diffusion engine."""
    od_config = None
    if hasattr(engine_client, "get_diffusion_od_config"):
        od_config = engine_client.get_diffusion_od_config()
    if od_config is None and diffusion_engine is not None:
        if hasattr(diffusion_engine, "get_diffusion_od_config"):
            od_config = diffusion_engine.get_diffusion_od_config()
        else:
            od_config = getattr(diffusion_engine, "od_config", None)
    return od_config


def is_single_stage_diffusion(engine_client: Any) -> bool:
    """Return True if the engine is a single-stage diffusion pipeline."""
    stage_configs = getattr(engine_client, "stage_configs", None) or []
    if len(stage_configs) != 1:
        return False
    return getattr(stage_configs[0], "stage_type", None) in ("diffusion", "DIFFUSION")
