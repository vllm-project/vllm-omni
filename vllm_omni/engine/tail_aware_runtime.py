# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Validate head-side admission before launching unchanged stage processes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from vllm_omni.config.omni_config import BaseVllmOmniStageConfig, VllmOmniDiffusionStageConfig


def prepare_tail_aware_stages(
    stages: list[BaseVllmOmniStageConfig],
    settings: dict[str, Any] | None,
    *,
    model: str,
    distributed: bool = False,
    async_chunk: bool = False,
    session_mode: str = "turn",
    api_client_count: int = 1,
) -> str | None:
    """Validate the production typed stages and return the calibrated model class.

    Admission state has one local owner; reject unsupported layouts before
    launching any stage process.
    """
    if not settings or not settings.get("enabled", False):
        return None
    if distributed or api_client_count != 1:
        raise ValueError("Tail-aware scheduling requires one local head with locally managed replicas")
    if len(stages) != 1 or async_chunk or session_mode != "turn":
        raise ValueError("Tail-aware scheduling currently supports one non-streaming diffusion stage")
    if stages[0].stage_type != "diffusion":
        raise ValueError("Tail-aware scheduling can only be enabled for a diffusion stage")
    stage = cast("VllmOmniDiffusionStageConfig", stages[0])
    execution = stage.diffusion_config
    if execution.streaming_output:
        raise ValueError("Tail-aware scheduling does not support streaming diffusion output")
    if (
        execution.custom_pipeline_args is not None
        or execution.diffusion_load_format == "diffusers"
        or execution.engine_backend not in (None, "default")
    ):
        raise ValueError("Tail-aware scheduling requires the default native diffusion engine and pipeline")
    model_class = execution.model_class_name
    if model_class is None:
        from vllm_omni.diffusion.data import resolve_model_class_name

        # Match worker projection: non-None model_config fields override the
        # diffusion projection, then an empty model falls back to the root.
        model_config = stage.model_config
        model_name = model_config.model if model_config.model is not None else execution.model
        revision = model_config.revision if model_config.revision is not None else execution.revision
        model_class = resolve_model_class_name(
            model=model_name or model,
            revision=revision,
            diffusion_load_format=execution.diffusion_load_format or "default",
        )
    if model_class not in {"QwenImagePipeline", "WanPipeline", "Wan22Pipeline"}:
        raise ValueError(f"Tail-aware scheduling is not supported for model class {model_class!r}")
    return model_class
