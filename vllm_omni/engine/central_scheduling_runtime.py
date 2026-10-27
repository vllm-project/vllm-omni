# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Validate head-side admission before launching unchanged stage processes."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from vllm_omni.diffusion.sched.dispatch.config import CentralSchedulingConfig

if TYPE_CHECKING:
    from vllm_omni.config.omni_config import BaseVllmOmniStageConfig, VllmOmniDiffusionStageConfig


def prepare_central_scheduling_stages(
    stages: list[BaseVllmOmniStageConfig],
    settings: CentralSchedulingConfig,
    *,
    model: str,
    distributed: bool = False,
    async_chunk: bool = False,
    session_mode: str = "turn",
    api_client_count: int = 1,
) -> str | None:
    """Validate the production typed stages and return the supported model class.

    Admission state has one local owner; reject unsupported layouts before
    launching any stage process.
    """
    if not settings.enabled:
        return None
    if distributed or api_client_count != 1:
        raise ValueError("Central scheduling requires one local head with locally managed replicas")
    if len(stages) != 1 or async_chunk or session_mode != "turn":
        raise ValueError("Central scheduling currently supports one non-streaming diffusion stage")
    if stages[0].stage_type != "diffusion":
        raise ValueError("Central scheduling can only be enabled for a diffusion stage")
    stage = cast("VllmOmniDiffusionStageConfig", stages[0])
    execution = stage.diffusion_config
    if execution.streaming_output:
        raise ValueError("Central scheduling does not support streaming diffusion output")
    if (
        execution.custom_pipeline_args is not None
        or execution.diffusion_load_format == "diffusers"
        or execution.engine_backend not in (None, "default")
    ):
        raise ValueError("Central scheduling requires the default native diffusion engine and pipeline")
    model_config = stage.model_config
    model_name = model_config.model if model_config.model is not None else execution.model
    revision = model_config.revision if model_config.revision is not None else execution.revision
    model_class = execution.model_class_name
    if model_class is None:
        from vllm_omni.diffusion.data import resolve_model_class_name

        # Match worker projection: non-None model_config fields override the
        # diffusion projection, then an empty model falls back to the root.
        model_class = resolve_model_class_name(
            model=model_name or model,
            revision=revision,
            diffusion_load_format=execution.diffusion_load_format or "default",
        )
    from vllm_omni.diffusion.sched.dispatch.estimation import validate_service_time_profile

    validate_service_time_profile(model_class, settings.hardware_profile)
    if model_class in {"WanPipeline", "Wan22Pipeline"}:
        from vllm_omni.diffusion.utils.hf_utils import get_diffusion_model_index

        try:
            index = get_diffusion_model_index(model_name or model, revision=revision)
        except Exception as exc:
            raise ValueError("Central Wan scheduling requires readable standard T2V model metadata") from exc
        if (
            not index
            or index.get("_class_name") not in {"WanPipeline", "Wan22Pipeline"}
            or index.get("expand_timesteps", False)
            or any(index.get("image_encoder") or ())
        ):
            raise ValueError("Central Wan scheduling supports standard T2V only, not TI2V, I2V or DMD")
    return model_class
