# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import inspect

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.model_metadata import get_diffusion_model_metadata
from vllm_omni.diffusion.models.diffusers_adapter.pipeline_utils import resolve_diffusers_pipeline_class
from vllm_omni.diffusion.registry import DiffusionModelRegistry


def supports_multimodal_input(od_config: OmniDiffusionConfig) -> tuple[bool, bool]:
    if (
        od_config.diffusion_load_format == "diffusers"
        and (pipe_cls := resolve_diffusers_pipeline_class(od_config)) is not None
    ):
        signature = inspect.signature(pipe_cls.__call__)
        support_image_input = "image" in signature.parameters
        support_audio_input = (
            "audio" in signature.parameters or "audio_latents" in signature.parameters
        )  # ref. LTX-2 format
        return support_image_input, support_audio_input

    supports_image_input = False
    supports_audio_input = False

    model_cls = DiffusionModelRegistry._try_load_model_cls(od_config.model_class_name)
    if model_cls is not None:
        supports_image_input = bool(getattr(model_cls, "support_image_input", False))
        supports_audio_input = bool(getattr(model_cls, "support_audio_input", False))
    return supports_image_input, supports_audio_input


def image_color_format(model_class_name: str) -> str:
    model_cls = DiffusionModelRegistry._try_load_model_cls(model_class_name)
    return getattr(model_cls, "color_format", "RGB")


def supports_audio_output(model_class_name: str) -> bool:
    model_cls = DiffusionModelRegistry._try_load_model_cls(model_class_name)
    if model_cls is None:
        return False
    return bool(getattr(model_cls, "support_audio_output", False))


def get_diffusion_output_type(model_class_name: str | None) -> str:
    """Return the declared final modality for a diffusion model."""
    declared_output_type = get_diffusion_model_metadata(model_class_name).final_output_type
    if declared_output_type is not None:
        return declared_output_type
    if model_class_name and supports_audio_output(model_class_name):
        return "audio"
    return "image"


def get_dummy_run_num_frames(model_class_name: str, supports_audio_input: bool) -> int:
    """Get num_frames for the dummy warmup run. Returns 0 to skip warmup."""

    model_cls = DiffusionModelRegistry._try_load_model_cls(model_class_name)
    if model_cls is not None and hasattr(model_cls, "dummy_run_num_frames"):
        return int(getattr(model_cls, "dummy_run_num_frames"))
    return 2 if supports_audio_input or supports_audio_output(model_class_name) else 1


def get_dummy_run_num_image_inputs(model_class_name: str) -> int:
    """Return the maximum advertised image-input count for profiling."""

    return get_diffusion_model_metadata(model_class_name).max_multimodal_image_inputs or 1
