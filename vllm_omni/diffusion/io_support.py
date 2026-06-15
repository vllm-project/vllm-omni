# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import inspect

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.registry import DiffusionModelRegistry


def supports_multimodal_input(od_config: OmniDiffusionConfig) -> tuple[bool, bool]:
    if od_config.diffusion_load_format == "diffusers" and (pipe_cls := od_config.diffusers_pipeline_cls) is not None:
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


def get_dummy_run_num_frames(model_class_name: str, supports_audio_input: bool) -> int:
    """Get num_frames for the dummy warmup run. Returns 0 to skip warmup."""

    model_cls = DiffusionModelRegistry._try_load_model_cls(model_class_name)
    if model_cls is not None and hasattr(model_cls, "dummy_run_num_frames"):
        return int(getattr(model_cls, "dummy_run_num_frames"))
    return 2 if supports_audio_input or supports_audio_output(model_class_name) else 1


def get_extra_body_params(model_class_name: str) -> frozenset[str]:
    """Return the set of extra_body keys accepted by a pipeline.

    Each pipeline can declare ``EXTRA_BODY_PARAMS: ClassVar[frozenset[str]]``
    to advertise which request-level parameters should be forwarded from
    ``extra_body`` to ``OmniDiffusionSamplingParams.extra_args``.
    Returns an empty frozenset when the pipeline does not declare any.
    """
    model_cls = DiffusionModelRegistry._try_load_model_cls(model_class_name)
    if model_cls is None:
        return frozenset()
    return frozenset(getattr(model_cls, "EXTRA_BODY_PARAMS", frozenset()))


def get_extra_output_params(model_class_name: str) -> frozenset[str]:
    """Return the set of custom_output keys to expose in API response metrics.

    Each pipeline can declare ``EXTRA_OUTPUT_PARAMS: ClassVar[frozenset[str]]``
    to advertise which ``DiffusionOutput.custom_output`` keys should be
    copied into the response ``metrics`` dict.
    Returns an empty frozenset when the pipeline does not declare any.
    """
    model_cls = DiffusionModelRegistry._try_load_model_cls(model_class_name)
    if model_cls is None:
        return frozenset()
    return frozenset(getattr(model_cls, "EXTRA_OUTPUT_PARAMS", frozenset()))
