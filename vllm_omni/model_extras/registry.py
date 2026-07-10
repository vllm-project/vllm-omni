# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from PIL import Image

from vllm_omni.model_extras.audiox import (
    AUDIOX_EXTRA_BODY_PARAMS,
    AUDIOX_EXTRA_OUTPUT_PARAMS,
)
from vllm_omni.model_extras.bagel import (
    BAGEL_EXTRA_BODY_PARAMS,
    BAGEL_EXTRA_OUTPUT_PARAMS,
    BAGEL_INIT_EXTRA_ARGS_FOR_NON_DIFFUSION_STAGES,
)
from vllm_omni.model_extras.bagel import (
    build_image_to_image_prompt as build_bagel_image_to_image_prompt,
)
from vllm_omni.model_extras.bagel import (
    build_text_to_image_prompt as build_bagel_text_to_image_prompt,
)
from vllm_omni.model_extras.cosmos3 import (
    COSMOS3_EXTRA_BODY_PARAMS,
    COSMOS3_EXTRA_OUTPUT_PARAMS,
)
from vllm_omni.model_extras.cosmos3 import (
    build_text_to_image_prompt as build_cosmos3_text_to_image_prompt,
)
from vllm_omni.model_extras.helios import (
    HELIOS_EXTRA_BODY_PARAMS,
    HELIOS_EXTRA_OUTPUT_PARAMS,
)
from vllm_omni.model_extras.magi_human import (
    MAGI_HUMAN_EXTRA_BODY_PARAMS,
    MAGI_HUMAN_EXTRA_OUTPUT_PARAMS,
)
from vllm_omni.model_extras.mammothmodal2_preview import (
    MAMMOTHMODA2_PREVIEW_EXTRA_BODY_PARAMS,
    MAMMOTHMODA2_PREVIEW_EXTRA_OUTPUT_PARAMS,
    MAMMOTHMODA2_PREVIEW_INIT_EXTRA_ARGS_FOR_NON_DIFFUSION_STAGES,
)
from vllm_omni.model_extras.mammothmodal2_preview import (
    build_text_to_image_prompt as build_mammothmoda2_text_to_image_prompt,
)
from vllm_omni.model_extras.ming_flash_omni import (
    MING_FLASH_OMNI_EXTRA_BODY_PARAMS,
    MING_FLASH_OMNI_EXTRA_OUTPUT_PARAMS,
    MING_FLASH_OMNI_INIT_EXTRA_ARGS_FOR_NON_DIFFUSION_STAGES,
)
from vllm_omni.model_extras.ming_flash_omni import (
    build_image_to_image_prompt as build_ming_flash_omni_image_to_image_prompt,
)
from vllm_omni.model_extras.ming_flash_omni import (
    build_text_to_image_prompt as build_ming_flash_omni_text_to_image_prompt,
)
from vllm_omni.model_extras.sensenova_u1 import (
    SENSENOVA_U1_EXTRA_BODY_PARAMS,
    SENSENOVA_U1_EXTRA_OUTPUT_PARAMS,
)
from vllm_omni.model_extras.vace import (
    VACE_EXTRA_BODY_PARAMS,
    VACE_EXTRA_OUTPUT_PARAMS,
)
from vllm_omni.model_extras.vace import (
    build_image_to_video_prompt as build_vace_image_to_video_prompt,
)

TextToImagePromptBuilder = Callable[
    [str, str | None, int | None, int | None],
    dict[str, Any],
]
ImageToImagePromptBuilder = Callable[
    [str, str | None, "Image.Image | list[Image.Image]", int | None, int | None],
    dict[str, Any],
]
ImageToVideoPromptBuilder = Callable[
    [
        str,
        str | None,
        "Mapping[str, Any]",
        int | None,
        int | None,
        int | None,
    ],
    dict[str, Any],
]


def default_text_to_image_prompt(
    prompt: str,
    negative_prompt: str | None,
    height: int | None = None,
    width: int | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {"prompt": prompt}
    if negative_prompt is not None:
        result["negative_prompt"] = negative_prompt
    return result


def default_image_to_image_prompt(
    prompt: str,
    negative_prompt: str | None,
    input_image: Image.Image | list[Image.Image],
    height: int | None = None,
    width: int | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "prompt": prompt,
        "multi_modal_data": {"image": input_image},
    }
    if negative_prompt is not None:
        result["negative_prompt"] = negative_prompt
    return result


def default_image_to_video_prompt(
    prompt: str,
    negative_prompt: str | None,
    media_inputs: Mapping[str, Any],
    height: int | None = None,
    width: int | None = None,
    num_frames: int | None = None,
) -> dict[str, Any]:
    del height, width, num_frames
    if set(media_inputs) != {"image"} or not isinstance(media_inputs["image"], Image.Image):
        raise ValueError("This model only supports a single --image input in the shared image-to-video example.")
    return default_image_to_image_prompt(prompt, negative_prompt, media_inputs["image"])


_EXTRA_SPECS: dict[str, dict[str, Any]] = {
    "AudioXPipeline": {
        "extra_body_params": AUDIOX_EXTRA_BODY_PARAMS,
        "extra_output_params": AUDIOX_EXTRA_OUTPUT_PARAMS,
    },
    "BagelPipeline": {
        "extra_body_params": BAGEL_EXTRA_BODY_PARAMS,
        "extra_output_params": BAGEL_EXTRA_OUTPUT_PARAMS,
        "init_extra_args_for_non_diffusion_stages": BAGEL_INIT_EXTRA_ARGS_FOR_NON_DIFFUSION_STAGES,
        "text_to_image_prompt_builder": build_bagel_text_to_image_prompt,
        "image_to_image_prompt_builder": build_bagel_image_to_image_prompt,
    },
    "SenseNovaU1Pipeline": {
        "extra_body_params": SENSENOVA_U1_EXTRA_BODY_PARAMS,
        "extra_output_params": SENSENOVA_U1_EXTRA_OUTPUT_PARAMS,
    },
    "Cosmos3OmniDiffusersPipeline": {
        "extra_body_params": COSMOS3_EXTRA_BODY_PARAMS,
        "extra_output_params": COSMOS3_EXTRA_OUTPUT_PARAMS,
        "text_to_image_prompt_builder": build_cosmos3_text_to_image_prompt,
    },
    "MagiHumanPipeline": {
        "extra_body_params": MAGI_HUMAN_EXTRA_BODY_PARAMS,
        "extra_output_params": MAGI_HUMAN_EXTRA_OUTPUT_PARAMS,
    },
    "HeliosPipeline": {
        "extra_body_params": HELIOS_EXTRA_BODY_PARAMS,
        "extra_output_params": HELIOS_EXTRA_OUTPUT_PARAMS,
    },
    "HeliosPyramidPipeline": {
        "extra_body_params": HELIOS_EXTRA_BODY_PARAMS,
        "extra_output_params": HELIOS_EXTRA_OUTPUT_PARAMS,
    },
    "WanVACEPipeline": {
        "extra_body_params": VACE_EXTRA_BODY_PARAMS,
        "extra_output_params": VACE_EXTRA_OUTPUT_PARAMS,
        "image_to_video_prompt_builder": build_vace_image_to_video_prompt,
    },
    "MammothModa2DiTPipeline": {
        "extra_body_params": MAMMOTHMODA2_PREVIEW_EXTRA_BODY_PARAMS,
        "extra_output_params": MAMMOTHMODA2_PREVIEW_EXTRA_OUTPUT_PARAMS,
        "init_extra_args_for_non_diffusion_stages": MAMMOTHMODA2_PREVIEW_INIT_EXTRA_ARGS_FOR_NON_DIFFUSION_STAGES,
        "text_to_image_prompt_builder": build_mammothmoda2_text_to_image_prompt,
    },
    "MingImagePipeline": {
        "extra_body_params": MING_FLASH_OMNI_EXTRA_BODY_PARAMS,
        "extra_output_params": MING_FLASH_OMNI_EXTRA_OUTPUT_PARAMS,
        "init_extra_args_for_non_diffusion_stages": MING_FLASH_OMNI_INIT_EXTRA_ARGS_FOR_NON_DIFFUSION_STAGES,
        "text_to_image_prompt_builder": build_ming_flash_omni_text_to_image_prompt,
        "image_to_image_prompt_builder": build_ming_flash_omni_image_to_image_prompt,
    },
}


def _get_spec(model_class_name: str | None) -> dict[str, Any] | None:
    if not model_class_name:
        return None
    return _EXTRA_SPECS.get(model_class_name)


def get_model_class_name(omni: Any) -> str | None:
    """Extract model_class_name from an Omni/AsyncOmni instance.

    This hides the internal ODConfig plumbing from example scripts.
    """
    engine = getattr(omni, "engine", None)
    if engine is None:
        return None
    od_config = getattr(engine, "od_config", None)
    if od_config is None and hasattr(engine, "get_diffusion_od_config"):
        od_config = engine.get_diffusion_od_config()
    return getattr(od_config, "model_class_name", None) if od_config else None


def get_extra_body_params(model_class_name: str | None) -> frozenset[str]:
    spec = _get_spec(model_class_name)
    return spec.get("extra_body_params", frozenset()) if spec is not None else frozenset()


def get_extra_output_params(model_class_name: str | None) -> frozenset[str]:
    spec = _get_spec(model_class_name)
    return spec.get("extra_output_params", frozenset()) if spec is not None else frozenset()


def should_init_extra_args_for_non_diffusion_stages(model_class_name: str | None) -> bool:
    spec = _get_spec(model_class_name)
    return bool(spec and spec.get("init_extra_args_for_non_diffusion_stages", False))


def build_text_to_image_prompt(
    model_class_name: str | None,
    prompt: str,
    negative_prompt: str | None,
    height: int | None = None,
    width: int | None = None,
) -> dict[str, Any]:
    spec = _get_spec(model_class_name)
    builder: TextToImagePromptBuilder = (
        spec.get("text_to_image_prompt_builder", default_text_to_image_prompt)
        if spec is not None
        else default_text_to_image_prompt
    )
    return builder(prompt, negative_prompt, height, width)


def build_image_to_image_prompt(
    model_class_name: str | None,
    prompt: str,
    negative_prompt: str | None,
    input_image: Image.Image | list[Image.Image],
    height: int | None = None,
    width: int | None = None,
) -> dict[str, Any]:
    spec = _get_spec(model_class_name)
    builder: ImageToImagePromptBuilder = (
        spec.get("image_to_image_prompt_builder", default_image_to_image_prompt)
        if spec is not None
        else default_image_to_image_prompt
    )
    return builder(prompt, negative_prompt, input_image, height, width)


def build_image_to_video_prompt(
    model_class_name: str | None,
    prompt: str,
    negative_prompt: str | None,
    media_inputs: Mapping[str, Any],
    height: int | None = None,
    width: int | None = None,
    num_frames: int | None = None,
) -> dict[str, Any]:
    spec = _get_spec(model_class_name)
    builder: ImageToVideoPromptBuilder = (
        spec.get("image_to_video_prompt_builder", default_image_to_video_prompt)
        if spec is not None
        else default_image_to_video_prompt
    )
    return builder(prompt, negative_prompt, media_inputs, height, width, num_frames)
