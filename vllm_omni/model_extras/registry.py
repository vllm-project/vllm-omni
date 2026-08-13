# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal

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
from vllm_omni.model_extras.bagel import build_x_to_text_prompt as build_bagel_x_to_text_prompt
from vllm_omni.model_extras.cosmos3 import (
    COSMOS3_EXTRA_BODY_PARAMS,
    COSMOS3_EXTRA_OUTPUT_PARAMS,
)
from vllm_omni.model_extras.helios import (
    HELIOS_EXTRA_BODY_PARAMS,
    HELIOS_EXTRA_OUTPUT_PARAMS,
)
from vllm_omni.model_extras.hunyuan_image3 import build_x_to_text_prompt as build_hunyuan_x_to_text_prompt
from vllm_omni.model_extras.lingbot_video import LINGBOT_VIDEO_EXTRA_BODY_PARAMS
from vllm_omni.model_extras.ltx2 import LTX_EXTRA_BODY_PARAMS, LTX_EXTRA_OUTPUT_PARAMS
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
from vllm_omni.model_extras.mammothmodal2_preview import (
    build_x_to_text_prompt as build_mammothmoda2_x_to_text_prompt,
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
XToTextPromptBuilder = Callable[[str, str, bool], tuple[dict[str, Any], list[int] | None]]
OutputTensorRange = Literal["negative_one_to_one", "zero_to_one"]


def default_x_to_text_prompt(
    model: str,
    prompt: str,
    has_image: bool,
) -> tuple[dict[str, Any], list[int] | None]:
    del model, has_image
    return {"prompt": prompt, "modalities": ["text"]}, None


_X_TO_TEXT_SPECS: dict[str, XToTextPromptBuilder] = {
    "bagel": build_bagel_x_to_text_prompt,
    "hunyuan_image3": build_hunyuan_x_to_text_prompt,
    "mammoth_moda2": build_mammothmoda2_x_to_text_prompt,
}


def get_x_to_text_model_family(model: str) -> str:
    """Resolve a text-output prompt family from a checkpoint's config.json."""
    from vllm.transformers_utils.config import get_hf_file_to_dict

    config = get_hf_file_to_dict("config.json", model) or {}
    model_type = str(config.get("model_type", "")).lower()
    architectures = {str(value).lower() for value in (config.get("architectures") or [])}
    if model_type == "bagel" or "bagelforconditionalgeneration" in architectures:
        return "bagel"
    if model_type == "hunyuan_image_3_moe" or any("hunyuanimage3" in value for value in architectures):
        return "hunyuan_image3"
    if "mammoth" in model_type or any("mammothmoda2" in value for value in architectures):
        return "mammoth_moda2"
    return "generic"


def build_x_to_text_prompt(
    model_family: str,
    model: str,
    prompt: str,
    has_image: bool,
) -> tuple[dict[str, Any], list[int] | None]:
    """Build a model-aware T2T/I2T prompt and optional stop-token ids."""
    builder = _X_TO_TEXT_SPECS.get(model_family, default_x_to_text_prompt)
    return builder(model, prompt, has_image)


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
        # The shared T2I example already supplies modalities=["image"].
    },
    "Cosmos3OmniPipeline": {
        "extra_body_params": COSMOS3_EXTRA_BODY_PARAMS,
        "extra_output_params": COSMOS3_EXTRA_OUTPUT_PARAMS,
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
    "LingBotVideoPipeline": {
        "extra_body_params": LINGBOT_VIDEO_EXTRA_BODY_PARAMS,
        "output_tensor_range": "zero_to_one",
        # Shared T2I/I2V envelopes select the output modality. LingBot's
        # pipeline owns model-specific validation and normalization.
    },
    **{
        model_class_name: {
            "extra_body_params": LTX_EXTRA_BODY_PARAMS,
            "extra_output_params": LTX_EXTRA_OUTPUT_PARAMS,
        }
        for model_class_name in (
            "LTX2Pipeline",
            "LTX2DistilledPipeline",
        )
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

# Multi-stage discovery reports the top-level wrapper rather than its DiT
# submodule, so both names must resolve to the same request builders.
_EXTRA_SPECS["MammothModa2ForConditionalGeneration"] = _EXTRA_SPECS["MammothModa2DiTPipeline"]
_EXTRA_SPECS["Mammothmoda2Model"] = _EXTRA_SPECS["MammothModa2DiTPipeline"]


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


def get_output_tensor_range(model_class_name: str | None) -> OutputTensorRange:
    """Return the declared range for floating-point tensor outputs.

    The default preserves the shared examples' historical handling. Pipelines
    that already return normalized tensors declare ``zero_to_one`` explicitly.
    """
    spec = _get_spec(model_class_name)
    if spec is None:
        return "negative_one_to_one"
    return spec.get("output_tensor_range", "negative_one_to_one")


def should_init_extra_args_for_non_diffusion_stages(model_class_name: str | None) -> bool:
    spec = _get_spec(model_class_name)
    return bool(spec and spec.get("init_extra_args_for_non_diffusion_stages", False))


def build_text_to_image_prompt(
    model_class_name: str | None,
    prompt: dict[str, Any],
    height: int | None = None,
    width: int | None = None,
) -> dict[str, Any]:
    """Build a model-specific T2I prompt from an example-owned envelope."""
    spec = _get_spec(model_class_name)
    builder: TextToImagePromptBuilder | None = spec.get("text_to_image_prompt_builder") if spec else None
    if builder is None:
        return prompt
    return builder(
        prompt=str(prompt["prompt"]),
        negative_prompt=prompt.get("negative_prompt"),
        height=height,
        width=width,
    )


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
    prompt: dict[str, Any],
    height: int | None = None,
    width: int | None = None,
    num_frames: int | None = None,
) -> dict[str, Any]:
    """Build a model-specific I2V prompt from an example-owned envelope."""
    spec = _get_spec(model_class_name)
    builder: ImageToVideoPromptBuilder | None = spec.get("image_to_video_prompt_builder") if spec else None
    if builder is None:
        return prompt
    media_inputs = prompt.get("multi_modal_data") or {}
    if not isinstance(media_inputs, Mapping):
        raise TypeError("Canonical I2V prompt multi_modal_data must be a mapping.")
    return builder(
        prompt=str(prompt["prompt"]),
        negative_prompt=prompt.get("negative_prompt"),
        media_inputs=media_inputs,
        height=height,
        width=width,
        num_frames=num_frames,
    )
