# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Portions adapted from Microsoft Mage:
# https://github.com/microsoft/Mage/tree/df7f84d9f8fc991d189d929f03cff623b430a4a2
# Copyright (c) Microsoft Corporation.
# Microsoft-derived portions are licensed under the MIT License.

"""Prompt formatting and request validation for Mage-Flow."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image

from vllm_omni.diffusion.model_metadata import (
    MAGE_FLOW_EXTRA_BODY_PARAMS,
    MAGE_FLOW_MAX_INPUT_IMAGES,
)
from vllm_omni.diffusion.utils.prompt_utils import validate_prompt_sequence_lengths

MAGE_FLOW_PROMPT_TEMPLATE = (
    "<|im_start|>system\n"
    "Describe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background:"
    "<|im_end|>\n<|im_start|>user\n{}"
    "<|im_end|>\n<|im_start|>assistant\n"
)
MAGE_FLOW_PROMPT_START_INDEX = 34

MAGE_FLOW_EDIT_PROMPT_TEMPLATE = (
    "<|im_start|>system\n"
    "Describe the key features of the input image (color, shape, size, "
    "texture, objects, background), then explain how the user's text "
    "instruction should alter or modify the image. Generate a new image that "
    "meets the user's requirements while maintaining consistency with the "
    "original input where appropriate.<|im_end|>\n"
    "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)
MAGE_FLOW_EDIT_PROMPT_START_INDEX = 64
MAGE_FLOW_EDIT_IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"

MAGE_FLOW_MIN_SIZE = 512
MAGE_FLOW_MAX_SIZE = 2048
MAGE_FLOW_SIZE_MULTIPLE = 16
MAGE_FLOW_MAX_ASPECT_RATIO = 4.0


@dataclass(frozen=True)
class MageFlowVariantDefaults:
    num_inference_steps: int
    guidance_scale: float
    # Only Edit checkpoints are trained to consume reference-image tokens; the
    # T2I variants must never see them, even though both share the pipeline.
    supports_reference_images: bool = False


def get_mage_flow_variant_defaults(
    model_name_or_path: str,
    *,
    model_index: dict[str, Any] | None = None,
) -> MageFlowVariantDefaults:
    """Resolve defaults from checkpoint identity, never from requested steps.

    ``model_index.json`` records the repository a checkpoint was published as
    whenever it carries ``_name_or_path``, so that wins when present. Otherwise
    the identity is the *last path component that names a Mage-Flow checkpoint*,
    which is neither the whole path nor simply the basename:

    - ``/turbo-disk/Mage-Flow`` resolves on ``Mage-Flow``, so an unrelated
      parent directory cannot pass the checkpoint off as the Turbo variant.
    - ``.../models/microsoft--Mage-Flow-Turbo/snapshots/master`` resolves on
      ``microsoft--Mage-Flow-Turbo``, because Hub and ModelScope caches both
      end in a revision component that names nothing.
    """
    identity = model_index.get("_name_or_path") if model_index is not None else None
    if not isinstance(identity, str) or not identity:
        identity = model_name_or_path
    parts = [part for part in identity.replace("\\", "/").lower().split("/") if part]
    named = [part for part in parts if "mage-flow" in part or "mage_flow" in part]
    identity = named[-1] if named else (parts[-1] if parts else "")
    is_edit = "edit" in identity
    if "turbo" in identity:
        return MageFlowVariantDefaults(4, 1.0, is_edit)
    if "mage-flow" in identity and not is_edit and "base" not in identity:
        return MageFlowVariantDefaults(20, 5.0)
    return MageFlowVariantDefaults(30, 5.0, is_edit)


def validate_mage_flow_size(height: int, width: int) -> None:
    for name, value in (("height", height), ("width", width)):
        if not MAGE_FLOW_MIN_SIZE <= value <= MAGE_FLOW_MAX_SIZE:
            raise ValueError(f"Mage-Flow {name} must be in [{MAGE_FLOW_MIN_SIZE}, {MAGE_FLOW_MAX_SIZE}], got {value}")
        if value % MAGE_FLOW_SIZE_MULTIPLE:
            raise ValueError(f"Mage-Flow {name} must be a multiple of {MAGE_FLOW_SIZE_MULTIPLE}, got {value}")
    aspect_ratio = max(height, width) / min(height, width)
    if aspect_ratio > MAGE_FLOW_MAX_ASPECT_RATIO:
        raise ValueError(f"Mage-Flow aspect ratio must not exceed {MAGE_FLOW_MAX_ASPECT_RATIO}:1, got {height}x{width}")


def validate_reference_image_count(images: list[Any] | None) -> None:
    if images is None:
        return
    if not 1 <= len(images) <= MAGE_FLOW_MAX_INPUT_IMAGES:
        raise ValueError(
            f"Mage-Flow Edit requires between 1 and {MAGE_FLOW_MAX_INPUT_IMAGES} reference images, got {len(images)}"
        )


def _reference_image_to_pil(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, torch.Tensor):
        value = image.detach().cpu()
        if value.ndim == 4 and value.shape[0] == 1:
            value = value[0]
        if value.ndim != 3:
            raise ValueError(
                f"Mage-Flow reference tensors must have shape [C,H,W], [H,W,C], or [1,C,H,W], got {tuple(value.shape)}"
            )
        if value.shape[0] in (1, 3, 4):
            value = value.permute(1, 2, 0)
        array = value.numpy()
    elif isinstance(image, np.ndarray):
        array = image
        if array.ndim == 4 and array.shape[0] == 1:
            array = array[0]
        if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
            array = np.moveaxis(array, 0, -1)
    else:
        raise TypeError(
            f"Mage-Flow reference images must be PIL images, NumPy arrays, or torch tensors, got {type(image).__name__}"
        )
    if array.ndim not in (2, 3):
        raise ValueError(f"Mage-Flow reference arrays must be 2-D or 3-D, got shape {array.shape}")
    if np.issubdtype(array.dtype, np.floating):
        array = np.nan_to_num(array)
        if array.size and float(array.min()) < 0:
            array = (array + 1.0) / 2.0
        if not array.size or float(array.max()) <= 1.0:
            array = array * 255.0
    array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    return Image.fromarray(array).convert("RGB")


def normalize_mage_flow_reference_images(images: Any | None) -> list[Image.Image]:
    """Normalize one to three PIL/NumPy/Tensor references while preserving order."""
    if images is None:
        return []
    values = list(images) if isinstance(images, (list, tuple)) else [images]
    validate_reference_image_count(values)
    return [_reference_image_to_pil(image) for image in values]


def resize_mage_flow_reference_for_vision(
    image: Image.Image,
    max_long_edge: int,
) -> Image.Image:
    """Match the official Qwen3-VL edit-conditioning resize."""
    width, height = image.size
    long_edge = max(width, height)
    if long_edge <= max_long_edge:
        return image
    scale = max_long_edge / long_edge
    resized = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    return image.resize(resized, Image.Resampling.BICUBIC)


def preprocess_mage_flow_reference_for_vae(
    image: Image.Image,
    *,
    height: int,
    width: int,
) -> torch.Tensor:
    """Resize one reference and normalize RGB pixels from [0, 255] to [-1, 1]."""
    resized = image.convert("RGB").resize((width, height), Image.Resampling.BICUBIC)
    array = np.asarray(resized, dtype=np.float32).copy()
    return torch.from_numpy(array).permute(2, 0, 1).div_(127.5).sub_(1.0)


def parse_mage_flow_extra_args(extra_args: dict[str, Any] | None) -> int:
    """Return the Edit visual-conditioning long edge from ``extra_args``."""
    extra_args = extra_args or {}
    # The serving layer advertises exactly these keys, so accept exactly these.
    unknown = sorted(key for key in extra_args if key.startswith("mage_") and key not in MAGE_FLOW_EXTRA_BODY_PARAMS)
    if unknown:
        raise ValueError("Unknown Mage-Flow extra_args: " + ", ".join(unknown))

    vision_long_edge = extra_args.get("mage_vision_long_edge", 384)
    if isinstance(vision_long_edge, bool) or not isinstance(vision_long_edge, int) or vision_long_edge <= 0:
        raise ValueError(f"mage_vision_long_edge must be a positive integer, got {vision_long_edge!r}")
    return vision_long_edge


def format_mage_flow_prompt(prompt: str, *, edit: bool = False) -> str:
    template = MAGE_FLOW_EDIT_PROMPT_TEMPLATE if edit else MAGE_FLOW_PROMPT_TEMPLATE
    return template.format(prompt)


def format_mage_flow_edit_prompt(prompt: str, num_references: int) -> str:
    validate_reference_image_count([None] * num_references)
    prefix = "".join(f"Image {index}: {MAGE_FLOW_EDIT_IMAGE_PLACEHOLDER}" for index in range(1, num_references + 1))
    return format_mage_flow_prompt(prefix + prompt, edit=True)


@torch.no_grad()
def encode_mage_flow_prompt(
    text_encoder,
    processor,
    prompt: str,
    *,
    device: torch.device,
    reference_images: list[Image.Image] | None = None,
    max_length: int = 2048,
    vision_long_edge: int = 384,
) -> torch.Tensor:
    """Encode a prompt without upstream's global Qwen monkeypatch.

    Passing ``reference_images`` switches to the Edit template and conditions on
    the ordered visual references; otherwise this is the plain T2I encode.
    """
    if reference_images:
        validate_reference_image_count(reference_images)
        formatted = format_mage_flow_edit_prompt(prompt, len(reference_images))
        start_index = MAGE_FLOW_EDIT_PROMPT_START_INDEX
        vision_keys = {"pixel_values", "image_grid_thw", "mm_token_type_ids"}
        error_context = "for Mage-Flow Edit (the count includes reference-image tokens)"
        processor_kwargs = {
            "images": [resize_mage_flow_reference_for_vision(image, vision_long_edge) for image in reference_images],
        }
    else:
        formatted = format_mage_flow_prompt(prompt)
        start_index = MAGE_FLOW_PROMPT_START_INDEX
        vision_keys = set()
        error_context = "for Mage-Flow"
        processor_kwargs = {}

    inputs = processor(
        text=[formatted],
        padding=True,
        return_tensors="pt",
        **processor_kwargs,
    )
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        # Bound the conditioning before it reaches the encoder rather than after.
        # The T2I path used to ask the processor to truncate, which quietly cut
        # the tail off an over-long prompt, and the Edit path had no bound at
        # all -- an unbounded instruction went straight into Qwen3-VL.
        validate_prompt_sequence_lengths(
            attention_mask,
            max_sequence_length=max_length,
            supported_max_sequence_length=max_length,
            prompt_name="prompt",
            length_offset=start_index,
            error_context=error_context,
        )
    model_inputs = {
        key: value
        for key, value in inputs.items()
        if key in {"input_ids", "attention_mask", "position_ids"} | vision_keys
    }
    outputs = text_encoder.model(
        **model_inputs,
        use_cache=False,
        return_dict=True,
    )
    hidden_states = outputs.last_hidden_state
    valid_length = int(attention_mask[0].sum().item()) if attention_mask is not None else hidden_states.shape[1]
    if valid_length <= start_index:
        raise ValueError("Mage-Flow prompt produced no conditioning tokens")
    return hidden_states[:, start_index:valid_length].contiguous()
