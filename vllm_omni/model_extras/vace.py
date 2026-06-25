# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from PIL import Image

# VACE conditioning is expressed through prompt media, and the pipeline does not
# currently read any model-specific values from sampling_params.extra_args.
VACE_EXTRA_BODY_PARAMS: frozenset[str] = frozenset()
VACE_EXTRA_OUTPUT_PARAMS: frozenset[str] = frozenset()

_MEDIA_KEYS = frozenset({"image", "last_image", "mask", "reference_images"})


def _base_prompt(prompt: str, negative_prompt: str | None) -> dict[str, Any]:
    result: dict[str, Any] = {"prompt": prompt}
    if negative_prompt is not None:
        result["negative_prompt"] = negative_prompt
    return result


def _require_dimensions(
    height: int | None,
    width: int | None,
    num_frames: int | None,
) -> tuple[int, int, int]:
    if height is None or width is None or num_frames is None:
        raise ValueError("VACE conditional inputs require height, width, and num_frames.")
    if height <= 0 or width <= 0 or num_frames <= 0:
        raise ValueError("VACE height, width, and num_frames must be positive.")
    return height, width, num_frames


def _validate_media_inputs(media_inputs: Mapping[str, Any]) -> None:
    unknown = set(media_inputs) - _MEDIA_KEYS
    if unknown:
        raise ValueError(f"Unsupported VACE media input(s): {', '.join(sorted(unknown))}.")

    has_references = bool(media_inputs.get("reference_images"))
    has_frame_conditioning = any(media_inputs.get(key) is not None for key in ("image", "last_image", "mask"))
    if has_references and has_frame_conditioning:
        raise ValueError("VACE reference images cannot be combined with image, last_image, or mask inputs.")
    if media_inputs.get("mask") is not None and media_inputs.get("image") is None:
        raise ValueError("VACE mask input requires an image input.")
    if media_inputs.get("mask") is not None and media_inputs.get("last_image") is not None:
        raise ValueError("VACE mask and last_image inputs cannot be combined.")
    if not has_references and not has_frame_conditioning:
        raise ValueError("VACE image-to-video generation requires a conditioning media input.")


def build_image_to_video_prompt(
    prompt: str,
    negative_prompt: str | None,
    media_inputs: Mapping[str, Any],
    height: int | None = None,
    width: int | None = None,
    num_frames: int | None = None,
) -> dict[str, Any]:
    """Build VACE's native video, mask, or reference-image conditioning."""
    _validate_media_inputs(media_inputs)
    result = _base_prompt(prompt, negative_prompt)

    reference_images = media_inputs.get("reference_images")
    if reference_images:
        if not isinstance(reference_images, list) or not all(
            isinstance(image, Image.Image) for image in reference_images
        ):
            raise ValueError("VACE reference_images must be a non-empty list of PIL images.")
        result["multi_modal_data"] = {"reference_images": reference_images}
        return result

    height, width, num_frames = _require_dimensions(height, width, num_frames)
    first_image = media_inputs.get("image")
    last_image = media_inputs.get("last_image")
    mask_image = media_inputs.get("mask")
    for name, image in (("image", first_image), ("last_image", last_image), ("mask", mask_image)):
        if image is not None and not isinstance(image, Image.Image):
            raise ValueError(f"VACE {name} must be a PIL image.")

    gray = Image.new("RGB", (width, height), (128, 128, 128))
    mask_black = Image.new("L", (width, height), 0)
    mask_white = Image.new("L", (width, height), 255)

    if mask_image is not None:
        mask = mask_image.convert("L")
        source = np.asarray(first_image).copy()
        source[np.asarray(mask) > 128] = 128
        frame = Image.fromarray(source)
        result["multi_modal_data"] = {
            "video": [frame.copy() for _ in range(num_frames)],
            "mask": [mask.copy() for _ in range(num_frames)],
        }
        return result

    if first_image is not None and last_image is not None:
        if num_frames < 2:
            raise ValueError("VACE first/last-frame conditioning requires at least two frames.")
        result["multi_modal_data"] = {
            "video": [first_image] + [gray] * (num_frames - 2) + [last_image],
            "mask": [mask_black] + [mask_white] * (num_frames - 2) + [mask_black],
        }
        return result

    if first_image is not None:
        result["multi_modal_data"] = {
            "video": [first_image] + [gray] * (num_frames - 1),
            "mask": [mask_black] + [mask_white] * (num_frames - 1),
        }
        return result

    result["multi_modal_data"] = {
        "video": [gray] * (num_frames - 1) + [last_image],
        "mask": [mask_white] * (num_frames - 1) + [mask_black],
    }
    return result
