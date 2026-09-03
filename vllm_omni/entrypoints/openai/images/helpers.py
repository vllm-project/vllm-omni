# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Image endpoint helpers peeled from ``api_server.py``.

PUT HERE:
  - ``/v1/images*`` request/engine helpers: limits, input loading, edit extras,
    result extraction/normalization, and other route-adjacent logic that used
    to live in ``api_server.py``.
  - May import ``image_api_utils`` for encode/validate (helpers → utils OK).

DO NOT PUT HERE:
  - Shared encode/size/layered validation — keep those in ``image_api_utils``.

LONGEVITY:
  - This package ``helpers.py`` is the **longer home** for image endpoint
    logic through the helper (P0.2) + router (P0.3) stages.
  - The later P1.2 image modality PR (#5227) may further split this file (e.g.
    validation / editing / responses) and absorb ``image_api_utils`` into
    this package; until then keep growing here, not in root utils.

See ``images/README.md`` (utils vs helpers, no overlap).
"""

import io
from http import HTTPStatus
from numbers import Integral
from typing import Any

import httpx
import numpy as np
import pybase64 as base64
from fastapi import HTTPException, Request
from PIL import Image
from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.app_state import _get_diffusion_od_config

logger = init_logger(__name__)


def _get_max_edit_input_images(raw_request: Request, engine_client: Any) -> int | None:
    od_config = _get_diffusion_od_config(raw_request, engine_client)
    if od_config is None:
        # Preserve the existing compatibility behavior when the diffusion
        # config is not exposed on the serving surface.
        return None

    supports_multimodal_inputs = getattr(od_config, "supports_multimodal_inputs", None)
    if not isinstance(supports_multimodal_inputs, bool):
        # Older serving surfaces and mocked engines may expose a placeholder
        # object instead of a real diffusion config. Treat that as "unknown"
        # so existing single-image flows keep working.
        return None

    if not supports_multimodal_inputs:
        return 1

    max_input_images = getattr(od_config, "max_multimodal_image_inputs", None)
    if max_input_images is None:
        return None
    if isinstance(max_input_images, bool) or not isinstance(max_input_images, Integral):
        return None
    if max_input_images < 1:
        return None
    return int(max_input_images)


def _check_max_generated_image_size(
    app_state_args: Any,
    width: int | None,
    height: int | None,
    resolution: int | None = None,
) -> None:
    """Raise 400 if the requested image size exceeds --max-generated-image-size."""
    max_generated_image_size = getattr(app_state_args, "max_generated_image_size", None)
    # Check max_generated_image_size
    if max_generated_image_size is None:
        return
    if width is not None and height is not None:
        if width * height > max_generated_image_size:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail=f"Requested image size {width}x{height} exceeds the maximum allowed "
                f"size of {max_generated_image_size} pixels. You can reduce the requested size "
                f"or increase the server's --max-generated-image-size limit.",
            )
    elif resolution is not None:
        # When resolution is set, the output size is resolution * resolution
        if resolution * resolution > max_generated_image_size:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail=f"Requested resolution {resolution} (max {resolution}x{resolution} pixels) "
                f"exceeds the maximum allowed size of {max_generated_image_size} pixels. "
                f"You can reduce the requested size or increase the server's --max-generated-image-size limit.",
            )


def _build_hunyuan_edit_extra_args(
    *,
    bot_task: str | None,
    sys_type: str | None,
    system_prompt: str | None,
) -> dict[str, Any]:
    """Map Hunyuan /v1/images/edits form fields to DiT ``extra_args``."""
    extra_args: dict[str, Any] = {}
    effective_use_system_prompt = sys_type
    if effective_use_system_prompt is None and bot_task is not None:
        from vllm_omni.diffusion.models.hunyuan_image3.prompt_utils import resolve_sys_type

        effective_use_system_prompt = resolve_sys_type(bot_task)
    if effective_use_system_prompt is not None:
        extra_args["use_system_prompt"] = effective_use_system_prompt
    if system_prompt is not None:
        extra_args["system_prompt"] = system_prompt
    if bot_task is not None:
        extra_args["bot_task"] = bot_task
    return extra_args


def _update_if_not_none(object: Any, key: str, val: Any) -> None:
    if val is not None:
        setattr(object, key, val)


def _normalize_image(image: Any) -> Any:
    """Normalize a single image output to a PIL-compatible format."""
    if isinstance(image, Image.Image):
        return image
    if not isinstance(image, np.ndarray):
        raise ValueError(f"Unsupported image type: {type(image)}")
    if not np.issubdtype(image.dtype, np.integer) and not np.issubdtype(image.dtype, np.floating):
        raise ValueError(f"Unsupported dtype: {image.dtype}")
    if isinstance(image, np.ndarray):
        while image.ndim > 3:
            image = image[0]
        if image.min() < 0:
            if image.min() < -1.01 or image.max() > 1.01:
                logger.warning(
                    f"Image float range [{image.min():.2f}, {image.max():.2f}] outside expected [-1, 1]. "
                    f"Clipping to [-1, 1] before normalization."
                )
            image = np.clip(image, -1.0, 1.0) * 0.5 + 0.5
        elif image.max() > 1.01:
            logger.warning(
                f"Image float range [{image.min():.2f}, {image.max():.2f}] outside expected [0, 1]. "
                f"Clipping to [0, 1] before normalization."
            )
        image = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
        image = Image.fromarray(image)
    return image


def _extract_images_from_result(result: Any) -> list[Any]:
    images = []
    if hasattr(result, "images") and result.images:
        images = result.images
    # Handle when generate more than one image
    if images and isinstance(images[0], np.ndarray) and images[0].shape[0] > 1 and images[0].ndim == 5:
        # Unwrap batch: (N, T, H, W, C) -> [img1, img2, ...]
        images = list(images[0])
    # Flatten nested lists (e.g., from layered models like Qwen-Image-Layered).
    # Note: This only flattens one level deep. Deeper nesting is not supported.
    flattened = []
    for img in images:
        if isinstance(img, list):
            flattened.extend(img)
        else:
            flattened.append(img)
    return [_normalize_image(img) for img in flattened]


async def _load_input_images(
    inputs: list[str],
    *,
    normalize_rgb: bool = True,
) -> list[Image.Image]:
    """
    convert to PIL.Image.Image list
    """
    if isinstance(inputs, str):
        inputs = [inputs]

    images: list[Image.Image] = []

    for inp in inputs:
        # 1. URL + base64
        if isinstance(inp, str) and inp.startswith("data:image"):
            try:
                _, b64_data = inp.split(",", 1)
                image_bytes = base64.b64decode(b64_data)
                img = Image.open(io.BytesIO(image_bytes))
                images.append(img)
            except Exception as e:
                raise ValueError(f"Invalid base64 image: {e}")

        # 2. URL
        elif isinstance(inp, str) and inp.startswith("http"):
            async with httpx.AsyncClient(timeout=60) as client:
                try:
                    resp = await client.get(inp)
                    resp.raise_for_status()
                    img = Image.open(io.BytesIO(resp.content))
                    images.append(img)
                except Exception as e:
                    raise ValueError(f"Failed to download image from URL {inp}: {e}")

        # 3. UploadFile
        elif hasattr(inp, "file"):
            try:
                img_data = await inp.read()
                img = Image.open(io.BytesIO(img_data))
                images.append(img)
            except Exception as e:
                raise ValueError(f"Failed to open uploaded file: {e}")

        else:
            raise ValueError(f"Unsupported input: {inp}")

    if not images:
        raise ValueError("No valid input images found")

    if not normalize_rgb:
        return images

    # Match the offline HunyuanImage3 image-edit example path, which eagerly
    # normalizes input files with ``Image.open(...).convert("RGB")`` before
    # they reach the AR stage. Keeping uploads as RGBA/P PIL objects makes
    # online IT2I observe a different visual input than offline (for example
    # transparent-logo uploads alpha-composited over white instead of black),
    # which is enough for HunyuanImage3 AR recaption to diverge before DiT
    # sees the request -- root cause of the "online 3 magnets vs offline 1
    # magnet" systematic semantic mismatch.
    return [img.convert("RGB") for img in images]


def _choose_output_format(output_format: str | None, background: str | None) -> str:
    # Normalize and choose extension
    fmt = (output_format or "").lower()
    if fmt in {"jpg", "png", "webp", "jpeg"}:
        return fmt
    # If transparency requested, prefer png
    if (background or "auto").lower() == "transparent":
        return "png"
    # Default
    return "jpeg"
