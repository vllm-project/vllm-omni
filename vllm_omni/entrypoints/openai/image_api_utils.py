# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Shared helper utilities for OpenAI-compatible image generation API.

This module provides common helper functions for the image generation endpoint.
All functions work with plain Python types to maintain separation from the
FastAPI HTTP layer.
"""

import base64
import io
import os

import PIL.Image

QWEN_IMAGE_MIN_EDGE = 16
SUPPORTED_LAYERED_RESOLUTIONS = (640, 1024)
SUPPORTED_LAYERED_LAYERS_RANGE = range(3, 11)
QWEN_IMAGE_MODEL_CLASSES = {
    "QwenImagePipeline",
    "QwenImageEditPipeline",
    "QwenImageEditPlusPipeline",
    "QwenImageLayeredPipeline",
}


def parse_size(size_str: str) -> tuple[int, int]:
    """Parse size string to width and height tuple.

    Args:
        size_str: Size in format "WIDTHxHEIGHT" (e.g., "1024x1024")

    Returns:
        Tuple of (width, height)

    Raises:
        ValueError: If size format is invalid
    """
    if not size_str or not isinstance(size_str, str):
        raise ValueError(
            f"Size must be a non-empty string in format 'WIDTHxHEIGHT' (e.g., '1024x1024'), got: {size_str}"
        )

    parts = size_str.split("x")
    if len(parts) != 2:
        raise ValueError(
            f"Invalid size format: '{size_str}'. Expected format: 'WIDTHxHEIGHT' (e.g., '1024x1024'). "
            f"Did you mean to use 'x' as separator?"
        )

    try:
        width = int(parts[0])
        height = int(parts[1])
    except ValueError:
        raise ValueError(f"Invalid size format: '{size_str}'. Width and height must be integers.")

    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid size: {width}x{height}. Width and height must be positive integers.")

    return width, height


def encode_image_base64(image: PIL.Image.Image) -> str:
    """Encode PIL Image to base64 PNG string.

    Args:
        image: PIL Image object

    Returns:
        Base64-encoded PNG image as string
    """
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def requires_qwen_image_min_size(
    model_name: str | None = None,
    model_class_name: str | None = None,
) -> bool:
    """Return whether the request targets a Qwen-Image family pipeline."""
    if model_class_name in QWEN_IMAGE_MODEL_CLASSES:
        return True

    if not model_name:
        return False

    normalized = os.path.basename(model_name).replace("_", "-").lower()
    return "qwen-image" in normalized


def validate_qwen_image_min_size(
    width: int,
    height: int,
    *,
    model_name: str | None = None,
    model_class_name: str | None = None,
) -> None:
    """Reject Qwen-Image requests that would quantize to zero-sized latents."""
    if not requires_qwen_image_min_size(model_name=model_name, model_class_name=model_class_name):
        return

    if width < QWEN_IMAGE_MIN_EDGE or height < QWEN_IMAGE_MIN_EDGE:
        raise ValueError(
            f"Requested image size {width}x{height} is too small for Qwen-Image models. "
            f"Width and height must each be at least {QWEN_IMAGE_MIN_EDGE} pixels."
        )


def validate_layered_layers(layers: int | None) -> int | None:
    """Validate the Qwen-Image-Layered ``layers`` parameter."""
    if layers is None:
        return None
    if layers not in SUPPORTED_LAYERED_LAYERS_RANGE:
        raise ValueError(
            f"Invalid layers value {layers}. layers must be between "
            f"{SUPPORTED_LAYERED_LAYERS_RANGE.start} and "
            f"{SUPPORTED_LAYERED_LAYERS_RANGE.stop - 1} inclusive."
        )
    return layers
