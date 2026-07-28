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

from PIL import Image

SUPPORTED_LAYERED_RESOLUTIONS = (640, 1024)
SUPPORTED_LAYERED_LAYERS_RANGE = range(2, 11)
SUPPORTED_OUTPUT_FORMATS = frozenset({"png", "jpeg", "jpg", "webp"})


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


def encode_image_base64(image: Image.Image, format: str = "png", output_compression: int = 100) -> str:
    """Encode PIL Image to a base64 image string.

    Args:
        image: PIL Image object.
        format: Output image format. One of ``png``, ``jpeg``/``jpg``, ``webp``.
        output_compression: Quality (1-100). 100 means best quality / least compression.

    Returns:
        Base64-encoded image bytes as a UTF-8 string.
    """
    fmt = (format or "png").lower()
    if fmt not in SUPPORTED_OUTPUT_FORMATS:
        raise ValueError(f"Unsupported output format: {format!r}. Supported: {sorted(SUPPORTED_OUTPUT_FORMATS)}")
    # PIL's save handler doesn't recognize 'JPG'; normalize to 'JPEG'.
    pil_fmt = "jpeg" if fmt == "jpg" else fmt

    image = prepare_image_for_output_format(image, pil_fmt)

    save_kwargs: dict = {}
    if pil_fmt in {"jpeg", "webp"}:
        save_kwargs["quality"] = output_compression
    elif pil_fmt == "png":
        # Map quality 0-100 → PNG compress_level 9-0 (PIL accepts 0..9, higher = more compression)
        save_kwargs["compress_level"] = max(0, min(9, 9 - output_compression // 11))

    buffer = io.BytesIO()
    image.save(buffer, format=pil_fmt, **save_kwargs)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def choose_output_format(output_format: str | None, background: str | None) -> str:
    """Resolve a final image format from the request's output_format / background.

    Returns one of {"png", "jpeg", "webp"} — "jpg" is canonicalised to "jpeg" so
    downstream MIME types (e.g. ``image/jpeg``) follow the IANA spelling.
    Falls back to PNG if transparency is requested, otherwise JPEG.
    """
    fmt = (output_format or "").lower()
    if fmt == "jpg":
        return "jpeg"
    if fmt in SUPPORTED_OUTPUT_FORMATS:
        return fmt
    if (background or "auto").lower() == "transparent":
        return "png"
    return "jpeg"


def get_vllm_image_params(vllm_xargs: dict | None) -> tuple[str, int, str]:
    """Extract image (format, compression, background) from chat-completion ``vllm_xargs``.

    Invalid or missing values fall back to safe defaults (png / 100 / auto).
    Compression is clamped into [1, 100].
    """
    if not vllm_xargs:
        return "png", 100, "auto"

    image_format = vllm_xargs.get("image_format")
    image_compression = vllm_xargs.get("image_compression")
    image_background = vllm_xargs.get("image_background")

    if not isinstance(image_format, str) or image_format.strip().lower() not in SUPPORTED_OUTPUT_FORMATS:
        image_format = "png"
    else:
        image_format = image_format.strip().lower()

    if isinstance(image_compression, bool) or not isinstance(image_compression, (int, float)):
        image_compression = 100
    else:
        image_compression = int(max(1, min(100, image_compression)))

    if not isinstance(image_background, str) or not image_background.strip():
        image_background = "auto"

    return image_format, image_compression, image_background


def prepare_image_for_output_format(image: Image.Image, format: str) -> Image.Image:
    fmt = format.lower()
    if fmt not in {"jpg", "jpeg"}:
        return image

    if image.mode == "RGB":
        return image

    if image.mode in {"RGBA", "LA"} or (image.mode == "P" and "transparency" in image.info):
        alpha_image = image.convert("RGBA")
        flattened = Image.new("RGB", alpha_image.size, (255, 255, 255))
        flattened.paste(alpha_image, mask=alpha_image.getchannel("A"))
        return flattened

    return image.convert("RGB")


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
