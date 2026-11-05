# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""
Shared media utilities for OpenAI-compatible image APIs.

PUT HERE:
  - Pure media helpers with no FastAPI Request / app-state / job semantics:
    size parsing, image encode/base64, layered-layer validation.

DO NOT PUT HERE:
  - ``/v1/images*`` request/engine/job helpers — those go in ``images.helpers``.

LONGEVITY:
  - This root utils file is a **temporary shared home**.
  - TODO(#5227, P1.2): tidy up / absorb into the images family (e.g. ``images/media.py``
    or equivalent) in the P1.2 image modality PR; do not treat this file as the
    long-term owner.
  - ``images.helpers`` is the longer home for endpoint-specific logic through
    P0.2/P0.3 until P1.2 further splits it.

See ``openai/README.md`` and ``images/README.md`` (utils vs helpers, no overlap).
"""

import base64
import io
import math

import PIL.Image

SUPPORTED_LAYERED_RESOLUTIONS = (640, 1024)
SUPPORTED_LAYERED_LAYERS_RANGE = range(2, 11)

# Target pixel area used to resolve aspect-ratio size strings (e.g., "16:9")
# into concrete dimensions. Matches the ~1 megapixel default canvas of the
# Qwen-Image family; keep in sync with calculate_dimensions() call sites in
# vllm_omni/diffusion/models/qwen_image/.
RATIO_SIZE_TARGET_AREA = 1024 * 1024
# Dimensions resolved from an aspect ratio are aligned to multiples of this
# many pixels so they stay valid for the Qwen-Image family VAE.
RATIO_SIZE_ALIGNMENT = 32


def parse_size(size_str: str, *, allow_ratio: bool = False) -> tuple[int, int]:
    """Parse size string to width and height tuple.

    Args:
        size_str: Size in format "WIDTHxHEIGHT" (e.g., "1024x1024"). When
            ``allow_ratio`` is True, aspect-ratio format "W:H" (e.g., "1:1",
            "16:9") is also accepted and resolved to the nearest
            RATIO_SIZE_ALIGNMENT-aligned dimensions with a total pixel area of
            about RATIO_SIZE_TARGET_AREA.
        allow_ratio: Whether to accept the "W:H" aspect-ratio format.

    Returns:
        Tuple of (width, height)

    Raises:
        ValueError: If size format is invalid
    """
    if not size_str or not isinstance(size_str, str):
        raise ValueError(
            f"Size must be a non-empty string in format 'WIDTHxHEIGHT' (e.g., '1024x1024'), got: {size_str}"
        )

    if allow_ratio and ":" in size_str:
        return _parse_ratio_size(size_str)

    parts = size_str.split("x")
    if len(parts) != 2:
        raise ValueError(
            f"Invalid size format: '{size_str}'. Expected format: 'WIDTHxHEIGHT' (e.g., '1024x1024')"
            + (" or aspect-ratio 'W:H' (e.g., '16:9')" if allow_ratio else "")
            + ". Did you mean to use 'x' as separator?"
        )

    try:
        width = int(parts[0])
        height = int(parts[1])
    except ValueError:
        raise ValueError(f"Invalid size format: '{size_str}'. Width and height must be integers.")

    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid size: {width}x{height}. Width and height must be positive integers.")

    return width, height


def _parse_ratio_size(size_str: str) -> tuple[int, int]:
    """Resolve an aspect-ratio size string (e.g., "16:9") to pixel dimensions.

    The ratio is scaled to RATIO_SIZE_TARGET_AREA total pixels and each side is
    rounded to the nearest multiple of RATIO_SIZE_ALIGNMENT.

    Raises:
        ValueError: If the ratio format is invalid or too extreme to resolve
            to aligned positive dimensions.
    """
    parts = size_str.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid size format: '{size_str}'. Aspect-ratio format is 'W:H' (e.g., '16:9').")

    try:
        ratio_w = int(parts[0])
        ratio_h = int(parts[1])
    except ValueError:
        raise ValueError(f"Invalid size format: '{size_str}'. Aspect-ratio values must be integers.")

    if ratio_w <= 0 or ratio_h <= 0:
        raise ValueError(f"Invalid size: '{size_str}'. Aspect-ratio values must be positive integers.")

    ratio = ratio_w / ratio_h
    width = round(math.sqrt(RATIO_SIZE_TARGET_AREA * ratio) / RATIO_SIZE_ALIGNMENT) * RATIO_SIZE_ALIGNMENT
    height = round(math.sqrt(RATIO_SIZE_TARGET_AREA / ratio) / RATIO_SIZE_ALIGNMENT) * RATIO_SIZE_ALIGNMENT

    if width <= 0 or height <= 0:
        raise ValueError(
            f"Invalid size: '{size_str}'. Aspect ratio is too extreme to resolve to dimensions "
            f"aligned to {RATIO_SIZE_ALIGNMENT} pixels."
        )

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


def prepare_image_for_output_format(image: PIL.Image.Image, format: str) -> PIL.Image.Image:
    fmt = format.lower()
    if fmt not in {"jpg", "jpeg"}:
        return image

    if image.mode == "RGB":
        return image

    if image.mode in {"RGBA", "LA"} or (image.mode == "P" and "transparency" in image.info):
        alpha_image = image.convert("RGBA")
        flattened = PIL.Image.new("RGB", alpha_image.size, (255, 255, 255))
        flattened.paste(alpha_image, mask=alpha_image.getchannel("A"))
        return flattened

    return image.convert("RGB")


def encode_image_base64_with_compression(
    image: PIL.Image.Image, format: str = "png", output_compression: int = 100
) -> str:
    """Encode a PIL image to a base64 image string.

    Args:
        image: PIL Image object.
        format: Output image format, such as "png", "jpeg", or "webp".
        output_compression: Compression level (0-100), where 100 keeps the
            best quality for lossy formats and uses the lowest PNG compression.

    Returns:
        Base64-encoded image string.
    """
    buffer = io.BytesIO()
    image = prepare_image_for_output_format(image, format)
    pil_format = "jpeg" if format == "jpg" else format
    save_kwargs = {}
    if format in ("jpg", "jpeg", "webp"):
        save_kwargs["quality"] = output_compression
    elif format == "png":
        save_kwargs["compress_level"] = max(0, min(9, 9 - output_compression // 11))

    image.save(buffer, format=pil_format, **save_kwargs)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


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
