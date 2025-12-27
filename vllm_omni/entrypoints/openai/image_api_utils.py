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

import PIL.Image


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
