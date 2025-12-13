# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Shared helper utilities for OpenAI-compatible image generation API.

This module provides common helper functions used by both the image generation
and image editing endpoints. All functions work with plain Python types to
maintain separation from the FastAPI HTTP layer.
"""

import base64
import io
import math
from typing import Optional

import PIL.Image
from fastapi import HTTPException, UploadFile, status
from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.image_model_profiles import DiffusionModelProfile
from vllm_omni.entrypoints.openai.protocol.images import ImageGenerationRequest

logger = init_logger(__name__)


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


def validate_image_file(file: UploadFile, max_size_mb: int = 4) -> None:
    """Validate uploaded image file.

    Args:
        file: Uploaded file from FastAPI
        max_size_mb: Maximum file size in megabytes

    Raises:
        HTTPException: If file is invalid or too large
    """
    if not file or not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Image file is required")

    allowed_types = {"image/png", "image/jpeg", "image/jpg", "image/webp"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image format: {file.content_type}. Supported: PNG, JPEG, WebP",
        )

    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)

    if file_size > max_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File too large: {file_size / 1024 / 1024:.1f}MB (max: {max_size_mb}MB)",
        )


async def read_image_file(file: UploadFile) -> PIL.Image.Image:
    """Read uploaded file into PIL Image.

    Args:
        file: Uploaded file from FastAPI

    Returns:
        PIL Image object

    Raises:
        HTTPException: If file cannot be read as image
    """
    try:
        contents = await file.read()
        image = PIL.Image.open(io.BytesIO(contents))

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to read image file: {str(e)}")


def build_generation_params(
    request: ImageGenerationRequest,
    profile: DiffusionModelProfile,
    width: int,
    height: int,
) -> dict:
    """Build generation kwargs for AsyncOmniDiffusion.generate().

    Args:
        request: Image generation request
        profile: Model profile with defaults and constraints
        width: Image width in pixels
        height: Image height in pixels

    Returns:
        Dictionary of kwargs to pass to diffusion_engine.generate()
    """
    gen_params = {
        "prompt": request.prompt,
        "height": height,
        "width": width,
        "num_outputs_per_prompt": request.n,
    }

    num_steps = request.num_inference_steps or profile.default_num_inference_steps
    if num_steps > profile.max_num_inference_steps:
        raise ValueError(
            f"num_inference_steps={num_steps} exceeds maximum for "
            f"{profile.model_name} (max={profile.max_num_inference_steps})"
        )
    gen_params["num_inference_steps"] = num_steps

    if request.negative_prompt and profile.supports_negative_prompt:
        gen_params["negative_prompt"] = request.negative_prompt

    if profile.supports_guidance_scale:
        if profile.force_guidance_scale is not None:
            gen_params["guidance_scale"] = profile.force_guidance_scale
            if request.guidance_scale is not None and request.guidance_scale != profile.force_guidance_scale:
                logger.warning(
                    f"Ignoring guidance_scale={request.guidance_scale}, "
                    f"{profile.model_name} requires guidance_scale={profile.force_guidance_scale}"
                )
        elif request.guidance_scale is not None:
            gen_params["guidance_scale"] = request.guidance_scale
        elif profile.default_guidance_scale is not None:
            gen_params["guidance_scale"] = profile.default_guidance_scale

    if profile.supports_true_cfg_scale:
        cfg = request.true_cfg_scale or profile.default_true_cfg_scale
        if cfg is not None:
            gen_params["true_cfg_scale"] = cfg
    elif request.true_cfg_scale is not None:
        logger.warning(
            f"Ignoring true_cfg_scale={request.true_cfg_scale}, {profile.model_name} doesn't support this parameter"
        )

    if request.seed is not None:
        gen_params["seed"] = request.seed

    return gen_params


def build_edit_params(
    prompt: str,
    image: PIL.Image.Image,
    profile: DiffusionModelProfile,
    negative_prompt: Optional[str],
    n: int,
    size: Optional[str],
    num_inference_steps: Optional[int],
    guidance_scale: Optional[float],
    true_cfg_scale: Optional[float],
    seed: Optional[int],
) -> dict:
    """Build edit generation kwargs for AsyncOmniDiffusion.generate().

    Args:
        prompt: Edit instruction
        image: Input PIL Image
        profile: Model profile with defaults and constraints
        negative_prompt: Optional negative prompt
        n: Number of images to generate
        size: Optional output size (if None, auto-calculate from input)
        num_inference_steps: Optional custom step count
        guidance_scale: Optional CFG scale
        true_cfg_scale: Optional true CFG scale
        seed: Optional random seed

    Returns:
        Dictionary of kwargs to pass to diffusion_engine.generate()
    """
    # Calculate dimensions
    if size:
        width, height = parse_size(size)
    else:
        # Auto-calculate maintaining ~1024x1024 area
        img_width, img_height = image.size
        aspect_ratio = img_width / img_height
        target_area = 1024 * 1024

        width = int(math.sqrt(target_area * aspect_ratio))
        height = int(width / aspect_ratio)

        # Round to multiples of 32 (VAE constraint)
        width = (width // 32) * 32
        height = (height // 32) * 32

        logger.info(
            f"Auto-calculated dimensions from input {img_width}x{img_height}: "
            f"{width}x{height} (aspect ratio: {aspect_ratio:.2f})"
        )

    gen_params = {
        "prompt": prompt,
        "pil_image": image,  # Pass PIL Image to AsyncOmniDiffusion
        "height": height,
        "width": width,
        "num_outputs_per_prompt": n,
    }

    num_steps = num_inference_steps or profile.default_num_inference_steps
    if num_steps > profile.max_num_inference_steps:
        raise ValueError(
            f"num_inference_steps={num_steps} exceeds maximum for "
            f"{profile.model_name} (max={profile.max_num_inference_steps})"
        )
    gen_params["num_inference_steps"] = num_steps

    if negative_prompt and profile.supports_negative_prompt:
        gen_params["negative_prompt"] = negative_prompt

    if profile.supports_guidance_scale:
        if profile.force_guidance_scale is not None:
            gen_params["guidance_scale"] = profile.force_guidance_scale
            if guidance_scale is not None and guidance_scale != profile.force_guidance_scale:
                logger.warning(
                    f"Ignoring guidance_scale={guidance_scale}, "
                    f"{profile.model_name} requires guidance_scale={profile.force_guidance_scale}"
                )
        elif guidance_scale is not None:
            gen_params["guidance_scale"] = guidance_scale
        elif profile.default_guidance_scale is not None:
            gen_params["guidance_scale"] = profile.default_guidance_scale

    if profile.supports_true_cfg_scale:
        cfg = true_cfg_scale or profile.default_true_cfg_scale
        if cfg is not None:
            gen_params["true_cfg_scale"] = cfg
    elif true_cfg_scale is not None:
        logger.warning(f"Ignoring true_cfg_scale={true_cfg_scale}, {profile.model_name} doesn't support this parameter")

    if seed is not None:
        gen_params["seed"] = seed

    return gen_params
