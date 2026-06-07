# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
OpenAI-compatible protocol definitions for image generation.

This module provides Pydantic models that follow the OpenAI DALL-E API specification
for text-to-image generation, with vllm-omni specific extensions.
"""

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from vllm_omni.entrypoints.openai.image_api_utils import validate_layered_layers


class ResponseFormat(str, Enum):
    """Image response format"""

    B64_JSON = "b64_json"
    URL = "url"  # Not implemented in PoC


class ImageGenerationRequest(BaseModel):
    """
    OpenAI DALL-E compatible image generation request.

    Follows the OpenAI Images API specification with vllm-omni extensions
    for advanced diffusion parameters.
    """

    # Required fields
    prompt: str = Field(..., description="Text description of the desired image(s)")
    bot_task: str | None = Field(
        None,
        description="Task mode for the model (e.g., 'cot' enables chain-of-thought generation). "
        "Only supported by specific diffusion models.",
    )

    # OpenAI standard fields
    model: str | None = Field(
        default=None,
        description="Model to use (optional, uses server's configured model if omitted)",
    )
    n: int = Field(default=1, ge=1, le=10, description="Number of images to generate")
    size: str | None = Field(
        default=None,
        description="Image dimensions in WIDTHxHEIGHT format (e.g., '1024x1024', uses model defaults if omitted)",
    )
    response_format: ResponseFormat = Field(default=ResponseFormat.B64_JSON, description="Format of the returned image")
    user: str | None = Field(default=None, description="User identifier for tracking")
    layers: int | None = Field(
        default=None,
        description="Number of output layers for layered image models. Supported range: 3-10.",
    )

    @field_validator("size")
    @classmethod
    def validate_size(cls, v):
        """Validate size parameter.

        Accepts any string in 'WIDTHxHEIGHT' format (e.g., '1024x1024', '512x768').
        No restrictions on specific dimensions - models can handle arbitrary sizes.
        """
        if v is None:
            return None
        # Validate string format
        if not isinstance(v, str) or "x" not in v:
            raise ValueError("size must be in format 'WIDTHxHEIGHT' (e.g., '1024x1024')")
        return v

    @field_validator("response_format")
    @classmethod
    def validate_response_format(cls, v):
        """Validate response format - only b64_json is supported."""
        if v is not None and v != ResponseFormat.B64_JSON:
            raise ValueError(f"Only 'b64_json' response format is supported, got: {v}")
        return v

    @field_validator("layers")
    @classmethod
    def validate_layers(cls, v):
        """Validate the layers parameter for layered image models."""
        return validate_layered_layers(v)

    # vllm-omni extensions for diffusion control
    negative_prompt: str | None = Field(default=None, description="Text describing what to avoid in the image")
    system_prompt: str | None = Field(
        default=None, description="Custom system prompt. Used when --use_system_prompt is custom"
    )
    use_system_prompt: str | None = Field(
        default=None,
        description="System prompt type. Options: None, dynamic, en_vanilla, "
        "en_recaption, en_think_recaption, en_unified, custom",
    )

    @field_validator("use_system_prompt")
    @classmethod
    def validate_use_system_prompt(cls, v):
        """Validate system prompt type."""
        valid_types = [None, "dynamic", "en_vanilla", "en_recaption", "en_think_recaption", "en_unified", "custom"]
        if v not in valid_types:
            raise ValueError(f"Invalid use_system_prompt type: {v}. Must be one of: {valid_types[1:] + [None]}")
        return v

    num_inference_steps: int | None = Field(
        default=None,
        ge=1,
        le=200,
        description="Number of diffusion sampling steps (uses model defaults if not specified)",
    )
    guidance_scale: float | None = Field(
        default=None,
        ge=0.0,
        le=20.0,
        description="Classifier-free guidance scale (uses model defaults if not specified)",
    )
    true_cfg_scale: float | None = Field(
        default=None,
        ge=0.0,
        le=20.0,
        description="True CFG scale (model-specific parameter, may be ignored if not supported)",
    )
    flow_shift: float | None = Field(
        default=None, description="Scheduler flow_shift (sigma shift) for flow-matching diffusion models."
    )
    extra_params: dict[str, Any] | None = Field(
        default=None,
        description="Optional model-specific parameters passed directly to the model's extra_args.",
    )
    seed: int | None = Field(default=None, description="Random seed for reproducibility")
    generator_device: str | None = Field(
        default=None,
        description="Device for the seeded torch.Generator (e.g. 'cpu', 'cuda'). Defaults to the runner's device.",
    )

    # vllm-omni extension for per-request LoRA.
    # This mirrors the `extra_body.lora` convention in /v1/chat/completions.
    lora: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Optional LoRA adapter for this request. Expected shape: "
            "{name/path/scale/int_id}. Field names are flexible "
            "(e.g. name|lora_name|adapter, path|lora_path|local_path, "
            "scale|lora_scale, int_id|lora_int_id)."
        ),
    )

    # VAE memory optimizations (set at model init, included for completeness)
    vae_use_slicing: bool | None = Field(default=False, description="Enable VAE slicing")
    vae_use_tiling: bool | None = Field(default=False, description="Enable VAE tiling")

    # Output format for generated images
    output_format: str | None = Field(
        default=None,
        description="Output image format: 'png', 'jpeg', or 'webp'. Defaults to 'png'.",
    )


class ImageData(BaseModel):
    """Single generated image data"""

    b64_json: str | None = Field(default=None, description="Base64-encoded PNG image")
    url: str | None = Field(default=None, description="Image URL (not implemented)")
    revised_prompt: str | None = Field(default=None, description="Revised prompt (OpenAI compatibility, always null)")


class ImageGenerationResponse(BaseModel):
    """
    OpenAI DALL-E compatible image generation response.

    Returns generated images with metadata.
    """

    created: int = Field(..., description="Unix timestamp of when the generation completed")
    data: list[ImageData] = Field(..., description="Array of generated images")
    output_format: str = Field(None, description="The output format of the image generation")
    size: str = Field(None, description="The size of the image generated")
    cot_output: str | None = Field(
        None,
        description="Chain-of-thought text output from the AR stage. "
        "Only present for image editing (IT2I) with CoT-enabled models.",
    )


class ImageEditARDeltaChunk(BaseModel):
    """Streaming chunk carrying a text delta from the image-edit AR stage."""

    object: Literal["image.edit.chunk"] = "image.edit.chunk"
    type: Literal["ar_delta"] = "ar_delta"
    delta: str = Field(..., description="Text delta generated by the AR stage")
    index: int = Field(default=0, description="Completion index for the AR stage output")
    created: int = Field(..., description="Unix timestamp of when the stream was created")
    model: str = Field(..., description="Model used for the image edit request")
    metrics: dict[str, Any] | None = Field(
        default=None,
        description="Optional vLLM-Omni per-stage metrics snapshot for benchmark clients.",
    )


class ImageEditImageChunk(BaseModel):
    """Streaming chunk carrying the final image-edit result."""

    object: Literal["image.edit.chunk"] = "image.edit.chunk"
    type: Literal["image"] = "image"
    data: list[ImageData] = Field(..., description="Array of generated images")
    output_format: str = Field(..., description="The output format of the image generation")
    size: str = Field(..., description="The generated image size")
    created: int = Field(..., description="Unix timestamp of when the stream was created")
    model: str = Field(..., description="Model used for the image edit request")
    metrics: dict[str, Any] | None = Field(
        default=None,
        description="Optional vLLM-Omni per-stage metrics snapshot for benchmark clients.",
    )


class ImageEditStreamError(BaseModel):
    """Streaming error chunk emitted before the terminal [DONE] event."""

    object: Literal["error"] = "error"
    created: int = Field(..., description="Unix timestamp of when the stream was created")
    model: str = Field(..., description="Model used for the image edit request")
    error: dict[str, Any] = Field(..., description="OpenAI-compatible streaming error payload")


ImageEditStreamResponse = ImageEditARDeltaChunk | ImageEditImageChunk | ImageEditStreamError
