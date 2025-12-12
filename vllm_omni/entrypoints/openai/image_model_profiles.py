# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Model profiles for diffusion models in the image generation API.

This module defines configuration profiles that encapsulate model-specific
behavior, parameters, and constraints for different diffusion models.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DiffusionModelProfile:
    """Configuration profile for a diffusion model.

    This profile encapsulates all model-specific behavior including default
    parameters, supported features, and constraints. It allows the image
    generation API to handle different models with varying capabilities
    through a unified interface.

    Attributes:
        model_name: HuggingFace model identifier (e.g., "Qwen/Qwen-Image")
        default_num_inference_steps: Default number of diffusion sampling steps
        max_num_inference_steps: Maximum allowed inference steps
        default_height: Default image height in pixels
        default_width: Default image width in pixels
        supports_guidance_scale: Whether model supports classifier-free guidance
        default_guidance_scale: Default guidance scale value (if supported)
        force_guidance_scale: Force guidance scale to specific value (overrides user input)
        supports_true_cfg_scale: Whether model supports Qwen-style true CFG
        default_true_cfg_scale: Default true CFG scale (for Qwen-Image)
        supports_negative_prompt: Whether model supports negative prompts
        omni_kwargs: Additional kwargs to pass to Omni() constructor
    """

    # Model identification
    model_name: str

    # Inference step configuration
    default_num_inference_steps: int
    max_num_inference_steps: int

    # Image dimensions
    default_height: int = 1024
    default_width: int = 1024

    # Classifier-free guidance (CFG) configuration
    supports_guidance_scale: bool = True
    default_guidance_scale: Optional[float] = None
    force_guidance_scale: Optional[float] = None  # Override user input if set

    # Qwen-specific true CFG scale
    supports_true_cfg_scale: bool = False
    default_true_cfg_scale: Optional[float] = None

    # Feature support
    supports_negative_prompt: bool = True

    # Omni constructor arguments
    omni_kwargs: dict = field(default_factory=dict)


# Model profile registry
MODEL_PROFILES: dict[str, DiffusionModelProfile] = {
    "Qwen/Qwen-Image": DiffusionModelProfile(
        model_name="Qwen/Qwen-Image",
        default_num_inference_steps=50,
        max_num_inference_steps=200,
        default_height=1024,
        default_width=1024,
        supports_guidance_scale=True,
        default_guidance_scale=1.0,
        force_guidance_scale=None,
        supports_true_cfg_scale=True,
        default_true_cfg_scale=4.0,
        supports_negative_prompt=True,
        omni_kwargs={
            "vae_use_slicing": True,  # Memory optimization for large images
            "vae_use_tiling": True,  # Memory optimization for large images
        },
    ),
    "Tongyi-MAI/Z-Image-Turbo": DiffusionModelProfile(
        model_name="Tongyi-MAI/Z-Image-Turbo",
        default_num_inference_steps=9,  # Turbo is optimized for ~9 steps
        max_num_inference_steps=16,  # Prevent excessive steps for Turbo
        default_height=1024,
        default_width=1024,
        supports_guidance_scale=True,
        default_guidance_scale=None,
        force_guidance_scale=0.0,  # Turbo is distilled for CFG=0, must force
        supports_true_cfg_scale=False,  # Qwen-specific, Z-Image doesn't use it
        supports_negative_prompt=True,
        omni_kwargs={},  # No special VAE settings needed for Z-Image
    ),
}


def get_model_profile(model_name: str) -> DiffusionModelProfile:
    """Get the configuration profile for a diffusion model.

    Args:
        model_name: HuggingFace model identifier

    Returns:
        DiffusionModelProfile for the specified model

    Raises:
        ValueError: If model_name is not in the supported models registry
    """
    if model_name not in MODEL_PROFILES:
        supported = list_supported_models()
        raise ValueError(f"Unsupported model: '{model_name}'. Supported models: {', '.join(supported)}")
    return MODEL_PROFILES[model_name]


def list_supported_models() -> list[str]:
    """Get a list of all supported diffusion model names.

    Returns:
        List of model names that can be used with the image generation API
    """
    return list(MODEL_PROFILES.keys())
