# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for image generation model profiles.
"""

import pytest

from vllm_omni.entrypoints.openai.image_model_profiles import (
    MODEL_PROFILES,
    DiffusionModelProfile,
    get_model_profile,
    list_supported_models,
)


def test_model_profiles_exist():
    """Test that model profiles are registered"""
    assert len(MODEL_PROFILES) >= 2
    assert "Qwen/Qwen-Image" in MODEL_PROFILES
    assert "Tongyi-MAI/Z-Image-Turbo" in MODEL_PROFILES


def test_qwen_image_profile():
    """Test Qwen-Image profile configuration"""
    profile = MODEL_PROFILES["Qwen/Qwen-Image"]

    assert profile.model_name == "Qwen/Qwen-Image"
    assert profile.default_num_inference_steps == 50
    assert profile.max_num_inference_steps == 200
    assert profile.supports_true_cfg_scale is True
    assert profile.default_true_cfg_scale == 4.0
    assert profile.default_guidance_scale == 1.0
    assert profile.force_guidance_scale is None
    assert profile.supports_negative_prompt is True
    assert "vae_use_slicing" in profile.omni_kwargs
    assert "vae_use_tiling" in profile.omni_kwargs


def test_zimage_turbo_profile():
    """Test Z-Image Turbo profile configuration"""
    profile = MODEL_PROFILES["Tongyi-MAI/Z-Image-Turbo"]

    assert profile.model_name == "Tongyi-MAI/Z-Image-Turbo"
    assert profile.default_num_inference_steps == 9
    assert profile.max_num_inference_steps == 16
    assert profile.supports_true_cfg_scale is False
    assert profile.default_true_cfg_scale is None
    assert profile.force_guidance_scale == 0.0  # Forced for Turbo
    assert profile.supports_negative_prompt is True
    assert profile.omni_kwargs == {}  # No special kwargs


def test_get_model_profile_success():
    """Test successful model profile lookup"""
    profile = get_model_profile("Qwen/Qwen-Image")
    assert isinstance(profile, DiffusionModelProfile)
    assert profile.model_name == "Qwen/Qwen-Image"

    profile = get_model_profile("Tongyi-MAI/Z-Image-Turbo")
    assert isinstance(profile, DiffusionModelProfile)
    assert profile.model_name == "Tongyi-MAI/Z-Image-Turbo"


def test_get_model_profile_unsupported():
    """Test that unsupported model raises ValueError"""
    with pytest.raises(ValueError, match="Unsupported model"):
        get_model_profile("UnknownModel/Test")

    with pytest.raises(ValueError, match="Supported models"):
        get_model_profile("invalid-model-name")


def test_list_supported_models():
    """Test listing supported models"""
    models = list_supported_models()

    assert isinstance(models, list)
    assert len(models) >= 2
    assert "Qwen/Qwen-Image" in models
    assert "Tongyi-MAI/Z-Image-Turbo" in models


def test_profile_dataclass_defaults():
    """Test DiffusionModelProfile dataclass defaults"""
    profile = DiffusionModelProfile(
        model_name="Test/Model",
        default_num_inference_steps=25,
        max_num_inference_steps=100,
    )

    # Test defaults
    assert profile.default_height == 1024
    assert profile.default_width == 1024
    assert profile.supports_guidance_scale is True
    assert profile.default_guidance_scale is None
    assert profile.force_guidance_scale is None
    assert profile.supports_true_cfg_scale is False
    assert profile.default_true_cfg_scale is None
    assert profile.supports_negative_prompt is True
    assert profile.omni_kwargs == {}


def test_qwen_zimage_differences():
    """Test key differences between Qwen and Z-Image profiles"""
    qwen = get_model_profile("Qwen/Qwen-Image")
    zimage = get_model_profile("Tongyi-MAI/Z-Image-Turbo")

    # Qwen uses true_cfg_scale, Z-Image doesn't
    assert qwen.supports_true_cfg_scale is True
    assert zimage.supports_true_cfg_scale is False

    # Z-Image forces guidance_scale to 0.0, Qwen doesn't
    assert qwen.force_guidance_scale is None
    assert zimage.force_guidance_scale == 0.0

    # Z-Image has lower max steps (Turbo model)
    assert qwen.max_num_inference_steps > zimage.max_num_inference_steps

    # Qwen has VAE optimizations, Z-Image doesn't
    assert len(qwen.omni_kwargs) > 0
    assert len(zimage.omni_kwargs) == 0
