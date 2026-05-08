# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
ErnieImage diffusion model for vLLM-Omni.

This module implements ERNIE-Image text-to-image generation with:
- ErnieImageTransformer2DModel: Custom DiT transformer
- ErnieImagePipeline: Full generation pipeline
"""

from vllm_omni.diffusion.models.ernie_image.ernie_image_transformer import ErnieImageTransformer2DModel
from vllm_omni.diffusion.models.ernie_image.pipeline_ernie_image import ErnieImagePipeline

__all__ = ["ErnieImageTransformer2DModel", "ErnieImagePipeline"]
