# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm_image/modeling_glm_image.py
# Copyright 2025 The vLLM team.
# Copyright 2025 The ZhipuAI Team.
# Copyright 2025 The HuggingFace Inc. team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only GLM-Image model compatible with HuggingFace weights."""

import os
from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BatchFeature
from transformers.models.glm_image.configuration_glm_image import (
    GlmImageConfig,
    GlmImageTextConfig,
    GlmImageVisionConfig,
    GlmImageVQVAEConfig,
)
from transformers.models.glm_image.processing_glm_image import GlmImageProcessor
from vllm.config import CacheConfig, MultiModalConfig, VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.distributed import utils as dist_utils
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention.mm_encoder_attention import (
    MMEncoderAttention,
)
from vllm.model_executor.layers.conv import Conv2dLayer
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import (
    SupportsMRoPE,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.qwen2 import Qwen2MLP as GlmImageTextMLP
from vllm.model_executor.models.utils import (
    WeightsMapper,
    make_empty_intermediate_tensors_factory,
    make_layers,
)
from vllm.model_executor.models.vision import get_vit_attn_backend
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFeatureSpec,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import ImageProcessorItems, MultiModalDataItems
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
)
from vllm.sequence import IntermediateTensors
from vllm.utils.tensor_schema import TensorSchema, TensorShape
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from vllm_omni.model_executor.models.output_templates import OmniOutput

logger = init_logger(__name__)


# === Multimodal Processing ===


class GlmImagePixelInputs(TensorSchema):
    """
    Schema for GLM-Image pixel inputs.

    Dimensions:
        - np: Number of patches (total across all images)
        - cpp: channels * patch_size * patch_size
        - ni: Number of images
        - g: Grid dimensions (3 for temporal, height, width)
    """

    type: Literal["pixel_values"] = "pixel_values"

    pixel_values: Annotated[torch.Tensor, TensorShape("np", "cpp")]
    image_grid_thw: Annotated[torch.Tensor, TensorShape("ni", 3)]


class GlmImageProcessingInfo(BaseProcessingInfo):
    """
    Processing information for GLM-Image model.

    GLM-Image is an image generation model that uses:
    - Vision encoder for encoding source images (image-to-image)
    - VQ-VAE for tokenizing image features
    - Text model for generating image tokens
    """

    def get_hf_config(self) -> GlmImageConfig:
        return self.ctx.get_hf_config(GlmImageConfig)

    def get_hf_processor(self, **kwargs: object):
        """Get the GlmImageProcessor.

        GLM-Image has a special directory structure where:
        - Model (AR) is in: {base}/vision_language_encoder/
        - Processor is in: {base}/processor/

        Since model_subdir is used to load the AR model, the model_config.model
        path points to vision_language_encoder/. We need to go up one level
        and into processor/ to load the GlmImageProcessor.
        """

        # Get the model path from config
        model_path = self.ctx.model_config.model

        # Check if we're in a subdirectory (vision_language_encoder)
        # and need to go to processor/ instead
        if model_path.endswith("vision_language_encoder") or "/vision_language_encoder" in model_path:
            # Go up one level and into processor/
            base_path = os.path.dirname(model_path.rstrip("/"))
            processor_path = os.path.join(base_path, "processor")
        else:
            # Try processor subdirectory of current path
            processor_path = os.path.join(model_path, "processor")
            if not os.path.exists(processor_path):
                processor_path = model_path

        # Load processor directly from the correct path
        return GlmImageProcessor.from_pretrained(
            processor_path,
            trust_remote_code=self.ctx.model_config.trust_remote_code,
            **kwargs,
        )

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        # GLM-Image is an image GENERATION model that supports:
        # - Text-to-image (t2i): no multimodal input needed
        # - Image-to-image (i2i): source images provided as input
        #
        # For i2i mode, we support up to 1 image as condition.
        # The model architecture supports multiple images but typical usage is 1.
        return {"image": 1}

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> int:
        """
        Calculate the number of image tokens for a given image size.

        GLM-Image processes images through a patch embedding with patch_size=16,
        then quantizes through VQ-VAE. The number of tokens is:
        (image_height // patch_size) * (image_width // patch_size)
        """
        hf_config = self.get_hf_config()
        vision_config = hf_config.vision_config
        patch_size = vision_config.patch_size

        # Number of patches in each dimension
        num_patches_h = image_height // patch_size
        num_patches_w = image_width // patch_size

        return num_patches_h * num_patches_w

    def get_max_image_tokens(self) -> int:
        """
        Get the maximum number of image tokens.

        Based on the default image size (2048x2048) and patch size (16).
        """
        hf_config = self.get_hf_config()
        vision_config = hf_config.vision_config

        # Default max size
        image_size = getattr(vision_config, "image_size", 2048)
        patch_size = getattr(vision_config, "patch_size", 16)

        max_patches = (image_size // patch_size) ** 2
        return max_patches

    def get_image_size_with_most_features(self) -> tuple[int, int]:
        """
        Get the image size that produces the most features.

        Returns:
            Tuple of (width, height)
        """
        hf_config = self.get_hf_config()
        vision_config = hf_config.vision_config
        image_size = getattr(vision_config, "image_size", 2048)
        return (image_size, image_size)


class GlmImageDummyInputsBuilder(BaseDummyInputsBuilder[GlmImageProcessingInfo]):
    """
    Builds dummy inputs for GLM-Image model profiling.

    GLM-Image is an image GENERATION model that supports:
    - Text-to-image (t2i): no multimodal input needed
    - Image-to-image (i2i): source images provided as input

    For profiling purposes, we need to provide dummy multimodal data when
    mm_counts["image"] > 0, which happens because get_supported_mm_limits
    declares image support.
    """

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        """
        Generate dummy text for profiling.

        When images are requested (i2i mode profiling), include image placeholders
        so that _get_prompt_updates can find and replace them. Each <|image|> token
        will be expanded to grid_h * grid_w tokens by the replacement function.
        """
        num_images = mm_counts.get("image", 0)

        if num_images > 0:
            # i2i mode: include image placeholders that will be expanded
            # The <|image|> placeholder will be tokenized to image_token_id (167855)
            # and then replaced by _get_prompt_updates with actual grid tokens
            return "<|image|>" * num_images + "A beautiful image."
        else:
            # t2i mode: simple text prompt, no image placeholders needed
            return "A beautiful image."

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        """
        Generate dummy multimodal data for profiling.

        When images are requested, provide actual dummy images so the vision
        encoder can be profiled. The image size is set to maximize features
        for accurate memory profiling.
        """
        num_images = mm_counts.get("image", 0)

        # No images requested: t2i mode, no multimodal data needed
        if num_images == 0:
            return {}

        hf_config = self.info.get_hf_config()
        vision_config = hf_config.vision_config

        # Use image size from config for maximum features profiling
        image_size = getattr(vision_config, "image_size", 2048)
        width = height = image_size

        image_overrides = mm_options.get("image") if mm_options else None

        return {
            "image": self._get_dummy_images(
                width=width,
                height=height,
                num_images=num_images,
                overrides=image_overrides,
            )
        }


class GlmImageMultiModalProcessor(BaseMultiModalProcessor[GlmImageProcessingInfo]):
    """
    Multimodal processor for GLM-Image.

    Handles:
    - Image preprocessing and tokenization
    - Prompt construction with image placeholders
    - Grid dimension calculation for M-RoPE position encoding
    """

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        """
        Call the HuggingFace processor.

        For text-to-image mode (no images), we need to:
        1. Build the prompt with target grid dimensions
        2. Build the image_grid_thw tensor for M-RoPE position encoding

        For image-to-image mode:
        1. Process source images through the image processor
        2. Build prompt with image placeholders expanded
        3. Build image_grid_thw including source and target grids
        """
        processor = self.info.get_hf_processor()

        # Get target dimensions from mm_kwargs or use defaults
        target_h = mm_kwargs.get("target_h", 1024) if mm_kwargs else 1024
        target_w = mm_kwargs.get("target_w", 1024) if mm_kwargs else 1024

        if not mm_data or not mm_data.get("images"):
            # Text-to-image mode
            if processor is not None:
                # Build messages format expected by processor
                messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

                # Use apply_chat_template which handles target dimensions
                hf_inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    target_h=target_h,
                    target_w=target_w,
                    return_dict=True,
                    return_tensors="pt",
                )

                return hf_inputs
            else:
                # Fallback: just tokenize (this won't work properly for generation)
                tokenizer = self.info.get_tokenizer()
                prompt_ids = tokenizer.encode(prompt)
                return BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")

        # Image-to-image mode
        # NOTE: Use "images" (plural) - this is what vLLM's ImageProcessorItems.get_processor_data() returns
        images = mm_data.get("images")
        if not isinstance(images, list):
            images = [images]

        logger.debug(
            f"_call_hf_processor i2i: num_images={len(images)}, image_types={[type(img).__name__ for img in images]}"
        )

        if processor is not None:
            # Build messages with image objects directly in content
            # This is how GlmImageProcessor expects images - embedded in the content dict
            # NOT as a separate images= parameter
            #
            # IMPORTANT: Remove <|image|> placeholders from prompt since apply_chat_template
            # will automatically insert them for each image in content. Having both leads to
            # index out of bounds when processing image_grid_thw.
            clean_prompt = prompt.replace("<|image|>", "")
            content = []
            for img in images:
                content.append({"type": "image", "image": img})
            content.append({"type": "text", "text": clean_prompt})
            messages = [{"role": "user", "content": content}]

            logger.debug(f"_call_hf_processor: calling apply_chat_template with {len(images)} images in content")

            # Use apply_chat_template - processor will process images when they're in content
            hf_inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                target_h=target_h,
                target_w=target_w,
                return_dict=True,
                return_tensors="pt",
            )

            logger.debug(f"_call_hf_processor: apply_chat_template returned keys: {list(hf_inputs.keys())}")

            # IMPORTANT (i2i): vLLM multimodal encoder must see source-only grids
            # (matching pixel_values and number of images), but M-RoPE needs full
            # grids (source + target) to compute correct decode positions.
            image_grid_thw = hf_inputs.get("image_grid_thw")
            if image_grid_thw is not None:
                # Preserve full grids for M-RoPE.
                hf_inputs["mrope_image_grid_thw"] = image_grid_thw

                # Expose source-only grids for MM.
                # In most i2i requests, we process one prompt at a time here,
                # so `len(images)` is the number of source images.
                num_source_images = len(images)
                if image_grid_thw.shape[0] != num_source_images:
                    source_grids = image_grid_thw[:num_source_images]
                    hf_inputs["image_grid_thw"] = source_grids
                    logger.debug(
                        "_call_hf_processor: adjusted image_grid_thw for MM from %s to %s (num_source_images=%d)",
                        tuple(image_grid_thw.shape),
                        tuple(source_grids.shape),
                        num_source_images,
                    )

            # Debug: Analyze input_ids for image tokens
            input_ids = hf_inputs.get("input_ids")
            if input_ids is not None:
                if hasattr(input_ids, "tolist"):
                    ids_list = input_ids.tolist()
                    if isinstance(ids_list[0], list):
                        ids_list = ids_list[0]  # Unbatch
                else:
                    ids_list = list(input_ids)

                # Get image token ID from config
                hf_config = self.info.get_hf_config()
                image_token_id = getattr(hf_config, "image_token_id", 167855)

                # Count image tokens
                image_token_count = ids_list.count(image_token_id)
                logger.debug(
                    f"_call_hf_processor: input_ids length={len(ids_list)}, "
                    f"image_token_id={image_token_id}, "
                    f"image_token_count={image_token_count}"
                )

                # Log first/last few tokens to understand structure
                logger.debug(f"_call_hf_processor: first 20 tokens: {ids_list[:20]}")
                logger.debug(f"_call_hf_processor: last 20 tokens: {ids_list[-20:]}")

                # Find positions of image tokens
                image_positions = [i for i, t in enumerate(ids_list) if t == image_token_id]
                if image_positions:
                    logger.debug(f"_call_hf_processor: image token positions (first 10): {image_positions[:10]}")

            return hf_inputs
        else:
            # Fallback without processor - this is not ideal but prevents crashes
            logger.warning("GlmImageProcessor not available, using fallback for i2i")
            tokenizer = self.info.get_tokenizer()
            hf_config = self.info.get_hf_config()

            # Get image token
            image_token_id = getattr(hf_config, "image_token_id", 167855)
            try:
                image_token = tokenizer.convert_ids_to_tokens(image_token_id)
            except Exception:
                image_token = "<|image|>"

            # Build prompt with image placeholders
            image_placeholders = image_token * len(images)
            full_prompt = f"{image_placeholders}{prompt}"
            prompt_ids = tokenizer.encode(full_prompt)
            return BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")

    def _apply_hf_processor_mm_only(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        """
        Apply the HF processor on the multi-modal data only.

        GLM-Image requires special handling because apply_chat_template always
        adds a target <|image|> placeholder in addition to source image placeholders.
        This causes an IndexError when the HF processor tries to find grid info
        for the target placeholder (which doesn't exist for source-only processing).

        Solution: Call the image processor directly to get pixel_values and
        image_grid_thw, bypassing apply_chat_template's target handling.
        """
        mm_counts = mm_items.get_all_counts()
        num_images = mm_counts.get("image", 0)

        if num_images == 0:
            # No images - call parent implementation
            return super()._apply_hf_processor_mm_only(
                mm_items=mm_items,
                hf_processor_mm_kwargs=hf_processor_mm_kwargs,
                tokenization_kwargs=tokenization_kwargs,
            )

        # For i2i mode, we need to process images directly with the image processor
        # to avoid the apply_chat_template target placeholder issue
        processor = self.info.get_hf_processor()
        image_processor = processor.image_processor

        # Get images from mm_items
        images = mm_items.get_items("image", ImageProcessorItems)
        image_list = [images.get(i) for i in range(images.get_count())]

        logger.debug(f"_apply_hf_processor_mm_only: processing {len(image_list)} images directly")

        # Process images directly with image processor
        image_inputs = image_processor(
            images=image_list,
            return_tensors="pt",
        )

        # Get grid info for source images only (no target)
        pixel_values = image_inputs.get("pixel_values")
        image_grid_thw = image_inputs.get("image_grid_thw")
        if image_grid_thw is not None and image_grid_thw.shape[0] != num_images:
            # Be defensive: some processors may include extra target grids.
            image_grid_thw = image_grid_thw[:num_images]
            image_inputs["image_grid_thw"] = image_grid_thw

        logger.debug(
            f"_apply_hf_processor_mm_only: pixel_values shape=\
                {pixel_values.shape if pixel_values is not None else None}, "
            f"image_grid_thw shape={image_grid_thw.shape if image_grid_thw is not None else None}"
        )

        # Build input_ids with image token placeholders
        # The _get_prompt_updates returns PromptReplacement(target=[image_token_id], ...)
        # which needs to find image tokens in input_ids to replace them.
        # We need to include one image_token_id per image so the replacement can work.
        tokenizer = self.info.get_tokenizer()
        image_token_id = tokenizer.convert_tokens_to_ids("<|image|>")

        # Build input_ids: [image_token] * num_images + tokenized text
        # This way _apply_prompt_updates can find the image tokens and replace them
        dummy_text = self.dummy_inputs.get_dummy_text(mm_counts)
        text_ids = tokenizer.encode(dummy_text, add_special_tokens=False)
        input_ids = [image_token_id] * num_images + text_ids

        logger.debug(
            f"_apply_hf_processor_mm_only: built input_ids with {num_images} image tokens + {len(text_ids)} text tokens"
        )

        return BatchFeature(
            dict(
                input_ids=[input_ids],
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            ),
            tensor_type="pt",
        )

    def _apply_hf_processor_main(
        self,
        prompt: str | list[int],
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
        *,
        enable_hf_prompt_update: bool,
    ) -> tuple[list[int], BatchFeature, bool]:
        """
        Override to handle GLM-Image i2i mode correctly.

        Problem: When vLLM processes cached mm items (enable_hf_prompt_update=False),
        the base implementation:
        1. Gets prompt_ids from _apply_hf_processor_text_only (no image tokens)
        2. Gets mm_data from _apply_hf_processor_mm_only
        3. Returns is_update_applied=False

        This causes _apply_prompt_updates to fail because prompt_ids has no image tokens.

        Solution: For i2i mode, we build prompt_ids that include image placeholders,
        and return is_update_applied=False so _apply_prompt_updates can expand them.
        """
        mm_counts = mm_items.get_all_counts()
        num_images = mm_counts.get("image", 0)

        logger.debug(f"_apply_hf_processor_main: mm_counts={mm_counts}, num_images={num_images}")

        if num_images == 0 or enable_hf_prompt_update:
            # t2i mode or normal flow - use parent implementation
            return super()._apply_hf_processor_main(
                prompt=prompt,
                mm_items=mm_items,
                hf_processor_mm_kwargs=hf_processor_mm_kwargs,
                tokenization_kwargs=tokenization_kwargs,
                enable_hf_prompt_update=enable_hf_prompt_update,
            )

        # i2i mode with enable_hf_prompt_update=False (cache miss scenario)
        # We need to build prompt_ids with image placeholders
        logger.debug(f"_apply_hf_processor_main: i2i mode with enable_hf_prompt_update=False, num_images={num_images}")

        # Get mm data from our overridden _apply_hf_processor_mm_only
        mm_processed_data = self._apply_hf_processor_mm_only(
            mm_items=mm_items,
            hf_processor_mm_kwargs=hf_processor_mm_kwargs,
            tokenization_kwargs=tokenization_kwargs,
        )

        # In this path we do NOT call HF apply_chat_template, so we must still
        # provide full grids (source + target) for M-RoPE to compute decode positions.
        # Keep `image_grid_thw` source-only for MM batching/validation.
        try:
            source_grid_thw = mm_processed_data.get("image_grid_thw")
            if source_grid_thw is not None and isinstance(source_grid_thw, torch.Tensor):
                # Compute target grid following HF GlmImageProcessor: factor=32.
                # Prefer explicit target_h/target_w if present, otherwise fall back.
                target_h = (
                    hf_processor_mm_kwargs.get("target_h")
                    if isinstance(hf_processor_mm_kwargs.get("target_h"), int)
                    else None
                )
                target_w = (
                    hf_processor_mm_kwargs.get("target_w")
                    if isinstance(hf_processor_mm_kwargs.get("target_w"), int)
                    else None
                )
                if target_h is None or target_w is None:
                    # Some callers pass generation size as height/width.
                    target_h = (
                        hf_processor_mm_kwargs.get("height")
                        if isinstance(hf_processor_mm_kwargs.get("height"), int)
                        else 1024
                    )
                    target_w = (
                        hf_processor_mm_kwargs.get("width")
                        if isinstance(hf_processor_mm_kwargs.get("width"), int)
                        else 1024
                    )

                factor = 32
                target_h = (target_h // factor) * factor
                target_w = (target_w // factor) * factor
                token_h = target_h // factor
                token_w = target_w // factor
                target_grid = torch.tensor([[1, token_h, token_w]], dtype=source_grid_thw.dtype)

                mm_processed_data["mrope_image_grid_thw"] = torch.cat([source_grid_thw, target_grid], dim=0)
        except Exception:
            # Best-effort only; M-RoPE has additional fallbacks.
            pass

        # Build prompt_ids with image placeholders
        # _apply_prompt_updates will replace each [image_token_id] with expanded tokens
        tokenizer = self.info.get_tokenizer()
        image_token_id = tokenizer.convert_tokens_to_ids("<|image|>")

        if isinstance(prompt, str):
            # Match HF GlmImageProcessor behavior: append target grid tokens + BOS.
            # This helps M-RoPE/grid parsing and keeps i2i vs t2i behavior aligned.
            try:
                grid_bos = getattr(tokenizer, "grid_bos_token", "")
                grid_eos = getattr(tokenizer, "grid_eos_token", "")
                bos = getattr(tokenizer, "bos_token", "")

                # Use the same target sizes we used for mrope grids when available.
                target_h = (
                    hf_processor_mm_kwargs.get("target_h")
                    if isinstance(hf_processor_mm_kwargs.get("target_h"), int)
                    else None
                )
                target_w = (
                    hf_processor_mm_kwargs.get("target_w")
                    if isinstance(hf_processor_mm_kwargs.get("target_w"), int)
                    else None
                )
                if target_h is None or target_w is None:
                    target_h = (
                        hf_processor_mm_kwargs.get("height")
                        if isinstance(hf_processor_mm_kwargs.get("height"), int)
                        else 1024
                    )
                    target_w = (
                        hf_processor_mm_kwargs.get("width")
                        if isinstance(hf_processor_mm_kwargs.get("width"), int)
                        else 1024
                    )

                factor = 32
                target_h = (target_h // factor) * factor
                target_w = (target_w // factor) * factor
                token_h = target_h // factor
                token_w = target_w // factor

                expanded_prompt = f"{prompt}{grid_bos}{token_h} {token_w}{grid_eos}{bos}"
                text_ids = tokenizer.encode(expanded_prompt, add_special_tokens=False)
            except Exception:
                text_ids = tokenizer.encode(prompt, add_special_tokens=False)
        else:
            text_ids = list(prompt)

        # Prepend image placeholders - one per image
        prompt_ids = [image_token_id] * num_images + text_ids

        logger.debug(f"_apply_hf_processor_main: built prompt_ids with {num_images} image placeholders")

        # Return is_update_applied=False so _apply_prompt_updates will expand the placeholders
        return prompt_ids, mm_processed_data, False

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        """
        Get the multimodal field configuration.

        For GLM-Image i2i mode:
        - image_grid_thw has been sliced in _call_hf_processor to only include source images
        - pixel_values has shape [total_patches, C, H, W] - only for source images

        For t2i mode:
        - No pixel_values, no source images - return empty config
        """
        result = {}

        # Get image_grid_thw if present (already sliced in _call_hf_processor)
        image_grid_thw = hf_inputs.get("image_grid_thw")

        if "pixel_values" in hf_inputs and image_grid_thw is not None:
            # i2i mode: pixel_values contains patches for source images
            # image_grid_thw has already been sliced to only include source grids
            num_source_images = len(image_grid_thw)
            logger.debug(
                f"_get_mm_fields_config: num_source_images={num_source_images}, image_grid_thw={image_grid_thw.shape}"
            )

            if num_source_images > 0:
                # Calculate grid sizes for source images
                image_grid_sizes = image_grid_thw.prod(-1)

                result["pixel_values"] = MultiModalFieldConfig.flat_from_sizes("image", image_grid_sizes)

                # Register image_grid_thw - it's been sliced in _call_hf_processor
                # to only include source image grids, so batching will work correctly
                result["image_grid_thw"] = MultiModalFieldConfig.batched("image")

        logger.debug(f"_get_mm_fields_config: result keys: {list(result.keys())}")

        return result

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        """
        Return whether the HF processor applies prompt updates.

        For GLM-Image i2i mode, the HF processor's apply_chat_template already
        expands <|image|> to N tokens (e.g., 4096 for 64x64 grid).

        By returning True, we tell vLLM that HF processor DID apply prompt updates,
        so vLLM will use _find_mm_placeholders to locate the expanded tokens
        instead of trying to apply replacements.

        For t2i mode (no images), there are no image placeholders to expand.
        """
        # Check if we have images (i2i mode)
        num_images = mm_items.get_all_counts().get("image", 0)
        if num_images > 0:
            logger.debug(
                f"_hf_processor_applies_updates: returning True for i2i mode "
                f"(num_images={num_images}) - HF processor already expanded tokens"
            )
            return True

        # For t2i mode (no images), use default behavior
        return True

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        """
        Get prompt updates for image tokens.

        For GLM-Image image-to-image mode, the HF processor's apply_chat_template
        already expands each <|image|> placeholder to the correct number of
        image tokens (grid_h * grid_w tokens per source image).

        The HF processor does:
        1. Replace each <|image|> with num_image_tokens copies of <|placeholder|>
        2. Replace all <|placeholder|> back to <|image|>

        So the tokenized input already has the expanded tokens. We use
        target=[image_token_id] to match each occurrence of the image token,
        similar to how Qwen2VL handles this pattern.

        We use image_grid_thw from out_mm_kwargs to get the actual processed grid
        size, following the Qwen2VL pattern. This is critical because the HF processor
        resizes images, so the original image size doesn't match the processed size.

        For t2i mode (no images), we return an empty list since there are no
        image placeholders to replace.
        """
        hf_config = self.info.get_hf_config()

        # Get image token ID - this is the token that appears multiple times
        # in the tokenized input after HF processor expansion
        image_token_id = getattr(hf_config, "image_token_id", 167855)

        # Debug: log mm_items info
        logger.debug(f"_get_prompt_updates: image_token_id={image_token_id}")
        logger.debug(f"_get_prompt_updates: mm_items modalities={list(mm_items.get_all_counts().keys())}")
        logger.debug(f"_get_prompt_updates: mm_items counts={mm_items.get_all_counts()}")
        logger.debug(
            f"_get_prompt_updates: out_mm_kwargs key={list(out_mm_kwargs.get_data().keys()) if out_mm_kwargs else None}"
        )

        # Check if there are any images to process
        num_images = mm_items.get_count("image", strict=False)
        if num_images == 0:
            # t2i mode: no images, no prompt updates needed
            logger.debug("_get_prompt_updates: no images, returning empty list (t2i mode)")
            return []

        def get_replacement_glm_image(item_idx: int) -> list[int]:
            """
            Return replacement token IDs for an image placeholder.

            For GLM-Image, each source image is represented by grid_h * grid_w tokens.
            These are placeholder tokens that will be replaced by actual VQ-VAE
            tokens during model forward pass.

            IMPORTANT: We use image_grid_thw from out_mm_kwargs to get the actual
            processed grid size. The HF processor resizes images, so the original
            image size (from mm_items) doesn't match the actual token count.
            """
            # Get grid info from out_mm_kwargs (set by _get_mm_fields_config)
            out_item = out_mm_kwargs["image"][item_idx]
            grid_thw = out_item.get("image_grid_thw")

            if grid_thw is not None:
                grid_data = grid_thw.data if hasattr(grid_thw, "data") else grid_thw
                if isinstance(grid_data, torch.Tensor):
                    # grid is [t, h, w] - for images, t=1, so num_tokens = h * w
                    num_tokens = int(grid_data.prod().item())
                else:
                    num_tokens = int(grid_data[0] * grid_data[1] * grid_data[2])
                logger.debug(
                    f"get_replacement_glm_image: item_idx={item_idx}, \
                        grid={grid_data.tolist() if isinstance(grid_data, torch.Tensor) else grid_data},\
                              num_tokens={num_tokens}"
                )
            else:
                # Fallback: use default 1024x1024 grid size
                # (1024/16) * (1024/16) = 64 * 64 = 4096 tokens
                num_tokens = 64 * 64
                logger.warning(
                    f"get_replacement_glm_image: item_idx={item_idx}, \
                    no grid_thw found, using default num_tokens={num_tokens}"
                )

            return [image_token_id] * num_tokens

        return [
            PromptReplacement(
                modality="image",
                # Use [token_id] to match each occurrence of image token
                # The HF processor has already expanded <|image|> to multiple tokens
                target=[image_token_id],
                replacement=get_replacement_glm_image,
            ),
        ]


# === VQ-VAE Components ===


class GlmImageVQVAEVectorQuantizer(nn.Module):
    """
    Vector Quantizer module for GLM-Image VQ-VAE (Inference-optimized).

    This module quantizes continuous latent vectors into discrete codebook vectors
    using L2-normalized distance computation for better stability.

    Key differences from Chameleon's VQ-VAE:
    - GLM-Image uses L2 normalization on both input and codebook embeddings
    - Distance is computed as cosine similarity in normalized space

    Optimizations for inference (compared to transformers implementation):
    1. Uses matmul + argmax(similarity) instead of einsum + argmin(distance)
       - Mathematically equivalent: argmin(2 - 2*sim) = argmax(sim)
       - More efficient and clearer for L2-normalized vectors
    2. Removes redundant normalization (transformers normalizes twice)
    3. Removes training-only components (loss, straight-through estimator, beta)
    4. Directly returns quantized vectors without gradient preservation

    Args:
        config: GlmImageVQVAEConfig containing:
            - num_embeddings: Number of codebook vectors (typically 16384)
            - embed_dim: Dimension of each embedding vector (typically 2048)

    Mathematical Verification:
        For L2-normalized vectors where ||z|| = ||e|| = 1:
        - distance = ||z - e||^2 = 2 - 2*(z·e) = 2(1 - cosine_similarity)
        - Therefore: argmin(distance) ≡ argmax(cosine_similarity)
        This equivalence has been verified numerically (see verify_vqvae_correctness.py)
    """

    def __init__(self, config: GlmImageVQVAEConfig):
        super().__init__()
        self.num_embeddings = config.num_embeddings
        self.embedding_dim = config.embed_dim

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)

    def forward(self, hidden_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize the input hidden states.

        Args:
            hidden_state: Input tensor of shape (batch, channels, height, width)

        Returns:
            Tuple of:
                - hidden_state_quant: Quantized tensor, same shape as input
                - min_encoding_indices: Codebook indices of shape
                  (batch * height * width,)
        """
        batch_size, channels, height, width = hidden_state.shape

        # Permute to (batch, height, width, channels) and flatten for processing
        hidden_state_flat = hidden_state.permute(0, 2, 3, 1).reshape(-1, self.embedding_dim)

        # L2 normalize both hidden states and embeddings
        # This is the key difference from Chameleon's implementation
        hidden_state_normalized = F.normalize(hidden_state_flat, p=2, dim=-1)
        embedding_normalized = F.normalize(self.embedding.weight, p=2, dim=-1)

        # Compute cosine similarity (since both are L2 normalized)
        # Higher similarity = closer match, so we negate for argmin
        # Using matmul for efficiency: (N, D) @ (D, K) -> (N, K)
        similarity = torch.matmul(hidden_state_normalized, embedding_normalized.t())

        # Find nearest codebook entry (highest similarity)
        min_encoding_indices = torch.argmax(similarity, dim=1)

        # Get quantized vectors using normalized embeddings
        # For inference, we directly return the quantized vectors without
        # straight-through estimator (no gradients needed)
        hidden_state_quant = embedding_normalized[min_encoding_indices]

        # Reshape back to (batch, height, width, channels)
        # then (batch, channels, height, width)
        hidden_state_quant = (
            hidden_state_quant.view(batch_size, height, width, self.embedding_dim).permute(0, 3, 1, 2).contiguous()
        )

        return hidden_state_quant, min_encoding_indices


class GlmImageVQVAE(nn.Module):
    """
    VQ-VAE module for GLM-Image.

    Unlike Chameleon's VQ-VAE which includes a full encoder, GLM-Image's VQ-VAE
    only contains:
    - quant_conv: Projects from latent_channels to embed_dim
    - quantize: Vector quantizer
    - post_quant_conv: Projects from embed_dim back to latent_channels

    The encoder functionality is handled by GlmImageVisionModel instead.

    This module is always in eval mode as the VQ-VAE is frozen during inference.

    Args:
        config: GlmImageVQVAEConfig
    """

    def __init__(self, config: GlmImageVQVAEConfig):
        super().__init__()
        self.config = config

        # Vector quantizer
        self.quantize = GlmImageVQVAEVectorQuantizer(config)

        # Convolutions for projecting to/from embedding space
        # Using vLLM's optimized Conv2dLayer
        self.quant_conv = Conv2dLayer(
            in_channels=config.latent_channels,
            out_channels=config.embed_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.post_quant_conv = Conv2dLayer(
            in_channels=config.embed_dim,
            out_channels=config.latent_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        # VQ-VAE is always frozen in GLM-Image
        self.eval()

    def encode(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input features into quantized latent codes.

        Args:
            hidden_states: Input tensor of shape (batch, latent_channels, height, width)
                          This is typically the output from GlmImageVisionModel reshaped
                          into spatial format.

        Returns:
            Tuple of:
                - quant: Quantized tensor of shape (batch, embed_dim, height, width)
                - indices: Codebook indices of shape (batch * height * width,)
        """
        # Project to embedding dimension
        hidden_states = self.quant_conv(hidden_states)

        # Quantize
        quant, indices = self.quantize(hidden_states)

        return quant, indices

    @property
    def dtype(self) -> torch.dtype:
        """Get the dtype of the model."""
        return self.quant_conv.weight.dtype

    @property
    def device(self) -> torch.device:
        """Get the device of the model."""
        return self.quant_conv.weight.device


# === Vision Model Components ===


class GlmImageVisionMLP(nn.Module):
    """
    MLP module for GLM-Image vision model.

    Uses GELU activation with standard fc1 -> fc2 structure.
    Key difference from Glm4vVisionMLP: uses GELU instead of SwiGLU.
    """

    def __init__(
        self,
        config: GlmImageVisionConfig,
        quant_config: QuantizationConfig | None = None,
        multimodal_config: MultiModalConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data" if multimodal_config else False
        self.fc1 = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1",
            disable_tp=use_data_parallel,
        )
        self.fc2 = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2",
            disable_tp=use_data_parallel,
        )
        self.act_fn = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.fc1(x)
        x = self.act_fn(x)
        x, _ = self.fc2(x)
        return x


class GlmImageVisionAttention(nn.Module):
    """
    Multi-headed attention for GLM-Image vision model.

    Key differences from Glm4vVisionAttention:
    - No RoPE - uses learned position embeddings instead
    - Uses standard qkv projection (not separate q, k, v)
    - attention_bias from config controls bias in linear layers
    """

    def __init__(
        self,
        config: GlmImageVisionConfig,
        quant_config: QuantizationConfig | None = None,
        multimodal_config: MultiModalConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data" if multimodal_config else False
        self.tp_size = 1 if use_data_parallel else get_tensor_model_parallel_world_size()

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        attention_bias = getattr(config, "attention_bias", True)

        self.num_heads_per_partition = dist_utils.divide(self.num_heads, self.tp_size)

        # QKV projection - uses bias based on config
        self.qkv = QKVParallelLinear(
            hidden_size=self.embed_dim,
            head_size=self.head_dim,
            total_num_heads=self.num_heads,
            total_num_kv_heads=self.num_heads,  # No GQA in vision model
            bias=attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv",
            disable_tp=use_data_parallel,
        )
        self.proj = RowParallelLinear(
            input_size=self.embed_dim,
            output_size=self.embed_dim,
            bias=attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.proj",
            disable_tp=use_data_parallel,
        )

        # MMEncoderAttention for efficient vision attention
        self.attn = MMEncoderAttention(
            num_heads=self.num_heads_per_partition,
            head_size=self.head_dim,
            scale=self.head_dim**-0.5,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # hidden_states: [seq_len, hidden_size] (no batch dim)
        seq_len = hidden_states.shape[0]

        # QKV projection
        qkv, _ = self.qkv(hidden_states)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for attention: [seq, hidden] -> [1, seq, heads, head_dim]
        q = q.view(seq_len, self.num_heads_per_partition, self.head_dim).unsqueeze(0)
        k = k.view(seq_len, self.num_heads_per_partition, self.head_dim).unsqueeze(0)
        v = v.view(seq_len, self.num_heads_per_partition, self.head_dim).unsqueeze(0)

        # No RoPE in GLM-Image vision model - position info comes from embeddings

        # Apply attention
        attn_output = self.attn(
            query=q,
            key=k,
            value=v,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        # Reshape back: [1, seq, heads, head_dim] -> [seq, hidden]
        attn_output = attn_output.view(seq_len, -1)

        # Output projection
        output, _ = self.proj(attn_output)
        return output


class GlmImageVisionPatchEmbed(nn.Module):
    """
    Patch embedding for GLM-Image vision model.

    Key difference from Glm4vVisionPatchEmbed:
    - Uses 2D convolution (no temporal dimension)
    - GLM-Image processes single images, not videos
    """

    def __init__(self, config: GlmImageVisionConfig) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size

        # 2D convolution for patch embedding
        self.proj = Conv2dLayer(
            in_channels=self.in_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding=0,
            bias=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: Packed pixel values of shape
                [total_patches, in_channels * patch_size * patch_size]

        Returns:
            Patch embeddings of shape [total_patches, embed_dim]
        """
        target_dtype = self.proj.weight.dtype
        # Reshape from [N, C*P*P] to [N, C, P, P]
        hidden_states = hidden_states.view(-1, self.in_channels, self.patch_size, self.patch_size)
        # Conv2d and flatten: [N, C, P, P] -> [N, embed_dim, 1, 1] -> [N, embed_dim]
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states


class GlmImageVisionEmbeddings(nn.Module):
    """
    Vision embeddings for GLM-Image.

    Uses learned 2D position embeddings with bilinear interpolation
    for variable resolution support.

    Key difference from Glm4vVisionEmbeddings:
    - Uses bilinear interpolation (not bicubic) for position embedding adaptation
    """

    def __init__(self, config: GlmImageVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

        # GLM-Image uses bilinear, Glm4v uses bicubic
        self.interpolation_mode = "bilinear"

    def forward(
        self,
        embeddings: torch.Tensor,
        lengths: list[int] | torch.Tensor,
        image_shapes: torch.Tensor,
        h_coords: torch.Tensor,
        w_coords: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add adapted position embeddings to patch embeddings.

        Args:
            embeddings: Patch embeddings [total_seq, embed_dim]
            lengths: Sequence length for each image
            image_shapes: [num_images, 3] with (t, h, w) for each image
            h_coords: Height coordinates for each patch [total_seq]
            w_coords: Width coordinates for each patch [total_seq]

        Returns:
            Embeddings with position encoding added [total_seq, embed_dim]
        """
        pos_embed_weight = self.position_embedding.weight
        hidden_size = pos_embed_weight.shape[1]
        total_seq = h_coords.shape[0]
        device = pos_embed_weight.device

        # Handle empty sequence case
        if total_seq == 0:
            adapted_pos_embed = torch.empty(0, hidden_size, device=device, dtype=pos_embed_weight.dtype)
        else:
            # Convert to tensors if needed
            if isinstance(lengths, list):
                lengths = torch.tensor(lengths, device=device, dtype=torch.long)
            if not isinstance(image_shapes, torch.Tensor):
                image_shapes = torch.tensor(image_shapes, device=device, dtype=torch.long)

            # Prepare 2D position embedding for interpolation
            orig_size_sq = pos_embed_weight.shape[0]
            orig_size = int(orig_size_sq**0.5)
            pos_embed_2d = (
                pos_embed_weight.view(orig_size, orig_size, hidden_size)
                .permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
                .unsqueeze(0)  # [1, C, H, W]
                .to(device=device, dtype=torch.float32)
            )

            # Calculate target dimensions for each patch
            target_h = torch.cat([image_shapes[i, 1].repeat(lengths[i]) for i in range(len(lengths))]).to(
                device=device, dtype=torch.float32
            )
            target_w = torch.cat([image_shapes[i, 2].repeat(lengths[i]) for i in range(len(lengths))]).to(
                device=device, dtype=torch.float32
            )

            # Normalize coordinates to [-1, 1] for grid_sample
            h_coords = h_coords.to(device=device, dtype=torch.float32)
            w_coords = w_coords.to(device=device, dtype=torch.float32)
            norm_w = ((w_coords + 0.5) / target_w) * 2 - 1
            norm_h = ((h_coords + 0.5) / target_h) * 2 - 1

            # Create sampling grid [1, total_seq, 1, 2]
            grid = torch.stack((norm_w, norm_h), dim=-1).unsqueeze(0).unsqueeze(2)

            # Bilinear interpolation (GLM-Image uses bilinear, not bicubic)
            interpolated_embed = F.grid_sample(
                pos_embed_2d,
                grid,
                mode=self.interpolation_mode,
                align_corners=False,
                padding_mode="border",
            )

            # Reshape: [1, C, total_seq, 1] -> [total_seq, C]
            adapted_pos_embed = (interpolated_embed.squeeze(0).squeeze(-1).permute(1, 0)).to(pos_embed_weight.dtype)

        # Add position embedding to patch embeddings
        embeddings = embeddings + adapted_pos_embed.to(embeddings.device)
        return embeddings


class GlmImageVisionBlock(nn.Module):
    """
    Transformer block for GLM-Image vision model.

    Key differences from Glm4vVisionBlock:
    - Uses LayerNorm instead of RMSNorm
    - No RoPE position embeddings (handled in GlmImageVisionEmbeddings)
    - Uses GELU MLP instead of SwiGLU
    """

    def __init__(
        self,
        config: GlmImageVisionConfig,
        quant_config: QuantizationConfig | None = None,
        multimodal_config: MultiModalConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = GlmImageVisionAttention(
            config,
            quant_config=quant_config,
            multimodal_config=multimodal_config,
            prefix=f"{prefix}.attn",
        )
        self.mlp = GlmImageVisionMLP(
            config,
            quant_config=quant_config,
            multimodal_config=multimodal_config,
            prefix=f"{prefix}.mlp",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Pre-norm attention
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attn(
            hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        hidden_states = residual + hidden_states

        # Pre-norm MLP
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class GlmImageVisionModel(nn.Module):
    """
    Vision encoder for GLM-Image.

    Key differences from Glm4vVisionTransformer:
    - No RoPE - uses learned position embeddings with bilinear interpolation
    - No merger, downsample, or post-processing layers
    - Uses LayerNorm instead of RMSNorm in blocks
    - No temporal dimension (images only, no video)
    """

    def __init__(
        self,
        config: GlmImageVisionConfig,
        quant_config: QuantizationConfig | None = None,
        multimodal_config: MultiModalConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.patch_size = config.patch_size
        self.spatial_merge_size = config.spatial_merge_size

        # Patch embedding
        self.patch_embed = GlmImageVisionPatchEmbed(config)

        # Position embeddings
        self.embeddings = GlmImageVisionEmbeddings(config)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                GlmImageVisionBlock(
                    config,
                    quant_config=quant_config,
                    multimodal_config=multimodal_config,
                    prefix=f"{prefix}.blocks.{i}",
                )
                for i in range(config.depth)
            ]
        )

        # Attention backend selection
        self.attn_backend = get_vit_attn_backend(
            head_size=self.head_dim,
            dtype=torch.get_default_dtype(),
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device

    def compute_position_ids(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Compute position IDs for each patch based on grid dimensions.

        Args:
            grid_thw: [num_images, 3] with (t, h, w) for each image

        Returns:
            Position IDs [total_patches, 2] with (h_pos, w_pos) for each patch
        """
        pos_ids = []
        for t, h, w in grid_thw:
            # Create h and w position grids
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)

            # Reshape for spatial merge
            hpos_ids = (
                hpos_ids.reshape(
                    h // self.spatial_merge_size,
                    self.spatial_merge_size,
                    w // self.spatial_merge_size,
                    self.spatial_merge_size,
                )
                .permute(0, 2, 1, 3)
                .flatten()
            )

            wpos_ids = (
                wpos_ids.reshape(
                    h // self.spatial_merge_size,
                    self.spatial_merge_size,
                    w // self.spatial_merge_size,
                    self.spatial_merge_size,
                )
                .permute(0, 2, 1, 3)
                .flatten()
            )

            # Stack and repeat for temporal dimension
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))

        return torch.cat(pos_ids, dim=0)

    def compute_attn_mask_seqlen(
        self,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor | None:
        """Compute max sequence length for flash attention."""
        if (
            self.attn_backend == AttentionBackendEnum.FLASH_ATTN
            or self.attn_backend == AttentionBackendEnum.ROCM_AITER_FA
        ):
            # Return as 1D tensor for vLLM 0.14.0+ API compatibility
            return (cu_seqlens[1:] - cu_seqlens[:-1]).max().unsqueeze(0)
        return None

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through vision encoder.

        Args:
            pixel_values: Packed pixel values
                [total_patches, num_channels * patch_size * patch_size]
            grid_thw: [num_images, 3] with (t, h, w) for each image

        Returns:
            Hidden states [total_patches, hidden_size]
        """
        # Patch embedding
        hidden_states = self.patch_embed(pixel_values.to(self.device, self.dtype))

        # Compute position IDs
        position_ids = self.compute_position_ids(grid_thw)

        # Compute cumulative sequence lengths for attention
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype=torch.int32
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        cu_seqlens = cu_seqlens.to(self.device)

        # Get sequence lengths for position embedding
        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()

        # Add position embeddings
        hidden_states = self.embeddings(
            hidden_states,
            seqlens,
            grid_thw,
            position_ids[:, 0].to(hidden_states.device),
            position_ids[:, 1].to(hidden_states.device),
        )

        # Compute max seqlen for flash attention
        max_seqlen = self.compute_attn_mask_seqlen(cu_seqlens)

        # Transformer blocks
        for blk in self.blocks:
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )

        return hidden_states


# === Text Model Components ===


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_glm_image_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply GLM-Image rotary position embedding to query and key tensors.

    Args:
        q: Query tensor [num_tokens, num_heads, head_dim]
        k: Key tensor [num_tokens, num_kv_heads, head_dim]
        cos: Cosine values [num_tokens, rotary_dim]
        sin: Sine values [num_tokens, rotary_dim]

    Returns:
        Tuple of (rotated_q, rotated_k) with same shapes as input
    """
    # cos/sin shape: [num_tokens, rotary_dim]
    # Need to unsqueeze for broadcasting with heads dimension
    cos = cos.unsqueeze(1)  # [num_tokens, 1, rotary_dim]
    sin = sin.unsqueeze(1)  # [num_tokens, 1, rotary_dim]

    rotary_dim = cos.shape[-1]

    # Split into rotary and pass-through parts
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    # Apply rotary embeddings
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    # Concatenate back
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)

    return q_embed, k_embed


class GlmImageRotaryEmbedding(nn.Module):
    """
    Custom Rotary Embedding for GLM-Image with M-RoPE support.

    GLM-Image uses a 3D position encoding (temporal, height, width) with
    M-RoPE sections [8, 12, 12]. This means:
    - First 8 dims use temporal positions
    - Next 12 dims use height positions
    - Next 12 dims use width positions
    - Pattern repeats for remaining dims

    Unlike vLLM's standard MRotaryEmbedding which uses cache-based lookup,
    this implementation computes cos/sin dynamically to handle arbitrary
    position values without cache size limitations.

    This follows the transformers reference implementation exactly:
    - inv_freq is expanded for matmul with position_ids
    - freqs = inv_freq @ position_ids (matrix multiplication)
    - apply_mrope interleaves frequency chunks from different dimensions
    """

    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int = 32768,
        rope_theta: float = 10000.0,
        partial_rotary_factor: float = 1.0,
        mrope_section: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta

        # Compute rotary dimension
        self.rotary_dim = int(head_dim * partial_rotary_factor)

        # Default mrope_section for GLM-Image
        self.mrope_section = mrope_section if mrope_section is not None else [8, 12, 12]

        # Compute inverse frequencies
        # inv_freq shape: [rotary_dim // 2]
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _apply_mrope(self, freqs: torch.Tensor) -> torch.Tensor:
        """
        Apply M-RoPE section interleaving.

        For mrope_section = [8, 12, 12]:
        - Split freqs into chunks of size [8, 12, 12, 8, 12, 12, ...]
        - Take chunk[i % 3] from each split (alternating T, H, W dimensions)
        - Concatenate back

        Args:
            freqs: Frequency tensor [3, num_tokens, rotary_dim // 2]

        Returns:
            Interleaved frequencies [num_tokens, rotary_dim // 2]
        """
        # freqs shape: [3, num_tokens, rotary_dim // 2]
        # Split along last dimension according to mrope_section
        chunks = freqs.split(self.mrope_section, dim=-1)

        # Take chunk[i % 3] from each split
        # chunks[i] has shape [3, num_tokens, section_size]
        # We select dimension 0 (T), 1 (H), or 2 (W) based on i % 3
        result = torch.cat([chunk[i % 3] for i, chunk in enumerate(chunks)], dim=-1)

        return result

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position embeddings to query and key.

        Args:
            positions: Position IDs
                - Shape [num_tokens] for 1D positions (text-only)
                - Shape [3, num_tokens] for 3D M-RoPE positions (T, H, W)
            query: Query tensor [num_tokens, num_heads * head_dim]
            key: Key tensor [num_tokens, num_kv_heads * head_dim]

        Returns:
            Tuple of (rotated_query, rotated_key) with same shapes as input
        """
        # Get dimensions
        if positions.ndim == 1:
            num_tokens = positions.shape[0]
        else:
            num_tokens = positions.shape[1]

        device = positions.device
        dtype = query.dtype

        # Ensure inv_freq is on same device
        inv_freq = self.inv_freq.to(device=device, dtype=torch.float32)

        if positions.ndim == 1:
            # 1D positions: expand to 3D with same values
            # Shape: [num_tokens] -> [3, num_tokens]
            positions_3d = positions.unsqueeze(0).expand(3, -1)
        else:
            # Already 3D: [3, num_tokens]
            positions_3d = positions

        # Follow reference implementation exactly:
        # Reference: inv_freq_expanded = self.inv_freq[None, None, :, None].expand(3, bs, -1, 1)
        # Reference: position_ids_expanded = position_ids[:, :, None, :].float()  # (3, bs, 1, positions)
        # Reference: freqs = (inv_freq_expanded @ position_ids_expanded).transpose(2, 3)
        #
        # For vLLM (no batch dim):
        # inv_freq: [rotary_dim // 2]
        # positions_3d: [3, num_tokens]
        #
        # We want: freqs[i, j, k] = positions_3d[i, j] * inv_freq[k]
        # So: freqs = positions_3d[:, :, None] * inv_freq[None, None, :]
        # Shape: [3, num_tokens, 1] * [1, 1, rotary_dim // 2] = [3, num_tokens, rotary_dim // 2]

        # Compute frequencies using broadcasting (equivalent to matmul in reference)
        positions_expanded = positions_3d.unsqueeze(-1).float()  # [3, num_tokens, 1]
        inv_freq_expanded = inv_freq.unsqueeze(0).unsqueeze(0)  # [1, 1, rotary_dim // 2]
        freqs = positions_expanded * inv_freq_expanded  # [3, num_tokens, rotary_dim // 2]

        # Apply M-RoPE interleaving
        # This selects different frequency dims from different position dims
        freqs = self._apply_mrope(freqs)  # [num_tokens, rotary_dim // 2]

        # Build cos/sin embeddings
        # Concatenate freqs with itself for full rotary_dim (real and imaginary parts)
        emb = torch.cat((freqs, freqs), dim=-1)  # [num_tokens, rotary_dim]
        cos = emb.cos().to(dtype)  # [num_tokens, rotary_dim]
        sin = emb.sin().to(dtype)  # [num_tokens, rotary_dim]

        # Reshape query and key for rotary application
        # query: [num_tokens, num_heads * head_dim] -> [num_tokens, num_heads, head_dim]
        query_shape = query.shape
        key_shape = key.shape

        query = query.view(num_tokens, -1, self.head_dim)
        key = key.view(num_tokens, -1, self.head_dim)

        # Apply rotary embeddings
        query, key = apply_glm_image_rotary_pos_emb(query, key, cos, sin)

        # Reshape back
        query = query.view(query_shape)
        key = key.view(key_shape)

        return query, key


class GlmImageTextAttention(nn.Module):
    """
    Multi-headed attention for GLM-Image text model.

    Uses Grouped Query Attention (GQA) with M-RoPE position embeddings.
    """

    def __init__(
        self,
        config: GlmImageTextConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position_embeddings: int = 32768,
        quant_config: QuantizationConfig | None = None,
        bias: bool = True,
        cache_config: CacheConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # M-RoPE for 3D position encoding (temporal, height, width)
        # Use custom GlmImageRotaryEmbedding instead of vLLM's get_rope
        # to properly handle 3D positions with mrope_section interleaving
        rope_parameters = getattr(config, "rope_parameters", None)
        rope_theta = 10000.0
        partial_rotary_factor = 1.0
        mrope_section = [8, 12, 12]  # Default for GLM-Image

        if rope_parameters is not None:
            rope_theta = rope_parameters.get("rope_theta", rope_theta)
            partial_rotary_factor = rope_parameters.get("partial_rotary_factor", partial_rotary_factor)
            mrope_section = rope_parameters.get("mrope_section", mrope_section)

        self.rotary_emb = GlmImageRotaryEmbedding(
            head_dim=self.head_dim,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            partial_rotary_factor=partial_rotary_factor,
            mrope_section=mrope_section,
        )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class GlmImageTextDecoderLayer(nn.Module):
    """
    Decoder layer for GLM-Image text model.

    Key difference from standard LLaMA-style decoder:
    - Uses 4 RMSNorm layers instead of 2:
      - input_layernorm: before self-attention
      - post_self_attn_layernorm: after self-attention, before residual add
      - post_attention_layernorm: before MLP
      - post_mlp_layernorm: after MLP, before residual add
    """

    def __init__(
        self,
        config: GlmImageTextConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        max_position_embeddings = getattr(config, "max_position_embeddings", 32768)
        attention_bias = getattr(config, "attention_bias", True)

        self.self_attn = GlmImageTextAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(config, "num_key_value_heads", config.num_attention_heads),
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=attention_bias,
            cache_config=cache_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = GlmImageTextMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

        # GLM-Image uses 4 RMSNorm layers per decoder layer
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_self_attn_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_mlp_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Save residual for first add
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        # Self Attention
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)

        # Post self-attention norm and residual add
        hidden_states = self.post_self_attn_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_mlp_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # Return hidden_states and None for residual (already added)
        return hidden_states, None


class GlmImageTextModel(nn.Module):
    """
    Text model (language backbone) for GLM-Image.

    This is the decoder-only transformer that generates discrete image tokens.
    Uses M-RoPE (3D position encoding) for multimodal position awareness.
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config: GlmImageTextConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.config = config
        self.quant_config = vllm_config.quant_config
        cache_config = vllm_config.cache_config

        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size

        # Embedding layer
        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
            )
        else:
            self.embed_tokens = None

        # Decoder layers
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: GlmImageTextDecoderLayer(
                config=config,
                cache_config=cache_config,
                quant_config=vllm_config.quant_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )

        # Final norm
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = None

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def get_input_embeddings(self) -> VocabParallelEmbedding:
        return self.embed_tokens

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed input token IDs into hidden states.

        This method is required by vLLM's SupportsMultiModal interface.
        The parent multimodal model's embed_input_ids calls
        get_language_model().embed_input_ids() to get text embeddings.
        """
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_tokens(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in self.layers[self.start_layer : self.end_layer]:
            hidden_states, residual = layer(positions, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states, "residual": residual})

        hidden_states = self.norm(hidden_states)
        return hidden_states


class GlmImageModel(nn.Module):
    """
    GLM-Image model that combines Vision Encoder, VQ-VAE, and Text Model.

    This model is used for image generation tasks:
    - Image-to-Image: Source image → Vision Encoder → VQ-VAE tokens → Text Model
    - Text-to-Image: Text tokens → Text Model → Generate image tokens

    Key components:
    - visual: GlmImageVisionModel for encoding input images
    - vqmodel: GlmImageVQVAE for tokenizing image features
    - language_model: GlmImageTextModel for text/token generation

    The model uses M-RoPE (3D position encoding) for multimodal position awareness:
    - temporal: constant for image tokens, incremental for text
    - height: row position for image tokens
    - width: column position for image tokens
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self.multimodal_config = multimodal_config

        # Vision encoder
        self.visual = GlmImageVisionModel(
            config.vision_config,
            quant_config=quant_config,
            multimodal_config=multimodal_config,
            prefix=f"{prefix}.visual" if prefix else "visual",
        )

        # VQ-VAE for image tokenization (frozen)
        self.vqmodel = GlmImageVQVAE(config.vq_config)

        # Text/Language model
        self.language_model = GlmImageTextModel(
            vllm_config=vllm_config,
            config=config.text_config,
            prefix=f"{prefix}.language_model" if prefix else "language_model",
        )

        # Store special token IDs
        self.image_token_id = config.image_token_id
        self.image_start_token_id = config.image_start_token_id
        self.image_end_token_id = config.image_end_token_id

        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors

    def get_input_embeddings(self) -> VocabParallelEmbedding:
        return self.language_model.get_input_embeddings()

    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract image features using the vision encoder.

        Args:
            pixel_values: Packed pixel values
                [total_patches, num_channels * patch_size * patch_size]
            image_grid_thw: [num_images, 3] with (t, h, w) for each image

        Returns:
            Image features [total_patches, hidden_size]
        """
        return self.visual(pixel_values, image_grid_thw)

    def get_image_tokens(
        self,
        hidden_states: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """
        Tokenize image features into discrete tokens using VQ-VAE.

        Args:
            hidden_states: Image features [total_patches, hidden_size]
            image_grid_thw: [num_images, 3] with (t, h, w) for each image

        Returns:
            Discrete token indices [total_patches]
        """
        hidden_size = hidden_states.shape[-1]
        split_sizes = (image_grid_thw.prod(dim=-1)).tolist()
        hidden_states_list = torch.split(hidden_states, split_sizes, dim=0)

        all_image_tokens = []
        for i, hs in enumerate(hidden_states_list):
            grid_t, grid_h, grid_w = image_grid_thw[i].tolist()
            # Reshape to spatial format: [t, h, w, c] -> [t, c, h, w]
            hs = hs.view(grid_t, grid_h, grid_w, hidden_size)
            hs = hs.permute(0, 3, 1, 2).contiguous()
            # Encode with VQ-VAE
            _, indices = self.vqmodel.encode(hs)
            all_image_tokens.append(indices)

        return torch.cat(all_image_tokens, dim=0)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | IntermediateTensors, dict | None]:
        """
        Forward pass through the GLM-Image model.

        For image-to-image generation:
        1. Encode source images with vision encoder
        2. Tokenize features with VQ-VAE
        3. Replace placeholder tokens with actual image tokens
        4. Run through language model

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            positions: Position IDs, shape (3, seq_len) for M-RoPE
            intermediate_tensors: For pipeline parallelism
            inputs_embeds: Pre-computed embeddings (optional)
            pixel_values: Source image pixels (for image-to-image)
            image_grid_thw: Grid dimensions for source images

        Returns:
            Tuple of (hidden_states, prior_token_image_ids_info)
            prior_token_image_ids_info is a dict with VQ-VAE tokens for i2i mode
        """
        prior_token_image_ids_info = None

        # Debug: log pixel_values presence
        has_pixel_values = pixel_values is not None
        has_image_grid_thw = image_grid_thw is not None
        logger.debug(
            f"GlmImageModel.forward: has_pixel_values={has_pixel_values}, has_image_grid_thw={has_image_grid_thw}"
        )

        # Handle intermediate tensors for pipeline parallelism
        if intermediate_tensors is not None:
            hidden_states = self.language_model(
                input_ids=None,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=None,
            )
            return hidden_states, None

        # Process source images if provided (image-to-image generation)
        if pixel_values is not None and image_grid_thw is not None:
            # Determine target device from available tensors
            if input_ids is not None:
                target_device = input_ids.device
            elif inputs_embeds is not None:
                target_device = inputs_embeds.device
            else:
                target_device = positions.device

            # Encode images
            image_features = self.get_image_features(pixel_values, image_grid_thw)
            # Tokenize with VQ-VAE
            image_tokens = self.get_image_tokens(image_features, image_grid_thw)
            image_tokens = image_tokens.to(target_device)

            # Store prior_token_image_ids for diffusion stage (i2i mode)
            # The tokens need to be upsampled from d32 to d16 (2x) for the DiT
            # We store the raw tokens here; upsampling happens in ar2diffusion
            split_sizes = (image_grid_thw.prod(dim=-1)).tolist()
            image_tokens_list = torch.split(image_tokens, split_sizes, dim=0)

            # Upsample each image's tokens for DiT (from d32 to d16)
            upsampled_token_ids = []
            for i, tokens in enumerate(image_tokens_list):
                grid_t, grid_h, grid_w = image_grid_thw[i].tolist()
                # Reshape to 2D grid
                tokens_2d = tokens.view(1, 1, grid_h, grid_w)
                # Upsample by 2x (nearest neighbor)
                tokens_upsampled = F.interpolate(tokens_2d.float(), scale_factor=2, mode="nearest").to(dtype=torch.long)
                upsampled_token_ids.append(tokens_upsampled.view(-1))

            prior_token_image_ids_info = {
                "prior_token_image_ids": upsampled_token_ids,
                "image_grid_thw": image_grid_thw.tolist(),
            }

            # Debug: log prior_token_image_ids_info
            shapes = [t.shape for t in upsampled_token_ids]
            logger.info(
                f"[GlmImageModel.forward] Built prior_token_image_ids_info: "
                f"num_images={len(upsampled_token_ids)}, shapes={shapes}, "
                f"image_grid_thw={image_grid_thw.tolist()}"
            )

            # Replace placeholder tokens with actual image tokens
            # Only do this if input_ids is provided (not during profile_run)
            if input_ids is not None:
                special_image_mask = input_ids == self.image_token_id
                if special_image_mask.sum() > 0:
                    input_ids = input_ids.clone()
                    input_ids[special_image_mask] = image_tokens

        # Get embeddings
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
            input_ids = None

        # Forward through language model
        hidden_states = self.language_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

        return hidden_states, prior_token_image_ids_info


@MULTIMODAL_REGISTRY.register_processor(
    GlmImageMultiModalProcessor,
    info=GlmImageProcessingInfo,
    dummy_inputs=GlmImageDummyInputsBuilder,
)
class GlmImageForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsPP, SupportsMRoPE):
    """
    GLM-Image model for conditional image generation.

    This is the main entry point for GLM-Image in vLLM. It wraps:
    - GlmImageModel (Vision Encoder + VQ-VAE + Text Model)
    - LM Head for token prediction

    Supports:
    - Multimodal inputs (images for image-to-image generation)
    - M-RoPE (3D position encoding) for multimodal generation
    - Pipeline Parallelism
    - Image-to-Image and Text-to-Image generation
    """

    # Explicit M-RoPE support flag (also inherited from SupportsMRoPE)
    supports_mrope = True

    # GLM-Image pre-computes M-RoPE positions for both prefill and decode
    # phases (2D spatial grid encoding for generated image tokens).  This
    # flag tells the runner to use those positions instead of the default
    # linear increments during decode.
    precomputed_mrope_decode = True

    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    # Weight mapping from HuggingFace to vLLM format
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "lm_head.": "lm_head.",
            "model.language_model.": "model.language_model.",
            "model.visual.": "model.visual.",
            "model.vqmodel.": "model.vqmodel.",
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: GlmImageConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.vllm_config = vllm_config

        # Main model (Vision + VQ-VAE + Text)
        self.model = GlmImageModel(
            vllm_config=vllm_config,
            prefix=f"{prefix}.model" if prefix else "model",
        )

        # LM head for token prediction
        # GLM-Image outputs to vision_vocab_size (16512) not full vocab
        self.lm_head = ParallelLMHead(
            config.text_config.vision_vocab_size,
            config.text_config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.lm_head" if prefix else "lm_head",
        )

        # Logits processor
        self.logits_processor = LogitsProcessor(
            config.text_config.vision_vocab_size,
            soft_cap=None,
        )

        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

        # Cache for prior_token_image_ids computed in embed_multimodal
        # This is needed because vLLM's multimodal flow calls embed_multimodal first,
        # then forward. We need to pass the VQ-VAE tokens from embed_multimodal to forward.
        self._prior_token_cache: dict | None = None

    def get_input_embeddings(self) -> VocabParallelEmbedding:
        return self.model.get_input_embeddings()

    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """Extract image features using vision encoder."""
        return self.model.get_image_features(pixel_values, image_grid_thw)

    def get_image_tokens(
        self,
        hidden_states: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """Tokenize image features with VQ-VAE."""
        return self.model.get_image_tokens(hidden_states, image_grid_thw)

    def _parse_and_validate_image_input(
        self,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        **kwargs: object,
    ) -> dict | None:
        """Parse and validate image inputs."""
        if pixel_values is None:
            return None
        return {
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }

    def _process_image_input(
        self,
        image_input: dict,
    ) -> tuple[list[torch.Tensor], dict]:
        """
        Process image input through vision encoder and VQ-VAE to get text embeddings.

        For GLM-Image, we:
        1. Extract features using the vision encoder (1536 dim)
        2. Quantize features to discrete tokens using VQ-VAE
        3. Embed tokens using text embedding layer (4096 dim)
        4. Upsample VQ-VAE tokens for diffusion stage (d32 -> d16)

        This follows the same pattern as Chameleon - returning text-space embeddings
        that can be directly scattered into the input_embeds tensor.

        Returns:
            Tuple of (image_embeddings_list, prior_token_info)
            - image_embeddings_list: List of embeddings per image
            - prior_token_info: Dict with upsampled VQ-VAE tokens for diffusion stage
        """
        pixel_values = image_input["pixel_values"]
        image_grid_thw = image_input["image_grid_thw"]

        # Get image features from vision encoder
        image_features = self.model.get_image_features(pixel_values, image_grid_thw)

        # Quantize to discrete tokens using VQ-VAE
        image_tokens = self.model.get_image_tokens(image_features, image_grid_thw)

        # Get text embeddings for the image tokens
        # This converts from vision token IDs to text-space embeddings (4096 dim)
        image_embeddings = self.model.language_model.embed_input_ids(image_tokens)

        # Split by image grid sizes
        split_sizes = (image_grid_thw.prod(dim=-1)).tolist()
        image_embeddings_list = torch.split(image_embeddings, split_sizes, dim=0)
        image_tokens_list = torch.split(image_tokens, split_sizes, dim=0)

        # Upsample VQ-VAE tokens for diffusion stage (from d32 to d16)
        # This is needed for the DiT model which operates at higher resolution
        upsampled_token_ids = []
        for i, tokens in enumerate(image_tokens_list):
            grid_t, grid_h, grid_w = image_grid_thw[i].tolist()
            # Reshape to 2D grid
            tokens_2d = tokens.view(1, 1, grid_h, grid_w)
            # Upsample by 2x (nearest neighbor)
            tokens_upsampled = F.interpolate(tokens_2d.float(), scale_factor=2, mode="nearest").to(dtype=torch.long)
            # Keep as CPU tensor for proper serialization through pooling_output
            upsampled_token_ids.append(tokens_upsampled.view(-1).detach().cpu().contiguous())

        # Note: We only include prior_token_image_ids in the info dict.
        # image_grid_thw is NOT included because:
        # 1. vLLM's pooling_output expects dict[str, torch.Tensor], not mixed types
        # 2. ar2diffusion doesn't need it - the grid info is already encoded in tensor shape
        prior_token_info = {
            "prior_token_image_ids": upsampled_token_ids,
        }

        # Debug: log prior_token_info
        shapes = [t.shape for t in upsampled_token_ids]
        logger.info(
            f"[_process_image_input] Built prior_token_info: "
            f"num_images={len(upsampled_token_ids)}, shapes={shapes}, "
            f"image_grid_thw={image_grid_thw.tolist()}"
        )

        return list(image_embeddings_list), prior_token_info

    def embed_multimodal(
        self,
        **kwargs: object,
    ) -> tuple[torch.Tensor, ...] | None:
        """
        Embed multimodal inputs (images) for vLLM's multimodal processing.

        For GLM-Image (similar to Chameleon), we:
        1. Extract features using the vision encoder (1536 dim)
        2. Quantize features to discrete tokens using VQ-VAE
        3. Embed tokens using text embedding layer (4096 dim)

        This returns text-space embeddings that can be directly scattered
        into the input_embeds tensor by vLLM's _merge_multimodal_embeddings.

        Returns:
            Tuple of image embedding tensors, one per image, each with shape
            [num_patches, text_hidden_size]
        """
        logger.debug(f"embed_multimodal called with kwargs keys: {list(kwargs.keys())}")

        # Parse image inputs - check for multiple possible keys
        pixel_values = kwargs.get("pixel_values")
        image_embeds = kwargs.get("image_embeds")  # Alternative key
        image_grid_thw = kwargs.get("image_grid_thw")

        # Debug: log what we found
        logger.debug(f"pixel_values type: {type(pixel_values)}, image_grid_thw type: {type(image_grid_thw)}")

        if pixel_values is None and image_embeds is None:
            # No image inputs
            logger.debug("No pixel_values or image_embeds found in kwargs")
            return ()

        # Use pixel_values if available, otherwise use image_embeds
        if pixel_values is not None:
            image_input = self._parse_and_validate_image_input(
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )
        else:
            # Handle image_embeds case - these are pre-computed embeddings
            if isinstance(image_embeds, torch.Tensor):
                # Split by image grid sizes if available
                if image_grid_thw is not None:
                    split_sizes = (image_grid_thw.prod(dim=-1)).tolist()
                    return tuple(torch.split(image_embeds, split_sizes, dim=0))
                else:
                    return (image_embeds,)
            return ()

        if image_input is None:
            return ()

        # Process images through vision encoder and get VQ-VAE tokens
        image_embeddings, prior_token_info = self._process_image_input(image_input)

        # Cache prior_token_info for retrieval in forward()
        # This is needed because vLLM doesn't pass pixel_values to forward
        self._prior_token_cache = prior_token_info
        logger.debug(
            f"embed_multimodal: cached prior_token_info with {len(prior_token_info['prior_token_image_ids'])} images"
        )

        return tuple(image_embeddings)

    def _parse_grid_from_tokens(
        self,
        input_tokens: list[int],
        hf_config,
    ) -> list[list[int]] | None:
        """
        Parse image grid dimensions from prompt tokens.

        For text-to-image, the prompt format is:
        "text<sop>H W<eop><sop>h w<eop><bos>"

        For image-to-image, the prompt format is:
        "text<sop>H W<eop><bos>"

        Where:
        - <sop> is grid_bos_token_id (start of phrase, marks grid dimension start)
        - <eop> is grid_eos_token_id (end of phrase, marks grid dimension end)
        - H W is large image grid (e.g., "32 32" for 1024x1024)
        - h w is small image grid for preview (t2i only)
        - <bos> is image_start_token_id (16384, marks start of image generation)

        Returns:
            List of grids [[1, H, W], ...] or None if parsing fails
        """
        try:
            # Get special token IDs from config or tokenizer
            # We need grid_bos_token_id and grid_eos_token_id
            # These are typically <sop> and <eop> tokens

            # First try to get from hf_config
            grid_bos_id = getattr(hf_config, "grid_bos_token_id", None)
            grid_eos_id = getattr(hf_config, "grid_eos_token_id", None)

            # If not in config, we need to infer from token patterns
            # For GLM-Image, looking at the processor code:
            # - grid_bos_token = tokenizer.grid_bos_token
            # - grid_eos_token = tokenizer.grid_eos_token
            # These are typically single-token markers

            if grid_bos_id is None or grid_eos_id is None:
                # Try to find pattern in tokens: look for repeated pattern of
                # [marker] [number] [number] [marker]
                # where numbers are small positive integers (grid dimensions like 16, 32)

                # Use heuristics: grid dimensions are typically between 8 and 128
                # represented as single tokens that decode to numbers

                # Cannot find grid tokens, let caller use defaults
                return None

            # Find all <sop>...<eop> regions
            grids = []
            i = 0
            while i < len(input_tokens):
                if input_tokens[i] == grid_bos_id:
                    # Found start of grid region, find end
                    j = i + 1
                    while j < len(input_tokens) and input_tokens[j] != grid_eos_id:
                        j += 1

                    if j < len(input_tokens):
                        # Extract tokens between <sop> and <eop>
                        grid_tokens = input_tokens[i + 1 : j]

                        # These should decode to "H W" format
                        # For now, we assume they're numeric token IDs that represent the dimensions
                        # This is a simplification - actual implementation would need tokenizer

                        if len(grid_tokens) >= 2:
                            # Assume first two tokens are H and W values
                            # This is a heuristic - actual values depend on tokenizer
                            # For GLM-Image with ChatGLM tokenizer, numbers are tokenized specially
                            h = grid_tokens[0] if grid_tokens[0] < 256 else 32  # fallback
                            w = grid_tokens[1] if grid_tokens[1] < 256 else 32  # fallback
                            grids.append([1, h, w])

                    i = j + 1
                else:
                    i += 1

            # Return grids if we found any (1 for i2i, 2 for t2i)
            if len(grids) >= 1:
                return grids

            return None

        except Exception:
            return None

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec] | None = None,
        image_grid_thw: list[list[int]] | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, int]:
        """Compute M-RoPE position IDs for GLM-Image.

        GLM-Image uses 3D positional encoding where text tokens have identical
        values across all dimensions, while image tokens use 2D grid positions.

        For text-to-image, also pre-computes decode positions for generated tokens.
        For image-to-image, uses image_start_token_id and image_end_token_id to
        find image boundaries (following reference implementation).

        Token structure for i2i:
        [text...] <image_start=16384> [image_token=167855 × N] <image_end=16385> [text...] <grid> <bos=16384>

        Returns:
            Tuple of (position_ids [3, total_len], mrope_position_delta)
        """
        hf_config = self.config
        image_token_id = hf_config.image_token_id  # 167855, repeated for each image patch
        image_start_token_id = hf_config.image_start_token_id  # 16384, marks image start / generation bos
        image_end_token_id = hf_config.image_end_token_id  # 16385, marks image end

        # Prefer full grids preserved by the processor for M-RoPE.
        # In i2i, vLLM multimodal batching must use *source-only* grids for pixel_values,
        # but M-RoPE needs *source + target* grids to precompute decode positions.
        mrope_grid = kwargs.get("mrope_image_grid_thw")
        if image_grid_thw is None and mrope_grid is not None:
            if isinstance(mrope_grid, torch.Tensor):
                image_grid_thw = [row.tolist() for row in mrope_grid]
            elif isinstance(mrope_grid, list):
                image_grid_thw = mrope_grid

        # Fallback: get image_grid_thw from mm_features (usually source-only grids).
        if image_grid_thw is None and mm_features is not None:
            feature_kwargs = MultiModalFeatureSpec.gather_kwargs(mm_features, {"image_grid_thw"})
            image_grid_thw = [item.tolist() for item in feature_kwargs.get("image_grid_thw", [])]

        if image_grid_thw is None:
            image_grid_thw = []

        # For text-to-image: parse grid info from input tokens if not provided
        # Input format: "text<sop>H W<eop><sop>h w<eop><bos>" where <bos>=image_start_token_id=16384
        # For 1024x1024: H=32, W=32 (large), h=16, w=16 (small preview)
        if not image_grid_thw:
            # Try to parse from kwargs (passed from processor)
            hf_config_arg = kwargs.get("hf_config")
            if hf_config_arg is not None and hasattr(hf_config_arg, "image_grid_thw"):
                image_grid_thw = hf_config_arg.image_grid_thw

            # If still empty, try to infer from input tokens
            if not image_grid_thw:
                # Check if this is a text-to-image request:
                # - Prompt ends with image_start_token_id (16384, the <bos> token for image generation)
                # - No image_end_token_id (16385) in prompt (no completed images)
                prompt_ends_with_start = len(input_tokens) > 0 and input_tokens[-1] == image_start_token_id
                has_end_token = image_end_token_id in input_tokens

                # Text-to-image: ends with start token but no end token
                if prompt_ends_with_start and not has_end_token:
                    # Parse grid dimensions from prompt tokens
                    image_grid_thw = self._parse_grid_from_tokens(input_tokens, hf_config)
                    if not image_grid_thw:
                        # Fallback to default 1024x1024 grids if parsing fails
                        image_grid_thw = [[1, 32, 32], [1, 16, 16]]

        seq_len = len(input_tokens)
        input_tokens_tensor = torch.tensor(input_tokens, dtype=torch.long)

        # Find image boundaries using image_start_token_id and image_end_token_id
        # This follows the reference implementation in modeling_glm_image.py get_rope_index()
        image_end_positions = torch.where(input_tokens_tensor == image_end_token_id)[0]
        image_start_positions = torch.where(input_tokens_tensor == image_start_token_id)[0]

        logger.debug(
            f"get_mrope_input_positions: seq_len={seq_len}, "
            f"image_start_positions={image_start_positions.tolist()}, "
            f"image_end_positions={image_end_positions.tolist()}"
        )

        # Filter start positions: only those followed by image tokens (not the final <bos>)
        # A valid image start is followed by image_token_id, not followed by end of sequence
        valid_start_positions = []
        for start_pos in image_start_positions:
            # Check if there's a token after this start and it's an image token
            if start_pos + 1 < seq_len and input_tokens[start_pos + 1] == image_token_id:
                valid_start_positions.append(start_pos.item() + 1)  # +1 to skip the start marker

        logger.debug(f"get_mrope_input_positions: valid_start_positions={valid_start_positions}")

        # Pair starts with ends to find complete image regions
        num_complete_images = min(len(valid_start_positions), len(image_end_positions))

        # Count source images for grid handling
        num_source_images = num_complete_images

        # For i2i mode: image_grid_thw may only contain source image grids
        # We need to add generation target grids for proper M-RoPE position calculation
        prompt_ends_with_start = len(input_tokens) > 0 and input_tokens[-1] == image_start_token_id
        if prompt_ends_with_start and len(image_grid_thw) == num_source_images and num_source_images > 0:
            # i2i mode: source grids exist but no target grids
            # Parse target grids from prompt tokens or use defaults
            parsed_grids = self._parse_grid_from_tokens(input_tokens, hf_config)
            if parsed_grids:
                # parsed_grids contains all grids mentioned in prompt
                # For i2i, add only the generation target grids
                if len(parsed_grids) > num_source_images:
                    image_grid_thw = list(image_grid_thw) + parsed_grids[num_source_images:]
                else:
                    # Fallback: add default 1024x1024 generation grids (1 target for i2i)
                    image_grid_thw = list(image_grid_thw) + [[1, 32, 32]]
            else:
                # Fallback to default 1024x1024 grids for generation
                image_grid_thw = list(image_grid_thw) + [[1, 32, 32]]

        llm_pos_ids_list: list[torch.Tensor] = []

        if image_grid_thw and num_source_images > 0:
            # Image-to-image mode: we have source images to encode
            # Build position IDs following reference implementation exactly
            current_pos = 0
            prev_image_end = 0  # Track position in input_ids of last image end

            # Process each complete image (source images)
            for img_idx in range(num_complete_images):
                start = valid_start_positions[img_idx]  # First image token position
                end = image_end_positions[img_idx].item()  # End marker position

                # Actual number of image tokens in input_ids
                actual_image_tokens = end - start

                logger.debug(
                    f"get_mrope_input_positions: processing image {img_idx}, "
                    f"start={start}, end={end}, actual_tokens={actual_image_tokens}, "
                    f"prev_image_end={prev_image_end}, current_pos={current_pos}"
                )

                # Get grid dimensions for this source image
                if img_idx < len(image_grid_thw):
                    t, h, w = image_grid_thw[img_idx]
                    expected_tokens = h * w
                    # Verify token count matches grid
                    if actual_image_tokens != expected_tokens:
                        logger.warning(
                            f"Image {img_idx}: token count mismatch! "
                            f"actual={actual_image_tokens}, expected={expected_tokens} (h={h}, w={w}). "
                            f"Using actual token count."
                        )
                        # Recalculate h, w from actual token count
                        h = w = int(actual_image_tokens**0.5)
                        if h * w != actual_image_tokens:
                            # Non-square, try to find factors
                            for factor in range(int(actual_image_tokens**0.5), 0, -1):
                                if actual_image_tokens % factor == 0:
                                    h = factor
                                    w = actual_image_tokens // factor
                                    break
                else:
                    # Fallback: estimate from token count
                    h = w = int(actual_image_tokens**0.5)
                    t = 1

                # Text tokens before this image (from prev_image_end to start)
                # Note: start points to first image token, so text is [prev_image_end, start)
                text_length = start - prev_image_end
                logger.debug(f"get_mrope_input_positions: text_length={text_length} (from {prev_image_end} to {start})")
                if text_length > 0:
                    # Text tokens get sequential 1D positions
                    text_positions = torch.arange(current_pos, current_pos + text_length, dtype=torch.long)
                    text_pos_3d = text_positions.unsqueeze(0).expand(3, -1)
                    llm_pos_ids_list.append(text_pos_3d)
                    current_pos += text_length

                # Image tokens with 2D spatial encoding
                # CRITICAL: Use actual_image_tokens to match input_ids length exactly
                # For an image with height H and width W:
                # - temporal: constant at current_pos
                # - height: cycles [current_pos, ..., current_pos+h-1] repeated w times each
                # - width: cycles [current_pos, ..., current_pos+w-1] repeated h times

                # Temporal: all tokens have same position
                position_temporal = torch.full((actual_image_tokens,), current_pos, dtype=torch.long)

                # Height: repeat_interleave pattern (clip to actual_image_tokens)
                position_height = torch.arange(current_pos, current_pos + h, dtype=torch.long).repeat_interleave(w)
                if len(position_height) != actual_image_tokens:
                    position_height = (
                        position_height[:actual_image_tokens]
                        if len(position_height) > actual_image_tokens
                        else F.pad(
                            position_height, (0, actual_image_tokens - len(position_height)), value=current_pos + h - 1
                        )
                    )

                # Width: repeat pattern (clip to actual_image_tokens)
                position_width = torch.arange(current_pos, current_pos + w, dtype=torch.long).repeat(h)
                if len(position_width) != actual_image_tokens:
                    position_width = (
                        position_width[:actual_image_tokens]
                        if len(position_width) > actual_image_tokens
                        else F.pad(
                            position_width, (0, actual_image_tokens - len(position_width)), value=current_pos + w - 1
                        )
                    )

                vision_position_ids = torch.stack([position_temporal, position_height, position_width], dim=0)
                llm_pos_ids_list.append(vision_position_ids)

                # Advance position by max(h, w) to maintain spatial coherence
                current_pos += max(h, w)

                # Update prev_image_end to the END marker position (not current_pos!)
                # This is the position in input_ids, used for text length calculation
                prev_image_end = end

            # Remaining text tokens after the last image (including grid tokens and final <bos>)
            remaining_length = seq_len - prev_image_end
            logger.debug(
                f"get_mrope_input_positions: remaining_length={remaining_length} "
                f"(seq_len={seq_len} - prev_image_end={prev_image_end})"
            )
            if remaining_length > 0:
                text_positions = torch.arange(current_pos, current_pos + remaining_length, dtype=torch.long)
                text_pos_3d = text_positions.unsqueeze(0).expand(3, -1)
                llm_pos_ids_list.append(text_pos_3d)
                current_pos += remaining_length

            prefill_positions = torch.cat(llm_pos_ids_list, dim=1)

            # Verify prefill positions length matches seq_len
            if prefill_positions.shape[1] != seq_len:
                logger.error(
                    f"Position length mismatch! prefill_positions.shape[1]={prefill_positions.shape[1]}, "
                    f"seq_len={seq_len}. This will cause incorrect attention. "
                    f"num_complete_images={num_complete_images}, image_grid_thw={image_grid_thw}"
                )

            # Pre-compute decode positions for images that will be generated
            # For i2i: source images are already encoded, generation targets are in remaining grids
            num_decode_grids = len(image_grid_thw) - num_source_images

            if num_decode_grids > 0:
                decode_pos_lists: list[torch.Tensor] = []
                decode_pos = current_pos

                # Process decode grids in REVERSE order (last grid first)
                # GLM-Image generates small image first (e.g., 16x16), then large (32x32)
                for i in range(1, num_decode_grids + 1):
                    grid_idx = len(image_grid_thw) - i  # -1, -2, ... from end
                    t, h, w = image_grid_thw[grid_idx]
                    total_tokens = h * w

                    # Build 2D positions following reference implementation
                    position_temporal = torch.full((total_tokens,), decode_pos, dtype=torch.long)
                    position_height = torch.arange(decode_pos, decode_pos + h, dtype=torch.long).repeat_interleave(w)
                    position_width = torch.arange(decode_pos, decode_pos + w, dtype=torch.long).repeat(h)

                    decode_pos_lists.append(torch.stack([position_temporal, position_height, position_width], dim=0))
                    decode_pos += max(h, w)

                # Add position for EOS token
                decode_pos_lists.append(torch.tensor([[decode_pos], [decode_pos], [decode_pos]]))

                decode_positions = torch.cat(decode_pos_lists, dim=1)

                # Concatenate prefill and decode positions
                llm_positions = torch.cat([prefill_positions, decode_positions], dim=1)
            else:
                llm_positions = prefill_positions

        elif image_grid_thw:
            # Text-to-image mode: no source images, just text + generation positions
            # Build position IDs considering image regions for decode phase
            current_pos = 0

            # All prefill tokens get sequential 1D positions
            prefill_positions = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(3, -1)
            current_pos = seq_len

            # Pre-compute decode positions for all grids (all for generation)
            decode_pos_lists: list[torch.Tensor] = []
            decode_pos = current_pos

            # For t2i with grids [[1,32,32], [1,16,16]]:
            # - First generate small image (16x16 = 256 tokens) from grid[-1]
            # - Then generate large image (32x32 = 1024 tokens) from grid[-2]
            # - Finally generate EOS
            # Process in reverse order for GLM-Image generation pattern
            for i in range(1, len(image_grid_thw) + 1):
                grid_idx = -i  # -1, -2, ... (last grid first)
                t, h, w = image_grid_thw[grid_idx]
                total_tokens = h * w

                # Build 2D positions following reference implementation
                position_temporal = torch.full((total_tokens,), decode_pos, dtype=torch.long)
                position_height = torch.arange(decode_pos, decode_pos + h, dtype=torch.long).repeat_interleave(w)
                position_width = torch.arange(decode_pos, decode_pos + w, dtype=torch.long).repeat(h)

                decode_pos_lists.append(torch.stack([position_temporal, position_height, position_width], dim=0))
                decode_pos += max(h, w)

            # Add position for EOS token
            decode_pos_lists.append(torch.tensor([[decode_pos], [decode_pos], [decode_pos]]))

            decode_positions = torch.cat(decode_pos_lists, dim=1)

            # Concatenate prefill and decode positions
            llm_positions = torch.cat([prefill_positions, decode_positions], dim=1)
        else:
            # Pure text - all dimensions same
            llm_positions = torch.arange(seq_len).view(1, -1).expand(3, -1)

        mrope_position_delta = (llm_positions.max() + 1 - seq_len).item()

        # Debug logging for M-RoPE position calculation
        logger.debug(
            f"get_mrope_input_positions: seq_len={seq_len}, "
            f"num_source_images={num_source_images}, "
            f"image_grid_thw={image_grid_thw}, "
            f"llm_positions.shape={llm_positions.shape}, "
            f"llm_positions.max={llm_positions.max().item()}, "
            f"mrope_position_delta={mrope_position_delta}"
        )

        return llm_positions, mrope_position_delta

    def get_language_model(self) -> torch.nn.Module:
        """Return the underlying language model for text embedding.

        This is required by vLLM's SupportsMultiModal interface.
        The embed_input_ids() method calls get_language_model().embed_input_ids()
        to get text token embeddings before merging with multimodal embeddings.
        """
        return self.model.language_model

    # Flag to indicate this model can output multimodal data (prior_token_image_ids for i2i)
    have_multimodal_outputs = True

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        **kwargs: object,
    ) -> OmniOutput | IntermediateTensors:
        """
        Forward pass through GLM-Image.

        Args:
            input_ids: Input token IDs [seq_len]
            positions: Position IDs, shape (3, seq_len) for M-RoPE
            intermediate_tensors: For pipeline parallelism
            inputs_embeds: Pre-computed embeddings
            pixel_values: Source image pixels (for image-to-image)
            image_grid_thw: Grid dimensions for images

        Returns:
            OmniOutput with hidden states and optional prior_token_image_ids for i2i
        """
        if intermediate_tensors is not None:
            inputs_embeds = None

        hidden_states, prior_token_image_ids_info = self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )

        # For intermediate tensors (PP), just return hidden states
        if isinstance(hidden_states, IntermediateTensors):
            return hidden_states

        # Build multimodal outputs for i2i mode
        # First check if model returned prior_token_image_ids_info (from pixel_values path)
        # If not, check the cache (from embed_multimodal path)
        multimodal_outputs = None
        if prior_token_image_ids_info is not None:
            multimodal_outputs = prior_token_image_ids_info
            logger.debug("forward: got prior_token_info from model (pixel_values path)")
        elif self._prior_token_cache is not None:
            # Retrieve cached prior_token_info from embed_multimodal
            multimodal_outputs = self._prior_token_cache
            self._prior_token_cache = None  # Clear after use
            logger.debug("forward: got prior_token_info from cache (embed_multimodal path)")

        return OmniOutput(
            text_hidden_states=hidden_states,
            multimodal_outputs=multimodal_outputs,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor | None:
        """Compute logits from hidden states."""
        logits = self.logits_processor(
            self.lm_head,
            hidden_states,
        )
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """
        Load weights from HuggingFace checkpoint.

        Handles weight mapping for:
        - Vision encoder weights
        - VQ-VAE weights
        - Text model weights
        - LM head weights
        """
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            # Handle stacked parameters (QKV, gate_up)
            is_stacked = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                stacked_name = name.replace(weight_name, param_name)
                if stacked_name not in params_dict:
                    break
                param = params_dict[stacked_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(stacked_name)
                is_stacked = True
                break

            if not is_stacked:
                # Regular weight loading
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        return loaded_params
