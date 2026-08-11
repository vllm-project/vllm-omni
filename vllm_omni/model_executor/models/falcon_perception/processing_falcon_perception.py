# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Falcon Perception multimodal input processing.

The published checkpoints ship no ``AutoProcessor``; the reference engine drives
a small numpy/PIL image pipeline directly. This module reimplements that
pipeline against vLLM's processing interfaces so the token counts, the pixel
normalisation and the placeholder layout all match the reference exactly.

Per image the prompt gains ``5 + (H/patch)*(W/patch) + 1`` tokens:
``[image_cls, reg1..reg4] + <|image|> * n_patches + [end_of_image]``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from PIL import Image
from vllm.inputs import MultiModalDataDict
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import ImageProcessorItems, MultiModalDataItems
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)

IMAGE_MEAN = (0.5, 0.5, 0.5)
IMAGE_STD = (0.5, 0.5, 0.5)


def resize_image_if_necessary(
    image: Image.Image,
    min_dimension: int = 256,
    max_dimension: int = 1024,
) -> Image.Image:
    """Soft-clamp both sides into ``[min_dimension, max_dimension]``.

    Aspect ratio is preserved and an already-conforming image is returned
    untouched. Mirrors ``falcon_perception.data.resize_image_if_necessary``,
    which the reference paged engine applies before the image processor.
    """
    width, height = image.size
    if min_dimension <= width <= max_dimension and min_dimension <= height <= max_dimension:
        return image

    aspect = width / height
    is_vertical = width < height
    if width < min_dimension or height < min_dimension:
        target = min_dimension
    else:
        target = max_dimension

    if is_vertical:
        new_width = target
        new_height = int(new_width / aspect)
    else:
        new_height = target
        new_width = int(new_height * aspect)

    if new_width > max_dimension:
        new_width = max_dimension
        new_height = int(new_width / aspect)
    if new_height > max_dimension:
        new_height = max_dimension
        new_width = int(new_height * aspect)

    return image.resize((new_width, new_height))


def smart_resize_dims(
    height: int,
    width: int,
    *,
    factor: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
    """Round both sides to a multiple of ``factor`` inside the pixel budget.

    Pure-arithmetic twin of ``falcon_perception.data.smart_resize`` so token
    counts can be computed without materialising the resized image.
    """
    if height < factor or width < factor:
        raise ValueError(f"height={height} and width={width} must both be >= {factor}")
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )

    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


class FalconPerceptionImageProcessor:
    """Reference-equivalent image pipeline: resize -> smart_resize -> rescale -> normalize."""

    def __init__(self, config: Any) -> None:
        self.patch_size = config.spatial_patch_size
        self.min_image_size = config.min_image_size
        self.max_image_size = config.max_image_size
        self.min_pixels = config.min_pixels
        self.max_pixels = config.max_pixels

    def target_grid(self, height: int, width: int) -> tuple[int, int]:
        """Patch-grid ``(gh, gw)`` an image of this size will produce."""
        image = Image.new("RGB", (width, height))
        image = resize_image_if_necessary(image, self.min_image_size, self.max_image_size)
        w0, h0 = image.size
        h_bar, w_bar = smart_resize_dims(
            h0,
            w0,
            factor=self.patch_size,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        return h_bar // self.patch_size, w_bar // self.patch_size

    def __call__(self, image: Image.Image) -> torch.Tensor:
        """Return a channels-last ``(H, W, 3)`` float tensor."""
        image = image.convert("RGB")
        image = resize_image_if_necessary(image, self.min_image_size, self.max_image_size)
        width, height = image.size
        h_bar, w_bar = smart_resize_dims(
            height,
            width,
            factor=self.patch_size,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        image = image.resize((w_bar, h_bar), Image.Resampling.BICUBIC)

        # Normalised in numpy, in place. Bit-identical to the torch expression
        # this replaces -- same ops, same order, same float32 rounding, checked
        # with torch.equal -- but ~3.5x faster (40 ms -> 11 ms on a 1024x681
        # image): torch's CPU kernels do not vectorise the size-3 trailing
        # broadcast, and each out-of-place step allocated another 8.4 MB.
        # Worth the extra lines because this runs on the stage-0 -> stage-1
        # handoff, which is serial CPU time with both stages' GPUs idle.
        pixels = _as_float_array(image)
        mean = np.array(IMAGE_MEAN, dtype=pixels.dtype)
        std = np.array(IMAGE_STD, dtype=pixels.dtype)
        np.divide(pixels, np.float32(255.0), out=pixels)
        np.subtract(pixels, mean, out=pixels)
        np.divide(pixels, std, out=pixels)
        return torch.from_numpy(pixels)


def _as_float_array(image: Image.Image) -> np.ndarray:
    # ``np.array`` rather than ``np.asarray``: PIL hands back a read-only view,
    # and the caller normalises in place.
    return np.array(image, dtype=np.float32)


class FalconPerceptionProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        from vllm_omni.model_executor.models.falcon_perception.configuration_falcon_perception import (
            FalconPerceptionConfig,
        )

        return self.ctx.get_hf_config(FalconPerceptionConfig)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        # The reference prompt builder inserts a single image block; multi-image
        # prompts are not part of any released Falcon Perception task.
        return {"image": 1}

    def get_image_processor(self) -> FalconPerceptionImageProcessor:
        return FalconPerceptionImageProcessor(self.get_hf_config())

    def get_image_grid(self, *, image_width: int, image_height: int) -> tuple[int, int]:
        return self.get_image_processor().target_grid(image_height, image_width)

    def get_num_image_tokens(self, *, image_width: int, image_height: int) -> int:
        gh, gw = self.get_image_grid(image_width=image_width, image_height=image_height)
        # 5 structural prefix tokens + patches + end_of_image
        return 5 + gh * gw + 1


class FalconPerceptionDummyInputsBuilder(BaseDummyInputsBuilder[FalconPerceptionProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return "<|image|>" * mm_counts.get("image", 0)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, Any] | None = None,
    ) -> MultiModalDataDict:
        config = self.info.get_hf_config()
        # Largest image the pixel budget allows, so profiling sees the worst case.
        side = int(math.sqrt(config.max_pixels))
        side = min(max(side, config.min_image_size), config.max_image_size)
        num_images = mm_counts.get("image", 0)
        return {"image": self._get_dummy_images(width=side, height=side, num_images=num_images)}


class FalconPerceptionMultiModalProcessor(BaseMultiModalProcessor[FalconPerceptionProcessingInfo]):
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> MultiModalKwargsItems | dict[str, Any]:
        tokenizer = self.info.get_tokenizer()
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

        images = mm_data.get("images") or []
        if not isinstance(images, list):
            images = [images]

        processor = self.info.get_image_processor()
        pixel_values: list[torch.Tensor] = []
        grids: list[list[int]] = []
        for image in images:
            pixels = processor(image)
            pixel_values.append(pixels)
            grids.append([pixels.shape[0] // processor.patch_size, pixels.shape[1] // processor.patch_size])

        out: dict[str, Any] = {"input_ids": [prompt_ids]}
        if pixel_values:
            # Kept as a list: images keep their native size, so this cannot be
            # stacked into one dense tensor.
            out["pixel_values"] = pixel_values
            out["image_grid_hw"] = torch.tensor(grids, dtype=torch.long)
        return out

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        # ``_call_hf_processor`` returns the raw prompt ids; vLLM inserts the
        # image placeholder run for us via ``_get_prompt_updates``.
        return False

    def _get_mm_fields_config(
        self,
        hf_inputs: Any,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return {
            "pixel_values": MultiModalFieldConfig.batched("image"),
            "image_grid_hw": MultiModalFieldConfig.batched("image"),
        }

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        config = self.info.get_hf_config()
        images = mm_items.get_items("image", ImageProcessorItems)

        def get_replacement(item_idx: int) -> PromptUpdateDetails[list[int]]:
            size = images.get_image_size(item_idx)
            gh, gw = self.info.get_image_grid(image_width=size.width, image_height=size.height)
            n_patches = gh * gw

            tokens = [
                *config.image_structural_token_ids,
                *([config.img_id] * n_patches),
                config.img_end_id,
            ]

            # ``is_embed`` marks the span that receives projected patch
            # embeddings AND, via ``extract_embeds_range``, the span vLLM makes
            # bidirectional for prefix-LM models. The reference mask covers
            # image_cls through the last patch but excludes end_of_image, so the
            # trailing False is load-bearing, not cosmetic.
            def is_embed(tokenizer: object, full: object, *, _n: int = n_patches) -> torch.Tensor:
                return torch.tensor([True] * (5 + _n) + [False])

            return PromptUpdateDetails(full=tokens, is_embed=is_embed)

        # The reference tokenizer splits the prompt on the literal "<|image|>"
        # and splices the block in its place, so this is a replacement of that
        # single token — not an insertion, which would leave the original
        # placeholder behind as a 1201st patch token.
        return [
            PromptReplacement(
                modality="image",
                target=[config.img_id],
                replacement=get_replacement,
            )
        ]


__all__ = [
    "FalconPerceptionDummyInputsBuilder",
    "FalconPerceptionImageProcessor",
    "FalconPerceptionMultiModalProcessor",
    "FalconPerceptionProcessingInfo",
    "resize_image_if_necessary",
    "smart_resize_dims",
]
