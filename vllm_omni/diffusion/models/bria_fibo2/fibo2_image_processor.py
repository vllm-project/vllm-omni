# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
#
# Image processor for Fibo-2.0, ported from the HuggingFace diffusers
# ``BriaFibo2ImageProcessor`` (diffusers PR #14991): preprocesses the images
# to edit as Fibo 2.0 was trained on them — cover-resize + center crop at the
# size Fibo 2.0 was trained at for the aspect ratio, with an optional masked
# region greyed out — and produces Qwen3-VL's downscaled copies.

import math

import PIL.Image
import torch
from diffusers.image_processor import VaeImageProcessor


class Fibo2ImageProcessor(VaeImageProcessor):
    """Preprocesses the images to edit, and postprocesses generated images.

    Args:
        vae_scale_factor: VAE (spatial) scale factor including the 2x2 latent
            packing. Sizes are multiples of it.
    """

    def __init__(self, vae_scale_factor: int = 16):
        super().__init__(vae_scale_factor=vae_scale_factor)

    def get_image_size(self, aspect_ratio: float, area: int = 1024 * 1024) -> tuple[int, int]:
        """The (width, height) Fibo 2.0 was trained at for an aspect ratio:
        of the sizes divisible by ``vae_scale_factor``, at least 4 of it a
        side and within 10% of ``area``, the one that crops the least from a
        picture with that aspect ratio, then the one closest to ``area``."""
        multiple = self.config.vae_scale_factor
        min_side = 4 * multiple
        sizes = [
            (width, height)
            for width in range(min_side, 2 * area // min_side, multiple)
            for height in range(min_side, 2 * area // width, multiple)
            if 0.9 * area <= width * height <= 1.1 * area
        ]

        def crop(size):
            ratio = size[0] / size[1]
            return 1 - min(ratio / aspect_ratio, aspect_ratio / ratio)

        return min(sizes, key=lambda size: (round(crop(size), 9), abs(size[0] * size[1] - area), -size[0]))

    def _resize_and_crop(self, image: PIL.Image.Image, width: int, height: int) -> PIL.Image.Image:
        """Scale the image to cover width x height, keeping its aspect ratio,
        then crop the center, as Fibo 2.0's training did."""
        image = image.convert("RGB")
        scale = max(height / image.height, width / image.width)
        resized_width, resized_height = math.ceil(image.width * scale), math.ceil(image.height * scale)
        image = image.resize((resized_width, resized_height), resample=PIL.Image.LANCZOS)
        left, top = (resized_width - width) // 2, (resized_height - height) // 2
        return image.crop((left, top, left + width, top + height))

    def preprocess(
        self,
        image: PIL.Image.Image | list[PIL.Image.Image],
        height: int | None = None,
        width: int | None = None,
        mask: PIL.Image.Image | None = None,
        area: int = 1024 * 1024,
        text_encoder_image_size: int = 768,
    ) -> tuple[list[torch.Tensor], list[PIL.Image.Image]]:
        """Preprocess the images to edit.

        With ``mask``, the region it marks in the single image is greyed out
        first. A single image is cropped to ``width`` x ``height`` (default:
        the trained size for its own aspect ratio); several images each to the
        trained size for their own aspect ratio. Returns the crops as the VAE
        takes them (one tensor each, sizes can differ) and Qwen3-VL's copies
        capped at ``text_encoder_image_size`` pixels a side.
        """
        images = image if isinstance(image, list) else [image]
        if mask is not None:
            images = [self.grey_out(images[0], mask)]
        if len(images) == 1:
            if height is None and width is None:
                width, height = self.get_image_size(images[0].width / images[0].height, area=area)
            sizes = [(width, height)]
        else:
            sizes = [self.get_image_size(img.width / img.height, area=area) for img in images]
        crops = [self._resize_and_crop(img, size[0], size[1]) for img, size in zip(images, sizes)]

        text_encoder_images = []
        for crop in crops:
            text_encoder_image = crop.copy()
            text_encoder_image.thumbnail((text_encoder_image_size, text_encoder_image_size))
            text_encoder_images.append(text_encoder_image)
        to_tensor = super().preprocess
        return [to_tensor(crop) for crop in crops], text_encoder_images

    @staticmethod
    def grey_out(image: PIL.Image.Image, mask: PIL.Image.Image) -> PIL.Image.Image:
        """The image with the region ``mask`` marks (white on black) greyed
        out as training did: flat (128, 128, 128) through the mask made binary
        at 128, at the image's own size, before any resize."""
        grey = PIL.Image.new("RGB", image.size, (128, 128, 128))
        binary_mask = mask.convert("L").point(lambda value: 255 if value >= 128 else 0)
        return PIL.Image.composite(grey, image.convert("RGB"), binary_mask)
