# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SenseNova-Vision multi-view / per-task pre-processing primitives.

This module ports the resize semantics the official SenseNova-Vision repo
applies through ``data/transforms.py::MaxLongEdgeMinShortEdgeResize`` and
``data/data_utils.py``, and the multi-view packing arithmetic from
``inference/inferencer.py::InterleaveInferencer.gen_image``. Everything here is
pure Python + NumPy + PIL (no torch, no GPU) so the arithmetic can be
unit-tested on CPU without downloading a model checkpoint.

Per-task transform targets are transcribed verbatim from
``SenseNova-Vision/inference/sensenova_vision.py``::

    recon3d_vae_transform   = ImageTransform(512, 256, 16)   # (max, min, stride)
    recon3d_vit_transform   = ImageTransform(448, 224, 14)
    camera_vit_transform    = ImageTransform(560, 378, 14)

``ImageTransform(max_size, min_size, stride)`` applies a
``MaxLongEdgeMinShortEdgeResize``: the long edge is scaled down to at most
``max_size``, the short edge is scaled up to at least ``min_size``, and the
result is rounded to a multiple of ``stride``. For the square input views used
by recon3d / camera-pose this yields a stride-aligned square whose side is the
largest multiple of ``stride`` that fits in ``max_size``. The VAE target side
must stay within ``max_latent_size`` latent cells after ``latent_downsample``
(``side / latent_downsample <= max_latent_size``); ``(512,512)`` is 64 latent
cells per side, ``(448,448)`` is 56, ``(560,560)`` is 70.

These sizes feed the conditioning prefill in the AR stage: the resized input
view H/W propagates into ``kv_metadata["image_shape"]`` so the DiT latent grid
matches, and the multi-view decode emits one PIL image per view at that single
shared shape.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from PIL import Image

ImageTransformFn = Callable[[Image.Image], Image.Image]

# Stride-aligned square target side (in pixels) per SenseNovaVision pipeline
# mode, driven by the upstream ``ImageTransform(max_size, min_size, stride)``
# keys. ``target_side = (max_size // stride) * stride`` for square inputs.
# Keyed by the mode string used by ``SenseNovaVisionPipeline._apply_mode_defaults``.
#
#   recon3d     : VAE 512 / ViT 448  (ImageTransform(512,256,16) / (448,224,14))
#   camera_pose: ViT 560             (ImageTransform(560,378,14)); no generation
#   default     : base pipeline VAE 1024 / ViT 980 (BAGEL checkpoint defaults)
PER_TASK_VAE_SIDE: dict[str, int | None] = {
    "recon3d": 512,
    "camera_pose": None,  # camera-pose is understanding-only (no VAE prefill)
}
PER_TASK_VIT_SIDE: dict[str, int | None] = {
    "recon3d": 448,
    "camera_pose": 560,
}


@dataclass(frozen=True)
class ResizeSpec:
    """A stride-aligned square target derivable from a ``(max, min, stride)`` key."""

    max_size: int
    min_size: int
    stride: int

    @property
    def target_side(self) -> int:
        """Largest stride-aligned square within ``max_size`` (>= ``min_size``)."""
        side = (self.max_size // self.stride) * self.stride
        return max(self.min_size, side)

    def vae_grid(self, latent_downsample: int) -> tuple[int, int]:
        """Latent grid ``(h, w)`` for a square of :attr:`target_side`."""
        side = self.target_side
        return side // latent_downsample, side // latent_downsample


def max_long_edge_resize(
    max_size: int,
    min_size: int,
    stride: int,
    interpolation: int = Image.BICUBIC,
) -> ImageTransformFn:
    """Port :class:`MaxLongEdgeMinShortEdgeResize` (upstream ``transforms.py``)."""

    def _make_divisible(value: int) -> int:
        return max(stride, int(round(value / stride) * stride))

    def transform(image: Image.Image) -> Image.Image:
        if image.mode != "RGB":
            image = image.convert("RGB")
        w, h = image.size
        scale = min(max_size / max(w, h), 1.0)
        scale = max(scale, min_size / min(w, h))
        new_w, new_h = _make_divisible(int(round(w * scale))), _make_divisible(int(round(h * scale)))
        # Clamp the longest edge back to max_size (matches upstream "ensure
        # longest edge does not exceed max_size").
        if max(new_w, new_h) > max_size:
            scale = max_size / max(new_w, new_h)
            new_w, new_h = _make_divisible(int(round(new_w * scale))), _make_divisible(int(round(new_h * scale)))
        return image.resize((new_w, new_h), resample=interpolation)

    return transform


def recon3d_packing(
    num_views: int,
    curr_kvlen: int,
    curr_rope: int,
) -> tuple[list[int], list[int]]:
    """Build the multi-view gen-context ``(kv_lens, ropes)`` for recon3d.

    Mirrors upstream ``gen_image`` packing:
    ``curr_kvlens = [curr_kvlen] + [0] * (num_views - 1)``
    ``curr_rope   = [curr_rope + x for x in range(num_views)]``
    i.e. the first view continues from the primary KV cache and each further
    view starts from an empty (0-length) KV prefix.

    Returns ``(kv_lens, ropes)`` length-N lists.
    """
    kv_lens = [curr_kvlen] + [0] * (num_views - 1)
    ropes = list(range(curr_rope, curr_rope + num_views))
    return kv_lens, ropes


def packed_seqlens(num_views: int, latent_h: int, latent_w: int) -> list[int]:
    """Per-branch ``packed_seqlens``: ``(h*w + 2)`` markers each."""
    num_tokens = latent_h * latent_w
    return [num_tokens + 2] * num_views
