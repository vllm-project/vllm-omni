"""Dense-image decoders for SenseNova-Vision.

The model returns dense predictions as ordinary images:

- Segmentation: a mask image (binary grayscale for ``binary_segment``, or an
  RGB mask whose colors are defined by the generated caption for GCG/panoptic
  variants).
- Depth: a grayscale image encoding relative depth (brighter = closer), as
  consumed by the Marigold-style depth evaluation.
- Normal: an RGB image encoding surface normals, as consumed by the
  Marigold-style normals evaluation.
- Point map (recon3d): a raw HxWx3 tensor in ``[-1, 1]``-ish space (the
  official ``inferencer.decode_image(..., output_raw_tensor=True)`` output)
  interpreted as per-pixel 3D points.

Ports the decoding paths from the official SenseNova-Vision repository
(``utils/mask.py``, the ``batch_*_segment.py`` benchmarks, the
Marigold ``depth/eval.py`` and ``normals/eval.py`` scripts, and
``inference/inferencer.py``). Everything is NumPy/PIL-only so the module can be
unit-tested without torch or a GPU.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PIL import Image

ImageLike = Image.Image | np.ndarray | str

__all__ = [
    "decode_segmentation",
    "decode_depth",
    "decode_normal",
    "decode_point_map",
]


def _to_rgb_array(image: ImageLike) -> np.ndarray:
    """Return an RGB uint8 array for a PIL image / ndarray / path input."""
    if isinstance(image, str):
        image = Image.open(image)
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"), dtype=np.uint8)
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected an RGB image, got array with shape {arr.shape}")
    return np.asarray(arr, dtype=np.uint8)


def _to_gray_array(image: ImageLike) -> np.ndarray:
    """Return a uint8 grayscale array for a PIL image / ndarray / path input."""
    if isinstance(image, str):
        image = Image.open(image)
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("L"), dtype=np.uint8)
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        return np.asarray(arr[:, :, 0], dtype=np.uint8)
    if arr.ndim == 2:
        return np.asarray(arr, dtype=np.uint8)
    raise ValueError(f"Expected a grayscale image, got array with shape {arr.shape}")


def _rgb2id(rgb: np.ndarray) -> np.ndarray:
    """Encode an RGB array into a single integer id per pixel."""
    if rgb.ndim == 2:
        return rgb.astype(np.int32)
    return (
        rgb[:, :, 0].astype(np.int32) + 256 * rgb[:, :, 1].astype(np.int32) + 256 * 256 * rgb[:, :, 2].astype(np.int32)
    )


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------


def decode_segmentation(
    image: ImageLike,
    class_define: Sequence[Sequence[int]] | None = None,
    threshold: int = 127,
) -> np.ndarray:
    """Decode a model-generated segmentation mask image.

    Args:
        image: Predicted mask image (PIL image, ndarray, or path). For the
            binary-segment benchmark this is a grayscale mask; for GCG /
            panoptic variants it is an RGB mask.
        class_define: Optional ``[N, 3]`` RGB palette defining the class
            colors (in class-index order). When provided, each pixel is
            assigned to the nearest palette color; black pixels are assigned
            to the background index ``len(class_define)`` (matching
            ``to_multi_mask``). When omitted, the input is treated as a binary
            mask and thresholded with ``to_binary_mask`` semantics.
        threshold: Grayscale threshold used for the binary path (default 127).

    Returns:
        An int32/uint8 class-index mask array with shape ``(H, W)``.
    """
    if class_define is not None:
        palette = np.asarray(class_define, dtype=np.float32)
        if palette.ndim != 2 or palette.shape[1] != 3:
            raise ValueError(f"class_define must have shape [N, 3], got {palette.shape}")
        pixels = _to_rgb_array(image).reshape(-1, 3).astype(np.float32)
        diff = pixels[:, None, :] - palette[None, :, :]
        class_mask = np.argmin(np.sum(diff * diff, axis=-1), axis=1)
        rgb = _to_rgb_array(image)
        class_mask = class_mask.reshape(rgb.shape[0], rgb.shape[1])
        black = np.all(rgb == 0, axis=-1)
        class_mask[black] = len(palette)
        return class_mask.astype(np.int32, copy=False)

    gray = _to_gray_array(image)
    return (gray > threshold).astype(np.uint8)


# ---------------------------------------------------------------------------
# Depth
# ---------------------------------------------------------------------------


def decode_depth(
    image: ImageLike,
    resample: int = Image.NEAREST,
    size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Decode a predicted relative-depth grayscale image into a depth map.

    The model outputs depth as a grayscale image where brightness indicates
    proximity. The Marigold-style evaluation decodes it as the per-pixel mean
    of the RGB channels divided by 255, i.e. a relative depth map in
    ``[0, 1]``.

    Args:
        image: Predicted depth image (PIL image, ndarray, or path).
        resample: Resampling filter used when ``size`` is provided.
        size: Optional ``(width, height)`` to resize the prediction to before
            decoding (the benchmark resizes with nearest-neighbor to the input
            image size).

    Returns:
        A float32 depth map with shape ``(H, W)`` in ``[0, 1]``.
    """
    if isinstance(image, str):
        image = Image.open(image)
    if isinstance(image, Image.Image):
        if size is not None:
            image = image.resize(size, resample=resample)
        arr = np.asarray(image.convert("RGB"), dtype=np.float32)
        return np.mean(arr, axis=2) / 255.0

    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        arr = arr[:, :, :3]
    elif arr.ndim == 2:
        arr = arr[:, :, None]
    else:
        raise ValueError(f"Expected a depth image, got array with shape {arr.shape}")
    if size is not None:
        arr = np.asarray(
            Image.fromarray(arr.astype(np.uint8)).resize(size, resample=resample),
            dtype=np.float32,
        )
        if arr.ndim == 2:
            arr = arr[:, :, None]
    return np.mean(arr, axis=2) / 255.0


# ---------------------------------------------------------------------------
# Normals
# ---------------------------------------------------------------------------


def decode_normal(
    image: ImageLike,
    flip_x: bool = True,
    resample: int = Image.NEAREST,
    size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Decode a predicted RGB normal map into unit normal vectors.

    The model outputs normals as an RGB image encoding ``(n + 1) / 2``. The
    Marigold-style evaluation decodes this with ``2 * (img / 255) - 1`` and
    flips the X channel (``-nx``) to match the ground-truth convention.

    Args:
        image: Predicted normal image (PIL image, ndarray, or path).
        flip_x: Whether to flip the X (red) channel (default True, matching
            the official evaluation).
        resample: Resampling filter used when ``size`` is provided.
        size: Optional ``(width, height)`` to resize the prediction to before
            decoding.

    Returns:
        A float32 normal map with shape ``(H, W, 3)`` in ``[-1, 1]``.
    """
    if isinstance(image, str):
        image = Image.open(image)
    if isinstance(image, Image.Image):
        if size is not None:
            image = image.resize(size, resample=resample)
        arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    else:
        arr = np.asarray(image, dtype=np.float32)
        if arr.ndim == 2:
            arr = np.repeat(arr[:, :, None], 3, axis=2)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Expected an RGB normal image, got array with shape {arr.shape}")
        if size is not None:
            arr = np.asarray(
                Image.fromarray(arr.astype(np.uint8)).resize(size, resample=resample),
                dtype=np.float32,
            )

    normals = arr / 255.0 * 2.0 - 1.0
    if flip_x:
        normals = normals.copy()
        normals[:, :, 0] = -normals[:, :, 0]
    return normals


# ---------------------------------------------------------------------------
# Point map (recon3d)
# ---------------------------------------------------------------------------


def decode_point_map(image: ImageLike) -> np.ndarray:
    """Decode a raw point-map tensor/image into per-pixel 3D points.

    Recon3D predictions are produced by ``inferencer.decode_image(...,
    output_raw_tensor=True)`` as HxWx3 float arrays (the VAE output, roughly in
    ``[-1, 1]`` space). This helper accepts either such an array directly or
    an 8-bit image (whose channels are rescaled from ``[0, 255]`` to
    ``[-1, 1]`` via ``x / 255 * 2 - 1``), and returns an HxWx3 float32 point
    map.

    Args:
        image: Raw point map (HxWx3 float ndarray), a PIL image, or a path.

    Returns:
        A float32 point map with shape ``(H, W, 3)``.
    """
    if isinstance(image, str):
        image = Image.open(image)
    if isinstance(image, Image.Image):
        arr = np.asarray(image.convert("RGB"), dtype=np.float32)
        return arr / 255.0 * 2.0 - 1.0

    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected an HxWx3 point map, got array with shape {arr.shape}")
    if arr.max() > 1.0 + 1e-6 or arr.min() < -1.0 - 1e-6:
        return arr / 255.0 * 2.0 - 1.0
    return arr
