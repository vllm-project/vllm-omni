# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Match restored SeedVR2 colours to the resized input.

The restoration transformer reproduces detail faithfully but shifts global
colour, so every method here transfers colour from the resized input while
keeping the restored detail. Tensors are ``[B, C, T, H, W]`` RGB in ``[0, 1]``.

``wavelet`` swaps the lowest frequency band, which carries the tint, and keeps
every higher band from the restoration. ``lab`` adds CIELAB chroma histogram
matching on top for casts a single band cannot express. ``adain`` only aligns
per-channel statistics and is the cheapest, coarsest option.
"""

import torch
from torch import Tensor
from torch.nn import functional as F

from vllm_omni.inputs.data import COLOR_CORRECTION_METHODS, DEFAULT_COLOR_CORRECTION_METHOD

# Decomposition depth: the fifth band already isolates the illumination tint.
WAVELET_LEVELS = 5
# Restoration luminance is the reason to run the model, so chroma is matched
# fully while luminance keeps this much of the restored value.
DEFAULT_LUMINANCE_WEIGHT = 0.8

# CIE 15 lightness transfer, sRGB (IEC 61966-2-1) primaries, D65 white point.
_CIELAB_DELTA = 6.0 / 29.0
_D65_WHITE = (0.95047, 1.0, 1.08883)
_SRGB_TO_XYZ = (
    (0.4124564, 0.3575761, 0.1804375),
    (0.2126729, 0.7151522, 0.0721750),
    (0.0193339, 0.1191920, 0.9503041),
)
_XYZ_TO_SRGB = (
    (3.2404542, -1.5371385, -0.4985314),
    (-0.9692660, 1.8760108, 0.0415560),
    (0.0556434, -0.2040259, 1.0572252),
)


def _binomial_lowpass(image: Tensor, radius: int) -> Tensor:
    """One a-trous low-pass step with the separable 1-2-1 binomial kernel."""
    channels = image.shape[1]
    weights = image.new_tensor([1.0, 2.0, 1.0]) / 4.0
    kernel = torch.outer(weights, weights).expand(channels, 1, 3, 3)
    padded = F.pad(image, (radius,) * 4, mode="replicate")
    return F.conv2d(padded, kernel, groups=channels, dilation=radius)


def _lowest_band(image: Tensor, levels: int = WAVELET_LEVELS) -> Tensor:
    """Residual band after ``levels`` doubling-radius low-pass steps.

    Summing the per-level high-frequency residuals telescopes to
    ``image - lowest_band(image)``, so only this band has to be materialized.
    """
    # Keep a dilated tap well inside the frame so it samples real signal rather
    # than replicated edges; this bound matches the reference decomposition.
    limit = max(1, min(image.shape[-2:]) // 8)
    for level in range(levels):
        image = _binomial_lowpass(image, min(1 << level, limit))
    return image


def _match_histogram(source: Tensor, reference: Tensor) -> Tensor:
    """Give ``source`` the value distribution of ``reference``, rank for rank.

    Both channels hold the same pixel count, so ranks map one to one and the
    match is exact without quantile interpolation.
    """
    order = source.flatten().argsort()
    matched = torch.empty_like(order, dtype=reference.dtype)
    matched[order] = reference.flatten().sort().values
    return matched.view_as(source)


def _srgb_to_lab(rgb: Tensor) -> Tensor:
    linear = torch.where(rgb > 0.04045, ((rgb + 0.055) / 1.055).pow(2.4), rgb / 12.92)
    matrix = rgb.new_tensor(_SRGB_TO_XYZ)
    xyz = torch.einsum("ij,njhw->nihw", matrix, linear)
    xyz = xyz / xyz.new_tensor(_D65_WHITE).view(1, 3, 1, 1)
    scaled = torch.where(
        xyz > _CIELAB_DELTA**3,
        xyz.clamp_min(0.0).pow(1.0 / 3.0),
        xyz / (3.0 * _CIELAB_DELTA**2) + 4.0 / 29.0,
    )
    x, y, z = scaled[:, 0], scaled[:, 1], scaled[:, 2]
    return torch.stack([116.0 * y - 16.0, 500.0 * (x - y), 200.0 * (y - z)], dim=1)


def _lab_to_srgb(lab: Tensor) -> Tensor:
    lightness, chroma_a, chroma_b = lab[:, 0], lab[:, 1], lab[:, 2]
    y = (lightness + 16.0) / 116.0
    scaled = torch.stack([y + chroma_a / 500.0, y, y - chroma_b / 200.0], dim=1)
    xyz = torch.where(
        scaled > _CIELAB_DELTA,
        scaled.pow(3.0),
        3.0 * _CIELAB_DELTA**2 * (scaled - 4.0 / 29.0),
    )
    xyz = xyz * xyz.new_tensor(_D65_WHITE).view(1, 3, 1, 1)
    linear = torch.einsum("ij,njhw->nihw", lab.new_tensor(_XYZ_TO_SRGB), xyz)
    rgb = torch.where(
        linear > 0.0031308,
        1.055 * linear.clamp_min(0.0).pow(1.0 / 2.4) - 0.055,
        12.92 * linear,
    )
    return rgb.clamp_(0.0, 1.0)


def _wavelet_transfer(restored: Tensor, reference: Tensor) -> Tensor:
    return (restored - _lowest_band(restored) + _lowest_band(reference)).clamp_(0.0, 1.0)


def _channel_stats(image: Tensor, eps: float) -> tuple[Tensor, Tensor]:
    dims = (2, 3)
    deviation = (image.var(dims, correction=0, keepdim=True) + eps).sqrt()
    return image.mean(dims, keepdim=True), deviation


def _adain_transfer(restored: Tensor, reference: Tensor, eps: float = 1e-5) -> Tensor:
    restored_mean, restored_std = _channel_stats(restored, eps)
    reference_mean, reference_std = _channel_stats(reference, eps)
    normalized = (restored - restored_mean) / restored_std
    return (normalized * reference_std + reference_mean).clamp_(0.0, 1.0)


def _lab_transfer(restored: Tensor, reference: Tensor, luminance_weight: float) -> Tensor:
    restored_lab = _srgb_to_lab(_wavelet_transfer(restored, reference))
    reference_lab = _srgb_to_lab(reference)
    lightness = restored_lab[:, 0]
    if luminance_weight < 1.0:
        matched = _match_histogram(lightness, reference_lab[:, 0])
        lightness = lightness * luminance_weight + matched * (1.0 - luminance_weight)
    corrected = torch.stack(
        [
            lightness,
            _match_histogram(restored_lab[:, 1], reference_lab[:, 1]),
            _match_histogram(restored_lab[:, 2], reference_lab[:, 2]),
        ],
        dim=1,
    )
    return _lab_to_srgb(corrected)


def correct_video_color(
    video: Tensor,
    reference: Tensor,
    method: str = DEFAULT_COLOR_CORRECTION_METHOD,
    luminance_weight: float = DEFAULT_LUMINANCE_WEIGHT,
) -> Tensor:
    """Transfer ``reference`` colour onto ``video``; both are ``[B,C,T,H,W]`` in ``[0,1]``.

    Frames are processed one at a time because histogram matching sorts every
    pixel of a channel. Inputs are never modified.
    """
    if method not in COLOR_CORRECTION_METHODS:
        raise ValueError(f"SeedVR2 color_correction_method must be one of {list(COLOR_CORRECTION_METHODS)}: {method!r}")
    if method == "none":
        return video
    if video.shape != reference.shape:
        raise ValueError(f"SeedVR2 color correction needs matching shapes, got {video.shape} and {reference.shape}")
    if not 0.0 <= luminance_weight <= 1.0:
        raise ValueError(f"SeedVR2 luminance_weight must be within [0, 1]: {luminance_weight!r}")

    # Colour math is scale sensitive, so it runs in fp32 regardless of the model dtype.
    frames = video.movedim(2, 0).flatten(0, 1).float()
    references = reference.movedim(2, 0).flatten(0, 1).float()
    corrected = torch.empty_like(frames)
    for index in range(frames.shape[0]):
        frame, guide = frames[index : index + 1], references[index : index + 1]
        if method == "wavelet":
            corrected[index : index + 1] = _wavelet_transfer(frame, guide)
        elif method == "adain":
            corrected[index : index + 1] = _adain_transfer(frame, guide)
        else:
            corrected[index : index + 1] = _lab_transfer(frame, guide, luminance_weight)
    corrected = corrected.unflatten(0, (video.shape[2], video.shape[0])).movedim(0, 2)
    return corrected.to(dtype=video.dtype)
