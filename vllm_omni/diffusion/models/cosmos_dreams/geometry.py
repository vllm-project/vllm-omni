# SPDX-License-Identifier: Apache-2.0
"""Pure runtime-resolution policy shared by Cosmos-Dreams request boundaries."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"Cosmos-Dreams {name} must be a positive integer, got {value!r}.")
    integer = int(value)
    if integer <= 0:
        raise ValueError(f"Cosmos-Dreams {name} must be a positive integer, got {value!r}.")
    return integer


@dataclass(frozen=True)
class CosmosDreamsGeometry:
    """Resolved pixel, latent, and patch-grid geometry for one request."""

    height: int
    width: int
    vae_spatial_compression_factor: int = 16
    latent_patch_size: int = 2

    def __post_init__(self) -> None:
        for name in ("height", "width", "vae_spatial_compression_factor", "latent_patch_size"):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))

    @property
    def latent_height(self) -> int:
        return math.ceil(self.height / self.vae_spatial_compression_factor)

    @property
    def latent_width(self) -> int:
        return math.ceil(self.width / self.vae_spatial_compression_factor)

    @property
    def patch_grid(self) -> tuple[int, int]:
        return (
            math.ceil(self.latent_height / self.latent_patch_size),
            math.ceil(self.latent_width / self.latent_patch_size),
        )

    @property
    def vision_tokens_per_frame(self) -> int:
        grid_height, grid_width = self.patch_grid
        return grid_height * grid_width

    def tokens_per_frame(self, conditioning_tokens_per_frame: int) -> int:
        if (
            isinstance(conditioning_tokens_per_frame, bool)
            or not isinstance(conditioning_tokens_per_frame, int)
            or conditioning_tokens_per_frame < 0
        ):
            raise ValueError("Cosmos-Dreams conditioning_tokens_per_frame must be a non-negative integer.")
        return self.vision_tokens_per_frame + conditioning_tokens_per_frame

    @property
    def session_key(self) -> tuple[int, int]:
        return self.height, self.width


@dataclass(frozen=True)
class CosmosDreamsResolutionPolicy:
    """Deployment-owned bounds for all request-resolved Cosmos-Dreams canvases."""

    default_resolution: tuple[int, int] = (720, 1280)
    max_pixels: int = 921_600
    alignment: int = 16
    min_aspect: float = 704 / 1280
    max_aspect: float = 1280 / 704
    vae_spatial_compression_factor: int = 16
    latent_patch_size: int = 2

    def __post_init__(self) -> None:
        if not isinstance(self.default_resolution, list | tuple) or len(self.default_resolution) != 2:
            raise ValueError("Cosmos-Dreams default_resolution must be [height, width].")
        default = (
            _positive_int(self.default_resolution[0], "default height"),
            _positive_int(self.default_resolution[1], "default width"),
        )
        object.__setattr__(self, "default_resolution", default)
        for name in ("max_pixels", "alignment", "vae_spatial_compression_factor", "latent_patch_size"):
            object.__setattr__(self, name, _positive_int(getattr(self, name), name))
        if not math.isfinite(self.min_aspect) or not math.isfinite(self.max_aspect):
            raise ValueError("Cosmos-Dreams aspect bounds must be finite.")
        if self.min_aspect <= 0 or self.min_aspect > self.max_aspect:
            raise ValueError(f"Cosmos-Dreams aspect bounds are invalid: [{self.min_aspect}, {self.max_aspect}].")
        self.resolve(*default)

    def resolve(self, height: Any, width: Any) -> CosmosDreamsGeometry:
        resolved_height = _positive_int(height, "height")
        resolved_width = _positive_int(width, "width")
        if resolved_height % self.alignment or resolved_width % self.alignment:
            raise ValueError(
                f"Cosmos-Dreams dimensions must be multiples of {self.alignment}, "
                f"got {resolved_height}x{resolved_width}."
            )
        pixels = resolved_height * resolved_width
        if pixels > self.max_pixels:
            raise ValueError(
                f"Cosmos-Dreams dimensions exceed max_pixels={self.max_pixels}: "
                f"{resolved_height}x{resolved_width}={pixels}."
            )
        aspect = resolved_width / resolved_height
        if not self.min_aspect <= aspect <= self.max_aspect:
            raise ValueError(
                "Cosmos-Dreams width/height aspect ratio must be in "
                f"[{self.min_aspect:.6g}, {self.max_aspect:.6g}], got {aspect:.6g}."
            )
        geometry = CosmosDreamsGeometry(
            resolved_height,
            resolved_width,
            vae_spatial_compression_factor=self.vae_spatial_compression_factor,
            latent_patch_size=self.latent_patch_size,
        )
        if geometry.latent_height <= 0 or geometry.latent_width <= 0 or min(geometry.patch_grid) <= 0:
            raise ValueError("Cosmos-Dreams dimensions do not produce a valid model patch grid.")
        return geometry

    def aspect_preserving_canvas(self, source_height: Any, source_width: Any) -> CosmosDreamsGeometry:
        source_height = _positive_int(source_height, "media height")
        source_width = _positive_int(source_width, "media width")
        aspect = source_width / source_height
        if not self.min_aspect <= aspect <= self.max_aspect:
            raise ValueError(
                "Cosmos-Dreams media width/height aspect ratio must be in "
                f"[{self.min_aspect:.6g}, {self.max_aspect:.6g}], got {aspect:.6g}."
            )
        height = math.floor(math.sqrt(self.max_pixels / aspect) / self.alignment) * self.alignment
        width = math.floor(math.sqrt(self.max_pixels * aspect) / self.alignment) * self.alignment
        return self.resolve(max(self.alignment, height), max(self.alignment, width))

    def iter_valid_geometries(self) -> Iterator[CosmosDreamsGeometry]:
        """Enumerate every aligned geometry admitted by this policy."""

        max_height = int(math.sqrt(self.max_pixels / self.min_aspect))
        for height in range(self.alignment, max_height + 1, self.alignment):
            min_width = max(self.alignment, math.ceil(height * self.min_aspect / self.alignment) * self.alignment)
            max_width = min(
                math.floor(height * self.max_aspect / self.alignment) * self.alignment,
                math.floor(self.max_pixels / height / self.alignment) * self.alignment,
            )
            for width in range(min_width, max_width + 1, self.alignment):
                yield self.resolve(height, width)


def _param(params: Any, name: str) -> Any:
    if isinstance(params, Mapping):
        return params.get(name)
    return getattr(params, name, None)


def _media_hw(media: Any) -> tuple[int, int] | None:
    if media is None:
        return None
    if isinstance(media, Mapping):
        height, width = media.get("height"), media.get("width")
        if height is not None and width is not None:
            return _positive_int(height, "media height"), _positive_int(width, "media width")
        for key in ("image", "video", "frames", "data", "control", "control_path"):
            if media.get(key) is not None:
                resolved = _media_hw(media[key])
                if resolved is not None:
                    return resolved
        return None
    if isinstance(media, Sequence) and not isinstance(media, str | bytes):
        if len(media) == 2 and all(isinstance(value, Integral) and not isinstance(value, bool) for value in media):
            return _positive_int(media[0], "media height"), _positive_int(media[1], "media width")
        return _media_hw(media[0]) if media else None
    try:
        from vllm_omni.diffusion.models.cosmos3.transfer import media_hw

        return media_hw(media)
    except (ImportError, OSError, TypeError, ValueError):
        return None


def resolve_cosmos_dreams_geometry(
    sampling_params: Any,
    media: Any,
    policy: CosmosDreamsResolutionPolicy,
) -> CosmosDreamsGeometry:
    """Resolve explicit dimensions, an inferred media canvas, or the deployment default."""

    if not isinstance(policy, CosmosDreamsResolutionPolicy):
        raise TypeError("Cosmos-Dreams geometry resolution requires CosmosDreamsResolutionPolicy.")
    height = _param(sampling_params, "height")
    width = _param(sampling_params, "width")
    if (height is None) != (width is None):
        raise ValueError("Cosmos-Dreams height and width must be supplied together.")
    if height is not None:
        return policy.resolve(height, width)
    source_hw = _media_hw(media)
    if source_hw is not None:
        return policy.aspect_preserving_canvas(*source_hw)
    return policy.resolve(*policy.default_resolution)
