# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LingBotVAETileGeometry:
    """Spatial tile geometry in decoded sample pixels."""

    tile_sample_min_height: int
    tile_sample_min_width: int
    tile_sample_stride_height: int
    tile_sample_stride_width: int

    def __post_init__(self) -> None:
        values = {
            "tile_sample_min_height": self.tile_sample_min_height,
            "tile_sample_min_width": self.tile_sample_min_width,
            "tile_sample_stride_height": self.tile_sample_stride_height,
            "tile_sample_stride_width": self.tile_sample_stride_width,
        }
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"LingBot VAE `{name}` must be a positive integer, got {value!r}.")
            if value % 8 != 0:
                raise ValueError(f"LingBot VAE `{name}` must be divisible by the spatial factor 8, got {value}.")

        if self.tile_sample_stride_height > self.tile_sample_min_height:
            raise ValueError("LingBot VAE tile height stride cannot exceed tile height.")
        if self.tile_sample_stride_width > self.tile_sample_min_width:
            raise ValueError("LingBot VAE tile width stride cannot exceed tile width.")

    @classmethod
    def from_vae(cls, vae: Any) -> LingBotVAETileGeometry:
        return cls(
            tile_sample_min_height=int(getattr(vae, "tile_sample_min_height", 256)),
            tile_sample_min_width=int(getattr(vae, "tile_sample_min_width", 256)),
            tile_sample_stride_height=int(getattr(vae, "tile_sample_stride_height", 192)),
            tile_sample_stride_width=int(getattr(vae, "tile_sample_stride_width", 192)),
        )

    def with_overrides(
        self,
        raw: Mapping[str, Any] | None,
        *,
        label: str,
    ) -> LingBotVAETileGeometry:
        if raw is None:
            raw = {}
        if not isinstance(raw, Mapping):
            raise ValueError(f"LingBot VAE tiling `{label}` profile must be a mapping.")
        supported = set(self.as_enable_kwargs())
        unknown = sorted(set(raw) - supported)
        if unknown:
            raise ValueError(f"Unsupported LingBot VAE tiling `{label}` options: {unknown}.")
        values = self.as_enable_kwargs()
        values.update(raw)
        return LingBotVAETileGeometry(**values)

    def as_enable_kwargs(self) -> dict[str, int]:
        return {
            "tile_sample_min_height": self.tile_sample_min_height,
            "tile_sample_min_width": self.tile_sample_min_width,
            "tile_sample_stride_height": self.tile_sample_stride_height,
            "tile_sample_stride_width": self.tile_sample_stride_width,
        }


def normalize_lingbot_vae_tiling(
    model_config: Mapping[str, Any] | None,
    *,
    base_geometry: LingBotVAETileGeometry,
) -> LingBotVAETileGeometry:
    model_config = model_config or {}
    raw = model_config.get("lingbot_vae_tiling", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ValueError("LingBot `model_config.lingbot_vae_tiling` must be a mapping.")

    unknown = sorted(set(raw) - {"base"})
    if unknown:
        raise ValueError(f"Unsupported LingBot VAE tiling profiles: {unknown}.")
    return base_geometry.with_overrides(raw.get("base"), label="base")


def configure_lingbot_vae_tiling(
    vae: Any,
    *,
    enabled: bool,
    geometry: LingBotVAETileGeometry,
) -> None:
    if not enabled:
        disable_tiling = getattr(vae, "disable_tiling", None)
        if callable(disable_tiling):
            disable_tiling()
        else:
            vae.use_tiling = False
        return

    enable_tiling = getattr(vae, "enable_tiling", None)
    if not callable(enable_tiling):
        raise RuntimeError("LingBot VAE tiling requires a VAE with enable_tiling().")
    enable_tiling(**geometry.as_enable_kwargs())
