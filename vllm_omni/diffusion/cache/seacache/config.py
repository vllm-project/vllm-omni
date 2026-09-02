# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Configuration for the SeaCache hook."""

from __future__ import annotations

from dataclasses import dataclass

from vllm_omni.diffusion.cache.seacache.filter import _NORM_MODES

# Transformers whose TeaCache extractor supplies the SEA inputs (sigma + latent
# grid). SeaCache can only run on models listed here.
_SUPPORTED_TRANSFORMERS = (
    "FluxTransformer2DModel",
    # Qwen-Image family (t2i / Edit / Edit-Plus): the extractor slices the
    # decision feature to the noise-latent grid.
    "QwenImageTransformer2DModel",
    # Flux2-Klein's extractor alias: both Flux2 variants share the
    # Flux2Transformer2DModel name, so the backend dispatches by pipeline class.
    "Flux2Klein",
)


@dataclass
class SeaCacheConfig:
    """Parameters for [SeaCache](https://arxiv.org/abs/2602.18993) caching.

    TeaCache's accumulate-and-refresh rule on an SEA-filtered distance, so the
    metric tracks content change rather than noise; `sea_thresh` is the only
    value worth tuning.

    Attributes:
        sea_thresh: Refresh threshold (paper: delta); larger skips more steps,
            0 never skips (bit-identical to the uncached path).
        sea_norm_mode: Filter gain normalization; ``mean`` (the paper default)
            makes distances comparable across timesteps, ``peak``/``none``
            reproduce the paper's ablations.
        transformer_type: Transformer class name; selects the extractor.
    """

    sea_thresh: float = 0.3
    sea_norm_mode: str = "mean"
    transformer_type: str = "FluxTransformer2DModel"

    def validate(self) -> None:
        import math

        if (
            isinstance(self.sea_thresh, bool)
            or not isinstance(self.sea_thresh, (int, float))
            or not math.isfinite(self.sea_thresh)
        ):
            raise ValueError(f"SeaCache sea_thresh must be a finite number, got {self.sea_thresh!r}.")
        if self.sea_thresh < 0:
            raise ValueError(f"SeaCache sea_thresh must be non-negative, got {self.sea_thresh}.")
        if self.sea_norm_mode not in _NORM_MODES:
            raise ValueError(f"SeaCache sea_norm_mode must be one of {_NORM_MODES}, got {self.sea_norm_mode!r}.")
        if self.transformer_type not in _SUPPORTED_TRANSFORMERS:
            raise ValueError(
                f"SeaCache does not support transformer {self.transformer_type!r}; "
                f"supported: {list(_SUPPORTED_TRANSFORMERS)}."
            )
