# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class MagCacheConfig:
    """
    Configuration for MagCache applied to transformer models.

    MagCache (Magnitude-based Cache) is an adaptive caching technique that speeds up
    diffusion model inference by reusing transformer block computations based on
    magnitude ratios between consecutive timesteps.

    Reference: https://github.com/Zehong-Ma/MagCache

    Args:
        threshold: Accumulated error threshold. Higher = more aggressive skipping (faster, lower quality).
            Default: 0.24
        max_skip_steps: Max consecutive skip steps (K).
            Default: 5
        retention_ratio: Fraction of initial steps where skipping is disabled (stability).
            Default: 0.1
        num_inference_steps: Total inference steps. Required for retention step calculation.
            Default: 28
        mag_ratios: Pre-computed magnitude ratios per step. Calibrate or use strategy defaults.
            Default: None
        mag_calibrate: If True, runs without skipping and logs norm_ratios for calibration.
            Default: False
        transformer_type: Transformer class name for logging.
            Default: "FluxTransformer2DModel"
    """

    threshold: float = 0.24
    max_skip_steps: int = 5
    retention_ratio: float = 0.1
    num_inference_steps: int = 28
    mag_ratios: torch.Tensor | list[float] | None = None
    mag_calibrate: bool = False
    transformer_type: str = "FluxTransformer2DModel"

    def __post_init__(self) -> None:
        """Validate and set default coefficients."""
        if self.threshold <= 0:
            raise ValueError(f"threshold must be positive, got {self.threshold}")

        if self.max_skip_steps <= 0:
            raise ValueError(f"max_skip_steps must be positive, got {self.max_skip_steps}")

        if not 0 < self.retention_ratio < 1:
            raise ValueError(f"retention_ratio must be in (0, 1), got {self.retention_ratio}")

        if self.num_inference_steps is None:
            raise ValueError(
                "num_inference_steps must be provided for MagCache. "
                "This is required to determine retention steps and interpolate mag_ratios. "
                "For Flux models, use num_inference_steps=28."
            )

        if self.num_inference_steps <= 0:
            raise ValueError(f"num_inference_steps must be positive, got {self.num_inference_steps}")

        if not self.mag_calibrate and self.mag_ratios is None:
            raise ValueError(
                "mag_ratios must be provided for MagCache inference because these ratios "
                "are model-dependent. To get them for your model:\n"
                "1. Initialize MagCacheConfig(mag_calibrate=True, ...)\n"
                "2. Run inference on your model once.\n"
                "3. Copy the printed ratios array and pass it to mag_ratios in the config.\n"
                "For Flux models, you can import FLUX_MAG_RATIOS from vllm_omni.diffusion.cache.magcache.strategy."
            )

        if not self.mag_calibrate and self.mag_ratios is not None:
            if not torch.is_tensor(self.mag_ratios):
                self.mag_ratios = torch.tensor(self.mag_ratios)
