# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Configuration for step-level DiT velocity caching."""

from __future__ import annotations

from dataclasses import dataclass

from vllm_omni.diffusion.data import DiffusionCacheConfig

# Pipeline attributes set by StepCacheBackend.enable().
STEP_CACHE_CONFIG_ATTR = "_stepcache_config"
STEP_CACHE_STATE_ATTR = "_stepcache_state"


@dataclass(frozen=True)
class StepCacheConfig:
    """Runtime config for velocity-based step skipping.

    Skips entire DiT forwards when successive video velocity predictions are
    highly aligned (cosine similarity above configured thresholds).

    Reference: DreamZero paper DiT Caching / dreamzero.git ``should_run_model``.
    """

    enabled: bool = True
    min_history_steps: int = 2
    max_history: int = 2
    sim_thresholds: tuple[float, ...] = (0.95, 0.93)
    skip_countdowns: tuple[int, ...] = (4, 2)

    @classmethod
    def from_diffusion_cache_config(cls, config: DiffusionCacheConfig) -> StepCacheConfig:
        enabled = config.step_cache_dit_enabled

        thresholds = tuple(config.velocity_sim_thresholds)
        countdowns = tuple(config.velocity_skip_countdowns)
        if len(thresholds) != len(countdowns):
            raise ValueError(
                "velocity_sim_thresholds and velocity_skip_countdowns must have the same length; "
                f"got {len(thresholds)} and {len(countdowns)}."
            )

        return cls(
            enabled=enabled,
            min_history_steps=config.step_cache_dit_min_history,
            max_history=config.step_cache_dit_max_history,
            sim_thresholds=thresholds,
            skip_countdowns=countdowns,
        )
