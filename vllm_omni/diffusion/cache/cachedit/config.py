# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Configuration types and conversion helpers for Cache-DiT."""

from __future__ import annotations

from dataclasses import dataclass

from cache_dit import (
    CalibratorConfig,
    DBCacheConfig,
    ForwardPattern,
    TaylorSeerCalibratorConfig,
)
from cache_dit.caching.cache_adapters.cache_adapter import CachedAdapter
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionCacheConfig

logger = init_logger(__name__)


@dataclass
class CacheDiTAdapterConfig:
    """Describe how Cache-DiT wraps a model's transformer blocks."""

    block_forward_patterns: dict[str, ForwardPattern]
    has_separate_cfg: bool = False
    cached_adapter_cls: type[CachedAdapter] | None = None
    check_forward_pattern: bool = True


@dataclass(frozen=True)
class CacheDiTConfig:
    """Cache-DiT-specific projection of the shared diffusion cache config."""

    Fn_compute_blocks: int = 1
    Bn_compute_blocks: int = 0
    max_warmup_steps: int = 4
    max_cached_steps: int = -1
    residual_diff_threshold: float = 0.24
    max_continuous_cached_steps: int = 3
    enable_taylorseer: bool = False
    taylorseer_order: int = 1
    scm_steps_mask_policy: str | None = None
    scm_steps_policy: str = "dynamic"
    force_refresh_step_hint: int | None = None
    force_refresh_step_policy: str = "once"

    @classmethod
    def from_diffusion_config(cls, config: DiffusionCacheConfig) -> CacheDiTConfig:
        """Project the shared config onto fields consumed by Cache-DiT."""

        return cls(
            Fn_compute_blocks=config.Fn_compute_blocks,
            Bn_compute_blocks=config.Bn_compute_blocks,
            max_warmup_steps=config.max_warmup_steps,
            max_cached_steps=config.max_cached_steps,
            residual_diff_threshold=config.residual_diff_threshold,
            max_continuous_cached_steps=config.max_continuous_cached_steps,
            enable_taylorseer=config.enable_taylorseer,
            taylorseer_order=config.taylorseer_order,
            scm_steps_mask_policy=config.scm_steps_mask_policy,
            scm_steps_policy=config.scm_steps_policy,
            force_refresh_step_hint=config.force_refresh_step_hint,
            force_refresh_step_policy=config.force_refresh_step_policy,
        )

    def to_db_cache_config(self) -> DBCacheConfig:
        """Build the third-party cache configuration used during installation."""

        return DBCacheConfig(
            # The request step count is supplied by ``CacheDiTBackend.refresh``.
            num_inference_steps=None,
            Fn_compute_blocks=self.Fn_compute_blocks,
            Bn_compute_blocks=self.Bn_compute_blocks,
            max_warmup_steps=self.max_warmup_steps,
            max_cached_steps=self.max_cached_steps,
            max_continuous_cached_steps=self.max_continuous_cached_steps,
            residual_diff_threshold=self.residual_diff_threshold,
            force_refresh_step_hint=self.force_refresh_step_hint,
            force_refresh_step_policy=self.force_refresh_step_policy,
        )

    def to_calibrator_config(self) -> CalibratorConfig | None:
        """Build the optional TaylorSeer calibrator configuration."""

        if not self.enable_taylorseer:
            return None

        logger.info("TaylorSeer enabled with order=%s", self.taylorseer_order)
        return TaylorSeerCalibratorConfig(taylorseer_order=self.taylorseer_order)


__all__ = ["CacheDiTAdapterConfig", "CacheDiTConfig"]
