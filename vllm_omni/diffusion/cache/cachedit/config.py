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

# Cache window validated for TaylorSeer order 2 on MiniMax-H3 Ref2VA (#8805).
VALIDATED_TAYLORSEER_MAX_CONTINUOUS_CACHED_STEPS = 4


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
        if self.taylorseer_order > 2:
            logger.warning(
                "TaylorSeer order=%s has no validated cache window (#8805 "
                "validated order 2 with max_continuous_cached_steps<=%s); "
                "disable enable_taylorseer unless this configuration has "
                "been validated.",
                self.taylorseer_order,
                VALIDATED_TAYLORSEER_MAX_CONTINUOUS_CACHED_STEPS,
            )
        elif (
            self.taylorseer_order == 2
            and self.max_continuous_cached_steps > VALIDATED_TAYLORSEER_MAX_CONTINUOUS_CACHED_STEPS
        ):
            logger.warning(
                "TaylorSeer order=2 with max_continuous_cached_steps=%s "
                "exceeds the validated cache window (order=2 with "
                "max_continuous_cached_steps=%s): order-2 extrapolation error "
                "grows quadratically with consecutive cached steps and can "
                "produce localized artifacts on Ref2VA videos (#8805). "
                "Disable enable_taylorseer for longer cache windows.",
                self.max_continuous_cached_steps,
                VALIDATED_TAYLORSEER_MAX_CONTINUOUS_CACHED_STEPS,
            )
        return TaylorSeerCalibratorConfig(taylorseer_order=self.taylorseer_order)


__all__ = [
    "CacheDiTAdapterConfig",
    "CacheDiTConfig",
    "VALIDATED_TAYLORSEER_MAX_CONTINUOUS_CACHED_STEPS",
]
