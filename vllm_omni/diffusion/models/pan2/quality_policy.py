# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request quality policy for PAN2: which Cache-DiT profile, if any, a request runs with."""

from __future__ import annotations

from vllm_omni.diffusion.cache.cachedit import CacheDiTRequestSpec
from vllm_omni.diffusion.data import DiffusionCacheConfig, OmniDiffusionConfig

PAN2_GENERIC_CACHE_KEY = "pan2.generic"
PAN2_HIGH_CACHE_KEY = "pan2.high"


def pan2_high_quality_cache_config() -> DiffusionCacheConfig:
    """PAN2's conservative Cache-DiT profile, selected by ``quality="high"``."""
    return DiffusionCacheConfig(
        Fn_compute_blocks=1,
        Bn_compute_blocks=0,
        max_warmup_steps=4,
        residual_diff_threshold=0.06,
        max_continuous_cached_steps=1,
        enable_taylorseer=False,
        scm_steps_mask_policy=None,
    )


class PAN2QualityPolicy:
    """Resolve a request's ``quality`` into the Cache-DiT profile it should run with.

    ``high`` selects PAN2's conservative profile and ``lossless`` runs without a cache. Omitted quality uses the
    server's Cache-DiT profile when the server started with ``cache_backend="cache_dit"``, and no cache otherwise.
    """

    def __init__(self, od_config: OmniDiffusionConfig) -> None:
        self._od_config = od_config

    def resolve(self, *, quality: str | None, num_inference_steps: int) -> CacheDiTRequestSpec | None:
        if quality == "high":
            key, cache_config = PAN2_HIGH_CACHE_KEY, pan2_high_quality_cache_config()
        elif quality is None and self._od_config.cache_backend == "cache_dit":
            key, cache_config = PAN2_GENERIC_CACHE_KEY, self._od_config.cache_config
        else:
            return None
        return CacheDiTRequestSpec(
            installation_key=key, cache_config=cache_config, num_inference_steps=num_inference_steps
        )


__all__ = ["PAN2_GENERIC_CACHE_KEY", "PAN2_HIGH_CACHE_KEY", "PAN2QualityPolicy", "pan2_high_quality_cache_config"]
