# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Model-owned request quality policy for MiniMax H3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vllm_omni.diffusion.cache.cachedit.runtime import (
    CacheDiTRequestSpec,
)
from vllm_omni.diffusion.data import DiffusionCacheConfig
from vllm_omni.errors import OmniClientError

MINIMAX_H3_GENERIC_CACHE_KEY = "minimax_h3.generic"
MINIMAX_H3_HIGH_CACHE_KEY = "minimax_h3.high"


def _high_quality_cache_config() -> DiffusionCacheConfig:
    return DiffusionCacheConfig(
        Fn_compute_blocks=1,
        Bn_compute_blocks=0,
        max_warmup_steps=4,
        residual_diff_threshold=0.04,
        max_continuous_cached_steps=1,
        enable_taylorseer=False,
        scm_steps_mask_policy=None,
    )


@dataclass(frozen=True)
class MiniMaxH3QualityPlan:
    """Resolved execution choices for one MiniMax H3 request."""

    cache_dit: CacheDiTRequestSpec | None


class MiniMaxH3QualityPolicy:
    """Resolve H3 request quality into a declarative Cache-DiT target.

    When the server starts with Cache-DiT, omitted quality selects the
    server-configured profile, ``lossless`` selects no cache, and ``high``
    selects H3's high-quality profile. Without startup Cache-DiT capability,
    omitted and ``lossless`` requests select no cache while ``high`` fails.
    Other registered quality intents select no cache until H3 defines a
    model-specific policy for them.
    The pipeline owns applying the resulting target at the request boundary.
    """

    def __init__(self, od_config: Any) -> None:
        self._od_config = od_config
        self._configured_backend = str(getattr(od_config, "cache_backend", "none") or "none").lower()

    def resolve(
        self,
        *,
        quality: str | None,
        num_inference_steps: int,
    ) -> MiniMaxH3QualityPlan:
        if quality == "high":
            if self._configured_backend != "cache_dit":
                raise OmniClientError(
                    'MiniMax-H3 quality="high" requires the server to start with cache_backend="cache_dit"'
                )
            return MiniMaxH3QualityPlan(
                cache_dit=CacheDiTRequestSpec(
                    installation_key=MINIMAX_H3_HIGH_CACHE_KEY,
                    cache_config=_high_quality_cache_config(),
                    num_inference_steps=num_inference_steps,
                ),
            )

        generic_requested = self._configured_backend == "cache_dit" and quality is None
        cache_dit = (
            CacheDiTRequestSpec(
                installation_key=MINIMAX_H3_GENERIC_CACHE_KEY,
                cache_config=self._od_config.cache_config,
                num_inference_steps=num_inference_steps,
            )
            if generic_requested
            else None
        )
        return MiniMaxH3QualityPlan(
            cache_dit=cache_dit,
        )


__all__ = [
    "MINIMAX_H3_GENERIC_CACHE_KEY",
    "MINIMAX_H3_HIGH_CACHE_KEY",
    "MiniMaxH3QualityPlan",
    "MiniMaxH3QualityPolicy",
]
