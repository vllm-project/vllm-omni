# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Model-owned request quality policy for MiniMax H3."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from numbers import Integral
from typing import Any

from vllm_omni.diffusion.cache.cachedit.runtime import (
    CacheDiTRequestSpec,
)
from vllm_omni.diffusion.data import DiffusionCacheConfig
from vllm_omni.errors import OmniClientError

MINIMAX_H3_GENERIC_CACHE_KEY = "minimax_h3.generic"
MINIMAX_H3_HIGH_CACHE_KEY = "minimax_h3.high"
MINIMAX_H3_FORCE_REFRESH_ARG = "force_refresh"
MINIMAX_H3_FORCE_REFRESH_POLICY_ARG = "force_refresh_policy"
MINIMAX_H3_FORCE_REFRESH_STEP_HINT_ARG = "force_refresh_step_hint"
MINIMAX_H3_FORCE_REFRESH_STEP_POLICY_ARG = "force_refresh_step_policy"
MINIMAX_H3_FORCE_REFRESH_POLICIES = ("once", "repeat")


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


def _resolve_force_refresh(
    extra_args: Mapping[str, Any] | None,
    *,
    num_inference_steps: int,
) -> tuple[int, str] | None:
    """Resolve H3's request-scoped Cache-DiT refresh hint.

    ``force_refresh`` is the short H3 request name.  The explicit
    ``force_refresh_step_hint``/``force_refresh_step_policy`` spellings are
    accepted as well so callers can use Cache-DiT's native terminology.
    These are model-specific ``extra_args`` rather than global sampling
    fields, keeping other diffusion models unchanged.
    """

    if not extra_args:
        return None

    short_hint = extra_args.get(MINIMAX_H3_FORCE_REFRESH_ARG)
    explicit_hint = extra_args.get(MINIMAX_H3_FORCE_REFRESH_STEP_HINT_ARG)
    if short_hint is not None and explicit_hint is not None and short_hint != explicit_hint:
        raise OmniClientError(
            "MiniMax H3 extra_args['force_refresh'] and "
            "extra_args['force_refresh_step_hint'] must match when both are provided"
        )
    raw_hint = explicit_hint if explicit_hint is not None else short_hint

    short_policy = extra_args.get(MINIMAX_H3_FORCE_REFRESH_POLICY_ARG)
    explicit_policy = extra_args.get(MINIMAX_H3_FORCE_REFRESH_STEP_POLICY_ARG)
    if short_policy is not None and explicit_policy is not None and short_policy != explicit_policy:
        raise OmniClientError(
            "MiniMax H3 extra_args['force_refresh_policy'] and "
            "extra_args['force_refresh_step_policy'] must match when both are provided"
        )
    raw_policy = explicit_policy if explicit_policy is not None else short_policy

    if raw_hint is None:
        if raw_policy is not None:
            raise OmniClientError(
                "MiniMax H3 force_refresh_policy requires a force_refresh step hint"
            )
        return None
    if isinstance(raw_hint, bool) or not isinstance(raw_hint, Integral):
        raise OmniClientError(
            "MiniMax H3 force_refresh must be a positive integer denoising step hint"
        )

    hint = int(raw_hint)
    if not 1 <= hint <= num_inference_steps:
        raise OmniClientError(
            "MiniMax H3 force_refresh must be between 1 and "
            f"num_inference_steps ({num_inference_steps}), got {hint}"
        )

    policy = "once" if raw_policy is None else raw_policy
    if not isinstance(policy, str) or policy not in MINIMAX_H3_FORCE_REFRESH_POLICIES:
        raise OmniClientError(
            "MiniMax H3 force_refresh_policy must be one of "
            f"{list(MINIMAX_H3_FORCE_REFRESH_POLICIES)}, got {policy!r}"
        )
    return hint, policy


def _with_force_refresh(
    cache_config: DiffusionCacheConfig,
    force_refresh: tuple[int, str] | None,
) -> DiffusionCacheConfig:
    if force_refresh is None:
        return cache_config
    hint, policy = force_refresh
    return replace(
        cache_config,
        force_refresh_step_hint=hint,
        force_refresh_step_policy=policy,
    )


def _cache_installation_key(base_key: str, force_refresh: tuple[int, str] | None) -> str:
    """Make refresh-policy transitions explicit to the request runtime.

    Cache-DiT cannot clear an existing ``force_refresh_step_hint`` through its
    incremental context update API because ``None`` is treated as "keep the
    old value".  Including the request hint in the installation key therefore
    makes a hint change perform a safe hook reinstall, while repeated requests
    with the same hint still only refresh the context.
    """

    if force_refresh is None:
        return base_key
    hint, policy = force_refresh
    return f"{base_key}:force_refresh={hint}:{policy}"


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
    H3-specific Cache-DiT refresh hints are read from request ``extra_args``;
    they do not change the global diffusion request contract.
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
        extra_args: Mapping[str, Any] | None = None,
    ) -> MiniMaxH3QualityPlan:
        force_refresh = _resolve_force_refresh(
            extra_args,
            num_inference_steps=num_inference_steps,
        )
        if quality == "high":
            if self._configured_backend != "cache_dit":
                raise OmniClientError(
                    'MiniMax-H3 quality="high" requires the server to start with cache_backend="cache_dit"'
                )
            return MiniMaxH3QualityPlan(
                cache_dit=CacheDiTRequestSpec(
                    installation_key=_cache_installation_key(MINIMAX_H3_HIGH_CACHE_KEY, force_refresh),
                    cache_config=_with_force_refresh(_high_quality_cache_config(), force_refresh),
                    num_inference_steps=num_inference_steps,
                ),
            )

        generic_requested = self._configured_backend == "cache_dit" and quality is None
        cache_dit = (
            CacheDiTRequestSpec(
                installation_key=_cache_installation_key(MINIMAX_H3_GENERIC_CACHE_KEY, force_refresh),
                cache_config=_with_force_refresh(self._od_config.cache_config, force_refresh),
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
