# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Model-owned request quality policy for MiniMax H3."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from numbers import Integral
from typing import Any

from vllm_omni.diffusion.cache.cachedit import CacheDiTRequestSpec
from vllm_omni.diffusion.cache.teacache.config import TeaCacheConfig
from vllm_omni.diffusion.data import DiffusionCacheConfig
from vllm_omni.errors import OmniClientError

MINIMAX_H3_GENERIC_CACHE_KEY = "minimax_h3.generic"
MINIMAX_H3_HIGH_CACHE_KEY = "minimax_h3.high"
MINIMAX_H3_FORCE_REFRESH_STEP_HINT_ARG = "force_refresh_step_hint"
MINIMAX_H3_FORCE_REFRESH_STEP_POLICY_ARG = "force_refresh_step_policy"
MINIMAX_H3_FORCE_REFRESH_POLICIES = ("once", "repeat")

_MINIMAX_H3_FL2VA_TEACACHE_COEFFICIENTS = [
    2.283704065852778e03,
    -7.775977277886368e02,
    9.408414741359490e01,
    -4.232669906169421e00,
    2.173782527946167e-01,
]
_MINIMAX_H3_REF2VA_TEACACHE_COEFFICIENTS = [
    2.1282372412524368e04,
    -2.2712379331933703e03,
    -5.8373015511205919e00,
    6.2854270826260406e00,
    2.5209538122600766e-01,
]
_MINIMAX_H3_FL2VA_TEACACHE_THRESHOLD = 0.17
_MINIMAX_H3_REF2VA_TEACACHE_THRESHOLD = 0.295
_MINIMAX_H3_REF2VA_TEACACHE_STEPS = 50
_MINIMAX_H3_REF2VA_TEACACHE_MIN_COMPUTE_STEPS = 35
_MINIMAX_H3_REF2VA_TEACACHE_MAX_CACHED_STEPS = 5


def _minimax_h3_teacache_config(
    config: DiffusionCacheConfig,
    *,
    ref2va: bool,
) -> TeaCacheConfig:
    coefficients = config.coefficients
    if coefficients is None:
        coefficients = _MINIMAX_H3_REF2VA_TEACACHE_COEFFICIENTS if ref2va else _MINIMAX_H3_FL2VA_TEACACHE_COEFFICIENTS
    threshold = config.rel_l1_thresh
    if threshold is None:
        threshold = _MINIMAX_H3_REF2VA_TEACACHE_THRESHOLD if ref2va else _MINIMAX_H3_FL2VA_TEACACHE_THRESHOLD
    return TeaCacheConfig(
        transformer_type="MiniMaxH3DiTModel",
        rel_l1_thresh=threshold,
        coefficients=coefficients,
        max_cached_steps=_MINIMAX_H3_REF2VA_TEACACHE_MAX_CACHED_STEPS if ref2va else None,
        min_compute_steps=_MINIMAX_H3_REF2VA_TEACACHE_MIN_COMPUTE_STEPS if ref2va else 0,
        calibrated_num_inference_steps=_MINIMAX_H3_REF2VA_TEACACHE_STEPS if ref2va else None,
    )


def minimax_h3_teacache_hook_configs(
    partition: str,
    config: DiffusionCacheConfig,
) -> dict[str, TeaCacheConfig]:
    """Declare TeaCache targets and calibrated policy for an H3 partition."""
    if partition == "fl2va":
        return {"transformer": _minimax_h3_teacache_config(config, ref2va=False)}
    if partition == "ref2va":
        return {"transformer": _minimax_h3_teacache_config(config, ref2va=True)}
    if partition == "combined":
        return {
            "transformer": _minimax_h3_teacache_config(config, ref2va=False),
            "transformers_ref": _minimax_h3_teacache_config(config, ref2va=True),
        }
    raise ValueError(f"Unsupported MiniMax-H3 partition for TeaCache: {partition!r}")


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

    These are model-specific ``extra_args`` rather than global sampling fields,
    keeping other diffusion models unchanged.
    """

    if not extra_args:
        return None

    raw_hint = extra_args.get(MINIMAX_H3_FORCE_REFRESH_STEP_HINT_ARG)
    raw_policy = extra_args.get(MINIMAX_H3_FORCE_REFRESH_STEP_POLICY_ARG)

    if raw_hint is None:
        if raw_policy is not None:
            raise OmniClientError("MiniMax H3 force_refresh_step_policy requires force_refresh_step_hint")
        return None
    if isinstance(raw_hint, bool) or not isinstance(raw_hint, Integral):
        raise OmniClientError("MiniMax H3 force_refresh_step_hint must be a positive integer")

    hint = int(raw_hint)
    if not 1 <= hint <= num_inference_steps:
        raise OmniClientError(
            "MiniMax H3 force_refresh_step_hint must be between 1 and "
            f"num_inference_steps ({num_inference_steps}), got {hint}"
        )

    policy = "once" if raw_policy is None else raw_policy
    if not isinstance(policy, str) or policy not in MINIMAX_H3_FORCE_REFRESH_POLICIES:
        raise OmniClientError(
            "MiniMax H3 force_refresh_step_policy must be one of "
            f"{list(MINIMAX_H3_FORCE_REFRESH_POLICIES)}, got {policy!r}"
        )
    return hint, policy


def _has_force_refresh_args(extra_args: Mapping[str, Any] | None) -> bool:
    if not extra_args:
        return False
    return any(
        extra_args.get(name) is not None
        for name in (
            MINIMAX_H3_FORCE_REFRESH_STEP_HINT_ARG,
            MINIMAX_H3_FORCE_REFRESH_STEP_POLICY_ARG,
        )
    )


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
    server-configured profile. Otherwise, omitted quality selects no cache.
    In either case, ``lossless`` selects no cache and ``high`` selects H3's
    high-quality profile. The policy therefore owns whether a request installs
    Cache-DiT; the startup backend only controls the omitted-quality default.
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
        if quality == "high":
            base_key = MINIMAX_H3_HIGH_CACHE_KEY
            base_config = _high_quality_cache_config()
        elif self._configured_backend == "cache_dit" and quality is None:
            base_key = MINIMAX_H3_GENERIC_CACHE_KEY
            base_config = self._od_config.cache_config
        else:
            if _has_force_refresh_args(extra_args):
                raise OmniClientError("MiniMax H3 force-refresh arguments require an active Cache-DiT request target")
            return MiniMaxH3QualityPlan(cache_dit=None)

        force_refresh = _resolve_force_refresh(
            extra_args,
            num_inference_steps=num_inference_steps,
        )
        return MiniMaxH3QualityPlan(
            cache_dit=CacheDiTRequestSpec(
                installation_key=_cache_installation_key(base_key, force_refresh),
                cache_config=_with_force_refresh(base_config, force_refresh),
                num_inference_steps=num_inference_steps,
            ),
        )


__all__ = [
    "MINIMAX_H3_GENERIC_CACHE_KEY",
    "MINIMAX_H3_HIGH_CACHE_KEY",
    "MiniMaxH3QualityPlan",
    "MiniMaxH3QualityPolicy",
    "minimax_h3_teacache_hook_configs",
]
