# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def apply_declared_extra_args(
    sampling_params: OmniDiffusionSamplingParams,
    declared_params: frozenset[str],
    user_kwargs: dict[str, object],
) -> None:
    """Route pipeline-declared request params into ``sampling_params.extra_args``.

    Both online serving and offline examples call this so that model-specific
    keys (e.g. ``cfg_text_scale`` for BAGEL) end up in ``extra_args`` instead
    of being silently dropped.

    This is a no-op when no declared params are present in ``user_kwargs``, so
    it is safe to call on non-diffusion (e.g. AR) sampling params whose
    ``extra_args`` defaults to ``None``.
    """
    declared = {key: user_kwargs[key] for key in declared_params if user_kwargs.get(key) is not None}
    if not declared:
        return
    sampling_params.extra_args = {**(sampling_params.extra_args or {}), **declared}


def parse_guidance_interval(value: object) -> tuple[float, float]:
    """Parse ``[lo, hi]`` in scheduler timestep units; raises ValueError unless both are numbers with lo <= hi."""
    message = f"Invalid guidance_interval={value!r}. Expected two numbers [lo, hi]."
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(message)
    try:
        lo, hi = float(value[0]), float(value[1])
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if not lo <= hi:
        raise ValueError(f"Invalid guidance_interval={value!r}. Expected lo <= hi in scheduler timestep units.")
    return lo, hi
