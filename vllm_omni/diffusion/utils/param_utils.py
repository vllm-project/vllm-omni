# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
