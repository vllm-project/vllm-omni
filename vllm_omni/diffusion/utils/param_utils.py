# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import logging

import torch

from vllm_omni.inputs.data import OmniDiffusionSamplingParams

logger = logging.getLogger(__name__)


def unwrap_request_generator(
    generator: torch.Generator | list[torch.Generator] | None,
    context: str = "request",
) -> torch.Generator | None:
    """Normalize a request-scoped generator slot to a single generator.

    ``OmniDiffusionSamplingParams.generator`` accepts a single generator or a
    list of them, but a per-request slot must resolve to exactly one
    generator: passing the raw list through (e.g. into diffusers
    ``randn_tensor``'s batch path, which reads ``generator[0].device`` and
    indexes entries per batch row) crashes on the nested list.
    """
    if not isinstance(generator, list):
        return generator
    if len(generator) > 1:
        logger.warning("%s carries %d generators; using the first", context, len(generator))
    return generator[0] if generator else None


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


def ar_grid_max_tokens(ar_width: int, ar_height: int) -> int | None:
    """Return the AR-stage ``max_tokens`` for MammothModa2-style visual-token grids.

    The AR stage emits one visual token per grid cell, one EOL token per row,
    and one final look-ahead token, so the generation budget is
    ``ar_height * (ar_width + 1) + 1``. Shared by the online serving path and
    the offline example so a grid-contract change cannot desynchronize one
    copy. Returns ``None`` when the grid is absent or non-positive, letting
    callers keep their existing default.
    """
    if ar_width <= 0 or ar_height <= 0:
        return None
    return ar_height * (ar_width + 1) + 1
