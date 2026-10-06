# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math

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


def glm_image_ar_max_tokens(height: int, width: int, *, is_i2i: bool = False, factor: int = 32) -> int | None:
    """Return the AR-stage ``max_tokens`` budget for GLM-Image online serving.

    GLM-Image AR emits a small preview grid (t2i only), a large target grid,
    and EOS. The deploy YAML ceiling (4353 for 2048x2048) is a hard upper
    bound; this helper returns the resolution-specific budget so Stage 0 stops
    near EOS instead of decoding past the codebook into special tokens (>=16384)
    that crash DiT ``prior_token_embedding``.

    Shared by the online serving path and offline examples so a grid-contract
    change cannot desynchronize one copy. Returns ``None`` when dimensions are
    absent or non-positive, letting callers keep their existing default.
    """
    if height <= 0 or width <= 0:
        return None

    token_h = height // factor
    token_w = width // factor
    large_tokens = token_h * token_w

    if is_i2i:
        return large_tokens + 1

    ratio = token_h / token_w if token_w > 0 else 1.0
    small_token_h = max(1, int(math.sqrt(ratio) * (factor // 2)))
    small_token_w = max(1, int(math.sqrt(1 / ratio) * (factor // 2)))
    small_tokens = small_token_h * small_token_w
    return small_tokens + large_tokens + 1
