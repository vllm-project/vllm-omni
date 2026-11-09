# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from typing import NamedTuple


class GridTokenLayout(NamedTuple):
    """Token-grid geometry for one GLM-Image target size at factor=32.

    ``token_h``/``token_w`` are the large grid's token dimensions and
    ``small_token_h``/``small_token_w`` the preview grid's; ``small_tokens``
    and ``large_tokens`` are the corresponding cell counts.
    """

    small_tokens: int
    large_tokens: int
    token_h: int
    token_w: int
    small_token_h: int
    small_token_w: int


def grid_token_layout(height: int, width: int, factor: int = 32) -> GridTokenLayout:
    """Return the preview + target grid geometry for a pixel target size.

    GLM-Image emits a small preview grid plus a large target grid at ``factor``
    downsampling; the preview keeps the target's aspect ratio at half the
    linear resolution (16x16 for any square target). Shared by the AR
    generation budget, the ar2diffusion parser, and the model's processor and
    M-RoPE grid construction so none of the copies can drift.
    """
    token_h = height // factor
    token_w = width // factor
    large_tokens = token_h * token_w

    ratio = token_h / token_w if token_w > 0 else 1.0
    small_token_h = max(1, int(math.sqrt(ratio) * (factor // 2)))
    small_token_w = max(1, int(math.sqrt(1 / ratio) * (factor // 2)))
    small_tokens = small_token_h * small_token_w
    return GridTokenLayout(small_tokens, large_tokens, token_h, token_w, small_token_h, small_token_w)


def propagate_target_size(prompt, sampling_params_list):
    if not isinstance(prompt, dict):
        return prompt
    dimensions = prompt.get("mm_processor_kwargs") or {}
    # Mirror the processor's _build_generation_grids resolution: the
    # target_h/target_w pair is used only when both are ints, and otherwise
    # the height/width pair (each defaulting to 1024) resolves both
    # dimensions. Resolving one dimension from each pair would give the
    # serving budget and the decode positions different grids.
    target_h = dimensions.get("target_h") if isinstance(dimensions.get("target_h"), int) else None
    target_w = dimensions.get("target_w") if isinstance(dimensions.get("target_w"), int) else None
    if target_h is None or target_w is None:
        target_h = dimensions.get("height") if isinstance(dimensions.get("height"), int) else 1024
        target_w = dimensions.get("width") if isinstance(dimensions.get("width"), int) else 1024
    params = sampling_params_list[0].clone()
    params.extra_args = {**(params.extra_args or {}), "target_h": target_h, "target_w": target_w}
    sampling_params_list[0] = params
    return prompt
