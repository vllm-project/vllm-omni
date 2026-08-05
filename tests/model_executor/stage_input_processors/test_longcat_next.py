# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm_omni.model_executor.models.longcat_next.longcat_next_utils import (
    IMG_END_TOKEN_ID,
    IMG_NEWLINE_TOKEN_ID,
    IMG_PAD_TOKEN_ID,
    IMG_START_TOKEN_ID,
    infer_visual_grid,
)


def test_infer_visual_grid_from_newlines():
    # One IMG_PAD placeholder per real-pixel step, not NUM_CODEBOOKS ids --
    # the visible stream never carries the real per-level codes (see
    # infer_visual_grid's docstring).
    stream = [IMG_START_TOKEN_ID]
    for i in range(6):
        stream.append(IMG_PAD_TOKEN_ID)
        if i % 3 == 2:  # newline after every 3 positions -> w=3
            stream.append(IMG_NEWLINE_TOKEN_ID)
    stream.append(IMG_END_TOKEN_ID)
    assert infer_visual_grid(stream) == (2, 3)
