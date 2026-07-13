# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from vllm_omni.diffusion.models.hidream_o1_image.pipeline_hidream_o1_image import (
    PATCH_SIZE,
    get_hidream_o1_image_post_process_func,
)


def main() -> None:
    # The current factory does not consume od_config.
    postprocess = get_hidream_o1_image_post_process_func(None)

    # Case 1: shape / type / mode
    # H=64, W=96, PATCH_SIZE=32
    # -> 2 * 3 = 6 patch tokens
    # -> patch_dim = 3 * 32 * 32 = 3072
    z = torch.zeros(
        (1, 6, 3 * PATCH_SIZE * PATCH_SIZE),
        dtype=torch.bfloat16,
    )

    img = postprocess((z, 64, 96))

    assert isinstance(img, Image.Image), type(img)
    assert img.mode == "RGB", img.mode
    assert img.size == (96, 64), img.size  # PIL uses (W, H)

    # z=0 -> (z+1)/2 = 0.5
    # 0.5*255 = 127.5 -> np.round(...) = 128
    arr = np.asarray(img)

    assert arr.shape == (64, 96, 3), arr.shape
    assert arr.dtype == np.uint8, arr.dtype
    assert np.all(arr == 128), (
        f"expected all pixels=128, "
        f"min={arr.min()} max={arr.max()}"
    )

    print(
        "case 1 shape/type       : "
        f"type={type(img).__name__} "
        f"mode={img.mode} "
        f"size={img.size} "
        f"pixel={arr[0, 0].tolist()}"
    )

    # Case 2: clipping
    # z=-2 -> (-2+1)/2=-0.5 -> clip -> 0
    # z=+2 -> (+2+1)/2=1.5 -> clip -> 255
    z_lo = torch.full(
        (1, 1, 3 * PATCH_SIZE * PATCH_SIZE),
        -2.0,
        dtype=torch.float32,
    )
    img_lo = postprocess((z_lo, 32, 32))
    arr_lo = np.asarray(img_lo)
    assert np.all(arr_lo == 0), (
        f"low clip failed: min={arr_lo.min()} max={arr_lo.max()}"
    )

    z_hi = torch.full(
        (1, 1, 3 * PATCH_SIZE * PATCH_SIZE),
        2.0,
        dtype=torch.float32,
    )
    img_hi = postprocess((z_hi, 32, 32))
    arr_hi = np.asarray(img_hi)
    assert np.all(arr_hi == 255), (
        f"high clip failed: min={arr_hi.min()} max={arr_hi.max()}"
    )

    print(
        "case 2 clipping         : "
        f"low={int(arr_lo[0, 0, 0])} "
        f"high={int(arr_hi[0, 0, 0])}"
    )

    # Case 3: patch placement
    # H=32, W=64 => two horizontal patches.
    # patch 0 = -1 -> black
    # patch 1 = +1 -> white
    patch_dim = 3 * PATCH_SIZE * PATCH_SIZE

    z_layout = torch.empty(
        (1, 2, patch_dim),
        dtype=torch.float32,
    )
    z_layout[:, 0, :] = -1.0
    z_layout[:, 1, :] = 1.0

    img_layout = postprocess((z_layout, 32, 64))
    arr_layout = np.asarray(img_layout)

    assert img_layout.size == (64, 32), img_layout.size

    left = arr_layout[:, :32, :]
    right = arr_layout[:, 32:, :]

    assert np.all(left == 0), (
        f"left patch expected black: "
        f"min={left.min()} max={left.max()}"
    )
    assert np.all(right == 255), (
        f"right patch expected white: "
        f"min={right.min()} max={right.max()}"
    )

    print(
        "case 3 patch placement  : "
        f"left={left[0, 0].tolist()} "
        f"right={right[0, 0].tolist()}"
    )

    print("pass")

if __name__ == "__main__":
    main()

# output:
# case 1 shape/type       : type=Image mode=RGB size=(96, 64) pixel=[128, 128, 128]
# case 2 clipping         : low=0 high=255
# case 3 patch placement  : left=[0, 0, 0] right=[255, 255, 255]
# pass
