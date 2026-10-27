# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import torch

from vllm_omni.diffusion.models.causvid_wan.pixels import float_bcthw_to_uint8_bthwc


def test_float_bcthw_to_uint8_bthwc_layout_and_range():
    video = torch.zeros(1, 3, 2, 4, 4, dtype=torch.float32)
    video[:, 0] = -1.0
    video[:, 1] = 0.0
    video[:, 2] = 1.0
    out = float_bcthw_to_uint8_bthwc(video)
    assert out.shape == (1, 2, 4, 4, 3)
    assert out.dtype == torch.uint8
    assert out[0, 0, 0, 0, 0].item() == 0
    assert out[0, 0, 0, 0, 1].item() == 128
    assert out[0, 0, 0, 0, 2].item() == 255
