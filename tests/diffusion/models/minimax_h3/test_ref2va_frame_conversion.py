# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import numpy as np
import pytest
import torch
from PIL import Image

from vllm_omni.model_executor.models.minimax_h3.encoder_processing import _frames_to_tensor

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize("reverse_rows", [False, True])
def test_rgb_video_array_matches_framewise_conversion_and_owns_data(reverse_rows):
    frames = np.arange(3 * 5 * 7 * 3, dtype=np.uint8).reshape(3, 5, 7, 3)
    if reverse_rows:
        frames = frames[:, ::-1]

    expected = _frames_to_tensor([Image.fromarray(frame) for frame in frames])
    actual = _frames_to_tensor(frames)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.is_contiguous()
    original_value = actual[0, 0, 0, 0].item()
    frames[0, 0, 0, 0] = (original_value + 1) % 256
    assert actual[0, 0, 0, 0].item() == original_value
