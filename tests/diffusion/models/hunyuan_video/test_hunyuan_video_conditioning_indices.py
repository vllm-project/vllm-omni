# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for HunyuanVideo-1.5 conditioning indices and token order."""

import pytest
import torch

from vllm_omni.diffusion.models.hunyuan_video.hunyuan_video_15_transformer import (
    _gather_valid_encoder_hidden_states,
)
from vllm_omni.diffusion.models.hunyuan_video.pipeline_hunyuan_video_1_5 import (
    _get_encoder_hidden_states_indices,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_conditioning_indices_preserve_token_order():
    streams = torch.arange(24, dtype=torch.float32).reshape(3, 1, 4, 2)
    masks = torch.tensor([[True, False, True, False], [False, True, True, False], [True, True, False, True]])[:, None]
    indices = _get_encoder_hidden_states_indices(*masks)
    assert indices is not None
    expected = torch.cat([stream[:, mask[0]] for stream, mask in zip(streams, masks)], dim=1)
    assert torch.equal(_gather_valid_encoder_hidden_states(*streams, indices), expected)
    assert _get_encoder_hidden_states_indices(*masks.repeat(1, 2, 1)) is None
