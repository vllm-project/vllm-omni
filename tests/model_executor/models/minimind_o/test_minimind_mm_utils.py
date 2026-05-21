# SPDX-License-Identifier: Apache-2.0
import torch

from vllm_omni.model_executor.models.minimind_o.minimind_mm_utils import (
    inject_audio_features,
    inject_vision_features,
)


def test_inject_audio_replaces_pad_run():
    tokens = torch.tensor([[1, 16, 16, 2]])
    hidden = torch.arange(4 * 3, dtype=torch.float32).view(1, 4, 3)
    audio = [torch.ones(2, 3)]
    out = inject_audio_features(tokens, hidden, audio, audio_marker=16)
    assert out.shape[0] == 1
    assert out.shape[1] >= 4


def test_inject_vision_replaces_image_pad():
    tokens = torch.tensor([[1, 12, 12, 2]])
    hidden = torch.zeros(1, 4, 4)
    vision = torch.ones(1, 1, 2, 4)
    out = inject_vision_features(tokens, hidden, vision, image_marker=12, seqlen=4)
    assert out.shape == hidden.shape
