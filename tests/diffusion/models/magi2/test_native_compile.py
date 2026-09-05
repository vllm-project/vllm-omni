# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import pytest
import torch

from vllm_omni.diffusion.models.magi2.sampler_magi2 import CFGConfig, Magi2PreviewSampler

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]


def _sampler_tensors() -> dict[str, torch.Tensor]:
    return {
        "latent": torch.randn(1, 4, 2, 1, 3),
        "audio_latent": torch.randn(1, 5, 4),
        "txt_feat": torch.randn(1, 3, 4),
        "null_txt_feat": torch.randn(1, 2, 4),
    }


def test_prepare_model_input_keeps_lengths_on_host() -> None:
    sampler = Magi2PreviewSampler(torch.nn.Identity())
    model_input = sampler.prepare_model_input(**_sampler_tensors(), t=torch.tensor([500.0]), cfg_config=CFGConfig())

    assert model_input.audio_feat_len == [5, 5]
    assert model_input.txt_feat_len == [3, 2]
    assert model_input.ref_audio_feat_len == [0, 0]
    assert model_input.ref_video_feat_len == [0, 0]

    positive, negative = Magi2PreviewSampler._split_cfg_model_input(model_input)
    assert positive.txt_feat_len == [3]
    assert negative.txt_feat_len == [2]
