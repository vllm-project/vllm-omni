# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_omni.diffusion.models.mova.mova_audio_transformer import MovaAudioTransformer
from vllm_omni.diffusion.models.mova.mova_video_transformer import MovaVideoTransformer
from vllm_omni.diffusion.models.mova.scheduling_mova import FlowMatchPairScheduler

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_pair_scheduler_uses_active_pair_sigmas_for_step_from_to() -> None:
    scheduler = FlowMatchPairScheduler(num_inference_steps=4, num_train_timesteps=1000, shift=3.0)
    scheduler.set_timesteps(4)
    scheduler.set_pair_postprocess_by_name("dual_sigma_shift", visual_shift=5.0, audio_shift=2.0)

    pair_timesteps = scheduler.get_pairs("timesteps")
    pair_sigmas = scheduler.get_pairs("sigmas")
    model_output = torch.ones(1)
    sample = torch.zeros(1)

    visual_step = scheduler.step_from_to(model_output, pair_timesteps[0, 0], pair_timesteps[1, 0], sample)
    audio_step = scheduler.step_from_to(model_output, pair_timesteps[0, 1], pair_timesteps[1, 1], sample)

    torch.testing.assert_close(visual_step, sample + (pair_sigmas[1, 0] - pair_sigmas[0, 0]))
    torch.testing.assert_close(audio_step, sample + (pair_sigmas[1, 1] - pair_sigmas[0, 1]))


@pytest.mark.parametrize("model_cls, patch_size", [(MovaVideoTransformer, (1, 2, 2)), (MovaAudioTransformer, (1,))])
def test_mova_transformers_accept_legacy_separated_timestep_key(model_cls, patch_size) -> None:
    legacy_timestep_key = "se" + "perated_timestep"
    model = model_cls(
        dim=12,
        in_dim=4,
        ffn_dim=24,
        out_dim=4,
        text_dim=8,
        freq_dim=6,
        eps=1e-6,
        patch_size=patch_size,
        num_heads=3,
        num_layers=1,
        has_image_input=False,
        **{legacy_timestep_key: True},
    )

    assert model.separated_timestep is True
    assert getattr(model, legacy_timestep_key) is True
