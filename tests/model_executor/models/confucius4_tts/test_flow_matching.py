# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.model_executor.models.confucius4_tts.configuration_confucius4_tts import (
    S2A_MAX_POSITIONS,
    Confucius4TTSConfig,
)
from vllm_omni.model_executor.models.confucius4_tts.flow.flow_matching import ConditionalCFM

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

MEL = 8


def _cfm(max_seq_len: int = 64) -> ConditionalCFM:
    torch.manual_seed(0)
    cfm = ConditionalCFM(
        hidden_dim=16,
        num_heads=2,
        depth=2,
        mel_dim=MEL,
        cond_dim=16,
        style_dim=4,
        final_layer="mlp",
        max_seq_len=max_seq_len,
    )
    return cfm.eval()


def _inputs(batch: int = 1, frames: int = 12, prompt_frames: int = 3):
    mu = torch.randn(batch, frames, 16)
    prompt = torch.randn(batch, MEL, prompt_frames)
    spks = torch.randn(batch, 4)
    x_lens = torch.full((batch,), frames, dtype=torch.long)
    return mu, x_lens, prompt, spks


@pytest.mark.parametrize("cfg_rate", [0.0, 0.7])
@pytest.mark.parametrize("batch", [1, 2])
def test_inference_runs_with_and_without_classifier_free_guidance(cfg_rate: float, batch: int) -> None:
    # cfg_rate == 0 skips the guided branch and used to hand the estimator a scalar timestep.
    mu, x_lens, prompt, spks = _inputs(batch=batch)

    with torch.no_grad():
        mel = _cfm().forward(mu, x_lens, prompt, spks, n_timesteps=2, inference_cfg_rate=cfg_rate)

    assert mel.shape == (batch, MEL, 12)
    assert torch.isfinite(mel).all()


def test_max_seq_len_sets_the_rotary_table() -> None:
    assert _cfm(max_seq_len=64).estimator.freqs_cis.shape[0] == 64
    # The decoder passes the config limit, so the table and the Stage 1 guard share one number.
    assert Confucius4TTSConfig().s2a_max_positions == S2A_MAX_POSITIONS == 4096
