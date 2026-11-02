# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import pytest
import torch
import torch.nn as nn

from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3AudioVAE, _AudioVAEDeterminismContext

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _graph_executor_optimize() -> bool:
    return torch._C._get_graph_executor_optimize()


class _RecordingDecoder:
    def __init__(self) -> None:
        self.optimize_during_decode: list[bool] = []

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        self.optimize_during_decode.append(_graph_executor_optimize())
        return torch.zeros(latent.shape[0], 1, 8)


def test_audio_vae_decode_runs_torchscript_without_executor_optimization():
    vae = MiniMaxH3AudioVAE.__new__(MiniMaxH3AudioVAE)
    nn.Module.__init__(vae)
    vae.config_dict = {"latent_channels": 2, "latents_mean": [0.0, 0.0], "latents_std": [1.0, 1.0]}
    vae.remote = _RecordingDecoder()
    before = _graph_executor_optimize()

    for _ in range(2):
        waveform = vae.decode_latent(torch.zeros(2, 2, 4))

    assert vae.remote.optimize_during_decode == [False, False]
    assert _graph_executor_optimize() == before
    assert tuple(waveform.shape) == (1, 2, 8)


def test_audio_vae_determinism_context_disables_and_restores_executor_optimization():
    before = _graph_executor_optimize()

    with _AudioVAEDeterminismContext():
        assert _graph_executor_optimize() is False

    assert _graph_executor_optimize() == before
