# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.mossimage.mossimage_transformer import MOSSImageTransformer

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@pytest.mark.parametrize(
    "device",
    [
        pytest.param("cpu", marks=pytest.mark.cpu),
        pytest.param("mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")),
    ],
)
@torch.inference_mode()
def test_timestep_embedding_matches_float64_reference(device):
    cpu_times = torch.tensor([1000.0, 976.5, 24.0, 0.001, 0.0], device="cpu", dtype=torch.float32)
    timesteps = cpu_times.to(device)
    freq_dim = 256
    dim = 8
    layers = SimpleNamespace(
        freq_dim=freq_dim,
        time_embedding=nn.Sequential(nn.Linear(freq_dim, dim, device=device)),
        shared_block_modulation=nn.Linear(dim, 6 * dim, device=device),
        video_head_modulation=nn.Linear(dim, 2 * dim, device=device),
    )
    half = freq_dim // 2
    frequencies = 10000 ** (-torch.arange(half, device="cpu", dtype=torch.float64) / half)
    phase = torch.outer(cpu_times.double(), frequencies)
    expected = torch.cat((phase.cos(), phase.sin()), dim=-1).float()

    mlp_inputs = []
    with layers.time_embedding.register_forward_pre_hook(lambda _, args: mlp_inputs.append(args[0])):
        block_mod, head_mod = MOSSImageTransformer._time_modulation(layers, timesteps, len(timesteps), timesteps.device)

    embedding = torch.cat(mlp_inputs, dim=0)
    assert embedding.device == block_mod.device == head_mod.device == timesteps.device
    torch.testing.assert_close(embedding.cpu(), expected, atol=0, rtol=0)
    assert block_mod.shape == (len(timesteps), 6, dim)
    assert head_mod.shape == (len(timesteps), 2, dim)
