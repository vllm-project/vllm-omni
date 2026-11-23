# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.overrides import TorchFunctionMode

from vllm_omni.diffusion.models.mossimage.mossimage_transformer import MOSSImageTransformer

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

MPS = torch.device("mps")
CPU = torch.device("cpu")


class _MPSOnCPU(TorchFunctionMode):
    """Route MPS allocations and transfers to CPU while enforcing its dtype limit."""

    def __init__(self):
        self.transfers_to_mps = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if kwargs.get("device") == MPS:
            if kwargs.get("dtype") == torch.float64:
                raise TypeError("MPS does not support float64")
            kwargs = {**kwargs, "device": CPU}
        elif func is torch.Tensor.to and len(args) > 1 and args[1] == MPS:
            if args[0].dtype == torch.float64:
                raise TypeError("MPS does not support float64")
            self.transfers_to_mps.append(args[0])
            args = (args[0], CPU, *args[2:])
        return func(*args, **kwargs)


@torch.inference_mode()
def test_mps_timestep_embedding_preserves_float64_cpu_result():
    timesteps = torch.tensor([1000.0, 976.5, 24.0, 0.001, 0.0], dtype=torch.float32)
    freq_dim = 256
    dim = 8
    layers = SimpleNamespace(
        freq_dim=freq_dim,
        time_embedding=nn.Sequential(nn.Linear(freq_dim, dim)),
        shared_block_modulation=nn.Linear(dim, 6 * dim),
        video_head_modulation=nn.Linear(dim, 2 * dim),
    )
    half = freq_dim // 2
    frequencies = 10000 ** (-torch.arange(half, dtype=torch.float64) / half)
    phase = torch.outer(timesteps.double(), frequencies)
    expected = torch.cat((phase.cos(), phase.sin()), dim=-1).float()

    with _MPSOnCPU() as mps:
        block_mod, head_mod = MOSSImageTransformer._time_modulation(layers, timesteps, len(timesteps), MPS)

    (embedding,) = mps.transfers_to_mps
    torch.testing.assert_close(embedding, expected, atol=0, rtol=0)
    assert block_mod.shape == (len(timesteps), 6, dim)
    assert head_mod.shape == (len(timesteps), 2, dim)
