# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stage-2 timesteps must reach the model with their fractional part intact.

Both stages build their schedule with ``np.linspace`` over a non-integer range,
so most timesteps are fractional. The stage-2 path used to cast the broadcast
timestep to ``int64``, which truncates it before the sinusoidal embedding and
made stage 2 disagree with stage 1 and with the reference schedule.
"""

import inspect

import pytest
import torch
from diffusers.models.embeddings import Timesteps

import vllm_omni.diffusion.models.helios.pipeline_helios as pipeline_helios
from vllm_omni.diffusion.models.helios.pipeline_helios import expand_timestep

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_fractional_values_survive_the_broadcast():
    timestep = torch.tensor([999.0, 950.5, 12.25])
    broadcast = expand_timestep(timestep, 3)
    assert broadcast.dtype == torch.float32
    assert broadcast.tolist() == [999.0, 950.5, 12.25]


def test_truncation_changes_the_timestep_embedding():
    project = Timesteps(num_channels=8, flip_sin_to_cos=True, downscale_freq_shift=0)
    timestep = torch.tensor([950.5, 12.25])
    assert not torch.equal(project(timestep), project(timestep.to(torch.int64)))


def test_no_stage_path_truncates_the_timestep():
    source = inspect.getsource(pipeline_helios)
    assert "t.expand(batch_size).to(torch.int64)" not in source
    assert "t.expand(latents.shape[0]).to(torch.int64)" not in source
    assert source.count("timestep = expand_timestep(") == 4
