# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.sana_video2.pipeline_sana_video2 import SanaVideo2Pipeline, validate_parallel_config

pytestmark = [pytest.mark.cpu, pytest.mark.diffusion, pytest.mark.core_model]


def _config(tp=1, sp=1, cfg=1, mode="strict"):
    parallel = SimpleNamespace(
        tensor_parallel_size=tp,
        sequence_parallel_size=sp,
        ulysses_degree=sp,
        ulysses_mode=mode,
        ring_degree=1,
        allgather_degree=1,
        cfg_parallel_size=cfg,
        pipeline_parallel_size=1,
        text_encoder_tp_size=1,
        vae_patch_parallel_size=1,
        use_hsdp=False,
    )
    return SimpleNamespace(parallel_config=parallel, cache_backend=None)


@pytest.mark.parametrize("tp,sp,cfg,mode", [(1, 1, 2, "strict"), (2, 1, 1, "strict"), (2, 2, 2, "advanced_uaa")])
def test_accepts_supported_tp_cfg_topologies(tp, sp, cfg, mode):
    validate_parallel_config(_config(tp, sp, cfg, mode))


def test_rejects_strict_sp_using_tp_sharded_anchor_heads():
    with pytest.raises(ValueError, match="advanced_uaa"):
        validate_parallel_config(_config(tp=2, sp=2))


class _Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))

    def forward(self, x, timestep, embeddings, mask):
        value = embeddings[:, 0, 0].to(torch.bfloat16).reshape(-1, 1, 1, 1, 1)
        return value.expand_as(x)


def _pipeline():
    vae = nn.Module()
    vae.config = SimpleNamespace(latent_channels=128, temporal_compression_ratio=8, spatial_compression_ratio=32)
    return SanaVideo2Pipeline(tokenizer=object(), text_encoder=nn.Identity(), vae=vae, transformer=_Transformer())


@pytest.mark.parametrize("noise_space", [False, True])
def test_single_branch_prediction_keeps_ti2v_flow_and_t2v_noise_dtype(noise_space):
    pipeline = _pipeline()
    x = torch.ones(1, 128, 2, 2, 2)
    time = torch.tensor(0.25) if noise_space else torch.tensor([[[[[0.25]], [[0.75]]]]])
    embeddings = torch.full((1, 2, 3), 2.0)
    mask = torch.ones(1, 2, dtype=torch.bool)
    pred = pipeline.predict_noise(x=x, time=time, embeddings=embeddings, mask=mask, noise_space=noise_space)
    expected = torch.full_like(x, 2.5) if noise_space else torch.full_like(x, 2.0, dtype=torch.bfloat16)
    torch.testing.assert_close(pred, expected, rtol=0, atol=0)
    assert pred.dtype == (torch.float32 if noise_space else torch.bfloat16)
