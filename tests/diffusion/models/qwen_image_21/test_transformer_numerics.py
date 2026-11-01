# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.qwen_image_21 import qwen_image_21_transformer as transformer

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@pytest.mark.parametrize(
    "device",
    [pytest.param("cpu", marks=pytest.mark.cpu), pytest.param("cuda", marks=[pytest.mark.cuda, pytest.mark.gpu])],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_qk_norm_matches_diffusers_rounding(monkeypatch, device, dtype):
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_tp_group", lambda: SimpleNamespace(rank_in_group=0, world_size=1)
    )
    monkeypatch.setattr(transformer, "Attention", lambda **kwargs: torch.nn.Identity())
    attention = transformer.QwenImage21Attention(dim=256, heads=2, dim_head=128).to(device=device, dtype=dtype)
    values = torch.randn(2, 17, 2, 128, generator=torch.Generator().manual_seed(42)).to(device=device, dtype=dtype)
    for norm in (attention.norm_q, attention.norm_k):
        with torch.no_grad():
            norm.weight.copy_(torch.linspace(0.5, 1.5, 128, device=device, dtype=dtype))
        variance = values.float().square().mean(-1, keepdim=True)
        expected = (values * torch.rsqrt(variance + norm.eps)).to(dtype) * norm.weight
        torch.testing.assert_close(norm(values), expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "device",
    [pytest.param("meta", marks=pytest.mark.cpu), pytest.param("cuda", marks=[pytest.mark.cuda, pytest.mark.gpu])],
)
def test_rope_frequencies_do_not_depend_on_loader_device(device):
    reference = transformer.QwenImage21Rope(theta=10000, axes_dim=[16, 56, 56])
    with torch.device(device):
        loaded = transformer.QwenImage21Rope(theta=10000, axes_dim=[16, 56, 56])
    for expected, actual in zip(reference.freqs, loaded.freqs):
        assert actual.device.type == "cpu"
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.gpu
def test_scheduler_matches_diffusers_on_latent_device():
    import numpy as np
    from diffusers import FlowMatchEulerDiscreteScheduler

    from vllm_omni.diffusion.models.qwen_image_21.pipeline_qwen_image_21 import QwenImage21Pipeline

    pipeline = object.__new__(QwenImage21Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cuda", 0)
    pipeline.scheduler = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True)
    timesteps, _ = pipeline.prepare_timesteps(50, None, 4096)
    reference = FlowMatchEulerDiscreteScheduler.from_config(pipeline.scheduler.config)
    reference.set_timesteps(sigmas=np.linspace(1.0, 1 / 50, 50), device=pipeline.device, mu=1.15)
    sample = torch.randn(1, 16, 64, generator=torch.Generator().manual_seed(42)).to(
        device=pipeline.device, dtype=torch.bfloat16
    )
    noise = sample.cos()
    actual = pipeline.scheduler.step(noise, timesteps[0], sample, return_dict=False)[0]
    expected = reference.step(noise, reference.timesteps[0], sample, return_dict=False)[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert pipeline.scheduler.sigmas.device == sample.device
    assert timesteps.device == sample.device


@pytest.mark.cuda
@pytest.mark.gpu
def test_timestep_frequencies_match_cpu_initialization():
    reference = transformer.QwenImage21TemporalTimesteps(timestep_dim=256)
    with torch.device("cuda"):
        loaded = transformer.QwenImage21TemporalTimesteps(timestep_dim=256)
    assert loaded.freqs.device.type == "cuda"
    torch.testing.assert_close(loaded.freqs.cpu(), reference.freqs, rtol=0, atol=0)
    timesteps = torch.tensor([0.0, 0.984375, 1.0], device="cuda")
    torch.testing.assert_close(loaded(timesteps), reference.to("cuda")(timesteps), rtol=0, atol=0)
