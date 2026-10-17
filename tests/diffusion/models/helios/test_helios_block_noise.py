# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from dataclasses import dataclass

import pytest
import torch

from vllm_omni.diffusion.models.helios.pipeline_helios import HeliosPipeline
from vllm_omni.platforms import current_omni_platform


@dataclass
class _BlockNoiseSchedulerConfig:
    gamma: float


@dataclass
class _BlockNoiseScheduler:
    config: _BlockNoiseSchedulerConfig


@dataclass
class _BlockNoisePipeline:
    scheduler: _BlockNoiseScheduler
    device: torch.device


def _pipeline_for_block_noise(device: torch.device, gamma: float = 1.0 / 3.0) -> _BlockNoisePipeline:
    return _BlockNoisePipeline(
        scheduler=_BlockNoiseScheduler(config=_BlockNoiseSchedulerConfig(gamma=gamma)),
        device=device,
    )


@pytest.mark.core_model
@pytest.mark.cpu
@pytest.mark.diffusion
@pytest.mark.parametrize("gamma", [1.0 / 3.0, 0.25])
def test_sample_block_noise_cpu(gamma: float) -> None:
    noise = HeliosPipeline.sample_block_noise(
        _pipeline_for_block_noise(torch.device("cpu"), gamma),
        batch_size=1,
        channel=1,
        num_frames=1,
        height=2,
        width=2,
    )

    assert noise.shape == (1, 1, 1, 2, 2)
    assert torch.isfinite(noise).all()


@pytest.mark.core_model
@pytest.mark.cpu
@pytest.mark.diffusion
def test_block_noise_covariance_has_float32_margin_cpu(monkeypatch) -> None:
    # Regression test for #6057. gamma=1/3 makes the covariance singular in the
    # all-ones direction, so the diagonal jitter is the only thing keeping
    # Cholesky alive -- it has to be larger than one float32 ULP around the
    # diagonal (1.19e-7) or it is rounded away and the factorisation is left to
    # the rounding luck of whichever potrf implementation runs it.
    captured: dict[str, torch.Tensor] = {}
    cholesky = torch.linalg.cholesky

    def spy(matrix: torch.Tensor) -> torch.Tensor:
        captured["cov"] = matrix.detach()
        return cholesky(matrix)

    monkeypatch.setattr(torch.linalg, "cholesky", spy)

    HeliosPipeline.sample_block_noise(
        _pipeline_for_block_noise(torch.device("cpu")),
        batch_size=1,
        channel=1,
        num_frames=1,
        height=2,
        width=2,
    )

    cov = captured["cov"]
    assert cov.dtype == torch.float32
    min_eig = torch.linalg.eigvalsh(cov.double()).min().item()
    assert min_eig > torch.finfo(torch.float32).eps, f"covariance has no float32 margin: {min_eig:.3e}"


@pytest.mark.core_model
@pytest.mark.cpu
@pytest.mark.diffusion
def test_sample_block_noise_matches_covariance_cpu() -> None:
    # Regression test for #6057. With gamma=1/3 the covariance is singular in
    # the all-ones direction, so Cholesky only succeeds while the diagonal
    # jitter survives float32 rounding -- and it must stay small enough that
    # the sampled statistics are unchanged.
    gamma = 1.0 / 3.0
    samples = 65536
    generator = torch.Generator().manual_seed(0)
    noise = HeliosPipeline.sample_block_noise(
        _pipeline_for_block_noise(torch.device("cpu"), gamma),
        batch_size=samples,
        channel=1,
        num_frames=1,
        height=2,
        width=2,
        generator=generator,
    )

    flat = noise.reshape(samples, -1)
    # Every entry has marginal variance (1 + gamma) - gamma + jitter == 1 + jitter, whichever block it is in.
    assert flat.var(dim=0).mean().item() == pytest.approx(1.0, abs=3e-2)
    # The all-ones direction stays degenerate: only the jitter leaks into the block mean.
    assert flat.mean(dim=1).var().item() < 1e-4


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.npu
@pytest.mark.skipif(not current_omni_platform.is_npu(), reason="requires an Ascend NPU")
def test_sample_block_noise_default_gamma_npu() -> None:
    generator = torch.Generator(device="npu").manual_seed(0)
    noise = HeliosPipeline.sample_block_noise(
        _pipeline_for_block_noise(torch.device("cpu")),
        batch_size=1,
        channel=1,
        num_frames=1,
        height=2,
        width=2,
        generator=generator,
    )
    torch.npu.synchronize()

    assert noise.device.type == "npu"
    assert noise.shape == (1, 1, 1, 2, 2)
    assert torch.isfinite(noise).all()
