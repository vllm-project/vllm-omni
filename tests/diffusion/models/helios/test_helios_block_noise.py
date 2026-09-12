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


def _pipeline_for_block_noise(device: torch.device) -> _BlockNoisePipeline:
    return _BlockNoisePipeline(
        scheduler=_BlockNoiseScheduler(config=_BlockNoiseSchedulerConfig(gamma=1.0 / 3.0)),
        device=device,
    )


@pytest.mark.core_model
@pytest.mark.cpu
@pytest.mark.diffusion
def test_sample_block_noise_default_gamma_cpu() -> None:
    noise = HeliosPipeline.sample_block_noise(
        _pipeline_for_block_noise(torch.device("cpu")),
        batch_size=1,
        channel=1,
        num_frames=1,
        height=2,
        width=2,
    )

    assert noise.shape == (1, 1, 1, 2, 2)
    assert torch.isfinite(noise).all()


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
