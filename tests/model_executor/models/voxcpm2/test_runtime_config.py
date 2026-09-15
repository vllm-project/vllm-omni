# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from dataclasses import dataclass

import pytest
import torch

from vllm_omni.model_executor.models.voxcpm2.runtime_config import _VoxCPM2RuntimeConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@dataclass
class _ModelConfig:
    voxcpm2_runtime_config: dict[str, object]


@dataclass
class _VllmConfig:
    model_config: _ModelConfig


@pytest.mark.parametrize(
    ("key", "value", "expected"),
    [
        ("inference_timesteps", 4, 4),
        ("inference_timesteps", "6", 6),
        ("inference_timesteps", -3, 2),
        ("inference_timesteps", 0, 2),
        ("inference_timesteps", 1, 2),
        ("inference_timesteps", 2, 2),
        ("cfg_value", "1.5", 1.5),
        ("cfg_cutoff_ratio", "0.5", 0.5),
        ("cfg_cutoff_ratio", -1.0, 0.0),
        ("cfg_cutoff_ratio", 2.0, 1.0),
    ],
)
def test_generation_config(key: str, value: object, expected: int | float) -> None:
    vllm_config = _VllmConfig(_ModelConfig({key: value}))

    config = _VoxCPM2RuntimeConfig.from_vllm_config(vllm_config)

    assert getattr(config, key) == expected


def test_generation_config_defaults() -> None:
    config = _VoxCPM2RuntimeConfig()

    assert config.inference_timesteps == 10
    assert config.cfg_value == 2.0
    assert config.cfg_cutoff_ratio == 1.0


class _CountingEstimator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, x, mu, t, cond, dt):
        self.calls += 1
        return mu[:, :1, None].expand_as(x)


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_cfg_value_rejects_nonfinite(value: str) -> None:
    with pytest.raises(ValueError, match="cfg_value must be finite"):
        _VoxCPM2RuntimeConfig.from_vllm_config(_VllmConfig(_ModelConfig({"cfg_value": value})))


@pytest.mark.parametrize("timesteps", [-3, 0, 1, 2, 4, 10])
@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("cfg_value", [1.5, 2.0])
@pytest.mark.parametrize("cutoff", [0.0, 1.0])
def test_generation_config_runs_estimator(timesteps: int, batch_size: int, cfg_value: float, cutoff: float) -> None:
    from vllm_omni.model_executor.models.voxcpm2.voxcpm2_talker import (
        _CFMBufferManager,
        _optimized_solve_euler_with_noise,
    )

    config = _VoxCPM2RuntimeConfig.from_vllm_config(
        _VllmConfig(
            _ModelConfig({"inference_timesteps": timesteps, "cfg_value": cfg_value, "cfg_cutoff_ratio": cutoff})
        )
    )
    cfm = torch.nn.Module()
    cfm.estimator = _CountingEstimator()
    buffers = _CFMBufferManager(
        device=torch.device("cpu"),
        dtype=torch.float32,
        feat_dim=2,
        patch_size=2,
        dit_hidden_size=2,
        max_batch_size=batch_size,
    )
    noise = torch.ones(batch_size, 2, 2)
    output = _optimized_solve_euler_with_noise(
        cfm,
        mu=torch.ones(batch_size, 2),
        patch_size=2,
        cond=torch.zeros_like(noise),
        noise=noise,
        n_timesteps=config.inference_timesteps,
        cfg_value=config.cfg_value,
        buffers=buffers,
        cfg_cutoff_ratio=config.cfg_cutoff_ratio,
    )

    assert cfm.estimator.calls == max(2, timesteps) - 1
    assert torch.isfinite(output).all()
    assert not torch.equal(output, noise)
    torch.testing.assert_close(noise, torch.ones_like(noise))
    if cutoff == 1.0:
        t_span = buffers.get_t_span(config.inference_timesteps)
        expected = noise - cfg_value * (t_span[1] - t_span[-1])
        torch.testing.assert_close(output, expected)
