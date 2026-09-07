# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from dataclasses import dataclass

import pytest

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
        ("inference_timesteps", 0, 1),
        ("cfg_value", "1.5", 1.5),
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
