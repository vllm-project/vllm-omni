# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest

from tests.model_tests.diffusion.config_types import (
    ACC_DEVICE_COUNT_KEYS,
    ACC_PARALLEL_KWARGS,
    DiffusionAccs,
    build_parallel_config_from_diff_accelerations,
    get_required_device_count,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize("acceleration", list(ACC_PARALLEL_KWARGS))
def test_parallel_accelerations_declare_device_count(acceleration: DiffusionAccs) -> None:
    assert acceleration in ACC_DEVICE_COUNT_KEYS


def test_pipeline_parallel_composes_with_cfg_within_ci_device_limit() -> None:
    accelerations = [DiffusionAccs.PIPELINE_PARALLEL, DiffusionAccs.CFG_PARALLEL]

    assert get_required_device_count(accelerations) == 4
    config = build_parallel_config_from_diff_accelerations(accelerations)
    assert config is not None
    assert config.pipeline_parallel_size == 2
    assert config.cfg_parallel_size == 2
