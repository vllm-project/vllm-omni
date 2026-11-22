# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Parallel configuration validation for Cosmos3-Nano-Sim-Auto."""

from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.models.cosmos3_nano_sim_auto.config import (
    validate_sim_auto_parallel_config,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_parallel_config_rejects_sp_and_cfg_parallel() -> None:
    ok = SimpleNamespace(parallel_config=SimpleNamespace(sequence_parallel_size=1, cfg_parallel_size=1))
    validate_sim_auto_parallel_config(ok)
    bad = SimpleNamespace(parallel_config=SimpleNamespace(sequence_parallel_size=2))
    with pytest.raises(ValueError, match="sequence_parallel_size>1"):
        validate_sim_auto_parallel_config(bad)
    bad = SimpleNamespace(parallel_config=SimpleNamespace(cfg_parallel_size=2))
    with pytest.raises(ValueError, match="cfg_parallel_size>1"):
        validate_sim_auto_parallel_config(bad)
