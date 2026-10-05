# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""``expert_parallel_size`` through the engine's flattened kwargs path.

The engine forwards ``expert_parallel_size`` into the diffusion parallel config, so a
head-sharded MoE can use an expert-parallel degree below the sequence-parallel
degree.  Without a value, ``parallel_state`` uses the sequence-parallel degree.
"""

from __future__ import annotations

import pytest

from vllm_omni.engine.omni_engine_base import OmniEngineBase

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]


def _parallel_config(**kwargs) -> dict:
    """Run the default diffusion stage builder and return its parallel config."""
    stage_cfg = OmniEngineBase._create_default_diffusion_stage_cfg(kwargs)
    return stage_cfg[0]["engine_args"]["parallel_config"]


def test_expert_parallel_size_survives_flattened_kwargs():
    config = _parallel_config(enable_expert_parallel=True, expert_parallel_size=4, ulysses_degree=8)
    assert config["expert_parallel_size"] == 4
    assert config["enable_expert_parallel"] is True
    assert config["sequence_parallel_size"] == 8


def test_expert_parallel_size_absent_stays_none():
    config = _parallel_config(enable_expert_parallel=True, ulysses_degree=8)
    assert config["expert_parallel_size"] is None


def test_dict_form_carries_expert_parallel_size():
    config = _parallel_config(
        parallel_config={"enable_expert_parallel": True, "expert_parallel_size": 4, "ulysses_degree": 8},
    )
    assert config["expert_parallel_size"] == 4


def test_invalid_expert_parallel_size_is_rejected():
    with pytest.raises(Exception):
        _parallel_config(enable_expert_parallel=True, expert_parallel_size=0)
