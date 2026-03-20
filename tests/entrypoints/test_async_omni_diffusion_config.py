# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm_omni.engine.async_omni_engine import AsyncOmniEngine

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_default_stage_config_includes_cache_backend():
    """Ensure cache knobs survive the default diffusion-stage builder."""
    stage_cfg = AsyncOmniEngine._create_default_diffusion_stage_cfg(
        {
            "cache_backend": "cache_dit",
            "cache_config": '{"Fn_compute_blocks": 2}',
            "vae_use_slicing": True,
            "ulysses_degree": 2,
        }
    )[0]

    engine_args = stage_cfg["engine_args"]
    assert stage_cfg["stage_type"] == "diffusion"
    assert engine_args["cache_backend"] == "cache_dit"
    assert engine_args["cache_config"]["Fn_compute_blocks"] == 2
    assert engine_args["vae_use_slicing"] is True
    assert engine_args["parallel_config"].ulysses_degree == 2
    assert engine_args["model_stage"] == "diffusion"


def test_default_cache_config_used_when_missing():
    """Ensure default cache_config is synthesized when only backend is given."""
    stage_cfg = AsyncOmniEngine._create_default_diffusion_stage_cfg(
        {
            "cache_backend": "cache_dit",
        }
    )[0]

    cache_config = stage_cfg["engine_args"]["cache_config"]
    assert cache_config is not None
    assert cache_config["Fn_compute_blocks"] == 1


def test_default_stage_devices_from_sequence_parallel():
    """Ensure runtime devices reflect computed diffusion world size."""
    stage_cfg = AsyncOmniEngine._create_default_diffusion_stage_cfg(
        {
            "ulysses_degree": 2,
            "ring_degree": 2,
        }
    )[0]

    assert stage_cfg["runtime"]["devices"] == "0,1,2,3"
