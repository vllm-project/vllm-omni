# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_omni.config.stage_config import deploy_override_field_names
from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
from vllm_omni.entrypoints.cli.serve import OmniServeCommand, _create_default_diffusion_stage_cfg

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


def test_default_stage_config_ignores_none_deploy_overrides():
    """Ensure nullified deploy override defaults do not alter diffusion defaults."""
    baseline = AsyncOmniEngine._create_default_diffusion_stage_cfg({})[0]
    nullified_overrides = {name: None for name in deploy_override_field_names()}
    stage_cfg = AsyncOmniEngine._create_default_diffusion_stage_cfg(nullified_overrides)[0]

    assert stage_cfg == baseline


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


def test_default_stage_config_uses_parallel_size_kwargs():
    """Ensure default diffusion parallel_config uses CLI/API parallel sizes."""
    stage_cfg = AsyncOmniEngine._create_default_diffusion_stage_cfg(
        {
            "pipeline_parallel_size": 2,
            "data_parallel_size": 3,
            "tensor_parallel_size": 4,
            "enable_expert_parallel": True,
        }
    )[0]

    parallel_config = stage_cfg["engine_args"]["parallel_config"]
    assert parallel_config.pipeline_parallel_size == 2
    assert parallel_config.data_parallel_size == 3
    assert parallel_config.tensor_parallel_size == 4
    assert parallel_config.enable_expert_parallel is True


def test_default_stage_config_defaults_nullified_parallel_size_kwargs():
    """Ensure nullified diffusion parallel-size kwargs fall back to defaults."""
    stage_cfg = AsyncOmniEngine._create_default_diffusion_stage_cfg(
        {
            "pipeline_parallel_size": None,
            "data_parallel_size": None,
            "tensor_parallel_size": None,
            "enable_expert_parallel": None,
            "enforce_eager": None,
        }
    )[0]

    parallel_config = stage_cfg["engine_args"]["parallel_config"]
    assert parallel_config.pipeline_parallel_size == 1
    assert parallel_config.data_parallel_size == 1
    assert parallel_config.tensor_parallel_size == 1
    assert parallel_config.enable_expert_parallel is False
    assert stage_cfg["engine_args"]["enforce_eager"] is False


def test_default_stage_config_propagates_ulysses_mode():
    """Ensure UAA mode survives default diffusion-stage creation."""
    stage_cfg = AsyncOmniEngine._create_default_diffusion_stage_cfg(
        {
            "ulysses_degree": 4,
            "ulysses_mode": "advanced_uaa",
        }
    )[0]

    parallel_config = stage_cfg["engine_args"]["parallel_config"]
    assert parallel_config.ulysses_degree == 4
    assert parallel_config.ulysses_mode == "advanced_uaa"


def test_default_stage_config_includes_default_sampling_params():
    """Ensure default sampling params survive the default diffusion-stage builder."""
    stage_cfg = AsyncOmniEngine._create_default_diffusion_stage_cfg(
        {
            "default_sampling_params": '{"0": {"generator_device":"cpu", "guidance_scale":7.5}}',
        }
    )[0]

    assert stage_cfg["default_sampling_params"] == {
        "generator_device": "cpu",
        "guidance_scale": 7.5,
    }


def test_default_stage_config_engine_args():
    """Ensure default diffusion-stage builder sets and propagates engine_args."""
    stage_cfg = AsyncOmniEngine._create_default_diffusion_stage_cfg(
        {
            "distributed_executor_backend": "ray",
            "boundary_ratio": 0.875,
            "flow_shift": 5.0,
            "trust_remote_code": True,
        }
    )[0]

    engine_args = stage_cfg["engine_args"]
    assert engine_args["distributed_executor_backend"] == "ray"
    assert engine_args["boundary_ratio"] == 0.875
    assert engine_args["flow_shift"] == 5.0
    assert engine_args["trust_remote_code"] is True


def test_default_stage_config_whitelist_none_fallback():
    """DeployConfig / StageDeployConfig whitelist fields with value None
    fall back to OmniDiffusionConfig dataclass defaults."""
    stage_cfg = AsyncOmniEngine._create_default_diffusion_stage_cfg(
        {
            # DeployConfig pipeline-wide
            "trust_remote_code": None,
            "distributed_executor_backend": None,
            "dtype": None,
            # StageDeployConfig
            "enforce_eager": None,
        }
    )[0]

    engine_args = stage_cfg["engine_args"]

    assert engine_args["trust_remote_code"] is False
    assert engine_args["distributed_executor_backend"] == "mp"
    assert engine_args["dtype"] == "auto"
    assert engine_args["enforce_eager"] is False


def test_serve_cli_accepts_ulysses_mode():
    """Ensure diffusion serve CLI exposes ulysses_mode and wires it to parallel_config."""
    parser = FlexibleArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    OmniServeCommand().subparser_init(subparsers)

    args = parser.parse_args(
        [
            "serve",
            "Qwen/Qwen-Image",
            "--omni",
            "--usp",
            "4",
            "--ulysses-mode",
            "advanced_uaa",
        ]
    )

    stage_cfg = _create_default_diffusion_stage_cfg(args)[0]
    parallel_config = stage_cfg["engine_args"]["parallel_config"]

    assert args.ulysses_mode == "advanced_uaa"
    assert parallel_config.ulysses_degree == 4
    assert parallel_config.ulysses_mode == "advanced_uaa"


def test_serve_cli_accepts_diffusion_pipeline_profiler_flag():
    """Ensure diffusion serve CLI exposes the profiler switch."""
    parser = FlexibleArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    OmniServeCommand().subparser_init(subparsers)

    args = parser.parse_args(
        [
            "serve",
            "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            "--omni",
            "--enable-diffusion-pipeline-profiler",
        ]
    )

    stage_cfg = _create_default_diffusion_stage_cfg(args)[0]

    assert args.enable_diffusion_pipeline_profiler is True
    assert stage_cfg["engine_args"]["enable_diffusion_pipeline_profiler"] is True
