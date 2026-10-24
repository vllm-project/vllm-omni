# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import msgspec
import pytest
import torch

from vllm_omni.config.omni_config import (
    VllmOmniConfig,
    _DiffusionConfigProjection,
    extract_diffusion_stage_config_kwargs,
)
from vllm_omni.config.stage_config import (
    DeployConfig,
    PipelineConfig,
    StageDeployConfig,
    StageExecutionType,
    StagePipelineConfig,
    build_stage_runtime_overrides,
    merge_pipeline_deploy,
)
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.engine.stage_engine_startup import _serialize_stage_config
from vllm_omni.engine.stage_init_utils import (
    build_engine_args_dict_from_omni_stage_config,
    build_legacy_engine_args_dict,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("value", [1, 2, 3, 4])
def test_typed_and_deploy_tile_batch_size(value):
    assert _DiffusionConfigProjection(vae_tile_batch_size=value).vae_tile_batch_size == value
    assert StageDeployConfig(stage_id=0, vae_tile_batch_size=value).vae_tile_batch_size == value


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "4"])
def test_invalid_typed_and_deploy_tile_batch_size(value):
    with pytest.raises(ValueError):
        _DiffusionConfigProjection(vae_tile_batch_size=value)
    with pytest.raises(ValueError):
        StageDeployConfig(stage_id=0, vae_tile_batch_size=value)


@pytest.mark.parametrize("reverse", [False, True])
def test_stage_scope_and_explicit_one_override(reverse):
    items = [("vae_tile_batch_size", 4), ("stage_1_vae_tile_batch_size", 1)]
    values = dict(reversed(items) if reverse else items)
    assert build_stage_runtime_overrides(1, values)["vae_tile_batch_size"] == 1
    assert build_stage_runtime_overrides(0, values)["vae_tile_batch_size"] == 4
    assert "vae_tile_batch_size" not in build_stage_runtime_overrides(0, {"vae_tile_batch_size": None})


@pytest.mark.parametrize("typed", [False, True])
@pytest.mark.parametrize("cli_value,expected", [(None, 4), (1, 1), (2, 2), (3, 3), (4, 4)])
def test_deploy_cli_final_worker_and_ipc(typed, cli_value, expected):
    pipeline = PipelineConfig(
        model_type="tile-batch-test",
        stages=(
            StagePipelineConfig(
                stage_id=0, model_stage="model", execution_type=StageExecutionType.DIFFUSION, final_output=True
            ),
        ),
    )
    deploy = DeployConfig(stages=[StageDeployConfig(stage_id=0, vae_tile_batch_size=4)])
    cli = {"vae_tile_batch_size": cli_value}
    if typed:
        stage = VllmOmniConfig.from_pipeline_config(
            pipeline, user_deploy_config=deploy, cli_overrides=cli
        ).stage_configs[0]
        serialized = msgspec.msgpack.decode(msgspec.msgpack.encode(_serialize_stage_config(stage)))
        assert serialized["diffusion_config"]["vae_tile_batch_size"] == expected
        args = build_engine_args_dict_from_omni_stage_config(stage, model="unused")
    else:
        stage = merge_pipeline_deploy(pipeline, deploy)[0]
        stage.runtime_overrides = build_stage_runtime_overrides(0, cli)
        args = build_legacy_engine_args_dict(stage.to_omegaconf(), model="unused")
    kwargs = extract_diffusion_stage_config_kwargs(args, stage_id=0, include_engine_adapter_metadata=True)
    kwargs.pop("model")
    assert OmniDiffusionConfig.from_kwargs(**kwargs).vae_tile_batch_size == expected


@pytest.mark.parametrize("use_tiling", [False, True])
def test_default_tile_batch_keeps_checkpoint_executor(monkeypatch, use_tiling):
    from vllm_omni.diffusion.models.minimax_h3 import vae as vae_module

    vae = object.__new__(vae_module.MiniMaxH3VideoVAE)
    torch.nn.Module.__init__(vae)
    vae.encode_only = False
    vae.use_tiling = use_tiling
    vae.parallel_size = 2
    vae._tile_batch_executor = None
    vae.model = torch.nn.Module()
    original_runner = object()
    vae.model._run_tile_tasks = original_runner

    def unexpected_batch_setup(*args, **kwargs):
        pytest.fail("K=1 must not access a control group or install a batch executor")

    monkeypatch.setattr(vae_module, "get_world_group", unexpected_batch_setup)
    monkeypatch.setattr(vae_module, "TileBatchExecutor", unexpected_batch_setup)

    vae.set_tile_batch_size(1)

    assert vae.model._run_tile_tasks is original_runner
    assert vae._tile_batch_executor is None


@pytest.mark.parametrize("batch_size", [1, 2])
def test_tile_batch_reconfiguration_is_rejected(batch_size):
    from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3VideoVAE

    vae = object.__new__(MiniMaxH3VideoVAE)
    torch.nn.Module.__init__(vae)
    vae.encode_only = False
    vae._tile_batch_executor = object()

    with pytest.raises(RuntimeError, match="already been configured"):
        vae.set_tile_batch_size(batch_size)


def test_rank_group_failure_stops_worker_before_device_cleanup(monkeypatch):
    from vllm_omni.diffusion.worker.diffusion_worker import _cleanup_after_execution_error

    class RankGroupFailureError(RuntimeError):
        requires_stage_shutdown = True

    # A poisoned device must not be synchronized or used for normal request
    # cleanup before the dedicated worker exits and its supervisor intervenes.
    def unexpected_gc():
        pytest.fail("request-scoped device cleanup ran for a fatal rank group")

    monkeypatch.setattr("vllm_omni.diffusion.worker.diffusion_worker.gc.collect", unexpected_gc)
    with pytest.raises(SystemExit) as exit_info:
        _cleanup_after_execution_error(RankGroupFailureError("injected rank-group failure"))
    assert exit_info.value.code == 1
