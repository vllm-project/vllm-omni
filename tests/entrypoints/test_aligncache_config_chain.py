# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Configuration integration using real engine/stage/worker modules, without model loading."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pipe

import pytest
import torch
from pytest_mock import MockerFixture
from vllm.config import DeviceConfig, VllmConfig

from vllm_omni.config.config_factory import StageConfigFactory
from vllm_omni.config.yaml_util import create_config
from vllm_omni.diffusion.cache.aligncache.config import AlignCacheConfig
from vllm_omni.diffusion.cache.selector import get_cache_backend
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker
from vllm_omni.engine import stage_init_utils
from vllm_omni.engine.async_omni_engine import AsyncOmniEngine

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def cpu_worker_boundary(mocker: MockerFixture, tmp_path):
    """Replace only device/distributed initialization; keep the actual worker and runner."""

    def init_device(worker: DiffusionWorker) -> None:
        worker.device = torch.device("cpu")
        worker.vllm_config = VllmConfig(device_config=DeviceConfig(device="cpu"))

    mocker.patch.object(DiffusionWorker, "init_device", init_device)
    mocker.patch.object(stage_init_utils.current_omni_platform, "get_device_count", return_value=1)
    mocker.patch.object(
        DiffusionWorker, "load_model", side_effect=AssertionError("configuration test must not load weights")
    )
    model_path = tmp_path / "model"
    model_path.mkdir()
    return str(model_path)


def _worker_roundtrip(config: OmniDiffusionConfig) -> OmniDiffusionConfig:
    """Use multiprocessing's real connection serializer, as used at worker startup."""
    reader, writer = Pipe(duplex=False)
    with reader, writer, ThreadPoolExecutor(max_workers=1) as executor:
        sent = executor.submit(writer.send, config)
        assert reader.poll(10), "worker config transport timed out"
        received = reader.recv()
        sent.result(timeout=10)
    assert isinstance(received, OmniDiffusionConfig)
    return received


@pytest.mark.parametrize("typed", [False, True], ids=["legacy", "typed"])
@pytest.mark.parametrize(
    "profile", ["wan21_reference_v1", "wan22_t2v_reference_v1", "qwen21_reference_v1", "minimax_h3_reference_v1"]
)
def test_aligncache_engine_stage_worker_namespace(cpu_worker_boundary, typed: bool, profile: str):
    cache_config = {"align_cache": {"profile": profile, "mode": "trace", "base_threshold": 0.031}}
    kwargs = {
        "cache_backend": "align_cache",
        "cache_config": json.dumps(cache_config),
        "enforce_eager": True,
        "diffusion_model_runner_cls": "vllm_omni.diffusion.worker.diffusion_model_runner.DiffusionModelRunner",
    }
    if typed:
        stage = StageConfigFactory.create_typed_default_diffusion(cpu_worker_boundary, kwargs).stage_configs[0]
        metadata = stage_init_utils.extract_stage_metadata_from_omni_stage_config(stage)
    else:
        stage = create_config(AsyncOmniEngine._create_default_diffusion_stage_cfg(kwargs)[0])
        metadata = stage_init_utils.extract_legacy_stage_metadata(stage)
    terminal = stage_init_utils.build_diffusion_config(cpu_worker_boundary, stage, metadata)
    transferred = _worker_roundtrip(terminal)
    worker = DiffusionWorker(local_rank=0, rank=0, od_config=transferred, skip_load_model=True, set_proc_title=False)
    try:
        assert isinstance(worker.model_runner, DiffusionModelRunner)
        assert worker.od_config is transferred
        assert worker.model_runner.od_config is transferred
        expected = AlignCacheConfig.parse(cache_config)
        backend = get_cache_backend(worker.od_config.cache_backend, worker.model_runner.od_config.cache_config)
        assert backend.align_config == expected
        assert backend.align_config.config_hash == expected.config_hash
        assert backend.align_config.base_threshold == 0.031
    finally:
        worker.shutdown()


@pytest.mark.parametrize("value", ["{", "[]", "null", {"align_cache": {"mode": "typo"}}, {"foreign": 1}])
def test_aligncache_typed_projection_rejects_invalid_config(value):
    stage = StageConfigFactory.create_typed_default_diffusion("unused", {}).stage_configs[0]
    projection = type(stage.diffusion_config)
    with pytest.raises(ValueError, match="cache_config|align_cache"):
        projection(cache_backend="align_cache", cache_config=value)


def test_aligncache_typed_projection_preserves_valid_json():
    stage = StageConfigFactory.create_typed_default_diffusion("unused", {}).stage_configs[0]
    projection = type(stage.diffusion_config)
    config = projection(cache_backend="align_cache", cache_config='{"align_cache":{"mode":"trace"}}')
    assert AlignCacheConfig.parse(config.cache_config).mode == "trace"


@pytest.mark.parametrize("documented_yaml", [False, True], ids=["python-deploy", "documented-deploy"])
def test_aligncache_cli_deploy_precedence_reaches_worker(cpu_worker_boundary, tmp_path, documented_yaml):
    from vllm_omni.config.omni_config import VllmOmniConfig
    from vllm_omni.config.stage_config import (
        DeployConfig,
        PipelineConfig,
        StageDeployConfig,
        StageExecutionType,
        StagePipelineConfig,
    )
    from vllm_omni.entrypoints.cli.serve import OmniServeCommand
    from vllm_omni.utils.tracking_parser import TrackingArgumentParser

    parser = TrackingArgumentParser()
    OmniServeCommand().subparser_init(parser.add_subparsers(dest="subparser"))
    args = parser.parse_args(
        [
            "serve",
            cpu_worker_boundary,
            "--omni",
            "--enforce-eager",
            "--cache-backend",
            "align_cache",
            "--cache-config",
            '{"align_cache":{"mode":"force_exact","warmup_steps":3}}',
        ]
    )
    explicit = args.get_explicit_kwargs_dict()
    pipeline = PipelineConfig(
        model_type="aligncache_test",
        stages=(
            StagePipelineConfig(
                stage_id=0,
                model_stage="diffusion",
                execution_type=StageExecutionType.DIFFUSION,
                final_output=True,
                final_output_type="video",
            ),
        ),
    )
    deploy = DeployConfig(
        stages=[
            StageDeployConfig(
                stage_id=0,
                engine_extras={
                    "cache_backend": "align_cache",
                    "cache_config": {"align_cache": {"mode": "trace", "warmup_steps": 7, "base_threshold": 0.021}},
                    "diffusion_model_runner_cls": "vllm_omni.diffusion.worker.diffusion_model_runner.DiffusionModelRunner",
                },
            )
        ]
    )
    if documented_yaml:
        from pathlib import Path

        from vllm_omni.config.stage_config import load_deploy_config

        guide = Path(__file__).resolve().parents[2] / "docs/user_guide/diffusion/cache_acceleration/aligncache.md"
        yaml_text = guide.read_text().split("```yaml\n", 1)[1].split("```", 1)[0]
        deploy_path = tmp_path / "aligncache.yaml"
        deploy_path.write_text(yaml_text)
        deploy = load_deploy_config(deploy_path)
        deploy.stages[0].engine_extras["diffusion_model_runner_cls"] = (
            "vllm_omni.diffusion.worker.diffusion_model_runner.DiffusionModelRunner"
        )
    resolved = VllmOmniConfig.from_pipeline_config(
        pipeline, user_deploy_config=deploy, cli_overrides={"model": cpu_worker_boundary, **explicit}
    )
    stage = resolved.stage_configs[0]
    metadata = stage_init_utils.extract_stage_metadata_from_omni_stage_config(stage)
    terminal = stage_init_utils.build_diffusion_config(cpu_worker_boundary, stage, metadata)
    worker = DiffusionWorker(
        local_rank=0, rank=0, od_config=_worker_roundtrip(terminal), skip_load_model=True, set_proc_title=False
    )
    try:
        actual = AlignCacheConfig.parse(worker.model_runner.od_config.cache_config)
        assert actual.mode == "force_exact"
        assert actual.warmup_steps == 3
        assert actual.base_threshold == (0.12 if documented_yaml else 0.021)
        assert actual.profile == ("qwen21_reference_v1" if documented_yaml else "wan21_reference_v1")
    finally:
        worker.shutdown()


def test_aligncache_resolver_launcher_delivers_config_to_worker(cpu_worker_boundary, mocker: MockerFixture):
    from dataclasses import dataclass

    from vllm_omni.config.resolver import resolve_omni_config
    from vllm_omni.diffusion import stage_diffusion_client
    from vllm_omni.engine import stage_runtime as runtime_module
    from vllm_omni.engine.stage_runtime import StageRuntime

    @dataclass
    class WorkerConfigClient:
        worker: DiffusionWorker
        input_address: str | None = None

        def shutdown(self):
            self.worker.shutdown()

    mocker.patch("vllm_omni.config.resolver.StageConfigFactory.create_from_model", return_value=None)
    mocker.patch(
        "vllm_omni.config.resolver._resolve_generic_diffusion_model_class", return_value=(True, "FakeDiffusionPipeline")
    )
    cache_config = {"align_cache": {"profile": "qwen21_reference_v1", "mode": "force_exact", "base_threshold": 0.017}}
    resolved = resolve_omni_config(
        cpu_worker_boundary,
        trust_remote_code=False,
        deploy_config_path=None,
        cli_overrides={
            "cache_backend": "align_cache",
            "cache_config": cache_config,
            "enforce_eager": True,
            "diffusion_model_runner_cls": "vllm_omni.diffusion.worker.diffusion_model_runner.DiffusionModelRunner",
        },
        stage_overrides=None,
        strategy_config_path=None,
    )
    mocker.patch.object(runtime_module, "prepare_engine_environment")
    mocker.patch.object(runtime_module, "load_omni_transfer_config_for_model", return_value=None)
    mocker.patch.object(runtime_module, "get_stage_connector_spec", return_value={})
    mocker.patch.object(runtime_module, "resolve_omni_kv_config_for_stage", return_value=(None, None, None))
    clients = []

    def create_client(model, od_config, metadata, stage_init_timeout, use_inline):
        worker = DiffusionWorker(
            local_rank=0, rank=0, od_config=_worker_roundtrip(od_config), skip_load_model=True, set_proc_title=False
        )
        client = WorkerConfigClient(worker)
        clients.append(client)
        return client

    mocker.patch.object(stage_diffusion_client, "create_diffusion_client", side_effect=create_client)
    runtime = StageRuntime(
        stage_configs=list(resolved.stage_configs),
        model=cpu_worker_boundary,
        config_path="",
        stage_init_timeout=10,
        async_chunk=False,
    )
    try:
        runtime.initialize()
        assert len(clients) == 1
        assert runtime.stage_pools[0].clients == clients
        worker = clients[0].worker
        assert worker.od_config.cache_backend == "align_cache"
        assert AlignCacheConfig.parse(worker.model_runner.od_config.cache_config) == AlignCacheConfig.parse(
            cache_config
        )
    finally:
        for client in clients:
            client.shutdown()


@pytest.mark.parametrize("typed", [False, True], ids=["legacy", "typed"])
def test_aligncache_environment_selection_preserves_json_errors(monkeypatch, typed):
    monkeypatch.setenv("DIFFUSION_CACHE_BACKEND", "align_cache")
    kwargs = {"cache_config": "{"}
    with pytest.raises(ValueError, match="cache_config"):
        if typed:
            StageConfigFactory.create_typed_default_diffusion("unused", kwargs)
        else:
            AsyncOmniEngine._create_default_diffusion_stage_cfg(kwargs)


def test_explicit_none_still_overrides_environment_for_worker(cpu_worker_boundary, monkeypatch):
    monkeypatch.setenv("DIFFUSION_CACHE_BACKEND", "align_cache")
    stage = StageConfigFactory.create_typed_default_diffusion(
        cpu_worker_boundary,
        {
            "cache_backend": "none",
            "diffusion_model_runner_cls": "vllm_omni.diffusion.worker.diffusion_model_runner.DiffusionModelRunner",
        },
    ).stage_configs[0]
    metadata = stage_init_utils.extract_stage_metadata_from_omni_stage_config(stage)
    config = stage_init_utils.build_diffusion_config(cpu_worker_boundary, stage, metadata)
    worker = DiffusionWorker(
        local_rank=0, rank=0, od_config=_worker_roundtrip(config), skip_load_model=True, set_proc_title=False
    )
    try:
        assert worker.model_runner.od_config.cache_backend == "none"
        assert get_cache_backend(worker.od_config.cache_backend, worker.od_config.cache_config) is None
    finally:
        worker.shutdown()
