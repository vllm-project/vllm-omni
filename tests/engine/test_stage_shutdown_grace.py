# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Keep stage cleanup policy through configuration and every LLM launch path."""

from contextlib import nullcontext
from unittest.mock import Mock

import pytest
from vllm.config import CacheConfig, ParallelConfig, VllmConfig
from vllm.v1.engine.utils import EngineZmqAddresses, EngineZmqBindAddresses
from vllm.v1.executor.uniproc_executor import UniProcExecutor

from vllm_omni.config.omni_config import VllmOmniConfig
from vllm_omni.config.stage_config import DeployConfig, PipelineConfig, StagePipelineConfig, merge_pipeline_deploy
from vllm_omni.engine import stage_engine_core_proc_manager, stage_engine_startup
from vllm_omni.engine.stage_engine_core_client import StageEngineCoreClientBase
from vllm_omni.model_executor.models.spark_tts.pipeline import SPARK_TTS_PIPELINE

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def resolved_stage(pipeline, config_kind):
    if config_kind == "structured":
        return VllmOmniConfig.from_pipeline_config(pipeline, user_deploy_config=DeployConfig()).stage_configs[0]
    return merge_pipeline_deploy(pipeline, DeployConfig())[0].to_omegaconf()


@pytest.mark.parametrize("config_kind", ["structured", "legacy"])
def test_spark_declares_cleanup_grace_only_on_its_codec(config_kind):
    if config_kind == "structured":
        stages = VllmOmniConfig.from_pipeline_config(SPARK_TTS_PIPELINE).stage_configs
    else:
        stages = [s.to_omegaconf() for s in merge_pipeline_deploy(SPARK_TTS_PIPELINE, DeployConfig())]
    assert [getattr(stage, "shutdown_grace_s", None) for stage in stages] == [None, 15.0]


@pytest.mark.parametrize("config_kind", ["structured", "legacy"])
@pytest.mark.parametrize("shutdown_grace_s", [None, 0.0, 12.5])
@pytest.mark.parametrize("launch", ["colocated", "coordinator", "without_coordinator", "headless"])
def test_pipeline_cleanup_grace_reaches_process_manager(mocker, config_kind, shutdown_grace_s, launch):
    pipeline = PipelineConfig(
        model_type="test_cleanup",
        stages=(
            StagePipelineConfig(
                stage_id=0, model_stage="generic", final_output=True, shutdown_grace_s=shutdown_grace_s
            ),
        ),
    )
    stage = resolved_stage(pipeline, config_kind)
    # The metadata belongs to the stage wrapper, never backend EngineArgs.
    assert getattr(stage, "shutdown_grace_s", None) == shutdown_grace_s
    if config_kind == "legacy":
        assert "shutdown_grace_s" not in stage.engine_args

    config = mocker.Mock(
        spec=VllmConfig,
        parallel_config=ParallelConfig(data_parallel_size_local=1),
        cache_config=CacheConfig(),
        needs_dp_coordinator=False,
    )
    factory_name = "StageCleanupProcManager" if launch == "without_coordinator" else "StageEngineCoreProcManager"
    manager_factory = mocker.patch.object(stage_engine_core_proc_manager, factory_name)
    mocker.patch.object(stage_engine_startup, "scoped_spawn_device_env", side_effect=lambda *a, **kw: nullcontext())
    mocker.patch.object(stage_engine_startup, "zmq_socket_ctx", side_effect=lambda *a, **kw: nullcontext())
    mocker.patch.object(stage_engine_startup, "wait_for_engine_startup_with_deadline")
    mocker.patch.object(
        stage_engine_startup,
        "get_engine_zmq_bind_addresses",
        return_value=EngineZmqBindAddresses(inputs=["ipc://unused-in"], outputs=["ipc://unused-out"]),
    )
    mocker.patch.object(
        stage_engine_startup,
        "register_stage_with_omni_master",
        return_value=stage_engine_startup.StageRegistrationResponse(
            handshake_address="ipc://unused-handshake",
            input_address="ipc://unused-in",
            output_address="ipc://unused-out",
            replica_id=0,
            coordinator_router_address="ipc://unused-coordinator",
        ),
    )
    if launch == "headless":
        stage_engine_startup.launch_headless_llm_replica(
            vllm_config=config,
            executor_class=UniProcExecutor,
            log_stats=False,
            omni_master_address="127.0.0.1",
            omni_master_port=26000,
            stage_id=0,
            stage_config=stage,
        )
    else:
        master = None
        if launch in ("coordinator", "without_coordinator"):
            master = mocker.Mock(spec=stage_engine_startup.OmniMasterServer)
            master.get_zmq_addresses.return_value = EngineZmqAddresses(
                inputs=["ipc://unused-in"], outputs=["ipc://unused-out"]
            )
        with stage_engine_startup.launch_stage_replica(
            vllm_config=config,
            executor_class=UniProcExecutor,
            log_stats=False,
            stage_id=0,
            stage_config=stage,
            omni_master_server=master,
            omni_coordinator_address=None if launch == "without_coordinator" else "ipc://unused-coordinator",
        ):
            pass
    manager_factory.assert_called_once()
    assert manager_factory.call_args.kwargs["shutdown_grace_s"] == shutdown_grace_s


class RecordingTransport:
    def shutdown(self, timeout=None):
        self.received_timeout = timeout


class Client(StageEngineCoreClientBase, RecordingTransport):
    def __init__(self, configured):
        self.vllm_config = Mock(shutdown_timeout=0)
        self._stage_shutdown_timeout = configured


@pytest.mark.parametrize(
    ("configured", "explicit", "expected"),
    [(30, None, 30), (30, 0, 0), (30, 5, 5), (0, None, 0), (None, None, None)],
)
def test_configured_grace_reaches_transport(configured, explicit, expected):
    client = Client(configured)
    client.shutdown(timeout=explicit)
    assert client.received_timeout == expected


def test_runtime_grace_validation():
    from vllm_omni.config.omni_config import OmniStageRuntimeConfig

    assert OmniStageRuntimeConfig(shutdown_timeout=30).shutdown_timeout == 30
    with pytest.raises(ValueError):
        OmniStageRuntimeConfig(shutdown_timeout=-1)
