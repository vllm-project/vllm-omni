# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Validate resolved payload edges before any stage worker is launched."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from vllm_omni.config.omni_config import VllmOmniConfig
from vllm_omni.config.pipeline_registry import resolve_pipeline_config
from vllm_omni.config.stage_config import load_deploy_config
from vllm_omni.distributed.omni_connectors.utils.config import ConnectorSpec, OmniTransferConfig
from vllm_omni.distributed.omni_connectors.utils.initialization import (
    resolve_omni_kv_config_for_stage,
    validate_stage_payload_connectors,
)
from vllm_omni.engine import stage_engine_startup, stage_init_utils, stage_runtime
from vllm_omni.entrypoints.utils import inject_omni_kv_config
from vllm_omni.model_executor.models.wan2_2.pipeline import WAN2_2_EGD_PIPELINE

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

DEPLOY_DIR = Path(__file__).resolve().parents[2] / "vllm_omni" / "deploy"


def _mixed_egd(incoming, outgoing):
    path = DEPLOY_DIR / "wan2_2_egd.yaml"
    stages = list(VllmOmniConfig.from_pipeline_config(WAN2_2_EGD_PIPELINE, deploy_config_path=str(path)).stage_configs)
    transfer = stage_init_utils.load_omni_transfer_config_for_model("unused", str(path))
    assert transfer is not None
    transfer.connectors[("0", "1")] = ConnectorSpec(incoming)
    transfer.connectors[("1", "2")] = ConnectorSpec(outgoing)
    return stages, transfer


@pytest.mark.parametrize("startup", ("regular", "headless"))
@pytest.mark.parametrize(
    ("incoming", "outgoing"),
    (
        ("MooncakeTransferEngineConnector", "SharedMemoryConnector"),
        ("NixlConnector", "SharedMemoryConnector"),
        ("SharedMemoryConnector", "NixlConnector"),
        ("SharedMemoryConnector", "MooncakeTransferEngineConnector"),
    ),
)
def test_mixed_egd_rejected_before_startup(startup, incoming, outgoing, mocker):
    stages, transfer = _mixed_egd(incoming, outgoing)
    if startup == "regular":
        mocker.patch.object(stage_runtime, "load_omni_transfer_config_for_model", return_value=transfer)
        mocker.patch.object(stage_runtime, "prepare_engine_environment")
        runtime = stage_runtime.StageRuntime(stages, "unused", None, stage_init_timeout=1, async_chunk=False)
        next_step = mocker.patch.object(
            runtime, "_build_logical_stage_init_plans", side_effect=AssertionError("must reject before building plans")
        )
        with pytest.raises(ValueError, match="Stage 1.*edge 0.*1"):
            runtime._prepare_stage_plans()
    else:
        mocker.patch.object(stage_init_utils, "load_omni_transfer_config_for_model", return_value=transfer)
        next_step = mocker.patch.object(
            stage_init_utils,
            "build_diffusion_config",
            side_effect=AssertionError("must reject before device discovery"),
        )
        with pytest.raises(ValueError, match="Stage 1.*edge 0.*1"):
            stage_engine_startup.launch_headless_diffusion_replicas(
                model="unused",
                stage_cfg=stages[1],
                stage_configs=stages,
                stage_id=1,
                omni_master_address="127.0.0.1",
                omni_master_port=1,
                omni_dp_size_local=1,
                per_replica_devices=["1"],
                config_path="unused",
            )
    next_step.assert_not_called()


@pytest.mark.parametrize(
    "path",
    sorted(
        path for path in DEPLOY_DIR.glob("*.yaml") if path.name.startswith(("wan2_2_eg", "minimax_h3_disaggregated"))
    ),
    ids=lambda path: path.name,
)
def test_shipped_payload_deploy_configs(path, monkeypatch, mocker):
    deploy = load_deploy_config(path)
    pipeline = resolve_pipeline_config(deploy.pipeline)
    assert pipeline is not None
    config = VllmOmniConfig.from_pipeline_config(pipeline, deploy_config_path=str(path))
    transfer = stage_init_utils.load_omni_transfer_config_for_model("unused", str(path))
    assert transfer is not None

    def skip_weight_resolution(model, engine_args):
        # Match the consumed adapter key while keeping this config test offline.
        engine_args.pop("model_path_resolver", None)
        return model

    resolver = mocker.patch.object(stage_init_utils, "_resolve_model_path", side_effect=skip_weight_resolution)
    stage_init_utils.validate_diffusion_stage_payload_topology(config.stage_configs, transfer)
    resolver.assert_not_called()
    # Test config construction only; no worker or native connector is created.
    monkeypatch.setattr(stage_init_utils.current_omni_platform, "get_device_count", lambda: 64)
    for stage in config.stage_configs:
        if stage.stage_type != "diffusion":
            continue
        connector, source, target = resolve_omni_kv_config_for_stage(transfer, stage.stage_id)
        if connector is not None:
            inject_omni_kv_config(stage, connector, source, target)
        metadata = stage_init_utils.extract_stage_metadata_from_omni_stage_config(stage)
        od_config = stage_init_utils.build_diffusion_config("unused", stage, metadata)
        assert od_config.stage_id == stage.stage_id


@pytest.mark.parametrize("connector", ("SharedMemoryConnector", "NixlConnector"))
def test_uniform_egd_topology_remains_allowed(connector):
    stages, transfer = _mixed_egd(connector, connector)
    stage_init_utils.validate_diffusion_stage_payload_topology(stages, transfer)


def test_every_incoming_edge_is_validated():
    transfer = OmniTransferConfig(
        connectors={
            ("0", "2"): ConnectorSpec("SharedMemoryConnector"),
            ("1", "2"): ConnectorSpec("MooncakeTransferEngineConnector"),
        }
    )
    with pytest.raises(ValueError, match="Stage 2.*edge 1→2.*MooncakeTransferEngineConnector"):
        validate_stage_payload_connectors(transfer, 2)


def test_legacy_payload_keys_are_validated():
    stage = OmegaConf.create(
        {"stage_id": 1, "stage_type": "diffusion", "engine_args": {"stage_input_payload_keys": ["prompt_embeds"]}}
    )
    _, transfer = _mixed_egd("NixlConnector", "SharedMemoryConnector")
    with pytest.raises(ValueError, match="edge 0→1.*edge 1→2.*Mixed connector types"):
        stage_init_utils.validate_diffusion_stage_payload_topology([stage], transfer)


def test_explicit_empty_input_keys_remain_allowed():
    stages, transfer = _mixed_egd("MooncakeTransferEngineConnector", "SharedMemoryConnector")
    stages[1].diffusion_config.stage_input_payload_keys = ()
    stage_init_utils.validate_diffusion_stage_payload_topology([stages[1]], transfer)
