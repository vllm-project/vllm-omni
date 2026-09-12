# SPDX-License-Identifier: Apache-2.0
"""Exercise deployment discovery across the simulation model registries."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from vllm_omni.config.pipeline_registry import resolve_pipeline_config
from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.config import Cosmos3NanoSimBimanualManifest
from vllm_omni.diffusion.registry import (
    _DIFFUSION_IR_OP_PRIORITY_FUNCS,
    _DIFFUSION_MODELS,
    _DIFFUSION_POST_PROCESS_FUNCS,
    _DIFFUSION_PRE_PROCESS_FUNCS,
)
from vllm_omni.model_extras import get_extra_body_params

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("model_type", ["cosmos3_nano_sim_bimanual", "cosmos3_nano_sim_transfer"])
def test_deployment_discovery(model_type: str) -> None:
    pipeline = resolve_pipeline_config(model_type)
    assert pipeline is not None
    deploy_path = Path(__file__).resolve().parents[4] / "vllm_omni/deploy" / pipeline.default_deploy_config_name
    deploy = yaml.safe_load(deploy_path.read_text())
    assert deploy["pipeline"] == model_type
    assert pipeline.model_arch == pipeline.diffusers_class_name
    assert pipeline.stages[0].model_arch == pipeline.model_arch
    assert deploy["stages"][0]["model_class_name"] == pipeline.model_arch

    folder, module_name, class_name = _DIFFUSION_MODELS[pipeline.model_arch]
    module = importlib.import_module(f"vllm_omni.diffusion.models.{folder}.{module_name}")
    assert getattr(module, class_name).__name__ == pipeline.model_arch
    for registry in (_DIFFUSION_PRE_PROCESS_FUNCS, _DIFFUSION_POST_PROCESS_FUNCS, _DIFFUSION_IR_OP_PRIORITY_FUNCS):
        assert callable(getattr(module, registry[pipeline.model_arch]))
    assert "session_id" in get_extra_body_params(pipeline.model_arch)


def test_legacy_identity_is_not_registered() -> None:
    assert resolve_pipeline_config("cosmos_dreams") is None
    assert "CosmosDreamsPipeline" not in _DIFFUSION_MODELS
    assert not get_extra_body_params("CosmosDreamsPipeline")


def test_legacy_artifact_requires_reexport() -> None:
    config = SimpleNamespace(tf_model_config={"cosmos_dreams": {"schema_version": 1}})
    with pytest.raises(ValueError, match="Re-export the checkpoint"):
        Cosmos3NanoSimBimanualManifest.from_od_config(config)


@pytest.mark.parametrize(
    "legacy_class",
    ["CosmosDreamsTransferPipeline", "CosmosDreamsTransferOmniPipeline", "Cosmos3TransferInteractivePipeline"],
)
def test_legacy_transfer_identity_is_not_registered(legacy_class: str) -> None:
    assert resolve_pipeline_config("cosmos_dreams_transfer") is None
    assert legacy_class not in _DIFFUSION_MODELS
    assert not get_extra_body_params(legacy_class)
