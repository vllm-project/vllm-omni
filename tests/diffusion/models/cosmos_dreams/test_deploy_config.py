# SPDX-License-Identifier: Apache-2.0
"""Registration and AR-engine routing contracts for Cosmos-Dreams."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.config.pipeline_registry import OMNI_PIPELINES
from vllm_omni.diffusion.diffusion_engine import DiffusionEngine
from vllm_omni.experimental.ar_diffusion.engine import ARDiffusionEngine
from vllm_omni.model_executor.models.cosmos_dreams.pipeline import (
    COSMOS_DREAMS_PIPELINE,
    COSMOS_DREAMS_TRANSFER_PIPELINE,
)
from vllm_omni.model_extras.registry import get_extra_body_params

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_cosmos_dreams_topology_and_deploy_route_to_ar_diffusion() -> None:
    deploy_path = Path(get_deploy_config_path("cosmos_dreams.yaml"))
    config = yaml.safe_load(deploy_path.read_text())

    assert config["pipeline"] == "cosmos_dreams"
    assert OMNI_PIPELINES["cosmos_dreams"] is COSMOS_DREAMS_PIPELINE
    assert COSMOS_DREAMS_PIPELINE.default_deploy_config_name == "cosmos_dreams.yaml"
    assert COSMOS_DREAMS_PIPELINE.diffusers_class_name == "CosmosDreamsPipeline"
    assert COSMOS_DREAMS_PIPELINE.validate() == []

    [stage] = config["stages"]
    assert stage["max_num_seqs"] == 1
    assert stage["enforce_eager"] is True
    assert stage["model_class_name"] == "CosmosDreamsPipeline"
    assert "default_domain_id" not in stage["model_config"]
    engine = DiffusionEngine.resolve_engine_class(SimpleNamespace(engine_backend=stage["engine_backend"]))
    assert issubclass(engine, ARDiffusionEngine)


def test_cosmos_dreams_deploy_keeps_artifact_fields_out_of_templates() -> None:
    config = yaml.safe_load(Path(get_deploy_config_path("cosmos_dreams.yaml")).read_text())
    manifest = config["stages"][0]["model_config"]["cosmos_dreams"]

    # These values must come from the validated model artifact. A deploy file
    # must not be able to masquerade as a completed Stage-2 export.
    assert "checkpoint_id" not in manifest
    assert "checkpoint_iteration" not in manifest
    assert "checkpoint_hash" not in manifest
    assert "normalizer_id" not in manifest
    assert "normalizer_source" not in manifest
    assert "action_normalizer" not in manifest
    assert "action_schema" not in manifest
    assert manifest["schema_version"] == 2


def test_cosmos_dreams_transfer_topology_and_dense_t1_deploy() -> None:
    deploy_path = Path(get_deploy_config_path("cosmos_dreams_transfer.yaml"))
    config = yaml.safe_load(deploy_path.read_text())

    assert config["pipeline"] == "cosmos_dreams_transfer"
    assert OMNI_PIPELINES["cosmos_dreams_transfer"] is COSMOS_DREAMS_TRANSFER_PIPELINE
    assert COSMOS_DREAMS_TRANSFER_PIPELINE.default_deploy_config_name == "cosmos_dreams_transfer.yaml"
    assert COSMOS_DREAMS_TRANSFER_PIPELINE.diffusers_class_name == "CosmosDreamsTransferPipeline"
    assert COSMOS_DREAMS_TRANSFER_PIPELINE.validate() == []

    [stage] = config["stages"]
    assert stage["model_class_name"] == "CosmosDreamsTransferPipeline"
    assert stage["enforce_eager"] is True
    assert "engine_backend" not in stage
    assert stage["model_config"]["action_gen"] is False
    assert stage["model_config"]["cosmos_dreams"] == {"schema_version": 3}

    extras = get_extra_body_params("CosmosDreamsTransferPipeline")
    assert {"control_video", "control_hint", "edge", "blur", "depth", "seg"} <= extras
    assert not {"action", "domain_id", "domain_name", "initial_latent"} & extras
