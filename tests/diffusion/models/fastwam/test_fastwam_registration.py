# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""FastWAM registration: model-class resolution for config-less checkpoints and deploy configs."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.config.pipeline_registry import OMNI_PIPELINES
from vllm_omni.diffusion.models.fastwam.config import PRESETS
from vllm_omni.diffusion.registry import _DIFFUSION_MODELS, _NO_CACHE_ACCELERATION
from vllm_omni.diffusion.utils.hf_utils import resolve_native_diffusion_model_class

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

DEPLOY_CONFIGS = ("cae_fastwam_libero.yaml", "fastwam_libero.yaml")


@pytest.mark.parametrize(
    "model", ["amd/CAE-FastWAM", "https://huggingface.co/amd/CAE-FastWAM/", "yuanty/fastwam", "AMD/cae-fastwam"]
)
def test_hub_ids_resolve_to_fastwam(model):
    assert resolve_native_diffusion_model_class(model) == "FastWAMPipeline"


def test_local_checkpoint_dirs_resolve_by_signature(tmp_path):
    cae = tmp_path / "cae"
    (cae / "stats").mkdir(parents=True)
    for name in ("policy.pt", "action_vae.pt", "stats/libero_spatial.json"):
        (cae / name).write_bytes(b"")
    original = tmp_path / "original"
    original.mkdir()
    for name in ("libero_uncond_2cam224.pt", "libero_uncond_2cam224_dataset_stats.json"):
        (original / name).write_bytes(b"")
    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "policy.pt").write_bytes(b"")
    assert resolve_native_diffusion_model_class(str(cae)) == "FastWAMPipeline"
    assert resolve_native_diffusion_model_class(str(original)) == "FastWAMPipeline"
    assert resolve_native_diffusion_model_class(str(partial)) is None
    assert resolve_native_diffusion_model_class("amd/some-other-model") is None


def test_registry_entries():
    assert _DIFFUSION_MODELS["FastWAMPipeline"] == ("fastwam", "pipeline_fastwam", "FastWAMPipeline")
    assert "FastWAMPipeline" in _NO_CACHE_ACCELERATION
    pipeline = OMNI_PIPELINES["fastwam"]
    assert pipeline.model_arch == "FastWAMPipeline"
    assert pipeline.default_deploy_config_name == "cae_fastwam_libero.yaml"


@pytest.mark.parametrize("name", DEPLOY_CONFIGS)
def test_deploy_configs(name):
    config = yaml.safe_load(Path(get_deploy_config_path(name)).read_text())
    assert config["pipeline"] == "fastwam" and config["async_chunk"] is False
    (stage,) = config["stages"]
    assert stage["model_class_name"] == "FastWAMPipeline"
    assert stage["max_num_seqs"] == 1 and stage["enforce_eager"] is True
    model_config = stage["model_config"]
    preset = PRESETS[model_config["fastwam_variant"]]
    handshake = model_config["policy_server_config"]
    assert handshake["action_horizon"] == preset.action_horizon
    assert handshake["action_dim"] == preset.action_dim
    assert handshake["image_resolution"] == list(preset.camera_size)
    # No silent suite default: suite-conditioned requests must name their suite
    # unless an operator pins one deliberately.
    assert "task_suite" not in model_config
