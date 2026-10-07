# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest

from vllm_omni.config.config_factory import StageConfigFactory
from vllm_omni.config.pipeline_registry import OMNI_PIPELINES, resolve_pipeline_config
from vllm_omni.config.resolver import resolve_omni_config
from vllm_omni.diffusion.models.waveserve_wan.pipeline_waveserve_wan import (
    HF_MODEL_ID,
    WaveServeWanPipeline,
)
from vllm_omni.diffusion.registry import DiffusionModelRegistry

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_waveserve_wan_registered_in_pipeline_registry():
    assert "waveserve_wan" in OMNI_PIPELINES
    pipeline = resolve_pipeline_config("waveserve_wan")
    assert pipeline is not None
    assert pipeline.model_arch == "WaveServeWanPipeline"
    assert pipeline.default_deploy_config_name == "waveserve_wan.yaml"
    # Diffusion registry must resolve the same class (model load path).
    cls = DiffusionModelRegistry._try_load_model_cls("WaveServeWanPipeline")
    assert cls is WaveServeWanPipeline
    assert HF_MODEL_ID.endswith("waveserve-wan2.1-1.3b-diffusers-rf-dev")


def _stub_hf_lookup(monkeypatch) -> None:
    monkeypatch.setattr(
        StageConfigFactory,
        "get_hf_config",
        classmethod(lambda _cls, model, trust_remote_code=True: None),
    )
    monkeypatch.setattr(
        "vllm_omni.config.config_factory.get_hf_file_to_dict",
        lambda *_args, **_kwargs: None,
    )


def test_waveserve_basename_infers_model_type(monkeypatch):
    _stub_hf_lookup(monkeypatch)
    model_type = StageConfigFactory.try_infer_model_type(
        "Physis-AI/waveserve-wan2.1-1.3b-diffusers-rf-dev",
        trust_remote_code=False,
    )
    assert model_type == "waveserve_wan"


def test_waveserve_path_resolves_default_deploy(monkeypatch):
    _stub_hf_lookup(monkeypatch)
    resolved = resolve_omni_config(
        "Physis-AI/waveserve-wan2.1-1.3b-diffusers-rf-dev",
        trust_remote_code=False,
        deploy_config_path=None,
        cli_overrides=None,
        stage_overrides=None,
        strategy_config_path=None,
    )
    diffusion_config = resolved.stage_configs[0].diffusion_config

    assert resolved.config_path is not None
    assert resolved.config_path.endswith("vllm_omni/deploy/waveserve_wan.yaml")
    assert diffusion_config.model_class_name == "WaveServeWanPipeline"
    assert "ARDiffusionEngine" in str(diffusion_config.engine_backend)
