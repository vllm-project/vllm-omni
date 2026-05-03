# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Tuna/Tuna-2 model recognition scaffolding."""

from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.registry import DiffusionModelRegistry
from vllm_omni.diffusion.utils import hf_utils

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_tuna_external_pipeline_registered():
    cls = DiffusionModelRegistry._try_load_model_cls("TunaExternalPipeline")
    assert cls is not None


def test_tuna_config_enriches_to_external_pipeline(monkeypatch):
    def fake_get_hf_file_to_dict(filename, model, revision=None):
        if filename == "model_index.json":
            return None
        if filename == "config.json":
            return {"model_type": "tuna_2_pixel"}
        return None

    monkeypatch.setattr("vllm.transformers_utils.config.get_hf_file_to_dict", fake_get_hf_file_to_dict)

    od_config = OmniDiffusionConfig(model="dummy-tuna")
    od_config.enrich_config()

    assert od_config.model_class_name == "TunaExternalPipeline"


def test_is_diffusion_model_detects_tuna_config(monkeypatch):
    monkeypatch.setattr("os.path.isdir", lambda _model: False)

    def fake_get_hf_file_to_dict(filename, model_name):
        if filename == "model_index.json":
            return None
        if filename == "config.json":
            return {"architectures": ["Tuna2PixelModel"]}
        return None

    monkeypatch.setattr(hf_utils, "get_hf_file_to_dict", fake_get_hf_file_to_dict)
    monkeypatch.setattr(hf_utils, "load_diffusers_config", lambda _model: (_ for _ in ()).throw(ValueError("nope")))
    hf_utils.is_diffusion_model.cache_clear()

    assert hf_utils.is_diffusion_model("dummy-tuna") is True


def test_tuna_external_pipeline_error_is_actionable():
    cls = DiffusionModelRegistry._try_load_model_cls("TunaExternalPipeline")
    od_config = SimpleNamespace()

    with pytest.raises(RuntimeError) as exc_info:
        cls(od_config=od_config)

    message = str(exc_info.value)
    assert "Tuna/Tuna-2 is recognized" in message
    assert "runtime integration is not available yet" in message
    assert "#3303" in message
    assert "https://github.com/facebookresearch/tuna-2" in message
