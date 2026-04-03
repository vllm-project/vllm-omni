# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import pytest

import vllm_omni.entrypoints.async_omni_diffusion as async_diffusion_module
from vllm_omni.entrypoints.async_omni_diffusion import AsyncOmniDiffusion

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_async_omni_diffusion_uses_registry_metadata_for_optional_transformer_config(monkeypatch) -> None:
    def fake_get_hf_file_to_dict(filename: str, model: str):
        assert model == "dummy-mova"
        if filename == "model_index.json":
            return {"_class_name": "MOVA"}
        if filename == "transformer/config.json":
            return None
        raise AssertionError(f"unexpected file lookup: {filename}")

    dummy_engine = Mock()
    dummy_engine.close = Mock()

    monkeypatch.setattr(async_diffusion_module, "get_hf_file_to_dict", fake_get_hf_file_to_dict)
    monkeypatch.setattr(async_diffusion_module.DiffusionEngine, "make_engine", Mock(return_value=dummy_engine))

    engine = AsyncOmniDiffusion(model="dummy-mova")

    assert engine.od_config.model_class_name == "MOVA"
    assert engine.od_config.tf_model_config.to_dict() == {}
    engine._weak_finalizer()
