# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import pytest

from vllm_omni.config.config_factory import StageConfigFactory
from vllm_omni.diffusion.data import OmniDiffusionConfig, resolve_model_class_name
from vllm_omni.entrypoints.utils import resolve_model_config_path

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _patch_hf_config(monkeypatch: pytest.MonkeyPatch, cfg: dict) -> None:
    from vllm.transformers_utils import config as hf_config_module

    def _get_hf_file_to_dict(filename: str, model: str, *args, **kwargs):
        del model, args, kwargs
        if filename == "config.json":
            return cfg
        return None

    monkeypatch.setattr(hf_config_module, "get_hf_file_to_dict", _get_hf_file_to_dict)


def _janus_config() -> dict:
    return {
        "model_type": "multi_modality",
        "architectures": ["MultiModalityCausalLM"],
        "language_config": {},
        "vision_config": {},
        "aligner_config": {},
        "gen_vision_config": {},
        "gen_aligner_config": {},
        "gen_head_config": {},
    }


def test_resolve_model_class_name_rejects_generic_multi_modality(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_hf_config(
        monkeypatch,
        {
            "model_type": "multi_modality",
            "architectures": ["MultiModalityCausalLM"],
        },
    )

    assert resolve_model_class_name("other-org/multi-modality") == "MultiModalityCausalLM"


def test_resolve_model_class_name_accepts_janus_config(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_hf_config(monkeypatch, _janus_config())

    assert resolve_model_class_name("other-org/local-copy") == "JanusPipeline"


def test_enrich_config_does_not_route_generic_multi_modality_to_janus(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_hf_config(
        monkeypatch,
        {
            "model_type": "multi_modality",
            "architectures": ["MultiModalityCausalLM"],
        },
    )

    config = OmniDiffusionConfig(model="other-org/multi-modality")
    config.enrich_config()

    assert config.model_class_name == "MultiModalityCausalLM"


def test_enrich_config_routes_janus_checkpoint_to_janus_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_hf_config(monkeypatch, _janus_config())

    config = OmniDiffusionConfig(model="deepseek-ai/Janus-Pro-7B")
    config.enrich_config()

    assert config.model_class_name == "JanusPipeline"


def test_resolve_model_config_path_routes_janus_to_single_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "vllm_omni.entrypoints.utils.get_config",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("unknown model_type")),
    )
    monkeypatch.setattr(
        "vllm_omni.entrypoints.utils.file_or_path_exists",
        lambda model, path, revision=None: path == "config.json",
    )
    monkeypatch.setattr(
        "vllm_omni.entrypoints.utils.get_hf_file_to_dict",
        lambda path, model, revision=None: _janus_config() if path == "config.json" else None,
    )

    path = resolve_model_config_path("deepseek-ai/Janus-1.3B")

    assert path.endswith("vllm_omni/deploy/deepseek_janus_single_stage.yaml")


def test_stage_config_factory_routes_janus_to_single_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    StageConfigFactory.try_infer_model_type.cache_clear()

    monkeypatch.setattr(
        StageConfigFactory,
        "get_hf_config",
        classmethod(lambda cls, model, trust_remote_code: None),
    )
    monkeypatch.setattr(
        "vllm_omni.config.config_factory.get_hf_file_to_dict",
        lambda path, model, revision=None: _janus_config() if path == "config.json" else None,
    )

    try:
        model_type = StageConfigFactory.try_infer_model_type("deepseek-ai/Janus-Pro-7B", True)
    finally:
        StageConfigFactory.try_infer_model_type.cache_clear()

    assert model_type == "deepseek_janus_single_stage"
