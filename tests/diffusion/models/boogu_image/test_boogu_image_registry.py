# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""L1 tests for checkpoint-to-pipeline routing of Boogu-Image Turbo."""

import json

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_turbo_registry_entry_loads_the_turbo_class():
    from vllm_omni.diffusion.models.boogu_image import BooguImageTurboPipeline
    from vllm_omni.diffusion.registry import DiffusionModelRegistry

    assert DiffusionModelRegistry._try_load_model_cls("BooguImageTurboPipeline") is BooguImageTurboPipeline


def test_turbo_entry_disables_request_batching():
    from vllm_omni.diffusion.models.boogu_image import BooguImagePipeline, BooguImageTurboPipeline

    assert BooguImagePipeline.supports_request_batch
    assert not BooguImageTurboPipeline.supports_request_batch


def test_turbo_process_funcs_resolve_to_the_shared_boogu_helpers():
    from vllm_omni.diffusion.models import boogu_image
    from vllm_omni.diffusion.registry import _DIFFUSION_POST_PROCESS_FUNCS, _DIFFUSION_PRE_PROCESS_FUNCS

    module = boogu_image.pipeline_boogu_image
    for funcs in (_DIFFUSION_PRE_PROCESS_FUNCS, _DIFFUSION_POST_PROCESS_FUNCS):
        name = funcs["BooguImageTurboPipeline"]
        assert name == funcs["BooguImagePipeline"]
        assert callable(getattr(module, name))


def test_turbo_inherits_the_base_multimodal_metadata():
    from vllm_omni.diffusion.model_metadata import get_diffusion_model_metadata

    metadata = get_diffusion_model_metadata("BooguImageTurboPipeline")

    assert metadata == get_diffusion_model_metadata("BooguImagePipeline")
    assert metadata.supports_multimodal_inputs
    assert metadata.max_multimodal_image_inputs == 1


def test_turbo_model_index_resolves_and_enriches(tmp_path):
    from vllm_omni.diffusion.data import OmniDiffusionConfig, resolve_model_class_name

    (tmp_path / "model_index.json").write_text(
        json.dumps({"_class_name": "BooguImageTurboPipeline"}),
        encoding="utf-8",
    )

    assert resolve_model_class_name(str(tmp_path)) == "BooguImageTurboPipeline"

    config = OmniDiffusionConfig(model=str(tmp_path))
    config.enrich_config()

    assert config.model_class_name == "BooguImageTurboPipeline"
    assert config.supports_multimodal_inputs
    assert config.max_multimodal_image_inputs == 1


def test_edit_turbo_override_selects_the_turbo_class(tmp_path):
    """Edit-Turbo publishes ``BooguImagePipeline``; ``--model-class-name`` must win."""
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.registry import DiffusionModelRegistry

    (tmp_path / "model_index.json").write_text(
        json.dumps({"_class_name": "BooguImagePipeline"}),
        encoding="utf-8",
    )

    config = OmniDiffusionConfig(model=str(tmp_path), model_class_name="BooguImageTurboPipeline")
    config.enrich_config()

    assert config.model_class_name == "BooguImageTurboPipeline"
    assert DiffusionModelRegistry._try_load_model_cls(config.model_class_name)._is_turbo


def test_edit_turbo_without_override_keeps_the_base_class(tmp_path):
    from vllm_omni.diffusion.data import OmniDiffusionConfig
    from vllm_omni.diffusion.registry import DiffusionModelRegistry

    (tmp_path / "model_index.json").write_text(
        json.dumps({"_class_name": "BooguImagePipeline"}),
        encoding="utf-8",
    )

    config = OmniDiffusionConfig(model=str(tmp_path))
    config.enrich_config()

    assert config.model_class_name == "BooguImagePipeline"
    assert not DiffusionModelRegistry._try_load_model_cls(config.model_class_name)._is_turbo
