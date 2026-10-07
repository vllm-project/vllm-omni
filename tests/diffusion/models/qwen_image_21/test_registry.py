# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""L1 tests for checkpoint-to-pipeline routing of Qwen-Image 2.1."""

import json

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_registry_entry_loads_the_pipeline_class():
    from vllm_omni.diffusion.models.qwen_image_21 import QwenImage21Pipeline
    from vllm_omni.diffusion.registry import DiffusionModelRegistry

    assert DiffusionModelRegistry._try_load_model_cls("QwenImage21Pipeline") is QwenImage21Pipeline


def test_pipeline_declares_request_batching_and_step_execution():
    from vllm_omni.diffusion.models.qwen_image_21 import QwenImage21Pipeline

    assert QwenImage21Pipeline.supports_request_batch
    assert QwenImage21Pipeline.supports_step_execution


def test_process_funcs_resolve_to_the_pipeline_module():
    from vllm_omni.diffusion.models.qwen_image_21 import pipeline_qwen_image_21
    from vllm_omni.diffusion.registry import _DIFFUSION_POST_PROCESS_FUNCS, _DIFFUSION_PRE_PROCESS_FUNCS

    pre_name = _DIFFUSION_PRE_PROCESS_FUNCS["QwenImage21Pipeline"]
    post_name = _DIFFUSION_POST_PROCESS_FUNCS["QwenImage21Pipeline"]
    assert pre_name == "get_qwen_image_21_pre_process_func"
    assert post_name == "get_qwen_image_21_post_process_func"
    assert callable(getattr(pipeline_qwen_image_21, pre_name))
    assert callable(getattr(pipeline_qwen_image_21, post_name))


def test_multimodal_metadata_caps_condition_images_at_four():
    from vllm_omni.diffusion.model_metadata import QWEN_IMAGE_21_MAX_INPUT_IMAGES, get_diffusion_model_metadata

    metadata = get_diffusion_model_metadata("QwenImage21Pipeline")

    assert metadata.supports_multimodal_inputs
    assert metadata.max_multimodal_image_inputs == 4
    assert metadata.max_multimodal_image_inputs == QWEN_IMAGE_21_MAX_INPUT_IMAGES


def test_model_index_resolves_and_enriches(tmp_path):
    from vllm_omni.diffusion.data import OmniDiffusionConfig, resolve_model_class_name

    (tmp_path / "model_index.json").write_text(
        json.dumps({"_class_name": "QwenImage21Pipeline"}),
        encoding="utf-8",
    )

    assert resolve_model_class_name(str(tmp_path)) == "QwenImage21Pipeline"

    config = OmniDiffusionConfig(model=str(tmp_path))
    config.enrich_config()

    assert config.model_class_name == "QwenImage21Pipeline"
    assert config.supports_multimodal_inputs
    assert config.max_multimodal_image_inputs == 4
