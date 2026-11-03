# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Checkpoint routing and processing for MOSS-Image."""

import json

import pytest
import torch

from vllm_omni.diffusion.data import OmniDiffusionConfig, resolve_model_class_name
from vllm_omni.diffusion.models.mossimage import MOSSImagePipeline
from vllm_omni.diffusion.registry import (
    DiffusionModelRegistry,
    get_diffusion_post_process_func,
    get_diffusion_pre_process_func,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_checkpoint_selects_native_pipeline_and_processing(tmp_path):
    class_name = "MOSSImagePipeline"
    (tmp_path / "model_index.json").write_text(json.dumps({"_class_name": class_name}), encoding="utf-8")
    assert resolve_model_class_name(str(tmp_path)) == class_name

    config = OmniDiffusionConfig(model=str(tmp_path))
    config.enrich_config()
    assert config.model_class_name == class_name
    assert DiffusionModelRegistry._try_load_model_cls(config.model_class_name) is MOSSImagePipeline

    request = OmniDiffusionRequest(
        prompt="A red cube",
        sampling_params=OmniDiffusionSamplingParams(extra_args={"flow_shift": 5.0}),
        request_id="moss-image",
    )
    preprocess = get_diffusion_pre_process_func(config)
    assert preprocess is not None
    assert preprocess(request) is request
    assert request.batch_compatibility_key is not None

    postprocess = get_diffusion_post_process_func(config)
    assert postprocess is not None
    images = postprocess(torch.zeros(1, 3, 16, 16))
    assert len(images) == 1
    assert images[0].size == (16, 16)
    assert images[0].getpixel((0, 0)) == (128, 128, 128)
