# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
import types
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

if "diffusers.models.transformers.transformer_glm_image" not in sys.modules:
    glm_image_module = types.ModuleType("diffusers.models.transformers.transformer_glm_image")

    class GlmImageCombinedTimestepSizeEmbeddings:
        def __init__(self, *args, **kwargs):
            del args, kwargs

    class GlmImageTransformer2DModel:
        pass

    glm_image_module.GlmImageCombinedTimestepSizeEmbeddings = GlmImageCombinedTimestepSizeEmbeddings
    glm_image_module.GlmImageTransformer2DModel = GlmImageTransformer2DModel
    sys.modules["diffusers.models.transformers.transformer_glm_image"] = glm_image_module

import vllm_omni.diffusion.cache.teacache.coefficient_estimator as coefficient_estimator
from vllm_omni.diffusion.cache.teacache.coefficient_estimator import (
    DataCollectionHook,
    TeaCacheCoefficientEstimator,
    _build_glm_prior_token_ids,
    _resolve_glm_transformer_config_path,
)
from vllm_omni.diffusion.cache.teacache.config import _MODEL_COEFFICIENTS, TeaCacheConfig
from vllm_omni.diffusion.cache.teacache.extractors import extract_glmimage_context, get_extractor
from vllm_omni.diffusion.request import OmniDiffusionRequest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_data_collection_hook_new_forward_casts_bfloat16_before_numpy():
    hidden_out = torch.tensor([[5.0, 6.0]], dtype=torch.bfloat16)
    encoder_out = torch.tensor([[7.0, 8.0]], dtype=torch.bfloat16)
    ctx = SimpleNamespace(
        modulated_input=torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16),
        hidden_states=torch.tensor([[3.0, 4.0]], dtype=torch.bfloat16),
        encoder_hidden_states=torch.tensor([[0.0, 0.0]], dtype=torch.bfloat16),
        run_transformer_blocks=lambda: (hidden_out, encoder_out),
        postprocess=Mock(side_effect=lambda h: ("ok", h.clone())),
    )

    hook = DataCollectionHook("GlmImageTransformer2DModel")
    hook.extractor_fn = lambda module, *args, **kwargs: ctx

    result = hook.new_forward(object())

    assert result[0] == "ok"
    ctx.postprocess.assert_called_once()
    postprocess_hidden = ctx.postprocess.call_args.args[0]
    assert torch.equal(postprocess_hidden, hidden_out)
    assert torch.equal(ctx.hidden_states, hidden_out)
    assert torch.equal(ctx.encoder_hidden_states, encoder_out)

    assert len(hook.current_trajectory) == 1
    modulated_np, output_np = hook.current_trajectory[0]
    assert modulated_np.dtype == np.float32
    assert output_np.dtype == np.float32
    np.testing.assert_allclose(modulated_np, np.array([[1.0, 2.0]], dtype=np.float32))
    np.testing.assert_allclose(output_np, np.array([[5.0, 6.0]], dtype=np.float32))


def test_collect_from_prompt_builds_request_and_appends_trajectory():
    estimator = TeaCacheCoefficientEstimator.__new__(TeaCacheCoefficientEstimator)
    estimator.hook = Mock()
    estimator.pipeline = Mock()
    estimator.transformer_type = "Bagel"
    estimator.collected_data = []
    trajectory = [(np.array([1.0], dtype=np.float32), np.array([2.0], dtype=np.float32))]
    estimator.hook.stop_collection.return_value = trajectory

    estimator.collect_from_prompt("hello", num_inference_steps=7, seed=123)

    estimator.hook.start_collection.assert_called_once_with()
    estimator.hook.stop_collection.assert_called_once_with()
    estimator.pipeline.forward.assert_called_once()

    req = estimator.pipeline.forward.call_args.args[0]
    assert isinstance(req, OmniDiffusionRequest)
    assert req.prompts == ["hello"]
    assert req.sampling_params.num_inference_steps == 7
    assert req.sampling_params.seed == 123
    assert estimator.collected_data == [trajectory]


def test_collect_from_prompt_skips_empty_trajectory():
    estimator = TeaCacheCoefficientEstimator.__new__(TeaCacheCoefficientEstimator)
    estimator.hook = Mock()
    estimator.pipeline = Mock()
    estimator.transformer_type = "Bagel"
    estimator.collected_data = []
    estimator.hook.stop_collection.return_value = []

    estimator.collect_from_prompt("hello")

    req = estimator.pipeline.forward.call_args.args[0]
    assert req.prompts == ["hello"]
    assert req.sampling_params.num_inference_steps == 20
    assert req.sampling_params.seed == 42
    assert estimator.collected_data == []


def test_collect_from_prompt_glmimage_injects_synthetic_prior_tokens():
    estimator = TeaCacheCoefficientEstimator.__new__(TeaCacheCoefficientEstimator)
    estimator.hook = Mock()
    estimator.pipeline = SimpleNamespace(
        default_sample_size=128,
        vae_scale_factor=8,
        _patch_size=2,
        transformer=SimpleNamespace(prior_token_embedding=SimpleNamespace(num_embeddings=11)),
        forward=Mock(),
    )
    estimator.transformer_type = "GlmImageTransformer2DModel"
    estimator.collected_data = []
    trajectory = [(np.array([1.0], dtype=np.float32), np.array([2.0], dtype=np.float32))]
    estimator.hook.stop_collection.return_value = trajectory

    estimator.collect_from_prompt("hello", num_inference_steps=7, seed=123, height=256, width=512)

    req = estimator.pipeline.forward.call_args.args[0]
    assert req.sampling_params.height == 256
    assert req.sampling_params.width == 512
    assert "prior_token_ids" in req.sampling_params.extra_args
    prior_token_ids = req.sampling_params.extra_args["prior_token_ids"]
    assert prior_token_ids.shape == (1, 512)
    assert prior_token_ids.dtype == torch.long
    assert torch.all(prior_token_ids >= 0)
    assert torch.all(prior_token_ids < 11)
    assert estimator.collected_data == [trajectory]


def test_build_glm_prior_token_ids_is_deterministic_and_in_range():
    pipeline = SimpleNamespace(
        vae_scale_factor=8,
        _patch_size=2,
        transformer=SimpleNamespace(prior_token_embedding=SimpleNamespace(num_embeddings=17)),
    )

    token_ids_a = _build_glm_prior_token_ids(pipeline, seed=7, height=256, width=256)
    token_ids_b = _build_glm_prior_token_ids(pipeline, seed=7, height=256, width=256)

    assert token_ids_a.shape == (1, 256)
    assert torch.equal(token_ids_a, token_ids_b)
    assert torch.all(token_ids_a >= 0)
    assert torch.all(token_ids_a < 17)


def test_resolve_glm_transformer_config_path_prefers_local_dir():
    with patch.object(coefficient_estimator.os.path, "isdir", return_value=True):
        path = _resolve_glm_transformer_config_path("/tmp/glm-model")

    assert path == "/tmp/glm-model/transformer/config.json"


def test_resolve_glm_transformer_config_path_uses_hub_for_remote():
    with (
        patch.object(coefficient_estimator.os.path, "isdir", return_value=False),
        patch.object(coefficient_estimator, "hf_hub_download", return_value="/tmp/downloaded-config.json") as download,
    ):
        path = _resolve_glm_transformer_config_path("zai-org/GLM-Image")

    download.assert_called_once_with("zai-org/GLM-Image", "transformer/config.json")
    assert path == "/tmp/downloaded-config.json"


def test_get_extractor_returns_glmimage_extractor():
    assert get_extractor("GlmImageTransformer2DModel") is extract_glmimage_context


def test_teacache_config_uses_glmimage_default_coefficients():
    config = TeaCacheConfig(transformer_type="GlmImageTransformer2DModel", coefficients=None)
    assert config.coefficients == _MODEL_COEFFICIENTS["GlmImageTransformer2DModel"]
