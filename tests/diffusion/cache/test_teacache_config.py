# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU tests for TeaCache model-specific defaults (coefficients and estimator adapters)."""

import pytest

from vllm_omni.diffusion.cache.teacache.coefficient_estimator import (
    _MODEL_ADAPTERS,
    DataCollectionHook,
    ZImageAdapter,
)
from vllm_omni.diffusion.cache.teacache.config import _MODEL_COEFFICIENTS, TeaCacheConfig
from vllm_omni.diffusion.cache.teacache.extractors import extract_zimage_context

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_zimage_has_calibrated_coefficients():
    """Z-Image must not fall back to the Qwen-Image placeholder polynomial (#8270)."""
    zimage = _MODEL_COEFFICIENTS["ZImageTransformer2DModel"]
    assert zimage != _MODEL_COEFFICIENTS["QwenImageTransformer2DModel"]
    # Pin the calibrated fit so a recalibration has to update this test deliberately.
    assert zimage == pytest.approx([-7.54613422e01, -9.23596156e01, 5.75318402e01, -3.76790311e00, 2.27176809e-01])

    config = TeaCacheConfig(transformer_type="ZImageTransformer2DModel")
    assert config.coefficients == zimage
    assert config.rel_l1_thresh == 0.2


def test_zimage_estimator_adapter_registered():
    assert _MODEL_ADAPTERS["ZImage"] is ZImageAdapter
    assert ZImageAdapter.model_class_name == "ZImagePipeline"
    assert ZImageAdapter.uses_tf_config is True


def test_data_collection_hook_resolves_extractor_at_init():
    """The estimator hook binds its extractor in __init__, so an unknown type fails before any forward."""
    hook = DataCollectionHook("ZImageTransformer2DModel")
    assert hook.extractor_fn is extract_zimage_context
    assert hook.current_trajectory == []

    with pytest.raises(ValueError, match="Unknown model type"):
        DataCollectionHook("NotARegisteredTransformer")
