# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
from vllm.model_executor.models.registry import ModelRegistry

from vllm_omni.model_executor.models.registry import OmniModelRegistry

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_breeze_architectures_resolve_to_omni_classes():
    assert "BreezeForConditionalGeneration" in OmniModelRegistry.get_supported_archs()
    assert "BreezeTTS2MimiCodec" in OmniModelRegistry.get_supported_archs()

    talker = OmniModelRegistry._try_load_model_cls("BreezeForConditionalGeneration")
    codec = OmniModelRegistry._try_load_model_cls("BreezeTTS2MimiCodec")
    assert talker.__name__ == "BreezeTTS2TalkerForGeneration"
    assert codec.__name__ == "BreezeTTS2MimiCodec"


def test_plugin_registration_publishes_architectures_to_upstream_registry():
    from vllm_omni.engine.arg_utils import register_omni_models_to_vllm

    register_omni_models_to_vllm()

    assert "BreezeForConditionalGeneration" in ModelRegistry.get_supported_archs()
    assert "BreezeTTS2MimiCodec" in ModelRegistry.get_supported_archs()
