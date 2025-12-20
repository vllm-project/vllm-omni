# SPDX-License-Identifier: Apache-2.0

import pytest


@pytest.mark.unit
def test_omni_model_registry_includes_bagel() -> None:
    # Importing the registry should be cheap and should not instantiate any model.
    from vllm.model_executor.models.registry import _LazyRegisteredModel

    from vllm_omni.model_executor.models.registry import OmniModelRegistry

    bagel = OmniModelRegistry.models["BagelForConditionalGeneration"]
    assert isinstance(bagel, _LazyRegisteredModel)
    assert bagel.module_name == "vllm_omni.model_executor.models.bagel.bagel"
    assert bagel.class_name == "BagelForConditionalGeneration"

    qwen2_bagel = OmniModelRegistry.models["Qwen2BagelForCausalLM"]
    assert isinstance(qwen2_bagel, _LazyRegisteredModel)
    assert qwen2_bagel.module_name == "vllm_omni.model_executor.models.bagel.qwen2_bagel"
    # In vllm-omni we reuse the local Qwen2 implementation for BAGEL weights.
    assert qwen2_bagel.class_name == "Qwen2ForCausalLM"
