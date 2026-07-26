from unittest.mock import Mock, patch

import pytest
import torch

from vllm_omni.diffusion.models.flux2.pipeline_flux2 import (
    Flux2Pipeline,
    _resolve_component_quant_config,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_resolve_component_quant_config_routes_components():
    text_encoder_config = object()
    component_config = Mock()
    component_config.resolve.side_effect = {
        "text_encoder": text_encoder_config,
        "transformer": None,
    }.get

    assert _resolve_component_quant_config(component_config, "text_encoder") is text_encoder_config
    assert _resolve_component_quant_config(component_config, "transformer") is None


def test_resolve_component_quant_config_preserves_global_config():
    global_config = object()
    assert _resolve_component_quant_config(global_config, "transformer") is global_config
    assert _resolve_component_quant_config(None, "text_encoder") is None


def test_offload_transformer_before_text_encoder_weights():
    pipeline = object.__new__(Flux2Pipeline)
    pipeline.transformer = Mock()
    weights = [
        ("transformer.layer.weight", torch.empty(1)),
        ("text_encoder.layer.weight", torch.empty(1)),
        ("text_encoder.layer.bias", torch.empty(1)),
    ]

    iterator = iter(pipeline._offload_transformer_before_text_encoder(weights))
    assert next(iterator)[0] == "transformer.layer.weight"
    pipeline.transformer.to.assert_not_called()

    with patch("vllm_omni.diffusion.models.flux2.pipeline_flux2.current_omni_platform.empty_cache") as empty_cache:
        assert next(iterator)[0] == "text_encoder.layer.weight"
        assert next(iterator)[0] == "text_encoder.layer.bias"

    pipeline.transformer.to.assert_called_once_with("cpu")
    empty_cache.assert_called_once_with()
