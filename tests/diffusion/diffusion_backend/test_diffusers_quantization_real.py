# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional real-backend checks for Diffusers quantization config conversion."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _load_quantization_utils():
    module_path = (
        Path(__file__).resolve().parents[3] / "vllm_omni/diffusion/models/diffusers_adapter/quantization_utils.py"
    )
    spec = importlib.util.spec_from_file_location("diffusers_quantization_utils_real_test", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_convert_diffusers_quantization_config_from_real_torchao_dict():
    diffusers = pytest.importorskip("diffusers")
    torchao_quantization = pytest.importorskip("torchao.quantization")
    quant_type_cls = getattr(torchao_quantization, "Int8DynamicActivationInt8WeightConfig", None)
    if quant_type_cls is None:
        pytest.skip("torchao does not expose Int8DynamicActivationInt8WeightConfig")

    quantization_utils = _load_quantization_utils()

    real_torchao_config = diffusers.TorchAoConfig(quant_type=quant_type_cls())
    serialized_torchao_config = real_torchao_config.to_dict()
    assert serialized_torchao_config["quant_type"].keys() == {"default"}

    load_kwargs = {
        "quantization_config": {
            "quant_mapping": {
                "transformer": serialized_torchao_config,
            },
        },
    }

    quantization_utils.convert_diffusers_quantization_config(load_kwargs)

    quant_config = load_kwargs["quantization_config"]
    component_config = quant_config.quant_mapping["transformer"]
    assert isinstance(quant_config, diffusers.PipelineQuantizationConfig)
    assert isinstance(component_config, diffusers.TorchAoConfig)
    assert isinstance(
        component_config.quant_type,
        quant_type_cls,
    )
