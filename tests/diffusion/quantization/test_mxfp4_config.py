# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for MXFP4 quantization config."""

import pytest
import torch

from vllm_omni.quantization import build_quant_config
from vllm_omni.quantization.factory import SUPPORTED_QUANTIZATION_METHODS
from vllm_omni.quantization.mxfp4_config import (
    DiffusionMXFP4Config,
    DiffusionMXFP4LinearMethod,
    _quantize_bf16_to_mxfp4,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


def test_mxfp4_config_creation():
    """Test that MXFP4 config can be created."""
    config = build_quant_config("mxfp4")
    assert config is not None
    assert config.get_name() == "mxfp4"


def test_mxfp4_config_custom_params():
    """Test MXFP4 config with custom parameters."""
    config = build_quant_config(
        "mxfp4",
        ignored_layers=["proj_out"],
        weight_block_size=32,
    )
    assert config is not None
    assert "proj_out" in config.ignored_layers
    assert config.weight_block_size == [32, 32]  # Converted to list internally


def test_mxfp4_config_weight_block_size_list():
    """Test MXFP4 config with list weight_block_size."""
    config = DiffusionMXFP4Config(weight_block_size=[64, 64])
    assert config.weight_block_size == [64, 64]
    assert config._block_size_scalar == 64


def test_supported_methods():
    """Test that mxfp4 is in supported methods list."""
    assert "mxfp4" in SUPPORTED_QUANTIZATION_METHODS


def test_quantize_bf16_to_mxfp4_basic():
    """Test basic BF16 to MXFP4 quantization."""
    out_features, in_features = 64, 128
    weight = torch.randn(out_features, in_features, dtype=torch.bfloat16)

    qweight, weight_scale = _quantize_bf16_to_mxfp4(weight, block_size=32)

    assert qweight.dtype == torch.uint8
    assert qweight.shape == (out_features, in_features // 2)
    assert weight_scale.dtype == torch.uint8
    assert weight_scale.shape == (out_features, in_features // 32)


def test_quantize_bf16_to_mxfp4_fp16():
    """Test FP16 to MXFP4 quantization."""
    out_features, in_features = 64, 128
    weight = torch.randn(out_features, in_features, dtype=torch.float16)

    qweight, weight_scale = _quantize_bf16_to_mxfp4(weight, block_size=32)

    assert qweight.dtype == torch.uint8
    assert qweight.shape == (out_features, in_features // 2)


def test_quantize_bf16_to_mxfp4_block_size():
    """Test quantization with different block sizes."""
    weight = torch.randn(64, 256, dtype=torch.bfloat16)

    qweight, weight_scale = _quantize_bf16_to_mxfp4(weight, block_size=16)
    assert weight_scale.shape == (64, 256 // 16)

    qweight, weight_scale = _quantize_bf16_to_mxfp4(weight, block_size=64)
    assert weight_scale.shape == (64, 256 // 64)


def test_quantize_bf16_to_mxfp4_zero_weights():
    """Test quantization with zero weights."""
    weight = torch.zeros(32, 64, dtype=torch.bfloat16)

    qweight, weight_scale = _quantize_bf16_to_mxfp4(weight, block_size=32)

    assert qweight.dtype == torch.uint8
    assert weight_scale.dtype == torch.uint8


def test_quantize_bf16_to_mxfp4_large_tensor():
    """Test quantization with large tensor (simulating real model size)."""
    out_features, in_features = 5120, 5120
    weight = torch.randn(out_features, in_features, dtype=torch.bfloat16)

    qweight, weight_scale = _quantize_bf16_to_mxfp4(weight, block_size=32)

    assert qweight.shape == (out_features, in_features // 2)
    assert weight_scale.shape == (out_features, in_features // 32)


def test_mxfp4_linear_method_creation():
    """Test MXFP4 Linear Method creation."""
    config = DiffusionMXFP4Config()
    method = DiffusionMXFP4LinearMethod(config)
    assert method.quant_config == config


def test_mxfp4_config_get_quant_method_cuda():
    """Test get_quant_method returns LinearMethod on CUDA."""
    from unittest.mock import patch

    config = DiffusionMXFP4Config()

    with patch("vllm_omni.quantization.mxfp4_config.current_omni_platform") as mock_platform:
        mock_platform.is_cuda.return_value = True
        mock_platform._omni_enum.value = "cuda"

        from vllm.model_executor.layers.linear import LinearBase

        class MockLinear(LinearBase):
            def __init__(self):
                pass

        method = config.get_quant_method(MockLinear(), "test_layer")
        assert isinstance(method, DiffusionMXFP4LinearMethod)


def test_mxfp4_config_ignored_layers():
    """Test that ignored layers return UnquantizedLinearMethod."""
    from unittest.mock import patch

    config = DiffusionMXFP4Config(ignored_layers=["proj_out"])

    with patch("vllm_omni.quantization.mxfp4_config.current_omni_platform") as mock_platform:
        mock_platform.is_cuda.return_value = True
        mock_platform._omni_enum.value = "cuda"

        from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod

        class MockLinear(LinearBase):
            def __init__(self):
                pass

        # Should return UnquantizedLinearMethod for ignored layer
        method = config.get_quant_method(MockLinear(), "proj_out")
        assert isinstance(method, UnquantizedLinearMethod)

        # Should return MXFP4 method for non-ignored layer
        method = config.get_quant_method(MockLinear(), "blocks.0.attn1.to_qkv")
        assert isinstance(method, DiffusionMXFP4LinearMethod)


def test_mxfp4_config_not_implemented_on_non_cuda():
    """Test that MXFP4 raises NotImplementedError on non-CUDA platforms."""
    from unittest.mock import patch

    config = DiffusionMXFP4Config()

    with patch("vllm_omni.quantization.mxfp4_config.current_omni_platform") as mock_platform:
        mock_platform.is_cuda.return_value = False
        mock_platform._omni_enum.value = "npu"

        from vllm.model_executor.layers.linear import LinearBase

        class MockLinear(LinearBase):
            def __init__(self):
                pass

        with pytest.raises(NotImplementedError, match="MXFP4 is not supported"):
            config.get_quant_method(MockLinear(), "test_layer")


def test_mxfp4_config_from_config():
    """Test config creation from dict."""
    config = DiffusionMXFP4Config.from_config(
        {"ignored_layers": ["proj_out"], "weight_block_size": 64}
    )
    assert "proj_out" in config.ignored_layers
    assert config.weight_block_size == [64, 64]


def test_mxfp4_config_from_config_defaults():
    """Test config creation with defaults."""
    config = DiffusionMXFP4Config.from_config({})
    assert config.ignored_layers == []
    assert config.weight_block_size == [32, 32]


def test_mxfp4_config_get_min_capability():
    """Test min capability is 75 (Turing)."""
    assert DiffusionMXFP4Config.get_min_capability() == 75


def test_mxfp4_config_get_supported_act_dtypes():
    """Test supported activation dtypes."""
    dtypes = DiffusionMXFP4Config.get_supported_act_dtypes()
    assert torch.bfloat16 in dtypes
    assert torch.float16 in dtypes
