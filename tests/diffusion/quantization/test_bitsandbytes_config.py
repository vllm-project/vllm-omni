# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for BitsAndBytes quantization config."""

import importlib.util
import sys
import types

import pytest
import torch
from pytest_mock import MockerFixture
from torch.nn import Module, Parameter
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.bitsandbytes import BitsAndBytesConfig

from vllm_omni.platforms import current_omni_platform
from vllm_omni.quantization import build_quant_config
from vllm_omni.quantization.factory import SUPPORTED_QUANTIZATION_METHODS

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]

cuda_available = pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="GPU platform not available.")

bitsandbytes_available = pytest.mark.skipif(
    importlib.util.find_spec("bitsandbytes") is None,
    reason="bitsandbytes package not installed",
)


def _ensure_bitsandbytes_importable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject a stub bitsandbytes package when the optional dep is absent.

    ``mocker.patch("bitsandbytes...")`` imports the target module; without a
    stub, core-model CI without bitsandbytes raises ModuleNotFoundError.
    """
    if importlib.util.find_spec("bitsandbytes") is not None:
        return
    bnb = types.ModuleType("bitsandbytes")
    bnb_functional = types.ModuleType("bitsandbytes.functional")
    bnb.functional = bnb_functional
    monkeypatch.setitem(sys.modules, "bitsandbytes", bnb)
    monkeypatch.setitem(sys.modules, "bitsandbytes.functional", bnb_functional)


def test_bitsandbytes_config_creation():
    """Test that BitsAndBytes config can be created."""
    config = build_quant_config("bitsandbytes")
    assert config is not None
    assert config.get_name() == "bitsandbytes"


def test_bitsandbytes_config_with_custom_params():
    """Test BitsAndBytes config with custom parameters."""
    from vllm_omni.quantization.bitsandbytes_config import DiffusionBitsAndBytesConfig

    config = build_quant_config(
        "bitsandbytes",
        quant_type="fp4",
        compress_statistics=False,
        ignored_layers=["to_out"],
    )
    assert isinstance(config, DiffusionBitsAndBytesConfig)
    assert config.quant_type == "fp4"
    assert config.compress_statistics is False
    assert "to_out" in config.ignored_layers


def test_bitsandbytes_checkpoint_config_uses_vllm_loader():
    """Pre-quantized HF checkpoints must use vLLM's BnB weight loader."""
    checkpoint_config = {
        "quant_method": "bitsandbytes",
        "_load_in_4bit": True,
        "_load_in_8bit": False,
        "load_in_4bit": True,
        "load_in_8bit": False,
        "bnb_4bit_compute_dtype": "bfloat16",
        "bnb_4bit_quant_storage": "uint8",
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "llm_int8_skip_modules": [
            "transformer_blocks.0.attn.to_q",
            "proj_out",
        ],
    }

    config = build_quant_config(checkpoint_config)

    assert isinstance(config, BitsAndBytesConfig)
    assert config.load_in_4bit is True
    assert config.load_in_8bit is False
    assert config.bnb_4bit_compute_dtype == "bfloat16"
    assert config.bnb_4bit_quant_type == "nf4"
    assert config.bnb_4bit_use_double_quant is True
    assert config.llm_int8_skip_modules == [
        "transformer_blocks.0.attn.to_q",
        "proj_out",
    ]


def test_bitsandbytes_online_config_stays_on_diffusion_path():
    from vllm_omni.quantization.bitsandbytes_config import DiffusionBitsAndBytesConfig

    config = build_quant_config(
        {
            "quant_method": "bitsandbytes",
            "quant_type": "nf4",
            "compress_statistics": True,
        }
    )

    assert isinstance(config, DiffusionBitsAndBytesConfig)


def test_bitsandbytes_checkpoint_config_is_auto_detected():
    from vllm_omni.diffusion.data import OmniDiffusionConfig, TransformerConfig

    transformer_config = TransformerConfig.from_dict(
        {
            "quantization_config": {
                "quant_method": "bitsandbytes",
                "_load_in_4bit": True,
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
            }
        }
    )

    config = OmniDiffusionConfig(
        model="test",
        tf_model_config=transformer_config,
    )

    assert isinstance(config.quantization_config, BitsAndBytesConfig)
    assert config.quantization_config.bnb_4bit_quant_type == "nf4"


def test_bitsandbytes_checkpoint_config_skips_fused_linear(mocker):
    config = build_quant_config(
        {
            "quant_method": "bitsandbytes",
            "_load_in_4bit": True,
            "load_in_4bit": True,
            "llm_int8_skip_modules": [
                "transformer_blocks.0.attn.to_q",
                "transformer_blocks.0.attn.to_k",
                "transformer_blocks.0.attn.to_v",
            ],
        }
    )
    config.packed_modules_mapping = {
        "to_qkv": ["to_q", "to_k", "to_v"],
    }

    method = config.get_quant_method(
        mocker.Mock(spec=LinearBase),
        "transformer_blocks.0.attn.to_qkv",
    )

    assert isinstance(method, UnquantizedLinearMethod)


def test_supported_methods():
    """Test that supported methods list includes bitsandbytes."""
    assert "bitsandbytes" in SUPPORTED_QUANTIZATION_METHODS


def test_quantization_integration():
    """Test end-to-end quantization flow through OmniDiffusionConfig."""
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    config = OmniDiffusionConfig(model="test", quantization_config="bitsandbytes")
    assert config.quantization_config is not None
    assert config.quantization_config.get_name() == "bitsandbytes"

    config2 = OmniDiffusionConfig(
        model="test",
        quantization_config={
            "method": "bitsandbytes",
            "quant_type": "nf4",
            "compress_statistics": True,
        },
    )
    assert config2.quantization_config is not None
    assert config2.quantization_config.get_name() == "bitsandbytes"
    assert config2.quantization_config.quant_type == "nf4"
    assert config2.quantization_config.compress_statistics is True


def test_quantization_dict_not_mutated():
    """Test that passing a dict to quantization_config doesn't mutate it."""
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    original_dict = {"method": "bitsandbytes", "quant_type": "nf4"}
    dict_copy = original_dict.copy()

    OmniDiffusionConfig(model="test", quantization_config=original_dict)

    assert original_dict == dict_copy


def test_get_quant_method(mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch):
    """Test get_quant_method routing for CUDA."""
    from vllm_omni.quantization.bitsandbytes_config import BnBOnlineLinearMethod

    config = build_quant_config("bitsandbytes")

    def _fake_init(self, quant_config):
        pass

    layer = mocker.Mock(spec=LinearBase)
    mocker.patch.object(BnBOnlineLinearMethod, "__init__", _fake_init)

    prefix = "test_layer"

    monkeypatch.setattr(current_omni_platform, "is_cuda", lambda: True)
    method = config.get_quant_method(layer, prefix)
    assert isinstance(method, BnBOnlineLinearMethod)

    config.ignored_layers = [prefix]
    method = config.get_quant_method(layer, prefix)
    assert isinstance(method, UnquantizedLinearMethod)


class TestBnBOnlineLinearMethod:
    @pytest.fixture
    def mock_quant_config(self, mocker):
        config = mocker.Mock()
        config.quant_type = "nf4"
        config.compress_statistics = True
        return config

    @pytest.fixture
    def mock_deps(self, mocker, monkeypatch: pytest.MonkeyPatch):
        _ensure_bitsandbytes_importable(monkeypatch)
        mock_qweight = torch.ones((64, 32), dtype=torch.uint8)
        mock_quant_state = mocker.Mock()
        mock_quant = mocker.patch(
            "bitsandbytes.functional.quantize_4bit",
            return_value=(mock_qweight, mock_quant_state),
            create=True,
        )
        # x is (2, 16, 32) -> flattened (32, 32); matmul must return (32, 64)
        # so apply can reshape back to (2, 16, 64).
        mock_matmul = mocker.patch(
            "bitsandbytes.matmul_4bit",
            return_value=torch.randn(32, 64),
            create=True,
        )
        return {
            "quant": mock_quant,
            "matmul": mock_matmul,
            "mock_qweight": mock_qweight,
            "mock_quant_state": mock_quant_state,
        }

    def test_process_weights_after_loading(self, mock_deps, mock_quant_config, mocker):
        from vllm_omni.quantization.bitsandbytes_config import BnBOnlineLinearMethod

        method = BnBOnlineLinearMethod(mock_quant_config)
        layer = Module()
        layer.weight = Parameter(torch.randn(64, 32))
        mocker.patch.object(torch.Tensor, "cuda", return_value=layer.weight.data)

        method.process_weights_after_loading(layer)

        mock_deps["quant"].assert_called_once()
        assert torch.equal(layer.weight.data, mock_deps["mock_qweight"])
        assert layer.quant_state is mock_deps["mock_quant_state"]
        assert layer.bnb_shape == (64, 32)

    def test_apply(self, mock_deps, mock_quant_config):
        from vllm_omni.quantization.bitsandbytes_config import BnBOnlineLinearMethod

        method = BnBOnlineLinearMethod(mock_quant_config)
        layer = Module()
        layer.weight = Parameter(torch.ones((64, 32), dtype=torch.uint8), requires_grad=False)
        layer.quant_state = mock_deps["mock_quant_state"]

        x = torch.randn(2, 16, 32, dtype=torch.float16)
        bias = torch.randn(64)
        output = method.apply(layer, x, bias)

        mock_deps["matmul"].assert_called_once()
        assert output.shape == (2, 16, 64)
        assert output.dtype == torch.float16


@pytest.fixture
def quant_config():
    from vllm_omni.quantization.bitsandbytes_config import DiffusionBitsAndBytesConfig

    return DiffusionBitsAndBytesConfig(
        quant_type="nf4",
        compress_statistics=True,
    )


@cuda_available
@bitsandbytes_available
class TestCudaBnBSmoke:
    """Smoke tests using real bitsandbytes CUDA kernels."""

    @pytest.fixture
    def real_layer(self):
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(
            torch.randn(128, 64, dtype=torch.float16, device="cuda"),
            requires_grad=False,
        )
        layer.logical_widths = [128]
        layer.input_size_per_partition = 64
        layer.output_size_per_partition = 128
        layer.orig_dtype = torch.float16
        return layer

    def test_real_cuda_quantize_4bit_shape_contract(self, quant_config):
        import bitsandbytes.functional as bnb_F

        weight = torch.randn(128, 64, dtype=torch.float16, device="cuda")
        qweight, quant_state = bnb_F.quantize_4bit(
            weight,
            quant_type=quant_config.quant_type,
            compress_statistics=quant_config.compress_statistics,
        )

        assert qweight.dtype == torch.uint8
        assert quant_state is not None

    def test_real_cuda_online_process_weights_after_loading(self, quant_config, real_layer):
        from vllm_omni.quantization.bitsandbytes_config import BnBOnlineLinearMethod

        method = BnBOnlineLinearMethod(quant_config)
        method.process_weights_after_loading(real_layer)

        assert real_layer.weight.dtype == torch.uint8
        assert hasattr(real_layer, "quant_state")
        assert real_layer.bnb_shape == (128, 64)

    def test_real_cuda_bnb_apply_forward(self, quant_config):
        from vllm_omni.quantization.bitsandbytes_config import BnBOnlineLinearMethod

        method = BnBOnlineLinearMethod(quant_config)

        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(
            torch.randn(128, 64, dtype=torch.float16, device="cuda"),
            requires_grad=False,
        )
        method.process_weights_after_loading(layer)

        x = torch.randn(2, 16, 64, dtype=torch.float16, device="cuda")
        output = method.apply(layer, x)

        assert output.shape == (2, 16, 128)
        assert output.dtype == torch.float16
