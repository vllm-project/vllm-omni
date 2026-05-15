# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for Int8 quantization config."""

import pytest
import torch
from pytest_mock import MockerFixture
from torch.nn import Module, Parameter
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod

from vllm_omni.platforms import current_omni_platform
from vllm_omni.quantization import build_quant_config
from vllm_omni.quantization.factory import SUPPORTED_QUANTIZATION_METHODS

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]

npu_available = pytest.mark.skipif(not current_omni_platform.is_npu(), reason="NPU platform not available.")

cuda_available = pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="GPU platform not available.")


def test_int8_config_creation():
    """Test that Int8 config can be created."""
    config = build_quant_config("int8")
    assert config is not None
    assert config.get_name() == "int8"


def test_none_quantization():
    """Test that None quantization returns None config."""
    config = build_quant_config(None)
    assert config is None


def test_invalid_quantization():
    """Test that invalid quantization method raises error."""
    with pytest.raises(ValueError, match="Unknown quantization method"):
        build_quant_config("invalid_method")


def test_int8_config_with_custom_params():
    """Test Int8 config with custom parameters."""
    config = build_quant_config(
        "int8",
        activation_scheme="dynamic",
        ignored_layers=["proj_out"],
    )
    assert config is not None
    assert config.activation_scheme == "dynamic"
    assert "proj_out" in config.ignored_layers


def test_supported_methods():
    """Test that supported methods list is correct."""
    assert "int8" in SUPPORTED_QUANTIZATION_METHODS


def test_quantization_integration():
    """Test end-to-end quantization flow through OmniDiffusionConfig."""
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    # Test with quantization_config string
    config = OmniDiffusionConfig(model="test", quantization_config="int8")
    assert config.quantization_config is not None
    assert config.quantization_config.get_name() == "int8"

    # Test with quantization_config dict
    config2 = OmniDiffusionConfig(
        model="test",
        quantization_config={"method": "int8", "activation_scheme": "dynamic"},
    )
    assert config2.quantization_config is not None
    assert config2.quantization_config.get_name() == "int8"
    assert config2.quantization_config.activation_scheme == "dynamic"


def test_quantization_dict_not_mutated():
    """Test that passing a dict to quantization_config doesn't mutate it."""
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    original_dict = {"method": "int8", "activation_scheme": "dynamic"}
    dict_copy = original_dict.copy()

    OmniDiffusionConfig(model="test", quantization_config=original_dict)

    # Original dict should be unchanged
    assert original_dict == dict_copy


def test_quantization_config_string_and_dict_equivalent():
    """Test that string and dict forms produce equivalent configs."""
    from vllm_omni.diffusion.data import OmniDiffusionConfig

    config_str = OmniDiffusionConfig(model="test", quantization_config="int8")
    config_dict = OmniDiffusionConfig(
        model="test",
        quantization_config={"method": "int8", "activation_scheme": "dynamic"},
    )
    assert config_str.quantization_config.get_name() == config_dict.quantization_config.get_name()
    assert config_str.quantization_config.activation_scheme == config_dict.quantization_config.activation_scheme


def test_get_quant_method(mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch):
    """Test for get_quant_method method for GPU"""
    from vllm_omni.quantization.int8_config import Int8OnlineLinearMethod

    config = build_quant_config("int8")

    def _fake_init(self, quant_config):
        pass

    layer = mocker.Mock(spec=LinearBase)
    mocker.patch.object(Int8OnlineLinearMethod, "__init__", _fake_init)

    prefix = "test_layer"

    # Mock the platform to be GPU
    monkeypatch.setattr(current_omni_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: False)
    method = config.get_quant_method(layer, prefix)
    assert isinstance(method, Int8OnlineLinearMethod)

    # Test skipping quantization for a layer
    config.ignored_layers = [prefix]
    method = config.get_quant_method(layer, prefix)
    assert isinstance(method, UnquantizedLinearMethod)


def test_get_npu_quant_method(mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch):
    """Test for get_quant_method method for NPU"""
    from vllm_omni.quantization.int8_config import NPUInt8OnlineLinearMethod

    config = build_quant_config("int8")

    layer = mocker.Mock(spec=LinearBase)
    prefix = "test_layer"

    # Mock the platform to be NPU
    monkeypatch.setattr(current_omni_platform, "is_cuda", lambda: False)
    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: True)
    method = config.get_quant_method(layer, prefix)
    assert isinstance(method, NPUInt8OnlineLinearMethod)

    # Test skipping quantization for a layer
    config.ignored_layers = [prefix]
    method = config.get_quant_method(layer, prefix)
    assert isinstance(method, UnquantizedLinearMethod)


class TestInt8LinearMethod:
    @pytest.fixture
    def mock_quant_config(self, mocker):
        return mocker.Mock()

    @pytest.fixture
    def mock_kernel(self, mocker):
        kernel = mocker.Mock()
        kernel.process_weights_after_loading = mocker.Mock()
        kernel.apply_weights = mocker.Mock(return_value=torch.randn(1, 10))
        return kernel

    @pytest.fixture
    def patch_deps(self, mocker, mock_kernel):
        # mock init_int8_linear_kernel
        mocker.patch("vllm_omni.quantization.int8_config.init_int8_linear_kernel", return_value=mock_kernel)
        return mock_kernel

    def test_init(self, patch_deps, mock_quant_config):
        # test for Int8LinearMethod init
        from vllm_omni.quantization.int8_config import Int8LinearMethod, init_int8_linear_kernel

        method = Int8LinearMethod(mock_quant_config)

        assert method.quant_config == mock_quant_config
        init_int8_linear_kernel.assert_called_once_with(
            is_channelwise=False, is_static_input_scheme=False, input_symmetric=True, module_name="Int8LinearMethod"
        )
        assert method.int8_linear == patch_deps

    def test_process_weights_after_loading(self, patch_deps, mock_quant_config):
        from vllm_omni.quantization.int8_config import Int8LinearMethod

        method = Int8LinearMethod(mock_quant_config)
        layer = Module()

        method.process_weights_after_loading(layer)
        patch_deps.process_weights_after_loading.assert_called_once_with(layer)

    def test_apply(self, patch_deps, mock_quant_config):
        from vllm_omni.quantization.int8_config import Int8LinearMethod

        method = Int8LinearMethod(mock_quant_config)
        layer = Module()
        x = torch.randn(1, 128)
        bias = torch.randn(128)

        output = method.apply(layer, x, bias)

        patch_deps.apply_weights.assert_called_once_with(layer, x, bias)
        assert isinstance(output, torch.Tensor)


class TestInt8OnlineLinearMethod:
    @pytest.fixture
    def mock_quant_config(self, mocker):
        return mocker.Mock()

    @pytest.fixture
    def mock_deps(self, mocker):
        # mock kernel
        mock_kernel = mocker.Mock()
        mock_kernel.layer_param_names = ("weight", "weight_scale", "input_scale", "input_zero_point", "azp_adj")
        mocker.patch("vllm_omni.quantization.int8_config.init_int8_linear_kernel", return_value=mock_kernel)
        mocker.patch("vllm_omni.quantization.int8_config.replace_parameter")

        # mock scaled_int8_quant return value
        mock_qweight = torch.ones((128, 64), dtype=torch.int8)
        mock_scale = torch.randn(128)
        mock_quant = mocker.patch(
            "vllm_omni.quantization.int8_config.ops.scaled_int8_quant", return_value=(mock_qweight, mock_scale, None)
        )
        return {"kernel": mock_kernel, "quant": mock_quant, "mock_qweight": mock_qweight, "mock_scale": mock_scale}

    def test_process_weights_after_loading(self, mock_deps, mock_quant_config):
        from vllm_omni.quantization.int8_config import Int8OnlineLinearMethod

        method = Int8OnlineLinearMethod(mock_quant_config)
        layer = Module()
        layer.weight = Parameter(torch.randn(128, 64))
        method.process_weights_after_loading(layer)
        mock_deps["quant"].assert_called_once_with(layer.weight, scale=None)


@npu_available
class TestNPUInt8LinearMethod:
    qweight_mock = torch.randn((128, 64)).to(dtype=torch.int8)
    scale_mock = torch.randn(128)
    out_mock = torch.randn((16, 128))

    @pytest.fixture
    def mock_torch_npu(self, mocker):
        torch_npu = mocker.MagicMock()

        mocker.patch("vllm_omni.quantization.int8_config.torch_npu", return_value=torch_npu)
        mocker.patch(
            "vllm_omni.quantization.int8_config.torch_npu.npu_dynamic_quant",
            return_value=(self.qweight_mock, self.scale_mock),
        )
        mocker.patch("vllm_omni.quantization.int8_config.torch_npu.npu_quant_matmul", return_value=self.out_mock)
        return torch_npu

    @pytest.fixture
    def mock_quant_config(self, mocker):
        return mocker.Mock()

    @pytest.fixture
    def mock_layer(self, mocker):
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(self.qweight_mock, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(self.scale_mock, requires_grad=False)
        return layer

    def test_npu_int8_process_weights_after_loading(self, mock_layer, mock_quant_config, mock_torch_npu):
        from vllm_omni.quantization.int8_config import NPUInt8LinearMethod

        method = NPUInt8LinearMethod(mock_quant_config)
        ori_weight_shape = mock_layer.weight.shape

        method.process_weights_after_loading(mock_layer)

        assert mock_layer.weight.shape == ori_weight_shape[::-1]
        assert mock_layer.weight.is_contiguous()

    def test_npu_int8_apply(self, mock_layer, mock_quant_config, mock_torch_npu):
        from vllm_omni.quantization.int8_config import NPUInt8LinearMethod

        method = NPUInt8LinearMethod(mock_quant_config)
        x = torch.randn(1, 16, 64)

        output = method.apply(mock_layer, x)
        assert output.shape == (1, 16, 128)

    def test_npu_int8_online_process_weights(self, mock_layer, mock_quant_config, mock_torch_npu):
        from vllm_omni.quantization.int8_config import NPUInt8OnlineLinearMethod

        method = NPUInt8OnlineLinearMethod(mock_quant_config)
        method.process_weights_after_loading(mock_layer)

        assert mock_layer.weight.shape == (64, 128)
        assert torch.equal(mock_layer.weight_scale, self.scale_mock)


@pytest.fixture
def quant_config():
    """Shared quant config fixture for smoke tests."""
    from vllm_omni.quantization.int8_config import DiffusionInt8Config

    return DiffusionInt8Config(
        is_checkpoint_int8_serialized=False,
        activation_scheme="dynamic",
    )


@npu_available
class TestNPUInt8Smoke:
    """Smoke tests using real torch_npu, only run on NPU."""

    @pytest.fixture
    def real_layer(self):
        """Create a real linear layer with fp16 weights on NPU"""
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(
            torch.randn(128, 64, dtype=torch.float16, device="npu"),
            requires_grad=False,
        )
        layer.logical_widths = [128]
        layer.input_size_per_partition = 64
        layer.output_size_per_partition = 128
        layer.orig_dtype = torch.float16
        return layer

    def test_real_npu_dynamic_quant_shape_contract(self, quant_config, real_layer):
        """Smoke test: verify npu_dynamic_quant returns correct shapes."""
        import torch_npu

        # Call real torch_npu.npu_dynamic_quant
        weight = real_layer.weight
        qweight, scale = torch_npu.npu_dynamic_quant(weight)

        assert qweight.shape == weight.shape
        assert qweight.dtype == torch.int8
        assert scale.shape == (weight.shape[0],)

    def test_real_npu_online_process_weights_after_loading(self, quant_config, real_layer):
        """Smoke test: full process_weights_after_loading with real torch_npu."""
        from vllm_omni.quantization.int8_config import NPUInt8OnlineLinearMethod

        method = NPUInt8OnlineLinearMethod(quant_config)

        method.process_weights_after_loading(real_layer)

        assert real_layer.weight.shape == (64, 128)
        assert real_layer.weight.dtype == torch.int8
        assert hasattr(real_layer, "weight_scale")
        assert real_layer.weight_scale.shape == (128,)

    def test_real_npu_int8_apply_forward(self, quant_config):
        """Smoke test: forward pass with real npu_quant_matmul."""
        import torch_npu

        from vllm_omni.quantization.int8_config import NPUInt8LinearMethod

        method = NPUInt8LinearMethod(quant_config)

        # Create layer with pre-processed weights on NPU
        layer = torch.nn.Module()
        weight_fp16 = torch.randn(128, 64, dtype=torch.float16, device="npu")
        qweight, scale = torch_npu.npu_dynamic_quant(weight_fp16)
        layer.weight = torch.nn.Parameter(qweight.t().contiguous(), requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(scale.squeeze(), requires_grad=False)

        # Forward pass on NPU
        x = torch.randn(2, 16, 64, dtype=torch.float16, device="npu")
        output = method.apply(layer, x)

        assert output.shape == (2, 16, 128)
        assert output.dtype == torch.float16


@cuda_available
class TestCudaInt8Smoke:
    """Smoke tests using real CUDA kernels, only on CUDA"""

    @pytest.fixture
    def real_layer(self):
        """Create a real linear layer with fp16 weights on CUDA"""
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

    def test_real_cuda_scaled_int8_quant_shape_contract(self, quant_config):
        """Smoke test: verify scaled_int8_quant returns correct shapes."""
        from vllm import _custom_ops as ops

        weight = torch.randn(128, 64, dtype=torch.float16, device="cuda")
        qweight, scale, _ = ops.scaled_int8_quant(weight, scale=None)

        assert qweight.shape == weight.shape
        assert qweight.dtype == torch.int8
        assert scale.shape == (weight.shape[0], 1)

    def test_real_cuda_online_process_weights_after_loading(self, quant_config, real_layer):
        """Smoke test: full process_weights_after_loading with real CUDA ops."""
        from vllm_omni.quantization.int8_config import Int8OnlineLinearMethod

        method = Int8OnlineLinearMethod(quant_config)

        method.process_weights_after_loading(real_layer)

        assert real_layer.weight.shape == (64, 128)
        assert real_layer.weight.dtype == torch.int8
        assert hasattr(real_layer, "weight_scale")

    def test_real_cuda_int8_apply_forward(self, quant_config):
        """Smoke test: forward pass with real CUDA int8 kernel."""
        from vllm import _custom_ops as ops

        from vllm_omni.quantization.int8_config import Int8LinearMethod

        method = Int8LinearMethod(quant_config)

        # Create layer with pre-processed weights
        layer = torch.nn.Module()
        weight_fp16 = torch.randn(128, 64, dtype=torch.float16, device="cuda")
        qweight, scale, _ = ops.scaled_int8_quant(weight_fp16, scale=None)
        layer.weight = torch.nn.Parameter(qweight.t(), requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(scale, requires_grad=False)

        # Set required attributes for kernel
        layer.input_scale = None
        layer.input_zero_point = None
        layer.azp_adj = None

        # Forward pass
        x = torch.randn(2, 16, 64, dtype=torch.float16, device="cuda")
        output = method.apply(layer, x)

        assert output.shape == (2, 16, 128)
        assert output.dtype == torch.float16
