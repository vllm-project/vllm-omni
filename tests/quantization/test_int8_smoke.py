# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Smoke tests for Int8 quantization on real hardware (CUDA / NPU).

These tests exercise the actual quantization kernels and require a GPU.
For pure config/factory unit tests, see test_int8_config.py.
"""

import pytest
import torch

from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.cuda, pytest.mark.L4]

npu_available = pytest.mark.skipif(
    not current_omni_platform.is_npu(),
    reason="NPU platform not available.",
)

cuda_available = pytest.mark.skipif(
    not current_omni_platform.is_cuda(),
    reason="GPU platform not available.",
)


@pytest.fixture
def quant_config():
    """Shared quant config fixture for smoke tests."""
    from vllm_omni.quantization.int8_config import DiffusionInt8Config

    return DiffusionInt8Config(
        is_checkpoint_int8_serialized=False,
        activation_scheme="dynamic",
    )


@npu_available
class TestNPUInt8LinearMethod:
    qweight_mock = torch.randn((128, 64)).to(dtype=torch.int8)
    scale_mock = torch.randn(128)
    out_mock = torch.randn((16, 128))

    @pytest.fixture
    def mock_torch_npu(self, mocker):
        torch_npu = mocker.MagicMock()

        mocker.patch(
            "vllm_omni.quantization.int8_config.torch_npu",
            return_value=torch_npu,
        )
        mocker.patch(
            "vllm_omni.quantization.int8_config.torch_npu.npu_dynamic_quant",
            return_value=(self.qweight_mock, self.scale_mock),
        )
        mocker.patch(
            "vllm_omni.quantization.int8_config.torch_npu.npu_quant_matmul",
            return_value=self.out_mock,
        )
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

        layer = torch.nn.Module()
        weight_fp16 = torch.randn(128, 64, dtype=torch.float16, device="npu")
        qweight, scale = torch_npu.npu_dynamic_quant(weight_fp16)
        layer.weight = torch.nn.Parameter(qweight.t().contiguous(), requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(scale.squeeze(), requires_grad=False)

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

        layer = torch.nn.Module()
        weight_fp16 = torch.randn(128, 64, dtype=torch.float16, device="cuda")
        qweight, scale, _ = ops.scaled_int8_quant(weight_fp16, scale=None)
        layer.weight = torch.nn.Parameter(qweight.t(), requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(scale, requires_grad=False)

        layer.input_scale = None
        layer.input_zero_point = None
        layer.azp_adj = None

        x = torch.randn(2, 16, 64, dtype=torch.float16, device="cuda")
        output = method.apply(layer, x)

        assert output.shape == (2, 16, 128)
        assert output.dtype == torch.float16
