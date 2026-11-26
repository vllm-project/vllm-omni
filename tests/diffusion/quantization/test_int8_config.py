# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for Int8 quantization config."""

import math
import types

import pytest
import torch
from pytest_mock import MockerFixture
from torch.nn import Module, Parameter
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod

from vllm_omni.platforms import current_omni_platform
from vllm_omni.quantization import build_quant_config
from vllm_omni.quantization.factory import SUPPORTED_QUANTIZATION_METHODS

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

npu_available = pytest.mark.skipif(not current_omni_platform.is_npu(), reason="NPU platform not available.")

gpu_available = pytest.mark.skipif(
    not (current_omni_platform.is_cuda() or current_omni_platform.is_rocm()),
    reason="CUDA or ROCm platform not available.",
)


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


@pytest.fixture(autouse=True)
def _no_sm100_fallback(request, mocker: MockerFixture):
    """Mocked kernel tests exercise the CUTLASS selection path, also on an SM100 host."""
    if "sm100" in request.node.name or (request.cls is not None and request.cls.__name__ == "TestGPUInt8Smoke"):
        return
    mocker.patch("vllm_omni.quantization.int8_config.sm100_int8_fallback_needed", return_value=False)


@pytest.mark.parametrize("platform", ["cuda", "rocm"])
@pytest.mark.parametrize("serialized", [False, True])
def test_get_quant_method(mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch, platform, serialized):
    """Both GPU backends delegate online and serialized INT8 to vLLM."""
    from vllm_omni.quantization.int8_config import Int8LinearMethod, Int8OnlineLinearMethod

    config = build_quant_config("int8", is_checkpoint_int8_serialized=serialized)
    kernel = mocker.Mock()
    init_kernel = mocker.patch("vllm_omni.quantization.int8_config.init_int8_linear_kernel", return_value=kernel)

    layer = mocker.Mock(spec=LinearBase)
    prefix = "test_layer"

    monkeypatch.setattr(current_omni_platform, "is_cuda", lambda: platform == "cuda")
    monkeypatch.setattr(current_omni_platform, "is_rocm", lambda: platform == "rocm")
    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: False)
    method = config.get_quant_method(layer, prefix)
    expected_method = Int8LinearMethod if serialized else Int8OnlineLinearMethod
    assert type(method) is expected_method
    assert method.int8_linear is kernel
    init_kernel.assert_called_once_with(
        is_channelwise=False,
        is_static_input_scheme=False,
        input_symmetric=True,
        module_name=expected_method.__name__,
    )

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
    monkeypatch.setattr(current_omni_platform, "is_rocm", lambda: False)
    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: True)
    method = config.get_quant_method(layer, prefix)
    assert isinstance(method, NPUInt8OnlineLinearMethod)

    # Test skipping quantization for a layer
    config.ignored_layers = [prefix]
    method = config.get_quant_method(layer, prefix)
    assert isinstance(method, UnquantizedLinearMethod)


def test_fused_layer_can_be_ignored_by_its_prefix(monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture):
    """A layer opts out of quantization under the prefix its module was built with.

    MiniMax H3 stores qkv as one checkpoint tensor, so the fused name is the
    only handle a user has on it.
    """
    from vllm_omni.quantization.int8_config import NPUInt8OnlineLinearMethod

    monkeypatch.setattr(current_omni_platform, "is_cuda", lambda: False)
    monkeypatch.setattr(current_omni_platform, "is_rocm", lambda: False)
    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: True)
    config = build_quant_config("int8", ignored_layers=["blocks.0.attn.qkv_proj"])
    layer = mocker.Mock(spec=LinearBase)

    assert isinstance(config.get_quant_method(layer, "blocks.0.attn.qkv_proj"), UnquantizedLinearMethod)
    assert isinstance(config.get_quant_method(layer, "blocks.1.attn.qkv_proj"), NPUInt8OnlineLinearMethod)
    assert isinstance(config.get_quant_method(layer, "blocks.0.mlp.fc1"), NPUInt8OnlineLinearMethod)


class TestNPUQuantMatmulShapeFallback:
    """npu_quant_matmul rejects wide outputs, so those layers stay unquantized.

    The limit applies to the per-rank shard, which is what the kernel sees. The
    same layer can therefore be quantized at a higher TP degree.
    """

    @pytest.fixture(autouse=True)
    def _mock_tp(self, mocker):
        # The fallback delegates to UnquantizedLinearMethod, which reads the TP
        # group; stand one in so the shape gating can be tested without a
        # distributed init.
        mock_group = mocker.Mock()
        mock_group.rank_in_group = 0
        mocker.patch("vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size", return_value=1)
        mocker.patch("vllm.model_executor.layers.linear.get_tensor_model_parallel_rank", return_value=0)
        mocker.patch("vllm.distributed.parallel_state.get_tp_group", return_value=mock_group)

    @pytest.fixture
    def quant_config(self):
        from vllm_omni.quantization.int8_config import DiffusionInt8Config

        return DiffusionInt8Config(is_checkpoint_int8_serialized=False, activation_scheme="dynamic")

    def create_weights(self, method, layer, output_partition_sizes):
        # Narrow input dim: the fallback path allocates for real, and only the
        # output dimension is under test.
        method.create_weights(
            layer,
            input_size_per_partition=8,
            output_partition_sizes=output_partition_sizes,
            input_size=8,
            output_size=sum(output_partition_sizes),
            params_dtype=torch.bfloat16,
            weight_loader=lambda param, loaded_weight, *args, **kwargs: None,
        )

    @pytest.mark.parametrize(
        "method_name",
        ["NPUInt8LinearMethod", "NPUInt8OnlineLinearMethod"],
    )
    def test_over_wide_layer_falls_back_to_unquantized(self, quant_config, method_name):
        import vllm_omni.quantization.int8_config as int8_config

        method = getattr(int8_config, method_name)(quant_config)
        layer = Module()
        layer.quant_method = method

        self.create_weights(method, layer, [int8_config.NPU_QUANT_MATMUL_MAX_OUT_FEATURES + 1])

        assert isinstance(layer.quant_method, UnquantizedLinearMethod)
        assert layer.weight.dtype == torch.bfloat16

    def test_layer_within_the_limit_keeps_int8(self, quant_config):
        from vllm_omni.quantization.int8_config import (
            NPU_QUANT_MATMUL_MAX_OUT_FEATURES,
            NPUInt8OnlineLinearMethod,
        )

        method = NPUInt8OnlineLinearMethod(quant_config)
        layer = Module()
        layer.quant_method = method

        self.create_weights(method, layer, [NPU_QUANT_MATMUL_MAX_OUT_FEATURES])

        assert layer.quant_method is method

    def test_tensor_parallel_sharding_brings_a_wide_layer_back(self, quant_config):
        from vllm_omni.quantization.int8_config import (
            NPU_QUANT_MATMUL_MAX_OUT_FEATURES,
            NPUInt8OnlineLinearMethod,
        )

        global_out_features = 4 * NPU_QUANT_MATMUL_MAX_OUT_FEATURES
        method = NPUInt8OnlineLinearMethod(quant_config)
        layer = Module()
        layer.quant_method = method

        self.create_weights(method, layer, [global_out_features // 4])

        assert layer.quant_method is method


class TestHostFallbackLoading:
    """Over-wide fallback layers load straight into host memory when the loader
    will offload the model back to CPU after online quantization (DLO).

    Without the context the eager path is preserved: the fallback weight is a
    real tensor created at construction time.
    """

    @pytest.fixture(autouse=True)
    def _mock_tp(self, mocker):
        # Same stand-in as TestNPUQuantMatmulShapeFallback: the eager fallback
        # delegates to UnquantizedLinearMethod, which reads the TP group.
        mock_group = mocker.Mock()
        mock_group.rank_in_group = 0
        mocker.patch("vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size", return_value=1)
        mocker.patch("vllm.model_executor.layers.linear.get_tensor_model_parallel_rank", return_value=0)
        mocker.patch("vllm.distributed.parallel_state.get_tp_group", return_value=mock_group)

    @pytest.fixture
    def quant_config(self):
        from vllm_omni.quantization.int8_config import DiffusionInt8Config

        return DiffusionInt8Config(is_checkpoint_int8_serialized=False, activation_scheme="dynamic")

    def _make_over_wide_layer(self, quant_config, weight_loader):
        from vllm_omni.quantization.int8_config import (
            NPU_QUANT_MATMUL_MAX_OUT_FEATURES,
            NPUInt8OnlineLinearMethod,
        )

        out_features = NPU_QUANT_MATMUL_MAX_OUT_FEATURES + 1
        method = NPUInt8OnlineLinearMethod(quant_config)
        layer = Module()
        layer.quant_method = method
        method.create_weights(
            layer,
            input_size_per_partition=8,
            output_partition_sizes=[out_features],
            input_size=8,
            output_size=out_features,
            params_dtype=torch.bfloat16,
            weight_loader=weight_loader,
        )
        return layer, out_features

    @staticmethod
    def _loaded_values(out_features):
        return torch.arange(out_features * 8, dtype=torch.float32).reshape(out_features, 8).to(torch.bfloat16)

    def test_fallback_without_context_keeps_eager_weights(self, quant_config):
        layer, _ = self._make_over_wide_layer(quant_config, lambda param, loaded_weight, *a, **k: None)

        assert type(layer.quant_method) is UnquantizedLinearMethod
        assert layer.weight.device.type != "meta"

    def test_fallback_under_context_defers_weight_to_meta(self, quant_config):
        from vllm_omni.quantization.int8_config import (
            UnquantizedHostLinearMethod,
            load_unquantizable_fallback_on_cpu,
        )

        with load_unquantizable_fallback_on_cpu():
            layer, _ = self._make_over_wide_layer(quant_config, lambda param, loaded_weight, *a, **k: None)

        assert isinstance(layer.quant_method, UnquantizedHostLinearMethod)
        assert layer.quant_method.uses_meta_device
        assert layer.weight.device.type == "meta"

    def test_fallback_weight_loads_straight_to_cpu(self, quant_config):
        from vllm_omni.quantization.int8_config import load_unquantizable_fallback_on_cpu

        def copy_loader(param, loaded_weight, *args, **kwargs):
            param.data.copy_(loaded_weight)

        with load_unquantizable_fallback_on_cpu():
            layer, out_features = self._make_over_wide_layer(quant_config, copy_loader)

        loaded = self._loaded_values(out_features)
        layer.weight.weight_loader(layer.weight, loaded)

        assert layer.weight.device.type == "cpu"
        assert torch.equal(layer.weight, loaded)
        # Marks the post-load sweep to skip the layer instead of bouncing the
        # host tensor to the accelerator and back for a no-op process call.
        assert layer._already_called_process_weights_after_loading

    def test_chunked_load_marks_completion_only_when_full(self, quant_config):
        from vllm_omni.quantization.int8_config import load_unquantizable_fallback_on_cpu

        state = {"offset": 0}

        def half_loader(param, loaded_weight, *args, **kwargs):
            start = state["offset"]
            param.data[start : start + loaded_weight.shape[0]].copy_(loaded_weight)
            state["offset"] += loaded_weight.shape[0]

        with load_unquantizable_fallback_on_cpu():
            layer, out_features = self._make_over_wide_layer(quant_config, half_loader)

        loaded = self._loaded_values(out_features)
        halves = loaded.chunk(2, dim=0)
        layer.weight.weight_loader(layer.weight, halves[0])
        assert not getattr(layer, "_already_called_process_weights_after_loading", False)

        layer.weight.weight_loader(layer.weight, halves[1])
        assert layer._already_called_process_weights_after_loading
        assert torch.equal(layer.weight, loaded)

    def test_context_does_not_leak(self, quant_config):
        from vllm_omni.quantization.int8_config import load_unquantizable_fallback_on_cpu

        with load_unquantizable_fallback_on_cpu():
            pass
        layer, _ = self._make_over_wide_layer(quant_config, lambda param, loaded_weight, *a, **k: None)

        assert type(layer.quant_method) is UnquantizedLinearMethod


class TestOffloadAfterQuant:
    def test_only_methods_advertising_the_capability_are_asked(self):
        from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
        from vllm_omni.quantization.int8_config import DiffusionInt8Config, NPUInt8OnlineLinearMethod

        class MetaDeviceOnlyMethod:
            """Stands in for upstream online FP8: lazy, but no offload hook."""

            uses_meta_device = True

        model = torch.nn.Sequential(Module(), Module())
        lazy_int8 = NPUInt8OnlineLinearMethod(
            DiffusionInt8Config(is_checkpoint_int8_serialized=False, activation_scheme="dynamic")
        )
        model[0].quant_method = lazy_int8
        model[1].quant_method = MetaDeviceOnlyMethod()

        marked = DiffusersPipelineLoader._request_offload_after_quant(model)

        assert marked == 1
        assert lazy_int8._offload_after_quant


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

    @pytest.mark.parametrize("shape", [(128,), (2, 128), (2, 3, 128), (2, 3, 4, 128)])
    @pytest.mark.parametrize("with_bias", [False, True])
    def test_apply(self, patch_deps, mock_quant_config, shape, with_bias):
        from vllm_omni.quantization.int8_config import Int8LinearMethod

        method = Int8LinearMethod(mock_quant_config)
        layer = Module()
        # Strided inputs exercise the same boundary as batched diffusion states.
        x = torch.arange(2 * math.prod(shape), dtype=torch.float32).reshape(*shape[:-1], 256)[..., ::2]
        weight = torch.arange(128 * 10, dtype=torch.float32).reshape(128, 10) / 128
        bias = torch.arange(10, dtype=torch.float32) if with_bias else None

        def matrix_kernel(actual_layer, matrix, actual_bias):
            assert actual_layer is layer
            assert matrix.ndim == 2
            assert matrix.is_contiguous()
            assert actual_bias is bias
            output = matrix @ weight
            return output if actual_bias is None else output + actual_bias

        patch_deps.apply_weights.side_effect = matrix_kernel

        output = method.apply(layer, x, bias)

        expected = x @ weight
        if bias is not None:
            expected = expected + bias
        torch.testing.assert_close(output, expected)
        patch_deps.apply_weights.assert_called_once()


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


@gpu_available
class TestGPUInt8Smoke:
    """Smoke tests using the upstream CUDA or ROCm INT8 kernels."""

    @pytest.fixture
    def real_layer(self):
        """Create a real linear layer with fp16 weights on the active GPU."""
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

    def test_real_gpu_scaled_int8_quant_shape_contract(self, quant_config):
        """Smoke test: verify scaled_int8_quant returns correct shapes."""
        from vllm import _custom_ops as ops

        weight = torch.randn(128, 64, dtype=torch.float16, device="cuda")
        qweight, scale, _ = ops.scaled_int8_quant(weight, scale=None)

        assert qweight.shape == weight.shape
        assert qweight.dtype == torch.int8
        assert scale.shape == (weight.shape[0], 1)

    def test_real_gpu_online_process_weights_after_loading(self, quant_config, real_layer):
        """Smoke test: full process_weights_after_loading with real GPU ops."""
        from vllm_omni.quantization.int8_config import Int8OnlineLinearMethod

        method = Int8OnlineLinearMethod(quant_config)

        method.process_weights_after_loading(real_layer)

        assert real_layer.weight.shape == (64, 128)
        assert real_layer.weight.dtype == torch.int8
        assert hasattr(real_layer, "weight_scale")

    def test_real_gpu_int8_apply_forward(self, quant_config):
        """Smoke test: batched forward pass with the real GPU INT8 kernel."""
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

        qx, x_scale, _ = ops.scaled_int8_quant(x.reshape(-1, 64), scale=None)
        expected = (qx.float() @ qweight.float().t()) * x_scale * scale.t()
        torch.testing.assert_close(output, expected.reshape(2, 16, 128).to(output.dtype), rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# Static per-tensor activations and SmoothQuant
# ---------------------------------------------------------------------------


def _static_config(**kwargs):
    return build_quant_config("int8", is_checkpoint_int8_serialized=True, **kwargs)


def test_from_config_static_and_smoothquant():
    from vllm_omni.quantization.int8_config import DiffusionInt8Config

    config = DiffusionInt8Config.from_config(
        {"quant_method": "int8", "activation_scheme": "static", "smoothquant": {"alpha": 0.5}}
    )
    assert config.activation_scheme == "static"
    assert config.smoothquant is True
    default = DiffusionInt8Config.from_config({"quant_method": "int8"})
    assert default.activation_scheme == "dynamic"
    assert default.smoothquant is False


def test_unknown_activation_scheme_is_rejected():
    from vllm_omni.quantization.int8_config import DiffusionInt8Config

    with pytest.raises(ValueError, match="Unsupported activation scheme"):
        DiffusionInt8Config.from_config({"quant_method": "int8", "activation_scheme": "per_group"})


def test_static_and_smoothquant_need_serialized_checkpoint():
    from vllm_omni.quantization.int8_config import DiffusionInt8Config

    with pytest.raises(ValueError, match="static"):
        DiffusionInt8Config(is_checkpoint_int8_serialized=False, activation_scheme="static")
    with pytest.raises(ValueError, match="smoothquant"):
        DiffusionInt8Config(is_checkpoint_int8_serialized=False, smoothquant=True)
    with pytest.raises(ValueError, match="smoothquant"):
        DiffusionInt8Config(is_checkpoint_int8_serialized=True, smoothquant="yes")


@pytest.fixture(autouse=False)
def _tp_stub(mocker: MockerFixture):
    group = mocker.Mock()
    group.rank_in_group = 0
    group.world_size = 1
    mocker.patch("vllm.model_executor.parameter.get_tensor_model_parallel_rank", return_value=0, create=True)
    mocker.patch("vllm.model_executor.parameter.get_tensor_model_parallel_world_size", return_value=1, create=True)
    mocker.patch("vllm.distributed.parallel_state.get_tp_group", return_value=group)


def _make_layer(config, mocker: MockerFixture, n=6, k=8, shards=(6,)):
    from vllm_omni.quantization.int8_config import Int8LinearMethod

    kernel = mocker.Mock()
    init_kernel = mocker.patch("vllm_omni.quantization.int8_config.init_int8_linear_kernel", return_value=kernel)
    method = Int8LinearMethod(config)
    layer = Module()
    method.create_weights(
        layer,
        input_size_per_partition=k,
        output_partition_sizes=list(shards),
        input_size=k,
        output_size=n,
        params_dtype=torch.bfloat16,
        weight_loader=None,
    )
    return method, layer, kernel, init_kernel


def test_dynamic_path_registers_no_extra_params(mocker: MockerFixture, _tp_stub):
    method, layer, _, init_kernel = _make_layer(_static_config(), mocker)
    assert layer.weight.dtype == torch.int8
    assert tuple(layer.weight_scale.shape) == (6, 1)
    assert not hasattr(layer, "input_scale")
    assert not hasattr(layer, "pre_quant_scale")
    assert init_kernel.call_args.kwargs["is_static_input_scheme"] is False


def test_static_registers_input_scale_per_shard(mocker: MockerFixture, _tp_stub):
    method, layer, _, init_kernel = _make_layer(
        _static_config(activation_scheme="static"), mocker, n=6, k=8, shards=(2, 2, 2)
    )
    assert init_kernel.call_args.kwargs["is_static_input_scheme"] is True
    assert layer.input_scale.dtype == torch.float32
    assert tuple(layer.input_scale.shape) == (3,)
    assert not hasattr(layer, "pre_quant_scale")


def test_smoothquant_registers_pre_quant_scale_and_loads(mocker: MockerFixture, _tp_stub):
    method, layer, _, _ = _make_layer(_static_config(smoothquant=True), mocker)
    assert layer.pre_quant_scale.dtype == torch.float32
    assert tuple(layer.pre_quant_scale.shape) == (8,)
    values = torch.arange(1, 9, dtype=torch.float32)
    layer.pre_quant_scale.weight_loader(layer.pre_quant_scale, values)
    assert torch.equal(layer.pre_quant_scale.data, values)
    with pytest.raises(ValueError, match="pre_quant_scale"):
        layer.pre_quant_scale.weight_loader(layer.pre_quant_scale, values[:4])


def test_smoothquant_apply_multiplies_input_before_kernel(mocker: MockerFixture, _tp_stub):
    method, layer, kernel, _ = _make_layer(_static_config(smoothquant=True, activation_scheme="static"), mocker)
    pre = torch.tensor([0.5, 1.0, 2.0, 4.0, 0.25, 8.0, 1.0, 1.0])
    layer.pre_quant_scale.data.copy_(pre)
    seen = {}

    def fake_apply(layer_, x2d, bias):
        seen["x"] = x2d
        return torch.zeros(x2d.shape[0], 6, dtype=x2d.dtype)

    kernel.apply_weights.side_effect = fake_apply
    x = torch.randn(2, 3, 8, dtype=torch.bfloat16)
    out = method.apply(layer, x)
    assert out.shape == (2, 3, 6)
    expected = (x * pre.to(x.dtype)).reshape(-1, 8)
    assert seen["x"].dtype == torch.bfloat16
    assert torch.equal(seen["x"], expected)


def test_dynamic_apply_passes_input_unchanged(mocker: MockerFixture, _tp_stub):
    method, layer, kernel, _ = _make_layer(_static_config(), mocker)
    seen = {}

    def fake_apply(layer_, x2d, bias):
        seen["x"] = x2d
        return torch.zeros(x2d.shape[0], 6, dtype=x2d.dtype)

    kernel.apply_weights.side_effect = fake_apply
    x = torch.randn(4, 8, dtype=torch.bfloat16)
    method.apply(layer, x)
    assert torch.equal(seen["x"], x)


def _ref_static_quant(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """CPU reference of symmetric static INT8 quantization."""
    return torch.clamp(torch.round(x.float() / scale), -128, 127).to(torch.int8)


def test_static_quant_reference_math():
    x = torch.tensor([[0.0, 1.0, -2.0, 127.0, -300.0]])
    scale = torch.tensor(1.0)
    assert _ref_static_quant(x, scale).tolist() == [[0, 1, -2, 127, -128]]
    # input_scale = amax / 127 maps amax to 127 and stays within 0.5 step elsewhere.
    x = torch.randn(64, 32)
    scale = x.abs().max() / 127
    q = _ref_static_quant(x, scale)
    assert int(q.abs().max()) == 127
    assert (q.float() * scale - x).abs().max() <= scale / 2 + 1e-6


def test_smoothquant_equivalence():
    """(x * p) @ (W * s)^T == x @ W^T when p = 1 / s."""
    torch.manual_seed(0)
    w = torch.randn(6, 8, dtype=torch.float64)
    x = torch.randn(5, 8, dtype=torch.float64)
    s = torch.rand(8, dtype=torch.float64) + 0.5
    pre = 1.0 / s
    assert torch.allclose((x * pre) @ (w * s).t(), x @ w.t())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="scaled_int8_quant needs CUDA")
def test_scaled_int8_quant_static_matches_reference_cuda():
    from vllm import _custom_ops as ops

    x = torch.randn(16, 64, device="cuda", dtype=torch.bfloat16)
    scale = (x.abs().max().float() / 127).reshape(1).to("cuda")
    q, x_scale, _ = ops.scaled_int8_quant(x, scale)
    ref = _ref_static_quant(x, scale)
    assert (q.int() - ref.int()).abs().max() <= 1
    assert torch.allclose(x_scale.float(), scale)


# ---------------------------------------------------------------------------
# SM100 torch._int_mm fallback
# ---------------------------------------------------------------------------


def _ref_weight_quant(w: torch.Tensor):
    """Per-output-channel INT8 weights, same formula as ops.scaled_int8_quant."""
    amax = w.abs().amax(dim=1, keepdim=True)
    q = torch.round(w * (127.0 / amax)).clamp(-128, 127).to(torch.int8)
    return q, amax / 127.0


def test_sm100_activation_quant_reference_dynamic_and_static():
    from vllm_omni.quantization.int8_sm100 import quantize_activations_reference

    x = torch.randn(5, 16)
    q, s = quantize_activations_reference(x, None)
    assert tuple(s.shape) == (5, 1)
    assert int(q.abs().amax(dim=1).min()) == 127
    assert torch.allclose(q.float() * s, x, atol=float(s.max()) / 2 + 1e-6)
    scale = x.abs().max() / 127
    q, s = quantize_activations_reference(x, scale.reshape(1))
    assert tuple(s.shape) == (1, 1)
    assert int(q.abs().max()) == 127


@pytest.mark.parametrize("static", [False, True])
def test_sm100_epilogue_matches_dequantized_reference(static):
    from vllm_omni.quantization.int8_sm100 import int8_epilogue, quantize_activations_reference

    torch.manual_seed(0)
    x = torch.randn(7, 32)
    w = torch.randn(24, 32)
    bias = torch.randn(24)
    w_q, w_s = _ref_weight_quant(w)
    i_s = (x.abs().max() / 127).reshape(1) if static else None
    x_q, x_s = quantize_activations_reference(x, i_s)
    acc = x_q.int() @ w_q.int().t()  # int32 [M, N]
    out = int8_epilogue(acc, x_s, w_s, bias, torch.float32)
    ref = (x_q.float() * x_s) @ (w_q.float() * w_s).t() + bias
    assert torch.allclose(out, ref, rtol=1e-5, atol=1e-4)
    # And it stays close to the unquantized GEMM.
    assert torch.allclose(out, x @ w.t() + bias, atol=0.5)


@pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10),
    reason="SM100 CUDA device required",
)
@pytest.mark.parametrize("m", [4, 64])
def test_sm100_kernel_matches_dequantized_reference_cuda(m):
    from vllm_omni.quantization.int8_sm100 import Int8Sm100ScaledMMLinearKernel

    torch.manual_seed(0)
    k, n = 256, 128
    w = torch.randn(n, k, device="cuda")
    w_q, w_s = _ref_weight_quant(w)
    layer = Module()
    layer.weight = Parameter(w_q.t(), requires_grad=False)  # [K, N] view, as after CUTLASS processing
    layer.weight_scale = Parameter(w_s, requires_grad=False)
    kernel = object.__new__(Int8Sm100ScaledMMLinearKernel)
    kernel.config = types.SimpleNamespace(input_symmetric=True)
    kernel.layer_param_names = ["weight", "weight_scale", "input_scale", "input_zero_point", "azp_adj"]
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    out = kernel.apply_weights(layer, x)
    ref = x.float() @ (w_q.float() * w_s).t()
    assert out.dtype == torch.bfloat16
    # Activation INT8 rounding noise (max ~0.45 over 8192 outputs, K=256) plus bf16 output rounding.
    assert torch.allclose(out.float(), ref, rtol=3e-2, atol=6e-1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernels need CUDA")
@pytest.mark.parametrize("m,k", [(1, 256), (37, 4096), (300, 12288)])
def test_sm100_triton_quant_matches_scaled_int8_quant_cuda(m, k):
    from vllm import _custom_ops as ops

    from vllm_omni.quantization.int8_sm100_triton import dynamic_per_token_quant

    torch.manual_seed(0)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    q_ref, s_ref, _ = ops.scaled_int8_quant(x, None, None, symmetric=True)
    q, s = dynamic_per_token_quant(x)
    assert torch.equal(q, q_ref)
    assert torch.equal(s, s_ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernels need CUDA")
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("static_scale", [False, True])
def test_sm100_triton_epilogue_matches_torch_epilogue_cuda(with_bias, static_scale):
    from vllm_omni.quantization.int8_sm100 import int8_epilogue
    from vllm_omni.quantization.int8_sm100_triton import int8_scale_epilogue

    torch.manual_seed(0)
    m, n = 53, 300
    acc = torch.randint(-(2**20), 2**20, (m, n), device="cuda", dtype=torch.int32)
    x_s = torch.rand(1 if static_scale else m, 1, device="cuda") + 0.01
    w_s = torch.rand(n, 1, device="cuda") + 0.01
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16) if with_bias else None
    ref = int8_epilogue(acc, x_s, w_s, bias, torch.float32)
    out = int8_scale_epilogue(acc, x_s, w_s, bias, torch.float32)
    assert torch.equal(out, ref)  # FMA is disabled in the kernel: bit-identical
    out_bf16 = int8_scale_epilogue(acc, x_s, w_s, bias, torch.bfloat16)
    assert out_bf16.dtype == torch.bfloat16
    assert torch.equal(out_bf16, int8_epilogue(acc, x_s, w_s, bias, torch.bfloat16))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernels need CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("m,k", [(1, 256), (37, 4096), (300, 12288)])
def test_sm100_triton_quant_fused_pre_quant_scale_bit_identical_cuda(m, k, dtype):
    from vllm_omni.quantization.int8_sm100_triton import dynamic_per_token_quant

    torch.manual_seed(0)
    x = torch.randn(m, k, device="cuda", dtype=dtype)
    pq = torch.rand(k, device="cuda", dtype=torch.float32) * 3 + 0.1
    q_ref, s_ref = dynamic_per_token_quant((x * pq.to(x.dtype)).contiguous())
    q, s = dynamic_per_token_quant(x, pq)
    assert torch.equal(q, q_ref)
    assert torch.equal(s, s_ref)


@pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10),
    reason="SM100 CUDA device required",
)
def test_sm100_kernel_apply_weights_fused_pre_quant_scale_cuda():
    from vllm_omni.quantization.int8_sm100 import Int8Sm100ScaledMMLinearKernel

    torch.manual_seed(0)
    k, n = 512, 256
    w_q, w_s = _ref_weight_quant(torch.randn(n, k, device="cuda"))
    layer = Module()
    layer.weight = Parameter(w_q.t(), requires_grad=False)
    layer.weight_scale = Parameter(w_s, requires_grad=False)
    kernel = object.__new__(Int8Sm100ScaledMMLinearKernel)
    kernel.config = types.SimpleNamespace(input_symmetric=True)
    kernel.layer_param_names = ["weight", "weight_scale", "input_scale", "input_zero_point", "azp_adj"]
    x = torch.randn(64, k, device="cuda", dtype=torch.bfloat16)
    pq = torch.rand(k, device="cuda") + 0.5
    assert kernel.fuses_pre_quant_scale(layer, x)
    fused = kernel.apply_weights(layer, x, pre_quant_scale=pq)
    unfused = kernel.apply_weights(layer, (x * pq.to(x.dtype)).contiguous())
    assert torch.equal(fused, unfused)
