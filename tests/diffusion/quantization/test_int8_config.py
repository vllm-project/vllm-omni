# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for Int8 quantization config."""

import os

import pytest
import torch
import torch.distributed as dist
from pytest_mock import MockerFixture
from torch.nn import Module, Parameter
from vllm.model_executor.layers.linear import (
    LinearBase,
    UnquantizedLinearMethod,
)

import vllm_omni.quantization.int8_config as int8_config
from tests.helpers.mark import hardware_test
from vllm_omni.platforms import current_omni_platform
from vllm_omni.quantization import build_quant_config
from vllm_omni.quantization.factory import SUPPORTED_QUANTIZATION_METHODS

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]

npu_available = pytest.mark.skipif(not current_omni_platform.is_npu(), reason="NPU platform not available.")

gpu_available = pytest.mark.skipif(
    not (current_omni_platform.is_cuda() or current_omni_platform.is_rocm()),
    reason="CUDA or ROCm platform not available.",
)


def _make_int8_layer(weight, *, input_size=None, scale_tp_group=None):
    layer = Module()
    layer.weight = Parameter(weight, requires_grad=False)
    layer.input_size = weight.shape[1] if input_size is None else input_size
    layer.input_size_per_partition = weight.shape[1]
    if scale_tp_group is not None:
        layer._int8_scale_tp_group = scale_tp_group
    return layer


def _distributed_int8_quant_worker(rank: int, init_method: str) -> None:
    # FileStore still asks Gloo for a local interface. Use loopback so this
    # regression does not depend on the host name being resolvable in CI.
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    dist.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=2,
    )
    try:
        full_weight = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],
                [-1.3984375, 2.796875, -2.0, 1.0],
            ],
            dtype=torch.bfloat16,
        )
        local_weight = full_weight.chunk(2, dim=1)[rank].contiguous()
        layer = _make_int8_layer(
            local_weight,
            input_size=full_weight.shape[1],
            scale_tp_group=dist.group.WORLD,
        )

        qweight, scale = int8_config._quantize_input_sharded_weight(layer)

        full_amax = full_weight.abs().amax(dim=1, keepdim=True).float()
        inv_scale = torch.where(
            full_amax == 0,
            torch.zeros_like(full_amax),
            torch.iinfo(torch.int8).max / full_amax,
        )
        expected = full_weight.float().mul(inv_scale).round().clamp(-127, 127).to(torch.int8)
        assert torch.equal(qweight, expected.chunk(2, dim=1)[rank])
        assert torch.equal(scale, full_amax / torch.iinfo(torch.int8).max)
    finally:
        dist.destroy_process_group()


def _distributed_accelerator_int8_quant_worker(rank: int, init_method: str) -> None:
    device = torch.device(current_omni_platform.device_type, rank)
    current_omni_platform.set_device(device)
    dist.init_process_group(
        "nccl",
        init_method=init_method,
        rank=rank,
        world_size=2,
    )
    try:
        full_weight = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],
                [-1.3984375, 2.796875, -2.0, 1.0],
            ],
            dtype=torch.bfloat16,
            device=device,
        )
        full_qweight, full_scale, _ = int8_config.ops.scaled_int8_quant(full_weight, scale=None)
        local_weight = full_weight.chunk(2, dim=1)[rank].contiguous()
        layer = _make_int8_layer(
            local_weight,
            input_size=full_weight.shape[1],
            scale_tp_group=dist.group.WORLD,
        )

        qweight, scale = int8_config._quantize_input_sharded_weight(layer)

        assert torch.equal(qweight, full_qweight.chunk(2, dim=1)[rank])
        assert torch.equal(scale, full_scale)
    finally:
        dist.destroy_process_group()


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


@pytest.mark.parametrize("platform", ["cuda", "rocm"])
def test_get_gpu_quant_method(mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch, platform):
    """CUDA and ROCm share vLLM's upstream INT8 linear method."""
    from vllm_omni.quantization.int8_config import (
        Int8LinearMethod,
        Int8OnlineLinearMethod,
    )

    config = build_quant_config("int8")

    def _fake_init(self, quant_config):
        pass

    layer = mocker.Mock(spec=LinearBase)
    mocker.patch.object(Int8OnlineLinearMethod, "__init__", _fake_init)
    mocker.patch.object(Int8LinearMethod, "__init__", _fake_init)

    prefix = "test_layer"

    monkeypatch.setattr(current_omni_platform, "is_cuda", lambda: platform == "cuda")
    monkeypatch.setattr(current_omni_platform, "is_rocm", lambda: platform == "rocm")
    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: False)
    method = config.get_quant_method(layer, prefix)
    assert isinstance(method, Int8OnlineLinearMethod)

    config.is_checkpoint_int8_serialized = True
    method = config.get_quant_method(layer, prefix)
    assert isinstance(method, Int8LinearMethod)

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
        kernel.apply_weights = mocker.Mock(
            side_effect=lambda layer, x, bias: torch.empty(x.shape[0], 10, dtype=x.dtype)
        )
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

    @pytest.mark.parametrize(
        ("input_shape", "kernel_input_shape", "output_shape"),
        [
            ((1, 128), (1, 128), (1, 10)),
            ((2, 16, 128), (32, 128), (2, 16, 10)),
        ],
    )
    def test_apply(self, patch_deps, mock_quant_config, input_shape, kernel_input_shape, output_shape):
        from vllm_omni.quantization.int8_config import Int8LinearMethod

        method = Int8LinearMethod(mock_quant_config)
        layer = Module()
        x = torch.randn(input_shape)
        bias = torch.randn(10)

        output = method.apply(layer, x, bias)

        kernel_input = patch_deps.apply_weights.call_args.args[1]
        assert kernel_input.shape == kernel_input_shape
        assert patch_deps.apply_weights.call_args.args[0] is layer
        assert patch_deps.apply_weights.call_args.args[2] is bias
        assert output.shape == output_shape
        assert output.dtype == x.dtype


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
        layer = _make_int8_layer(torch.randn(128, 64))
        method.process_weights_after_loading(layer)
        mock_deps["quant"].assert_called_once_with(layer.weight, scale=None)


@pytest.mark.cpu
class TestInt8OnlineTensorParallelScales:
    @staticmethod
    def _make_method(mocker, method_name):
        kernel = mocker.Mock()
        kernel.layer_param_names = ("weight", "weight_scale", "input_scale", "input_zero_point", "azp_adj")
        mocker.patch.object(int8_config, "init_int8_linear_kernel", return_value=kernel)
        return getattr(int8_config, method_name)(mocker.Mock())

    @pytest.mark.parametrize(
        ("input_size_per_partition", "expected"),
        [(2, True), (4, False)],
    )
    def test_create_weights_records_full_and_local_input_sizes(self, mocker, input_size_per_partition, expected):
        mocker.patch("vllm.model_executor.parameter.get_tensor_model_parallel_rank", return_value=0)
        mocker.patch("vllm.model_executor.parameter.get_tensor_model_parallel_world_size", return_value=1)
        layer = Module()
        method = self._make_method(mocker, "Int8OnlineLinearMethod")

        method.create_weights(
            layer,
            input_size_per_partition=input_size_per_partition,
            output_partition_sizes=[2],
            input_size=4,
            output_size=2,
            params_dtype=torch.bfloat16,
            weight_loader=mocker.Mock(),
        )

        assert layer.input_size == 4
        assert layer.input_size_per_partition == input_size_per_partition
        assert (layer.input_size != layer.input_size_per_partition) is expected

    @pytest.mark.parametrize("method_name", ["Int8OnlineLinearMethod", "NPUInt8OnlineLinearMethod"])
    def test_bare_input_sharded_weight_matches_unsharded_quantization(self, mocker, method_name):
        full_weight = torch.tensor(
            [[1, -2, 8, -64], [2, -3, 7, -56]],
            dtype=torch.bfloat16,
        )
        int8_max = torch.iinfo(torch.int8).max
        full_amax = full_weight.abs().amax(dim=1, keepdim=True).float()
        full_scale = full_amax / int8_max
        full_inv_scale = int8_max / full_amax
        full_qweight = full_weight.float().mul(full_inv_scale).round().clamp(-int8_max, int8_max).to(torch.int8)

        # BAGEL's gen_exp is a plain Module that owns an input-sharded weight.
        local_layer = _make_int8_layer(full_weight[:, :2], input_size=full_weight.shape[1])
        local_amax = local_layer.weight.abs().amax(dim=1, keepdim=True).float()
        local_scale = local_amax / int8_max
        local_qweight = (
            local_layer.weight.float().mul(int8_max / local_amax).round().clamp(-int8_max, int8_max).to(torch.int8)
        )
        if method_name == "Int8OnlineLinearMethod":
            native_quant = mocker.patch.object(
                int8_config.ops,
                "scaled_int8_quant",
                return_value=(local_qweight, local_scale, None),
            )
        else:
            torch_npu = mocker.Mock()
            native_quant = torch_npu.npu_dynamic_quant
            native_quant.return_value = (local_qweight, local_scale.squeeze(-1))
            mocker.patch.object(int8_config, "torch_npu", torch_npu)

        tp_group = mocker.Mock()
        tp_group.device_group = mocker.sentinel.tp_group
        mocker.patch("vllm_omni.quantization.int8_config.get_tp_group", return_value=tp_group)

        def max_reduce(row_amax, op, group):
            assert op is torch.distributed.ReduceOp.MAX
            assert group is tp_group.device_group
            row_amax.copy_(full_amax)

        all_reduce = mocker.patch("torch.distributed.all_reduce", side_effect=max_reduce)

        method = self._make_method(mocker, method_name)
        method.process_weights_after_loading(local_layer)

        all_reduce.assert_called_once()
        native_quant.assert_not_called()
        assert torch.equal(local_layer.weight.t(), full_qweight[:, :2])
        expected_scale = full_scale if method_name == "Int8OnlineLinearMethod" else full_scale.squeeze(-1)
        assert torch.equal(local_layer.weight_scale, expected_scale)

    @pytest.mark.parametrize("method_name", ["Int8OnlineLinearMethod", "NPUInt8OnlineLinearMethod"])
    def test_input_sharded_weight_uses_explicit_group(self, mocker, method_name):
        weight = torch.tensor([[1, -2]], dtype=torch.bfloat16)
        scale_tp_group = mocker.sentinel.minimax_text_encoder_group
        layer = _make_int8_layer(weight, input_size=4, scale_tp_group=scale_tp_group)

        if method_name == "Int8OnlineLinearMethod":
            native_quant = mocker.patch.object(int8_config.ops, "scaled_int8_quant")
        else:
            torch_npu = mocker.Mock()
            native_quant = torch_npu.npu_dynamic_quant
            mocker.patch.object(int8_config, "torch_npu", torch_npu)

        get_tp_group = mocker.patch("vllm_omni.quantization.int8_config.get_tp_group")
        all_reduce = mocker.patch("torch.distributed.all_reduce")

        method = self._make_method(mocker, method_name)
        method.process_weights_after_loading(layer)

        get_tp_group.assert_not_called()
        native_quant.assert_not_called()
        all_reduce.assert_called_once()
        assert all_reduce.call_args.kwargs["group"] is scale_tp_group

    @pytest.mark.parametrize("method_name", ["Int8OnlineLinearMethod", "NPUInt8OnlineLinearMethod"])
    def test_non_input_sharded_weight_does_not_reduce(self, mocker, method_name):
        local_weight = torch.tensor([[1, -2, 8, -64]], dtype=torch.bfloat16)
        layer = _make_int8_layer(local_weight)
        qweight = torch.ones_like(local_weight, dtype=torch.int8)
        weight_scale = torch.ones((1, 1), dtype=torch.float32)

        if method_name == "Int8OnlineLinearMethod":
            native_quant = mocker.patch.object(
                int8_config.ops,
                "scaled_int8_quant",
                return_value=(qweight, weight_scale, None),
            )
        else:
            weight_scale = weight_scale.squeeze(-1)
            torch_npu = mocker.Mock()
            native_quant = torch_npu.npu_dynamic_quant
            native_quant.return_value = (qweight, weight_scale)
            mocker.patch.object(int8_config, "torch_npu", torch_npu)

        all_reduce = mocker.patch("torch.distributed.all_reduce")
        method = self._make_method(mocker, method_name)
        original_weight = layer.weight
        method.process_weights_after_loading(layer)

        if method_name == "Int8OnlineLinearMethod":
            native_quant.assert_called_once_with(original_weight, scale=None)
        else:
            native_quant.assert_called_once_with(original_weight)
        all_reduce.assert_not_called()

    def test_shared_quantizer_matches_native_rounding_and_zero_row(self, mocker):
        weight = torch.tensor(
            [
                [0.0, 0.0],
                [-1.3984375, 2.796875],
            ],
            dtype=torch.bfloat16,
        )
        layer = _make_int8_layer(
            weight,
            input_size=4,
            scale_tp_group=mocker.sentinel.tp_group,
        )
        mocker.patch("torch.distributed.all_reduce")

        qweight, scale = int8_config._quantize_input_sharded_weight(layer)

        assert torch.equal(
            qweight,
            torch.tensor([[0, 0], [-64, 127]], dtype=torch.int8),
        )
        assert torch.equal(
            scale,
            torch.tensor([[0.0], [2.796875 / 127]], dtype=torch.float32),
        )

    def test_shared_quantizer_runs_real_two_rank_collective(self, tmp_path):
        init_method = f"file://{tmp_path / 'int8-gloo-init'}"
        torch.multiprocessing.spawn(
            _distributed_int8_quant_worker,
            args=(init_method,),
            nprocs=2,
            join=True,
        )


@hardware_test(
    res={"cuda": "L4", "rocm": "MI325"},
    num_cards=2,
)
@pytest.mark.skipif(current_omni_platform.get_device_count() < 2, reason="Test requires two GPUs.")
def test_shared_quantizer_matches_native_kernel_on_two_gpus(tmp_path):
    init_method = f"file://{tmp_path / 'int8-accelerator-init'}"
    torch.multiprocessing.spawn(
        _distributed_accelerator_int8_quant_worker,
        args=(init_method,),
        nprocs=2,
        join=True,
    )


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
        layer = _make_int8_layer(torch.randn(128, 64, dtype=torch.float16, device="npu"))
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

    def test_real_npu_shared_quantizer_matches_dynamic_quant(self, mocker):
        """The shared-scale arithmetic must match the vendor dynamic quantizer."""
        import torch_npu

        weight = torch.tensor(
            [
                [0.0, 0.0],
                [-1.3984375, 2.796875],
            ],
            dtype=torch.bfloat16,
            device="npu",
        )
        native_qweight, native_scale = torch_npu.npu_dynamic_quant(weight)
        layer = _make_int8_layer(
            weight,
            input_size=4,
            scale_tp_group=mocker.sentinel.npu_tp_group,
        )
        mocker.patch("torch.distributed.all_reduce")

        qweight, scale = int8_config._quantize_input_sharded_weight(layer)

        assert torch.equal(qweight, native_qweight)
        assert torch.equal(scale.squeeze(-1), native_scale)

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
@hardware_test(res={"cuda": "L4", "rocm": "MI325"})
class TestGPUInt8Smoke:
    """Smoke tests using real upstream INT8 kernels on CUDA or ROCm."""

    @pytest.fixture
    def real_layer(self):
        """Create a real linear layer with fp16 weights on the GPU."""
        layer = _make_int8_layer(torch.randn(128, 64, dtype=torch.float16, device=current_omni_platform.device_type))
        layer.logical_widths = [128]
        layer.input_size_per_partition = 64
        layer.output_size_per_partition = 128
        layer.orig_dtype = torch.float16
        return layer

    def test_real_gpu_scaled_int8_quant_shape_contract(self, quant_config):
        """Smoke test: verify scaled_int8_quant returns correct shapes."""
        from vllm import _custom_ops as ops

        weight = torch.randn(128, 64, dtype=torch.float16, device=current_omni_platform.device_type)
        qweight, scale, _ = ops.scaled_int8_quant(weight, scale=None)

        assert qweight.shape == weight.shape
        assert qweight.dtype == torch.int8
        assert scale.shape == (weight.shape[0], 1)

    def test_real_gpu_online_process_weights_after_loading(self, quant_config, real_layer):
        """Smoke test: process weights with real upstream GPU ops."""
        from vllm_omni.quantization.int8_config import Int8OnlineLinearMethod

        method = Int8OnlineLinearMethod(quant_config)

        method.process_weights_after_loading(real_layer)

        assert real_layer.weight.shape == (64, 128)
        assert real_layer.weight.dtype == torch.int8
        assert hasattr(real_layer, "weight_scale")

    def test_real_gpu_int8_apply_forward(self, quant_config):
        """Smoke test: forward pass with a real upstream GPU kernel."""
        from vllm import _custom_ops as ops

        from vllm_omni.quantization.int8_config import Int8LinearMethod

        method = Int8LinearMethod(quant_config)

        # Create layer with pre-processed weights
        layer = torch.nn.Module()
        weight_fp16 = torch.randn(128, 64, dtype=torch.float16, device=current_omni_platform.device_type)
        qweight, scale, _ = ops.scaled_int8_quant(weight_fp16, scale=None)
        layer.weight = torch.nn.Parameter(qweight.t(), requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(scale, requires_grad=False)

        # Set required attributes for kernel
        layer.input_scale = None
        layer.input_zero_point = None
        layer.azp_adj = None

        # Forward pass
        x = torch.randn(2, 16, 64, dtype=torch.float16, device=current_omni_platform.device_type)
        output = method.apply(layer, x)

        assert output.shape == (2, 16, 128)
        assert output.dtype == torch.float16
