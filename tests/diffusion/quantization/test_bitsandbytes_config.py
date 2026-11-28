# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for BitsAndBytes quantization config."""

import importlib.util
import sys
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
    setattr(bnb, "functional", bnb_functional)
    monkeypatch.setitem(sys.modules, "bitsandbytes", bnb)
    monkeypatch.setitem(sys.modules, "bitsandbytes.functional", bnb_functional)


def test_bitsandbytes_config_creation():
    """Test that BitsAndBytes config can be created."""
    config = build_quant_config("bitsandbytes")
    assert config is not None
    assert config.get_name() == "bitsandbytes"


def test_bitsandbytes_config_with_custom_params():
    """Test BitsAndBytes config with custom parameters."""
    config = build_quant_config(
        "bitsandbytes",
        quant_type="fp4",
        compress_statistics=False,
        ignored_layers=["to_out"],
    )
    assert config is not None
    assert config.quant_type == "fp4"
    assert config.compress_statistics is False
    assert "to_out" in config.ignored_layers


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


def test_bitsandbytes_ignored_layers_match_mode(
    mocker: MockerFixture,
    monkeypatch: pytest.MonkeyPatch,
):
    """`ignored_layers` matches full prefixes by default and short patterns in substring mode."""
    from vllm_omni.quantization.bitsandbytes_config import (
        BnBOnlineLinearMethod,
        DiffusionBitsAndBytesConfig,
    )

    layer = mocker.Mock(spec=LinearBase)
    monkeypatch.setattr(current_omni_platform, "is_cuda", lambda: True)

    short_config = build_quant_config("bitsandbytes", ignored_layers=["to_out"])
    assert isinstance(short_config, DiffusionBitsAndBytesConfig)
    assert short_config.ignored_layers_match_mode == "exact"
    assert isinstance(short_config.get_quant_method(layer, "layers.0.attention.to_out.0"), BnBOnlineLinearMethod)

    short_config.ignored_layers_match_mode = "substring"
    assert isinstance(short_config.get_quant_method(layer, "layers.0.attention.to_out.0"), UnquantizedLinearMethod)
    assert isinstance(short_config.get_quant_method(layer, "layers.0.attention.to_qkv"), BnBOnlineLinearMethod)

    full_config = build_quant_config("bitsandbytes", ignored_layers=["layers.0.attention.to_out.0"])
    assert isinstance(full_config, DiffusionBitsAndBytesConfig)
    assert isinstance(full_config.get_quant_method(layer, "layers.0.attention.to_out.0"), UnquantizedLinearMethod)
    assert isinstance(full_config.get_quant_method(layer, "layers.1.attention.to_out.0"), BnBOnlineLinearMethod)


def test_enable_pattern_ignored_layers():
    """Shared helper raises substring mode only on configs with non-empty `ignored_layers`."""
    from vllm_omni.quantization.bitsandbytes_config import DiffusionBitsAndBytesConfig
    from vllm_omni.quantization.component_config import (
        ComponentQuantizationConfig,
        enable_pattern_ignored_layers,
    )

    assert enable_pattern_ignored_layers(None) is None

    empty_config = build_quant_config("bitsandbytes")
    assert isinstance(empty_config, DiffusionBitsAndBytesConfig)
    assert enable_pattern_ignored_layers(empty_config) is empty_config
    assert empty_config.ignored_layers_match_mode == "exact"

    component_config = build_quant_config(
        {"transformer": {"method": "bitsandbytes", "ignored_layers": ["to_out"]}, "vae": None}
    )
    assert isinstance(component_config, ComponentQuantizationConfig)
    assert enable_pattern_ignored_layers(component_config) is component_config
    transformer_cfg = component_config.component_configs["transformer"]
    assert isinstance(transformer_cfg, DiffusionBitsAndBytesConfig)
    assert transformer_cfg.ignored_layers_match_mode == "substring"


def test_zimage_transformer_honours_bitsandbytes_ignored_layers(
    mocker: MockerFixture,
    monkeypatch: pytest.MonkeyPatch,
):
    """ZImageTransformer2DModel raises BitsAndBytes to substring mode so documented short names skip layers."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.model_executor.layers.utils import default_unquantized_gemm
    from vllm.utils.network_utils import get_file_store_init_method

    import vllm_omni.diffusion.models.z_image.z_image_transformer as z_image_module
    from vllm_omni.quantization.bitsandbytes_config import (
        BnBOnlineLinearMethod,
        DiffusionBitsAndBytesConfig,
    )

    mocker.patch(
        "vllm.model_executor.layers.linear.dispatch_unquantized_gemm",
        return_value=default_unquantized_gemm,
    )
    monkeypatch.setattr(current_omni_platform, "is_cuda", lambda: True)
    mocker.patch.object(z_image_module, "Attention", return_value=torch.nn.Identity())

    def _build_model(quant_config: DiffusionBitsAndBytesConfig):
        return z_image_module.ZImageTransformer2DModel(
            all_patch_size=(2,),
            all_f_patch_size=(1,),
            in_channels=4,
            dim=64,
            n_layers=2,
            n_refiner_layers=1,
            n_heads=2,
            n_kv_heads=2,
            cap_feat_dim=32,
            axes_dims=[8, 12, 12],
            axes_lens=[16, 16, 16],
            quant_config=quant_config,
        )

    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method=get_file_store_init_method(),
        backend="gloo",
    )
    try:
        with set_current_vllm_config(VllmConfig()):
            initialize_model_parallel()

            short_cfg = build_quant_config("bitsandbytes", ignored_layers=["to_out"])
            assert isinstance(short_cfg, DiffusionBitsAndBytesConfig)
            short_model = _build_model(short_cfg)
            for block in (*short_model.noise_refiner, *short_model.context_refiner, *short_model.layers):
                assert isinstance(block.attention.to_out[0].quant_method, UnquantizedLinearMethod)
                assert isinstance(block.attention.to_qkv.quant_method, BnBOnlineLinearMethod)
                assert isinstance(block.feed_forward.w13.quant_method, BnBOnlineLinearMethod)
                assert isinstance(block.feed_forward.w2.quant_method, BnBOnlineLinearMethod)
            assert short_cfg.ignored_layers_match_mode == "substring"

            full_cfg = build_quant_config("bitsandbytes", ignored_layers=["layers.0.attention.to_out.0"])
            assert isinstance(full_cfg, DiffusionBitsAndBytesConfig)
            full_model = _build_model(full_cfg)
            assert isinstance(full_model.layers[0].attention.to_out[0].quant_method, UnquantizedLinearMethod)
            assert isinstance(full_model.layers[1].attention.to_out[0].quant_method, BnBOnlineLinearMethod)
            assert isinstance(full_model.noise_refiner[0].attention.to_out[0].quant_method, BnBOnlineLinearMethod)
    finally:
        cleanup_dist_env_and_memory()


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
        layer.weight = Parameter(torch.randn(64, 32), requires_grad=False)
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
