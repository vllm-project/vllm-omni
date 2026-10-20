# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for MXFP8 quantization config and linear method dispatch.

Coverage:
- DiffusionMXFP8Config.from_config roundtrips (CPU, no NPU required)
- get_quant_method dispatch (mocked platform)
- MXFPLinearMethodBase.apply() reshape skeleton (CPU)
- Weight / scale shape-transform arithmetic from process_weights_after_loading (CPU)
- build_quant_config integration
- MXFP8_QUANT_CONFIG structure as the auto-detection contract
- Offload-after-quant capability of the shared online-method mixin (CPU)
"""

import pytest
import torch
from pytest_mock import MockerFixture
from torch.nn import Module
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod

from vllm_omni.platforms import current_omni_platform
from vllm_omni.quantization import build_quant_config

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

npu_available = pytest.mark.skipif(not current_omni_platform.is_npu(), reason="NPU platform not available")


# ---------------------------------------------------------------------------
# DiffusionMXFP8Config — from_config roundtrips
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value, expected", [(None, 0), ("0", 0), ("1", 1)])
def test_mxfp8_scale_alg(monkeypatch, value, expected):
    from vllm_omni.quantization.mxfp8_config import _mx_scale_alg

    monkeypatch.delenv("VLLM_OMNI_MXFP8_SCALE_ALG", raising=False)
    if value is not None:
        monkeypatch.setenv("VLLM_OMNI_MXFP8_SCALE_ALG", value)
    assert _mx_scale_alg() == expected


def test_mxfp8_activation_passes_scale_alg(monkeypatch, mocker: MockerFixture):
    import sys
    from types import SimpleNamespace

    from vllm_omni.quantization.mxfp8_config import DiffusionMXFP8Config, NPUMxfp8LinearMethod

    monkeypatch.setenv("VLLM_OMNI_MXFP8_SCALE_ALG", "1")
    quantize = mocker.Mock(return_value=("quantized", "scale"))
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(npu_dynamic_mx_quant=quantize, float8_e4m3fn="fp8"))
    method = NPUMxfp8LinearMethod(DiffusionMXFP8Config())
    x = torch.ones(2, 64, dtype=torch.bfloat16)
    assert method._quantize_activation(x) == ("quantized", "scale")
    quantize.assert_called_once_with(x, dst_type="fp8", scale_alg=1)


def _make_mxfp8_prefix_method(steps, mode="cpu"):
    from vllm_omni.quantization.mxfp8_config import DiffusionMXFP8Config, NPUMxfp8OnlineLinearMethod

    config = DiffusionMXFP8Config(prefix_bf16_steps=steps, prefix_bf16_master_mode=mode)
    return NPUMxfp8OnlineLinearMethod(config)


def _set_mxfp8_prefix_context(monkeypatch, step, available=True):
    import sys
    from types import SimpleNamespace

    context = SimpleNamespace(denoise_step_idx=step)
    monkeypatch.setitem(
        sys.modules,
        "vllm_omni.diffusion.forward_context",
        SimpleNamespace(get_forward_context=lambda: context, is_forward_context_available=lambda: available),
    )


@pytest.mark.parametrize("steps", [-1, 1.5, "1", True])
def test_mxfp8_prefix_rejects_invalid_steps(steps):
    from vllm_omni.quantization.mxfp8_config import DiffusionMXFP8Config

    expected_error = ValueError if isinstance(steps, int) and not isinstance(steps, bool) else TypeError
    with pytest.raises(expected_error, match="prefix_bf16_steps"):
        DiffusionMXFP8Config(prefix_bf16_steps=steps)


def test_mxfp8_prefix_rejects_unknown_mode():
    from vllm_omni.quantization.mxfp8_config import DiffusionMXFP8Config

    with pytest.raises(ValueError, match="Unknown MXFP8 prefix master mode"):
        DiffusionMXFP8Config(prefix_bf16_master_mode="invalid")


def test_mxfp8_prefix_rejects_serialized_checkpoint():
    from vllm_omni.quantization.mxfp8_config import DiffusionMXFP8Config

    with pytest.raises(ValueError, match="requires online MXFP8 weights"):
        DiffusionMXFP8Config(is_checkpoint_mxfp8_serialized=True, prefix_bf16_steps=1)


def test_mxfp8_prefix_resident_requires_npu():
    from vllm_omni.quantization.mxfp8_config import DiffusionMXFP8Config, _mxfp8_save_prefix_master

    layer = Module()
    layer.register_parameter("weight", torch.nn.Parameter(torch.ones(4, 64), requires_grad=False))
    with pytest.raises(RuntimeError, match="loading NPU"):
        _mxfp8_save_prefix_master(
            layer,
            DiffusionMXFP8Config(prefix_bf16_steps=1, prefix_bf16_master_mode="resident"),
        )


@pytest.mark.parametrize("steps", [0, 1, 5])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_mxfp8_prefix_saves_original_weights(monkeypatch, mocker: MockerFixture, steps, dtype):
    import sys
    from types import SimpleNamespace

    method = _make_mxfp8_prefix_method(steps)
    monkeypatch.setenv("VLLM_OMNI_MXFP8_SCALE_ALG", "1")
    layer = Module()
    weight = torch.arange(256, dtype=dtype).reshape(4, 64)
    expected_master = weight.to(torch.bfloat16).transpose(0, 1).contiguous().clone()
    layer.register_parameter("weight", torch.nn.Parameter(weight, requires_grad=False))

    def quantize(weight, **kwargs):
        weight.zero_()
        return weight, torch.ones(4, 2, dtype=torch.uint8)

    quantize_mock = mocker.Mock(side_effect=quantize)
    monkeypatch.setitem(
        sys.modules, "torch_npu", SimpleNamespace(npu_dynamic_mx_quant=quantize_mock, float8_e4m3fn="fp8")
    )
    original = layer.weight
    method.process_weights_after_loading(layer)
    quantize_mock.assert_called_once_with(original, dst_type="fp8", scale_alg=1)
    assert layer.weight.shape == (64, 4)
    assert layer.weight_scale.shape == (1, 4, 2)
    assert hasattr(layer, "_mxfp8_prefix_master") == bool(steps)
    if steps:
        master = layer._mxfp8_prefix_master
        assert master.dtype == torch.bfloat16
        assert master.device == torch.device("cpu")
        assert not master.requires_grad
        assert master.shape == (64, 4)
        assert master.is_contiguous()
        assert torch.equal(master, expected_master)
        assert layer._mxfp8_prefix_master_mode == "cpu"
        assert "_mxfp8_prefix_master" not in layer.state_dict()
    method.process_weights_after_loading(layer)
    assert quantize_mock.call_count == 1


@pytest.mark.parametrize(
    "steps, step, fallback",
    [
        (0, None, False),
        (1, 0, True),
        (1, 1, False),
        (5, -1, False),
        (5, 0, True),
        (5, 4, True),
        (5, 5, False),
        (50, 49, True),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("with_bias", [False, True])
def test_mxfp8_prefix_step_boundary(monkeypatch, mocker: MockerFixture, steps, step, fallback, dtype, with_bias):
    method = _make_mxfp8_prefix_method(steps)
    _set_mxfp8_prefix_context(monkeypatch, step)
    layer = Module()
    layer._mxfp8_prefix_master = torch.tensor([[1, 3], [2, 4]], dtype=torch.bfloat16)
    layer._mxfp8_prefix_master_mode = "cpu"
    x = torch.tensor([[[1, 2], [2, 3]]], dtype=dtype)
    bias = torch.tensor([1, 2], dtype=torch.float32) if with_bias else None
    quantize = mocker.patch.object(method, "_quantize_activation", return_value=(x.reshape(-1, 2), None))
    matmul = mocker.patch.object(method, "_quant_matmul", return_value=torch.zeros(2, 2, dtype=dtype))
    result = method.apply(layer, x, bias)
    assert result.shape == x.shape
    assert result.dtype == dtype
    if fallback:
        expected = torch.tensor([[[5, 11], [8, 18]]], dtype=dtype)
        if with_bias:
            expected += bias.to(dtype)
        assert torch.equal(result, expected)
        quantize.assert_not_called()
        matmul.assert_not_called()
    else:
        quantize.assert_called_once()
        matmul.assert_called_once()
        assert torch.count_nonzero(result) == 0


@pytest.mark.parametrize("available, step", [(False, 0), (True, None)])
def test_mxfp8_prefix_without_step_context_uses_mxfp8(monkeypatch, mocker: MockerFixture, available, step):
    method = _make_mxfp8_prefix_method(1)
    _set_mxfp8_prefix_context(monkeypatch, step, available=available)
    x = torch.ones(1, 2, dtype=torch.bfloat16)
    quantize = mocker.patch.object(method, "_quantize_activation", return_value=(x, None))
    matmul = mocker.patch.object(method, "_quant_matmul", return_value=torch.zeros_like(x))
    assert torch.equal(method.apply(Module(), x), torch.zeros_like(x))
    quantize.assert_called_once()
    matmul.assert_called_once()


def test_mxfp8_prefix_requires_saved_master(monkeypatch):
    method = _make_mxfp8_prefix_method(1)
    _set_mxfp8_prefix_context(monkeypatch, 0)
    with pytest.raises(RuntimeError, match="without the original master weight"):
        method.apply(Module(), torch.ones(1, 2, dtype=torch.bfloat16))


def test_mxfp8_prefix_resident_device_mismatch(monkeypatch):
    method = _make_mxfp8_prefix_method(1, "resident")
    _set_mxfp8_prefix_context(monkeypatch, 0)
    layer = Module()
    layer._mxfp8_prefix_master = torch.empty(2, 2, device="meta", dtype=torch.bfloat16)
    layer._mxfp8_prefix_master_mode = "resident"
    with pytest.raises(RuntimeError, match="device mismatch"):
        method.apply(layer, torch.ones(1, 2, dtype=torch.bfloat16))


def test_mxfp8_prefix_repeated_calls_and_requests(monkeypatch, mocker: MockerFixture):
    method = _make_mxfp8_prefix_method(1, "resident")
    layer = Module()
    layer._mxfp8_prefix_master = torch.eye(2, dtype=torch.bfloat16)
    layer._mxfp8_prefix_master_mode = "resident"
    x = torch.ones(1, 2, dtype=torch.bfloat16)
    mocker.patch.object(method, "_quantize_activation", return_value=(x, None))
    matmul = mocker.patch.object(method, "_quant_matmul", return_value=torch.zeros_like(x))
    for step in (0, 0, 1, 0, 0):
        _set_mxfp8_prefix_context(monkeypatch, step)
        assert torch.equal(method.apply(layer, x), x if step == 0 else torch.zeros_like(x))
    assert matmul.call_count == 1


def test_mxfp8_prefix_steps_come_from_quant_config(monkeypatch, mocker: MockerFixture):
    method = _make_mxfp8_prefix_method(1)
    layer = Module()
    x = torch.ones(1, 2, dtype=torch.bfloat16)
    layer._mxfp8_prefix_master = torch.eye(2, dtype=torch.bfloat16)
    layer._mxfp8_prefix_master_mode = "cpu"
    _set_mxfp8_prefix_context(monkeypatch, 0)
    mocker.patch.object(method, "_quantize_activation", return_value=(x, None))
    matmul = mocker.patch.object(method, "_quant_matmul", return_value=torch.zeros_like(x))
    monkeypatch.setenv("VLLM_OMNI_MXFP8_PREFIX_BF16_STEPS", "0")
    assert torch.equal(method.apply(layer, x), x)
    monkeypatch.setenv("VLLM_OMNI_MXFP8_PREFIX_BF16_STEPS", "100")
    assert torch.equal(method.apply(layer, x), x)
    assert matmul.call_count == 0


def test_mxfp8_prefix_fp16_uses_activation_dtype(monkeypatch):
    method = _make_mxfp8_prefix_method(1)
    _set_mxfp8_prefix_context(monkeypatch, 0)
    layer = Module()
    layer._mxfp8_prefix_master = torch.ones(1, 1, dtype=torch.bfloat16)
    layer._mxfp8_prefix_master_mode = "cpu"
    x = torch.tensor([[1.001]], dtype=torch.float16)
    assert torch.equal(method.apply(layer, x), x)
    assert not torch.equal(x.to(torch.bfloat16).to(torch.float16), x)


def test_mxfp8_config_get_name():
    from vllm_omni.quantization.mxfp8_config import DiffusionMXFP8Config

    assert DiffusionMXFP8Config.get_name() == "mxfp8"


def test_mxfp8_config_from_config_defaults():
    from vllm_omni.quantization.mxfp8_config import DiffusionMXFP8Config

    cfg = DiffusionMXFP8Config.from_config({})
    assert cfg.is_checkpoint_mxfp8_serialized is False
    assert cfg.ignored_layers == []
    assert cfg.prefix_bf16_steps == 0
    assert cfg.prefix_bf16_master_mode == "cpu"


def test_mxfp8_config_from_config_prefix_bf16_options():
    from vllm_omni.quantization.mxfp8_config import DiffusionMXFP8Config

    cfg = DiffusionMXFP8Config.from_config({"prefix_bf16_steps": 4, "prefix_bf16_master_mode": "resident"})
    assert cfg.prefix_bf16_steps == 4
    assert cfg.prefix_bf16_master_mode == "resident"


def test_mxfp8_config_from_config_serialized():
    from vllm_omni.quantization.mxfp8_config import DiffusionMXFP8Config

    cfg = DiffusionMXFP8Config.from_config({"is_checkpoint_mxfp8_serialized": True})
    assert cfg.is_checkpoint_mxfp8_serialized is True


def test_mxfp8_config_from_config_ignored_layers():
    from vllm_omni.quantization.mxfp8_config import DiffusionMXFP8Config

    cfg = DiffusionMXFP8Config.from_config({"ignored_layers": ["proj_out"]})
    assert cfg.ignored_layers == ["proj_out"]


def test_mxfp8_config_from_config_modules_to_not_convert_fallback():
    """modules_to_not_convert must be accepted as an alias for ignored_layers."""
    from vllm_omni.quantization.mxfp8_config import DiffusionMXFP8Config

    cfg = DiffusionMXFP8Config.from_config({"modules_to_not_convert": ["proj_out"]})
    assert cfg.ignored_layers == ["proj_out"]


# ---------------------------------------------------------------------------
# build_quant_config integration
# ---------------------------------------------------------------------------


def test_build_quant_config_mxfp8_string():
    from vllm_omni.quantization.mxfp8_config import DiffusionMXFP8Config

    cfg = build_quant_config("mxfp8")
    assert isinstance(cfg, DiffusionMXFP8Config)
    assert cfg.get_name() == "mxfp8"
    assert cfg.is_checkpoint_mxfp8_serialized is False


def test_build_quant_config_mxfp8_dict():
    from vllm_omni.quantization.mxfp8_config import DiffusionMXFP8Config

    cfg = build_quant_config({"method": "mxfp8", "is_checkpoint_mxfp8_serialized": True})
    assert isinstance(cfg, DiffusionMXFP8Config)
    assert cfg.is_checkpoint_mxfp8_serialized is True


def test_build_quant_config_mxfp8_prefix_bf16_options():
    from vllm_omni.quantization.mxfp8_config import DiffusionMXFP8Config

    cfg = build_quant_config({"method": "mxfp8", "prefix_bf16_steps": 3, "prefix_bf16_master_mode": "resident"})
    assert isinstance(cfg, DiffusionMXFP8Config)
    assert cfg.prefix_bf16_steps == 3
    assert cfg.prefix_bf16_master_mode == "resident"


def test_build_quant_config_mxfp8_prefix_bf16_checkpoint_metadata():
    from vllm_omni.quantization.mxfp8_config import DiffusionMXFP8Config

    cfg = build_quant_config({"quant_method": "mxfp8", "prefix_bf16_steps": 2, "prefix_bf16_master_mode": "cpu"})
    assert isinstance(cfg, DiffusionMXFP8Config)
    assert cfg.prefix_bf16_steps == 2


def test_resolve_quant_config_from_disk_rejects_prefix_fallback_for_serialized_weights():
    from vllm_omni.quantization.factory import resolve_quant_config_from_disk

    active_config = build_quant_config({"method": "mxfp8", "prefix_bf16_steps": 2})
    with pytest.raises(ValueError, match="requires online MXFP8 weights"):
        resolve_quant_config_from_disk(
            active_config,
            {"quant_method": "mxfp8", "is_checkpoint_mxfp8_serialized": True},
        )


def test_build_quant_config_mxfp8_config_json_format():
    """Verify that the exact quantization_config injected by merge_mxfp8_checkpoint.py
    is accepted by build_quant_config and selects the offline (serialized) path.

    This is the critical auto-detection contract: TransformerConfig.from_dict()
    reads quant_method + is_checkpoint_mxfp8_serialized to pick NPUMxfp8LinearMethod.
    """
    from vllm_omni.quantization.mxfp8_config import DiffusionMXFP8Config
    from vllm_omni.quantization.tools.merge_mxfp8_checkpoint import MXFP8_QUANT_CONFIG

    cfg = build_quant_config(MXFP8_QUANT_CONFIG)
    assert isinstance(cfg, DiffusionMXFP8Config)
    assert cfg.is_checkpoint_mxfp8_serialized is True


def test_mxfp8_quant_config_structure():
    """MXFP8_QUANT_CONFIG must contain exactly the keys that auto-detection reads."""
    from vllm_omni.quantization.tools.merge_mxfp8_checkpoint import MXFP8_QUANT_CONFIG

    assert MXFP8_QUANT_CONFIG.get("quant_method") == "mxfp8"
    assert MXFP8_QUANT_CONFIG.get("is_checkpoint_mxfp8_serialized") is True


# ---------------------------------------------------------------------------
# get_quant_method dispatch
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not current_omni_platform.is_npu(), reason="Native MXFP8 offline only supported on NPU")
def test_get_quant_method_offline_npu(mocker: MockerFixture):
    """Offline (serialized) path must return NPUMxfp8LinearMethod on NPU."""
    from vllm_omni.quantization.mxfp8_config import DiffusionMXFP8Config

    config = DiffusionMXFP8Config(is_checkpoint_mxfp8_serialized=True)
    layer = mocker.Mock(spec=LinearBase)

    method = config.get_quant_method(layer, "blocks.0.attn1.to_q")
    assert type(method).__name__ == "NPUMxfp8LinearMethod"


@pytest.mark.skipif(not current_omni_platform.is_xpu(), reason="XPU platform not available")
def test_get_quant_method_offline_xpu_raises(mocker: MockerFixture):
    """XPU offline mode must raise NotImplementedError (use AutoRound MXFP8 instead)."""
    from vllm_omni.quantization.mxfp8_config import DiffusionMXFP8Config

    config = DiffusionMXFP8Config(is_checkpoint_mxfp8_serialized=True)
    layer = mocker.Mock(spec=LinearBase)

    with pytest.raises(NotImplementedError, match="Native MXFP8 offline mode is not supported on XPU"):
        config.get_quant_method(layer, "blocks.0.attn1.to_q")


@pytest.mark.skipif(
    not (current_omni_platform.is_npu() or current_omni_platform.is_xpu()), reason="MXFP8 only supported on NPU and XPU"
)
def test_get_quant_method_online(mocker: MockerFixture):
    """Online (BF16 checkpoint) path must return platform-specific method on current platform."""
    from vllm_omni.quantization.mxfp8_config import DiffusionMXFP8Config

    # Mock the vLLM online method to avoid config dependency for XPU
    if current_omni_platform.is_xpu():
        mock_inner = mocker.Mock()
        mocker.patch(
            "vllm_omni.quantization.mxfp8_config.VllmMxfp8OnlineLinearMethod.__init__",
            lambda self: setattr(self, "_inner", mock_inner),
        )

    config = DiffusionMXFP8Config(is_checkpoint_mxfp8_serialized=False)
    layer = mocker.Mock(spec=LinearBase)

    method = config.get_quant_method(layer, "blocks.0.attn1.to_q")

    if current_omni_platform.is_npu():
        assert type(method).__name__ == "NPUMxfp8OnlineLinearMethod"
    elif current_omni_platform.is_xpu():
        assert type(method).__name__ == "VllmMxfp8OnlineLinearMethod"


def test_xpu_online_method_keeps_vllm_weight_lifecycle():
    """XPU online MXFP8 must not replace vLLM's create_weights: that is where vLLM
    builds the meta-device weight and selects the shape-dependent MXFP8 kernel."""
    from vllm.model_executor.layers.quantization.online.mxfp8 import Mxfp8OnlineLinearMethod

    from vllm_omni.quantization.mxfp8_config import VllmMxfp8OnlineLinearMethod

    for name in ("create_weights", "process_weights_after_loading"):
        assert getattr(VllmMxfp8OnlineLinearMethod, name) is getattr(Mxfp8OnlineLinearMethod, name)


def test_get_quant_method_unsupported_platform_raises(mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch):
    """Unsupported platform (CUDA, ROCm) must raise NotImplementedError."""
    from vllm_omni.quantization.mxfp8_config import DiffusionMXFP8Config

    config = DiffusionMXFP8Config()
    layer = mocker.Mock(spec=LinearBase)
    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: False)
    monkeypatch.setattr(current_omni_platform, "is_xpu", lambda: False)
    monkeypatch.setattr(current_omni_platform, "is_cuda", lambda: False)

    with pytest.raises(NotImplementedError):
        config.get_quant_method(layer, "blocks.0.attn1.to_q")


def test_get_quant_method_ignored_layer(mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch):
    """A prefix in ignored_layers must bypass quantization → UnquantizedLinearMethod."""
    from vllm_omni.quantization.mxfp8_config import DiffusionMXFP8Config

    config = DiffusionMXFP8Config(ignored_layers=["proj_out"])
    layer = mocker.Mock(spec=LinearBase)
    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: True)
    monkeypatch.setattr(current_omni_platform, "is_xpu", lambda: False)
    monkeypatch.setattr(current_omni_platform, "is_cuda", lambda: False)

    method = config.get_quant_method(layer, "proj_out")
    assert isinstance(method, UnquantizedLinearMethod)


def test_get_quant_method_non_linear_returns_none(monkeypatch: pytest.MonkeyPatch):
    """Non-LinearBase layers (norms, embeddings) must get None → no quantization."""
    from vllm_omni.quantization.mxfp8_config import DiffusionMXFP8Config

    config = DiffusionMXFP8Config()
    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: True)

    norm_layer = torch.nn.LayerNorm(64)
    assert config.get_quant_method(norm_layer, "blocks.0.norm1") is None


# ---------------------------------------------------------------------------
# MXFPLinearMethodBase.apply() — reshape skeleton (CPU, no NPU)
# ---------------------------------------------------------------------------


def test_apply_reshape_skeleton():
    """apply() must flatten batch dims → _apply_inner → restore original leading dims."""
    from vllm_omni.quantization.mxfp8_config import MXFPLinearMethodBase

    OUT_FEATURES = 4

    class _StubMethod(MXFPLinearMethodBase):
        def create_weights(
            self,
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        ):
            pass

        def _quantize_activation(self, x):
            return x, None

        def _quant_matmul(self, x_q, x_scale, layer, bias, ori_dtype):
            return torch.zeros(x_q.shape[0], OUT_FEATURES, dtype=ori_dtype)

    method = _StubMethod()
    x = torch.randn(2, 3, 8)  # (batch=2, seq=3, K=8)
    out = method.apply(None, x)
    assert out.shape == (2, 3, OUT_FEATURES)


def test_apply_reshape_with_bias():
    """apply() must pass bias through to _apply_inner unchanged."""
    from vllm_omni.quantization.mxfp8_config import MXFPLinearMethodBase

    received_bias = []

    class _StubMethod(MXFPLinearMethodBase):
        def create_weights(
            self,
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        ):
            pass

        def _quantize_activation(self, x):
            return x, None

        def _quant_matmul(self, x_q, x_scale, layer, bias, ori_dtype):
            received_bias.append(bias)
            return torch.zeros(x_q.shape[0], 4, dtype=ori_dtype)

    method = _StubMethod()
    bias = torch.zeros(4)
    method.apply(None, torch.randn(2, 8), bias=bias)
    assert received_bias[0] is bias


# ---------------------------------------------------------------------------
# process_weights_after_loading shape arithmetic (pure torch, no NPU ops)
#
# These tests replicate the CPU-safe portions of process_weights_after_loading
# to guard the key layout contract: (N,K) weight → (K,N) and (N,S) scale →
# (S/2,N,2). They do NOT call NPU ops; they test only the math.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n, k",
    [(64, 128), (32, 64), (16, 96)],
    ids=["64x128", "32x64", "16x96"],
)
def test_weight_transpose_contract(n: int, k: int):
    """Weight must be transposed from (N, K) to (K, N) and be contiguous."""
    w = torch.zeros(n, k, dtype=torch.uint8)
    w = w.transpose(0, 1).contiguous()
    assert w.shape == (k, n)
    assert w.is_contiguous()


@pytest.mark.parametrize(
    "n, k_groups, expected_groups",
    [
        (64, 4, 2),  # even K_groups — no padding
        (64, 3, 2),  # odd  K_groups — padded to 4
        (32, 1, 1),  # odd  K_groups — padded to 2
    ],
    ids=["even", "odd-to-4", "odd-to-2"],
)
def test_weight_scale_reshape_contract(n: int, k_groups: int, expected_groups: int):
    """Scale must be reshaped from (N, K_groups) to (K_groups_even//2, N, 2).

    Odd K_groups must be padded to even before the reshape.
    """
    s = torch.zeros(n, k_groups, dtype=torch.uint8)
    if k_groups % 2 == 1:
        s = torch.cat([s, torch.zeros(n, 1, dtype=s.dtype)], dim=1)
        k_groups += 1
    s = s.reshape(n, k_groups // 2, 2).transpose(0, 1).contiguous()
    assert s.shape == (expected_groups, n, 2)
    assert s.is_contiguous()


def test_num_groups_formula():
    """K_groups formula: ceil(K / 32) — spot-check boundary values."""
    assert (31 + 31) // 32 == 1  # K=31 → 1 group
    assert (32 + 31) // 32 == 1  # K=32 → 1 group
    assert (33 + 31) // 32 == 2  # K=33 → 2 groups
    assert (128 + 31) // 32 == 4  # K=128 → 4 groups (even)
    assert (96 + 31) // 32 == 3  # K=96  → 3 groups (odd → needs padding)


# ---------------------------------------------------------------------------
# SUPPORTED_QUANTIZATION_METHODS
# ---------------------------------------------------------------------------


def test_supported_methods_include_mxfp8():
    from vllm_omni.quantization import SUPPORTED_QUANTIZATION_METHODS

    assert "mxfp8" in SUPPORTED_QUANTIZATION_METHODS


# ---------------------------------------------------------------------------
# Offload-after-quant capability (CPU, no NPU ops)
# ---------------------------------------------------------------------------


class TestOffloadAfterQuant:
    """Online MXFP methods advertise ``supports_offload_after_quant`` so the
    diffusion loader can stream each layer back to host memory right after it
    is quantized, instead of accumulating the whole model on the accelerator
    until the final ``model.to("cpu")``.

    The capability lives on ``_LazyWeightMixin``, which is shared by all
    online MXFP8/MXFP4 methods, so pinning it here covers mxfp4 too.
    """

    @pytest.fixture(autouse=True)
    def _mock_tp(self, mocker: MockerFixture):
        # create_weights resolves the TP group while computing per-partition
        # shapes; these tests never initialize torch.distributed.
        mock_group = mocker.Mock()
        mock_group.rank_in_group = 0
        mocker.patch("vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size", return_value=1)
        mocker.patch("vllm.model_executor.layers.linear.get_tensor_model_parallel_rank", return_value=0)
        mocker.patch("vllm.distributed.parallel_state.get_tp_group", return_value=mock_group)

    def _make_method(self):
        from vllm_omni.quantization.mxfp8_config import (
            DiffusionMXFP8Config,
            NPUMxfp8OnlineLinearMethod,
        )

        return NPUMxfp8OnlineLinearMethod(DiffusionMXFP8Config(is_checkpoint_mxfp8_serialized=False))

    def test_only_methods_advertising_the_capability_are_asked(self):
        from vllm_omni.diffusion.model_loader.diffusers_loader import (
            DiffusersPipelineLoader,
        )

        class MetaDeviceOnlyMethod:
            """Stands in for upstream online FP8: lazy, but no offload hook."""

            uses_meta_device = True

        model = torch.nn.Sequential(Module(), Module())
        lazy_mxfp8 = self._make_method()
        model[0].quant_method = lazy_mxfp8
        model[1].quant_method = MetaDeviceOnlyMethod()

        marked = DiffusersPipelineLoader._request_offload_after_quant(model)

        assert marked == 1
        assert lazy_mxfp8._offload_after_quant

    def _load_layer_in_one_chunk(self, method, layer):
        method.create_weights(
            layer,
            input_size_per_partition=8,
            output_partition_sizes=[4],
            input_size=8,
            output_size=4,
            params_dtype=torch.bfloat16,
            weight_loader=lambda param, loaded_weight, *args, **kwargs: param.copy_(loaded_weight),
        )
        full_weight = torch.randn(4, 8, dtype=torch.bfloat16)
        layer.weight.weight_loader(layer.weight, full_weight)

    def test_finished_layer_returns_to_cpu_when_enabled(self, mocker: MockerFixture):
        method = self._make_method()
        method.enable_offload_after_quant()
        mocker.patch.object(method, "process_weights_after_loading")

        layer = Module()
        moved_to = []
        layer.to = lambda device, *args, **kwargs: moved_to.append(device)  # type: ignore[method-assign]

        self._load_layer_in_one_chunk(method, layer)

        method.process_weights_after_loading.assert_called_once_with(layer)
        assert layer._already_called_process_weights_after_loading
        assert moved_to == ["cpu"]

    def test_finished_layer_stays_put_by_default(self, mocker: MockerFixture):
        method = self._make_method()
        mocker.patch.object(method, "process_weights_after_loading")

        layer = Module()
        moved_to = []
        layer.to = lambda device, *args, **kwargs: moved_to.append(device)  # type: ignore[method-assign]

        self._load_layer_in_one_chunk(method, layer)

        method.process_weights_after_loading.assert_called_once_with(layer)
        assert layer._already_called_process_weights_after_loading
        assert moved_to == []
