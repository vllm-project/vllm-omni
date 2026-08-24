# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MXFP8 quantization config and linear method dispatch.

Coverage:
- DiffusionMXFP8Config.from_config roundtrips (CPU, no NPU required)
- get_quant_method dispatch (mocked platform)
- MXFPLinearMethodBase.apply() reshape skeleton (CPU)
- Weight / scale shape-transform arithmetic from process_weights_after_loading (CPU)
- build_quant_config integration
- MXFP8_QUANT_CONFIG structure as the auto-detection contract
"""

import sys
import types
from typing import Any

import pytest
import torch
from pytest_mock import MockerFixture
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod

from vllm_omni.platforms import current_omni_platform
from vllm_omni.quantization import build_quant_config

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

npu_available = pytest.mark.skipif(not current_omni_platform.is_npu(), reason="NPU platform not available")


def _install_ascend_fused_moe_fake(monkeypatch: pytest.MonkeyPatch):
    """Stub vllm_ascend.ops.fused_moe.fused_moe.AscendUnquantizedFusedMoEMethod
    so get_quant_method's FusedMoE branch can be exercised without the real
    vllm-ascend package. Returns the fake class so tests can inspect calls."""
    calls: list[tuple[Any, dict]] = []

    class _FakeAscendUnquantizedFusedMoEMethod:
        def __init__(self, moe_config, *, tid2eid=None):
            self.moe_config = moe_config
            self.tid2eid = tid2eid
            calls.append((moe_config, {"tid2eid": tid2eid}))

    for name in ("vllm_ascend", "vllm_ascend.ops", "vllm_ascend.ops.fused_moe"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    fused_moe_module = types.ModuleType("vllm_ascend.ops.fused_moe.fused_moe")
    fused_moe_module.AscendUnquantizedFusedMoEMethod = _FakeAscendUnquantizedFusedMoEMethod
    monkeypatch.setitem(sys.modules, "vllm_ascend.ops.fused_moe.fused_moe", fused_moe_module)

    return _FakeAscendUnquantizedFusedMoEMethod, calls


# ---------------------------------------------------------------------------
# DiffusionMXFP8Config — from_config roundtrips
# ---------------------------------------------------------------------------


def test_mxfp8_config_get_name():
    from vllm_omni.quantization.mxfp8_config import DiffusionMXFP8Config

    assert DiffusionMXFP8Config.get_name() == "mxfp8"


def test_mxfp8_config_from_config_defaults():
    from vllm_omni.quantization.mxfp8_config import DiffusionMXFP8Config

    cfg = DiffusionMXFP8Config.from_config({})
    assert cfg.is_checkpoint_mxfp8_serialized is False
    assert cfg.ignored_layers == []


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
# OmniNPUMxfp8Config — NPU AR-stage identity, distinct from vLLM's own "mxfp8"
# ---------------------------------------------------------------------------


def test_omni_npu_mxfp8_config_get_name():
    from vllm_omni.quantization.mxfp8_config import OmniNPUMxfp8Config

    assert OmniNPUMxfp8Config.get_name() == "omni_npu_mxfp8"


def test_build_quant_config_omni_npu_mxfp8_registered():
    """omni_npu_mxfp8 must resolve through vLLM's quantization registry,
    the same lifecycle vLLM's own ModelConfig uses for "mxfp8"."""
    from vllm_omni.quantization.mxfp8_config import OmniNPUMxfp8Config

    cfg = build_quant_config("omni_npu_mxfp8")
    assert isinstance(cfg, OmniNPUMxfp8Config)
    assert cfg.is_checkpoint_mxfp8_serialized is False


def test_build_quant_config_omni_npu_mxfp8_local_serialized_checkpoint():
    """A local pre-quantized checkpoint's config.json (quant_method="omni_npu_mxfp8",
    is_checkpoint_mxfp8_serialized=True) must build the offline variant."""
    from vllm_omni.quantization.mxfp8_config import OmniNPUMxfp8Config

    cfg = build_quant_config(
        {"method": "omni_npu_mxfp8", "is_checkpoint_mxfp8_serialized": True, "ignored_layers": ["proj_out"]}
    )
    assert isinstance(cfg, OmniNPUMxfp8Config)
    assert cfg.is_checkpoint_mxfp8_serialized is True
    assert cfg.ignored_layers == ["proj_out"]


@pytest.mark.parametrize("quant_method", ["mxfp8", "omni_npu_mxfp8"])
def test_omni_npu_mxfp8_override_quantization_method_on_npu(quant_method: str, monkeypatch: pytest.MonkeyPatch):
    """On NPU, a checkpoint declaring either the vLLM-reserved "mxfp8" name (e.g. a
    Hugging Face checkpoint uploaded before the remap existed) or the already-remapped
    "omni_npu_mxfp8" (a checkpoint saved locally by this integration) must both resolve
    to "omni_npu_mxfp8" so vLLM's ModelConfig loads the Omni-specific config class."""
    from vllm_omni.quantization.mxfp8_config import OmniNPUMxfp8Config

    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: True)

    result = OmniNPUMxfp8Config.override_quantization_method(
        hf_quant_cfg={"quant_method": quant_method},
        user_quant=None,
    )
    assert result == "omni_npu_mxfp8"


def test_omni_npu_mxfp8_override_quantization_method_off_npu_returns_none(monkeypatch: pytest.MonkeyPatch):
    """The remap must not apply on non-NPU platforms, even with a matching config."""
    from vllm_omni.quantization.mxfp8_config import OmniNPUMxfp8Config

    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: False)

    result = OmniNPUMxfp8Config.override_quantization_method(
        hf_quant_cfg={"quant_method": "mxfp8"},
        user_quant=None,
    )
    assert result is None


def test_omni_npu_mxfp8_override_quantization_method_mismatched_method_returns_none(
    monkeypatch: pytest.MonkeyPatch,
):
    """An unrelated quant_method in the checkpoint config must not be hijacked."""
    from vllm_omni.quantization.mxfp8_config import OmniNPUMxfp8Config

    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: True)

    result = OmniNPUMxfp8Config.override_quantization_method(
        hf_quant_cfg={"quant_method": "int8"},
        user_quant=None,
    )
    assert result is None


def test_omni_npu_mxfp8_override_quantization_method_non_dict_returns_none(monkeypatch: pytest.MonkeyPatch):
    """hf_quant_cfg may be None (unquantized model) or a non-dict; must not raise."""
    from vllm_omni.quantization.mxfp8_config import OmniNPUMxfp8Config

    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: True)

    assert OmniNPUMxfp8Config.override_quantization_method(hf_quant_cfg=None, user_quant=None) is None
    assert OmniNPUMxfp8Config.override_quantization_method(hf_quant_cfg="mxfp8", user_quant=None) is None


# ---------------------------------------------------------------------------
# get_quant_method — FusedMoE / tid2eid forwarding (NPU AR-stage MoE routing)
# ---------------------------------------------------------------------------


def test_get_quant_method_fused_moe_forwards_tid2eid(monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture):
    """tid2eid (logical-to-physical expert id mapping) must be forwarded explicitly
    to AscendUnquantizedFusedMoEMethod, not silently dropped via **kwargs."""
    from vllm_omni.quantization.mxfp8_config import DiffusionMXFP8Config

    fake_cls, calls = _install_ascend_fused_moe_fake(monkeypatch)
    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: True)

    config = DiffusionMXFP8Config()
    layer = mocker.Mock(spec=FusedMoE)
    layer.moe_config = mocker.sentinel.moe_config
    sentinel_tid2eid = mocker.sentinel.tid2eid

    method = config.get_quant_method(layer, "blocks.0.ffn.experts", tid2eid=sentinel_tid2eid)

    assert isinstance(method, fake_cls)
    assert len(calls) == 1
    moe_config_arg, kwargs = calls[0]
    assert moe_config_arg is mocker.sentinel.moe_config
    assert kwargs["tid2eid"] is sentinel_tid2eid


def test_get_quant_method_fused_moe_without_tid2eid_passes_none(monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture):
    """When no logical-to-physical mapping is active, tid2eid must default to None
    rather than being omitted or raising."""
    from vllm_omni.quantization.mxfp8_config import DiffusionMXFP8Config

    fake_cls, calls = _install_ascend_fused_moe_fake(monkeypatch)
    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: True)

    config = DiffusionMXFP8Config()
    layer = mocker.Mock(spec=FusedMoE)
    layer.moe_config = mocker.sentinel.moe_config

    method = config.get_quant_method(layer, "blocks.0.ffn.experts")

    assert isinstance(method, fake_cls)
    assert calls[0][1]["tid2eid"] is None
