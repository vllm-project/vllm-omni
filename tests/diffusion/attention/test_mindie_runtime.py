# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contract tests for MindIE adapters; no NPU kernels are mocked in production."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_omni.diffusion.attention import layer as layer_mod
from vllm_omni.diffusion.attention.backends import flash_attn
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.data import AttentionSpec, AttnQuantSpec

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.fixture
def runtime(monkeypatch):
    mod = ModuleType("mindiesd")

    def execute(q, k, v, *, precision, layout, scale, q_rot=None, k_rot=None):
        return q

    setattr(mod, "quant_attention", Mock(side_effect=execute))
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    for module in (flash_attn, layer_mod):
        monkeypatch.setattr(module, "current_omni_platform", SimpleNamespace(is_npu=lambda: True, device_name="npu"))
    monkeypatch.setattr(flash_attn, "get_current_diffusion_config_or_none", lambda: None)
    yield mod


def flash(method="mxfp8", fallback=(), layout="BSND"):
    return flash_attn.FlashAttentionImpl(
        num_heads=2,
        head_size=64,
        softmax_scale=0.37,
        qkv_layout=layout,
        backend_kwargs={"quant": {"method": method, "fallback": list(fallback)}},
    )


@pytest.mark.parametrize("method", ["fp8", "mxfp8", "mxfp4", "float"])
def test_method_configuration_reaches_backend(method):
    spec = AttentionSpec(backend="FLASH_ATTN", quant={"method": method})
    assert spec.quant.enabled
    assert spec.backend_kwargs()["quant"] == {"method": method, "fallback": []}


@pytest.mark.parametrize(
    "quant",
    [
        {"method": "invalid"},
        {"method": "fp8", "fallback": ["fp8"]},
        {"method": "fp8", "fallback": ["float", "mxfp8"]},
        {"fallback": ["float"]},
        {"method": "mxfp8", "dtype_qk": "int8"},
        {"method": "fp8", "rotation_seed": True},
    ],
)
def test_invalid_quant_configuration(quant):
    with pytest.raises(ValueError):
        AttnQuantSpec(**quant)


@pytest.mark.parametrize("method", ["fp8", "mxfp8", "mxfp4"])
@pytest.mark.parametrize("layout", ["BSND", "BNSD"])
def test_exact_runtime_layout_scale_seed(runtime, method, layout):
    q = torch.randn(1, 128, 2, 64, dtype=torch.bfloat16)
    if layout == "BNSD":
        q = q.transpose(1, 2)
    impl = flash(method, layout=layout)
    assert impl.forward_npu(q, q, q) is q
    args, kwargs = runtime.quant_attention.call_args
    assert args[0] is q  # no unconditional transpose or extra quantization
    assert kwargs["layout"] == layout and kwargs["scale"] == 0.37 and kwargs["precision"] == method
    assert "rotation_seed" not in kwargs and "attn_mask" not in kwargs
    if method == "mxfp4":
        assert "q_rot" not in kwargs
    else:
        from vllm_omni.platforms.npu.quant.kv_quant_npu import get_quant_attention_rotation

        assert kwargs["q_rot"] is kwargs["k_rot"]
        torch.testing.assert_close(kwargs["q_rot"], get_quant_attention_rotation(q.device, q.dtype, 64, 425500))


def test_mask_is_preserved_for_explicit_float_fallback(runtime, monkeypatch):
    q = torch.randn(1, 130, 2, 64, dtype=torch.bfloat16)
    mask = torch.arange(130)[None, :] < 129
    call = Mock(side_effect=lambda q, k, v, **kw: q)
    monkeypatch.setattr(runtime, "attention_forward", call, raising=False)
    flash("mxfp4", ["mxfp8", "float"]).forward_npu(q, q, q, AttentionMetadata(attn_mask=mask))
    runtime.quant_attention.assert_not_called()
    forwarded = call.call_args.kwargs["attn_mask"]
    assert forwarded.shape == (1, 1, 130, 130)
    assert forwarded[..., :129].all() and not forwarded[..., 129].any()


def test_missing_runtime_falls_back_only_when_configured(runtime):
    q = torch.randn(1, 128, 2, 64, dtype=torch.bfloat16)
    del runtime.quant_attention
    with pytest.raises(ImportError, match="compatible MindIE"):
        flash("mxfp4").forward_npu(q, q, q)
    impl = flash("mxfp4", ["mxfp8", "float"])
    impl.forward_fa_npu = Mock(return_value=q)
    assert impl.forward_npu(q, q, q) is q
    impl.forward_fa_npu.assert_called_once()


def test_operator_failure_never_falls_back(runtime):
    q = torch.randn(1, 128, 2, 64, dtype=torch.bfloat16)
    runtime.quant_attention.side_effect = RuntimeError("operator failure")
    with pytest.raises(RuntimeError, match="operator failure"):
        flash("mxfp4", ["mxfp8", "float"]).forward_npu(q, q, q)
    runtime.quant_attention.assert_called_once()


def test_unsupported_shape_requires_explicit_fallback(runtime):
    q = torch.randn(1, 128, 2, 96, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="power-of-two"):
        flash("mxfp8").forward_npu(q, q, q)
    impl = flash("mxfp8", ["float"])
    impl.forward_fa_npu = Mock(return_value=q)
    assert impl.forward_npu(q, q, q) is q
    impl.forward_fa_npu.assert_called_once()
    runtime.quant_attention.assert_not_called()


def test_packed_metadata_never_reaches_quant_runtime(runtime):
    q = torch.randn(1, 128, 2, 64, dtype=torch.bfloat16)
    metadata = AttentionMetadata(extra={"cu_seqlens_q": torch.tensor([0, 64, 128])})
    with pytest.raises(ValueError, match="packed/varlen"):
        flash().forward_npu(q, q, q, metadata)
    runtime.quant_attention.assert_not_called()


def test_float_fallback_does_not_drop_packed_boundaries(runtime):
    q = torch.randn(1, 128, 2, 64, dtype=torch.bfloat16)
    metadata = AttentionMetadata(extra={"cu_seqlens_q": torch.tensor([0, 64, 128])})
    with pytest.raises(ValueError, match="explicit mask"):
        flash(fallback=["float"]).forward_npu(q, q, q, metadata)


@pytest.mark.parametrize("layout", ["BSND", "BNSD"])
def test_float_fallback_preserves_mask_layout_and_scale(runtime, monkeypatch, layout):
    call = Mock(side_effect=lambda q, k, v, **kw: q)
    monkeypatch.setattr(sys.modules["mindiesd"], "attention_forward", call, raising=False)
    q = torch.randn(1, 130, 2, 96, dtype=torch.bfloat16)
    if layout == "BNSD":
        q = q.transpose(1, 2)
    mask = torch.arange(130)[None, :] < 129
    flash(fallback=["float"], layout=layout).forward_npu(q, q, q, AttentionMetadata(attn_mask=mask))
    kwargs = call.call_args.kwargs
    assert kwargs["layout"] == layout and kwargs["scale"] == 0.37
    assert kwargs["head_first"] == (layout == "BNSD")
    assert kwargs["attn_mask"].shape == (1, 1, 130, 130)
    assert kwargs["attn_mask"][..., :129].all() and not kwargs["attn_mask"][..., 129].any()


def test_gpu_variant_only_configuration_is_preserved():
    spec = AttentionSpec(backend="FLASHINFER_ATTN", quant={"flashinfer_backend": "trtllm-gen"})
    assert spec.quant.enabled
    assert spec.backend_kwargs()["quant"] == {"flashinfer_backend": "trtllm-gen"}


@pytest.mark.parametrize("layout", ["BSND", "BNSD"])
def test_float_fallback_preserves_native_causal_attention(runtime, monkeypatch, layout):
    npu = ModuleType("torch_npu")
    native = Mock(side_effect=lambda q, k, v, **kw: (q, None))
    monkeypatch.setattr(npu, "npu_fusion_attention", native, raising=False)
    monkeypatch.setitem(sys.modules, "torch_npu", npu)
    q = torch.randn(1, 4, 2, 64, dtype=torch.bfloat16)
    kv = q[:, :2]
    if layout == "BNSD":
        q, kv = q.transpose(1, 2), kv.transpose(1, 2)
    impl = flash(fallback=["float"], layout=layout)
    impl.causal = True
    assert impl.forward_npu(q, kv, kv).shape == q.shape
    kwargs = native.call_args.kwargs
    assert kwargs["input_layout"] == "BSND" and kwargs["sparse_mode"] == 3
    assert kwargs["inner_precise"] == 2 and kwargs["scale"] == 0.37
    runtime.quant_attention.assert_not_called()


def test_custom_attention_initialization_keeps_upstream_optout(monkeypatch):
    monkeypatch.setattr(layer_mod, "get_current_diffusion_config_or_none", lambda: None)
    monkeypatch.setattr(layer_mod, "build_parallel_attention_strategy", lambda **kw: object())
    monkeypatch.setattr(layer_mod, "NoParallelAttention", lambda: object())
    custom = torch.nn.Identity()
    layer = layer_mod.Attention(
        num_heads=2,
        head_size=64,
        softmax_scale=0.125,
        causal=False,
        custom_attention=custom,
        skip_sequence_parallel=True,
    )
    assert layer.attention is custom and layer._kv_cache_dtype is None
