# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tests.diffusion.models.magi2.test_native_preview import _initialize_tiny_model, _tiny_config
from vllm_omni.diffusion.models.magi2 import attention
from vllm_omni.diffusion.models.magi2.attention import VarlenHandler
from vllm_omni.diffusion.models.magi2.modeling_magi2 import Magi2PreviewTransformer, Modality

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model]


def _inputs(dtype=torch.bfloat16, head_dim=64, kv_heads=2):
    generator = torch.Generator().manual_seed(19)
    q = torch.randn(9, 4, head_dim, generator=generator).to(dtype)
    k = torch.randn(11, kv_heads, head_dim, generator=generator).to(dtype)
    v = torch.randn(11, kv_heads, head_dim, generator=generator).to(dtype)
    cu_q = torch.tensor([0, 3, 9], dtype=torch.int64)
    cu_k = torch.tensor([0, 4, 11], dtype=torch.int64)
    return q, k, v, VarlenHandler(cu_q, cu_k, 6, 7)


def _platform(monkeypatch, kind="musa", tensor_device="cpu"):
    # CPU dispatch tests emulate the active backend without pretending to run GPU kernels.
    monkeypatch.setattr(
        attention,
        "current_omni_platform",
        SimpleNamespace(
            is_cuda=lambda: kind == "cuda",
            is_musa=lambda: kind == "musa",
            device_type=tensor_device,
        ),
    )
    monkeypatch.delenv("MAGI2_FLASH_ATTN_VERSION", raising=False)
    monkeypatch.delenv("MAGI2_DETERMINISTIC", raising=False)
    # These CPU tests emulate provider dispatch even with five packed tokens.
    monkeypatch.setattr(attention, "_MUSA_FA3_MIN_TOKENS", 0)


@pytest.mark.cpu
@pytest.mark.parametrize("sink_count", [0, 1, 2])
def test_musa_dispatch_reuses_fp32_sink_correction(monkeypatch, sink_count):
    _platform(monkeypatch)
    q, k, v, varlen = _inputs()
    out = torch.full_like(q, 0.5)
    lse = torch.linspace(-2, 3, 36).reshape(4, 9)
    sink = None if not sink_count else torch.linspace(-0.98765, 0.12345, sink_count * 4).reshape(sink_count, 4)
    original_sink = None if sink is None else sink.clone()
    provider = Mock(return_value=(out, lse))
    monkeypatch.setattr(attention, "flash_attn_3_varlen", provider)
    monkeypatch.setattr(
        attention, "_resolve_flash_attn_version", Mock(side_effect=AssertionError("CUDA resolver used"))
    )
    monkeypatch.setattr(
        attention, "torch_varlen_attention_with_sink", Mock(side_effect=AssertionError("fallback used"))
    )
    monkeypatch.setenv("MAGI2_FLASH_ATTN_VERSION", "3")
    monkeypatch.setenv("MAGI2_DETERMINISTIC", "1")
    actual = attention.packed_attention_with_sink(q, k, v, varlen, softcap=2.0, sink=sink)
    expected = (
        out
        if sink is None
        else (out.float() * torch.sigmoid(lse.transpose(0, 1) - torch.logsumexp(sink, dim=0)).unsqueeze(-1)).to(
            out.dtype
        )
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    kwargs = provider.call_args.kwargs
    assert kwargs["return_softmax_lse"] and kwargs["deterministic"]
    assert kwargs["softcap"] == 2.0 and kwargs["softmax_scale"] == 64**-0.5
    assert kwargs["max_seqlen_q"] == 6 and kwargs["max_seqlen_k"] == 7
    assert kwargs["cu_seqlens_q"].dtype == kwargs["cu_seqlens_k"].dtype == torch.int32
    assert not kwargs["causal"] and kwargs.get("sinks") is None
    if sink is not None:
        assert sink.dtype == torch.float32 and torch.equal(sink, original_sink)


@pytest.mark.cpu
def test_musa_cuda_alias_does_not_select_cuda_resolver(monkeypatch):
    _platform(monkeypatch)
    q, k, v, varlen = _inputs()
    alias_q = SimpleNamespace(shape=q.shape, dtype=q.dtype, device=q.device, is_cuda=True)
    provider = Mock(return_value=(q, torch.zeros(4, 9)))
    monkeypatch.setattr(attention, "flash_attn_3_varlen", provider)
    monkeypatch.setattr(
        attention, "_resolve_flash_attn_version", Mock(side_effect=AssertionError("CUDA resolver used"))
    )
    assert attention.packed_attention_with_sink(alias_q, k, v, varlen) is q
    provider.assert_called_once()


@pytest.mark.cpu
@pytest.mark.parametrize(
    "exception", [ImportError("provider missing"), NotImplementedError("deterministic unsupported")]
)
def test_expected_unavailability_falls_back(monkeypatch, exception):
    _platform(monkeypatch)
    q, k, v, varlen = _inputs()
    expected = torch.empty_like(q)
    fallback = Mock(return_value=expected)
    monkeypatch.setattr(attention, "flash_attn_3_varlen", Mock(side_effect=exception))
    monkeypatch.setattr(attention, "torch_varlen_attention_with_sink", fallback)
    assert attention.packed_attention_with_sink(q, k, v, varlen) is expected
    fallback.assert_called_once()


@pytest.mark.cpu
@pytest.mark.parametrize("exception", [RuntimeError("device failure"), ValueError("bad shape"), TypeError("bad ABI")])
def test_provider_failures_are_not_hidden(monkeypatch, exception):
    _platform(monkeypatch)
    q, k, v, varlen = _inputs()
    fallback = Mock()
    monkeypatch.setattr(attention, "flash_attn_3_varlen", Mock(side_effect=exception))
    monkeypatch.setattr(attention, "torch_varlen_attention_with_sink", fallback)
    with pytest.raises(type(exception), match=str(exception)):
        attention.packed_attention_with_sink(q, k, v, varlen)
    fallback.assert_not_called()


@pytest.mark.cpu
@pytest.mark.parametrize("version", ["2", "4", "invalid"])
def test_musa_does_not_redirect_other_versions(monkeypatch, version):
    _platform(monkeypatch)
    q, k, v, varlen = _inputs()
    provider = Mock()
    monkeypatch.setattr(attention, "flash_attn_3_varlen", provider)
    monkeypatch.setenv("MAGI2_FLASH_ATTN_VERSION", version)
    with pytest.raises(ValueError):
        attention.packed_attention_with_sink(q, k, v, varlen)
    provider.assert_not_called()


@pytest.mark.cpu
@pytest.mark.parametrize("case", ["fp32", "host_tensor", "empty_query", "empty_key"])
def test_ineligible_inputs_keep_reference_path(monkeypatch, case):
    _platform(monkeypatch, tensor_device="musa" if case == "host_tensor" else "cpu")
    q, k, v, varlen = _inputs(torch.float32 if case == "fp32" else torch.bfloat16)
    q = q[:0] if case == "empty_query" else q
    k = k[:0] if case == "empty_key" else k
    provider, fallback = Mock(), Mock(return_value=q)
    monkeypatch.setattr(attention, "flash_attn_3_varlen", provider)
    monkeypatch.setattr(attention, "torch_varlen_attention_with_sink", fallback)
    assert attention.packed_attention_with_sink(q, k, v, varlen) is q
    provider.assert_not_called()
    fallback.assert_called_once()


@pytest.mark.cpu
def test_cuda_still_uses_bundled_version_selection(monkeypatch):
    _platform(monkeypatch, kind="cuda")
    q, k, v, varlen = _inputs()
    cuda_q = SimpleNamespace(shape=q.shape, dtype=q.dtype, device=q.device, is_cuda=True)
    resolver, bundled = Mock(return_value=2), Mock(return_value=(q, torch.zeros(4, 9)))
    monkeypatch.setattr(attention, "_resolve_flash_attn_version", resolver)
    monkeypatch.setattr(attention, "vllm_flash_attn_varlen_with_lse", bundled)
    monkeypatch.setattr(attention, "flash_attn_3_varlen", Mock(side_effect=AssertionError("standalone path used")))
    assert attention.packed_attention_with_sink(cuda_q, k, v, varlen, softcap=2) is q
    resolver.assert_called_once_with()
    assert bundled.call_args.kwargs["fa_version"] == 2


@pytest.mark.musa
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim,kv_heads,sink_count,softcap", [(64, 4, 0, -1.0), (128, 2, 1, -1.0), (64, 2, 2, 2.0)])
def test_real_musa_fa3_matches_sink_oracle(monkeypatch, dtype, head_dim, kv_heads, sink_count, softcap):
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("requires a MUSA device")
    q, k, v, varlen = _inputs(dtype, head_dim, kv_heads)
    sink = None if not sink_count else torch.linspace(-1.98765, 2.12345, sink_count * 4).reshape(sink_count, 4)
    expected = attention.torch_varlen_attention_with_sink(
        q,
        k,
        v,
        cu_seqlens_q=varlen.cu_seqlens_q,
        cu_seqlens_k=varlen.cu_seqlens_k,
        sink=sink,
        softcap=softcap,
    )
    provider = Mock(wraps=attention.flash_attn_3_varlen)
    monkeypatch.setattr(attention, "flash_attn_3_varlen", provider)
    monkeypatch.setattr(
        attention, "torch_varlen_attention_with_sink", Mock(side_effect=AssertionError("fallback is not validation"))
    )
    monkeypatch.setenv("MAGI2_FLASH_ATTN_VERSION", "3")
    monkeypatch.setenv("MAGI2_DETERMINISTIC", "1")
    actual = attention.packed_attention_with_sink(
        q.to("musa"),
        k.to("musa"),
        v.to("musa"),
        varlen,
        sink=None if sink is None else sink.to("musa"),
        softcap=softcap,
    )
    provider.assert_called_once()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.02, atol=0.003)


@pytest.mark.musa
def test_real_musa_tiny_transformer_uses_exact_short_sequence_fallback(monkeypatch):
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("requires a MUSA device")
    config = replace(_tiny_config(torch.bfloat16, num_layers=2), hidden_size=128, head_dim=64)
    model = Magi2PreviewTransformer(config).eval()
    _initialize_tiny_model(model, seed=119)
    model = model.to("musa")
    generator = torch.Generator().manual_seed(13)
    packed = torch.randn(6, 4, generator=generator).to(device="musa", dtype=torch.bfloat16)
    coords = torch.ones(6, 9, device="musa")
    modalities = torch.tensor(
        [Modality.VIDEO, Modality.VIDEO, Modality.AUDIO, Modality.AUDIO, Modality.TEXT, Modality.TEXT], device="musa"
    )
    cu = torch.tensor([0, 6], dtype=torch.int32, device="musa")
    varlen = VarlenHandler(cu, cu, 6, 6)
    provider = Mock(wraps=attention.flash_attn_3_varlen)
    monkeypatch.setattr(attention, "flash_attn_3_varlen", provider)
    monkeypatch.setenv("MAGI2_FLASH_ATTN_VERSION", "3")
    with torch.no_grad():
        actual = model(packed, coords, modalities, varlen)
    assert provider.call_count == 0
    assert torch.isfinite(actual).all()
