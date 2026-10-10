# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tests.diffusion.models.magi2.test_native_preview import _initialize_tiny_model, _tiny_config
from tests.helpers.mark import hardware_test
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
    # These CPU tests emulate provider dispatch with nine packed tokens and an available provider.
    monkeypatch.setattr(attention, "_MUSA_FA3_MIN_TOKENS", 0)
    monkeypatch.setattr(attention, "flash_attn_3_varlen_unsupported", lambda softcap, return_softmax_lse: None)


@pytest.mark.cpu
@pytest.mark.parametrize("sink_count", [0, 1, 2])
def test_musa_dispatch_reuses_fp32_sink_correction(monkeypatch, sink_count):
    _platform(monkeypatch)
    q, k, v, varlen = _inputs()
    out = torch.linspace(0.1, 0.9, q.numel()).reshape(q.shape).to(q.dtype)
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
    assert kwargs["return_softmax_lse"]
    assert kwargs["softcap"] == 2.0 and kwargs["softmax_scale"] == 64**-0.5
    assert kwargs["max_seqlen_q"] == 6 and kwargs["max_seqlen_k"] == 7
    assert kwargs["cu_seqlens_q"].dtype == kwargs["cu_seqlens_k"].dtype == torch.int32
    assert not kwargs.get("causal") and kwargs.get("sinks") is None
    if sink is not None:
        assert sink.dtype == torch.float32 and torch.equal(sink, original_sink)
        # A BF16 correction rounds differently, so the exact comparison above pins the FP32 path.
        bf16 = out * torch.sigmoid(lse.transpose(0, 1) - torch.logsumexp(sink, dim=0)).to(out.dtype).unsqueeze(-1)
        assert not torch.equal(bf16, expected)


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
def test_unavailable_fa3_falls_back_and_logs_a_stable_reason(monkeypatch):
    _platform(monkeypatch)
    q, k, v, varlen = _inputs()
    expected = torch.empty_like(q)
    fallback = Mock(return_value=expected)
    logger = Mock()
    monkeypatch.setattr(attention, "flash_attn_3_varlen_unsupported", Mock(return_value="provider missing"))
    monkeypatch.setattr(
        attention, "flash_attn_3_varlen", Mock(side_effect=AssertionError("unavailable provider called"))
    )
    monkeypatch.setattr(attention, "torch_varlen_attention_with_sink", fallback)
    monkeypatch.setattr(attention, "logger", logger)
    for _ in range(2):
        assert attention.packed_attention_with_sink(q, k, v, varlen) is expected
    assert fallback.call_count == 2
    # warning_once deduplicates on its arguments, so they must repeat exactly and hold no tensors.
    first, second = logger.warning_once.call_args_list
    assert first == second and all(isinstance(arg, str) for arg in first.args)


@pytest.mark.cpu
def test_reference_fallback_reads_host_sequence_bounds(monkeypatch):
    # Accelerator-resident q/k/v: the Torch reference must still get host bounds.
    _platform(monkeypatch, tensor_device="meta")
    q, k, v, varlen = _inputs()
    q, k, v = (tensor.to("meta") for tensor in (q, k, v))
    seen = []

    def fallback(q, k, v, *, cu_seqlens_q, cu_seqlens_k, **kwargs):
        seen.append((cu_seqlens_q.device.type, cu_seqlens_k.device.type, cu_seqlens_q.tolist(), cu_seqlens_k.tolist()))
        return q

    monkeypatch.setattr(attention, "flash_attn_3_varlen_unsupported", Mock(return_value="provider missing"))
    monkeypatch.setattr(attention, "torch_varlen_attention_with_sink", fallback)
    attention.packed_attention_with_sink(q, k, v, varlen)
    assert seen == [("cpu", "cpu", [0, 3, 9], [0, 4, 11])]


@pytest.mark.cpu
@pytest.mark.parametrize("cu,max_q,uses_fa3", [([0, 20, 40], 20, False), ([0, 40], 40, True), ([0, 8, 40], 32, True)])
def test_short_sequence_threshold_uses_the_longest_query_sequence(monkeypatch, cu, max_q, uses_fa3):
    _platform(monkeypatch)
    monkeypatch.setattr(attention, "_MUSA_FA3_MIN_TOKENS", 32)
    generator = torch.Generator().manual_seed(3)
    q = torch.randn(40, 4, 64, generator=generator).to(torch.bfloat16)
    k = torch.randn(40, 2, 64, generator=generator).to(torch.bfloat16)
    v = torch.randn(40, 2, 64, generator=generator).to(torch.bfloat16)
    cu_seqlens = torch.tensor(cu, dtype=torch.int64)
    provider = Mock(return_value=(q, torch.zeros(4, 40)))
    fallback = Mock(return_value=q)
    monkeypatch.setattr(attention, "flash_attn_3_varlen", provider)
    monkeypatch.setattr(attention, "torch_varlen_attention_with_sink", fallback)
    attention.packed_attention_with_sink(q, k, v, VarlenHandler(cu_seqlens, cu_seqlens, max_q, max_q))
    assert provider.called is uses_fa3
    assert fallback.called is not uses_fa3


@pytest.mark.cpu
@pytest.mark.parametrize(
    "exception",
    [
        RuntimeError("device failure"),
        ValueError("bad shape"),
        TypeError("bad ABI"),
        ImportError("late import"),
        NotImplementedError("late option"),
    ],
)
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
@pytest.mark.parametrize("version", ["2", "4"])
def test_musa_ignores_cuda_flash_attn_version(monkeypatch, version):
    _platform(monkeypatch)
    q, k, v, varlen = _inputs()
    provider = Mock(return_value=(q, torch.zeros(4, 9)))
    monkeypatch.setattr(attention, "flash_attn_3_varlen", provider)
    monkeypatch.setenv("MAGI2_FLASH_ATTN_VERSION", version)
    assert attention.packed_attention_with_sink(q, k, v, varlen) is q
    provider.assert_called_once()


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


@hardware_test(res={"musa": "S5000"}, num_cards=1)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim,kv_heads,sink_count,softcap", [(64, 4, 0, -1.0), (128, 2, 1, -1.0), (64, 2, 2, 2.0)])
def test_real_musa_fa3_matches_sink_oracle(monkeypatch, dtype, head_dim, kv_heads, sink_count, softcap):
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("requires a MUSA device")
    if attention.flash_attn_3_varlen_unsupported(softcap > 0, True) is not None:
        pytest.skip("standalone FA3 provider with LSE is not installed")
    # Force FA3 for operator coverage; the production short-sequence policy is
    # exercised separately by the tiny-transformer test below.
    monkeypatch.setattr(attention, "_MUSA_FA3_MIN_TOKENS", 0)
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
    # BF16 uses MATE's own FMHA tolerance: its Mubin kernel truncates the probabilities.
    tolerance = dict(rtol=0.01, atol=0.015) if dtype == torch.bfloat16 else dict(rtol=0.02, atol=0.003)
    torch.testing.assert_close(actual.cpu(), expected, **tolerance)


@hardware_test(res={"musa": "S5000"}, num_cards=1)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("sink_count", [1, 2])
def test_real_musa_fa3_matches_sink_oracle_across_tiles(monkeypatch, dtype, sink_count):
    """Uneven sequences longer than one tile, at the default short-sequence threshold."""
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("requires a MUSA device")
    if attention.flash_attn_3_varlen_unsupported(False, True) is not None:
        pytest.skip("standalone FA3 provider with LSE is not installed")
    generator = torch.Generator().manual_seed(23)
    q = torch.randn(300, 6, 128, generator=generator).to(dtype)
    k = torch.randn(330, 6, 128, generator=generator).to(dtype)
    v = torch.randn(330, 6, 128, generator=generator).to(dtype)
    cu_q = torch.tensor([0, 37, 300], dtype=torch.int64)
    cu_k = torch.tensor([0, 41, 330], dtype=torch.int64)
    sink = torch.linspace(-1.5, 2.5, sink_count * 6).reshape(sink_count, 6)
    expected = attention.torch_varlen_attention_with_sink(q, k, v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, sink=sink)
    provider = Mock(wraps=attention.flash_attn_3_varlen)
    monkeypatch.setattr(attention, "flash_attn_3_varlen", provider)
    actual = attention.packed_attention_with_sink(
        q.to("musa"), k.to("musa"), v.to("musa"), VarlenHandler(cu_q, cu_k, 263, 289), sink=sink.to("musa")
    )
    provider.assert_called_once()
    tolerance = dict(rtol=0.01, atol=0.015) if dtype == torch.bfloat16 else dict(rtol=0.02, atol=0.003)
    torch.testing.assert_close(actual.cpu(), expected, **tolerance)


@hardware_test(res={"musa": "S5000"}, num_cards=1)
def test_real_musa_tiny_transformer_uses_exact_short_sequence_fallback(monkeypatch):
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("requires a MUSA device")
    if attention.flash_attn_3_varlen_unsupported(False, True) is not None:
        pytest.skip("standalone FA3 provider with LSE is not installed")
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
    with torch.no_grad():
        actual = model(packed, coords, modalities, varlen)
    assert provider.call_count == 0
    assert torch.isfinite(actual).all()
