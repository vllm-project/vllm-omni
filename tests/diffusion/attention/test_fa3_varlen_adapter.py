# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import inspect
import sys
from types import ModuleType
from typing import Any

import pytest
import torch

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.attention.backends.utils import fa

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model]


@pytest.fixture(autouse=True)
def clear_provider_cache():
    fa._external_fa3_varlen.cache_clear()
    fa.flash_attn_3_varlen_unsupported.cache_clear()
    yield
    fa._external_fa3_varlen.cache_clear()
    fa.flash_attn_3_varlen_unsupported.cache_clear()


def inputs():
    q = torch.randn(5, 2, 64)
    k = torch.randn(7, 1, 64)
    v = torch.randn_like(k)
    kwargs = dict(
        cu_seqlens_q=torch.tensor([0, 2, 5], dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, 3, 7], dtype=torch.int32),
        max_seqlen_q=3,
        max_seqlen_k=4,
    )
    return q, k, v, kwargs


def set_provider(monkeypatch, func):
    monkeypatch.setattr(fa, "_external_fa3_varlen", lambda: (func, frozenset(inspect.signature(func).parameters)))


def install_module(monkeypatch, **attributes):
    module = ModuleType("flash_attn_interface")
    for name, value in attributes.items():
        setattr(module, name, value)
    monkeypatch.setitem(sys.modules, "flash_attn_interface", module)


@pytest.mark.cpu
def test_provider_resolution_is_cached(monkeypatch):
    def provider(q, *, return_attn_probs=False, **kwargs):
        return q

    install_module(monkeypatch, flash_attn_varlen_func=provider)
    func, parameters = fa._external_fa3_varlen()
    assert func is provider and "return_attn_probs" in parameters
    monkeypatch.setitem(sys.modules, "flash_attn_interface", None)
    assert fa._external_fa3_varlen()[0] is provider


@pytest.mark.cpu
@pytest.mark.parametrize("installed", [False, True])
def test_missing_provider_is_reported_once_with_an_actionable_error(monkeypatch, installed):
    if installed:
        install_module(monkeypatch)  # the module exists but has no flash_attn_varlen_func
    else:
        monkeypatch.setitem(sys.modules, "flash_attn_interface", None)
    q, k, v, kwargs = inputs()
    assert fa._external_fa3_varlen() is None
    reason = fa.flash_attn_3_varlen_unsupported(False, True)
    assert "flash_attn_interface" in reason
    assert fa.flash_attn_3_varlen_unsupported(False, True) is reason
    with pytest.raises(ImportError, match="standalone FlashAttention-3"):
        fa.flash_attn_3_varlen(q, k, v, **kwargs, return_softmax_lse=True)


@pytest.mark.cpu
def test_uninspectable_provider_is_reported_as_unsupported(monkeypatch):
    class Provider:
        __signature__ = "not a signature"  # inspect.signature raises TypeError

        def __call__(self, *args, **kwargs):
            raise AssertionError("an uninspectable provider must not be called")

    install_module(monkeypatch, flash_attn_varlen_func=Provider())
    q, k, v, kwargs = inputs()
    reason = fa.flash_attn_3_varlen_unsupported(False, True)
    assert reason is not None and "signature" in reason
    with pytest.raises(NotImplementedError, match="signature"):
        fa.flash_attn_3_varlen(q, k, v, **kwargs, return_softmax_lse=True)


@pytest.mark.cpu
@pytest.mark.parametrize("returns_tuple", [False, True])
def test_output_only_contract(monkeypatch, returns_tuple):
    q, k, v, kwargs = inputs()
    seen: dict[str, Any] = {}

    def provider(**kw):
        seen.update(kw)
        return (q, torch.zeros(2, 5)) if returns_tuple else q

    set_provider(monkeypatch, provider)
    assert fa.flash_attn_3_varlen(q, k, v, **kwargs, softmax_scale=0.2, causal=True) is q
    assert seen["q"] is q and seen["cu_seqlens_q"] is kwargs["cu_seqlens_q"]
    assert seen["softmax_scale"] == 0.2 and seen["causal"] is True
    assert "return_attn_probs" not in seen and "return_softmax_lse" not in seen


@pytest.mark.cpu
@pytest.mark.parametrize("flag", ["return_attn_probs", "return_softmax_lse"])
def test_lse_request_aliases(monkeypatch, flag):
    q, k, v, kwargs = inputs()
    lse = torch.randn(2, 5)
    seen: dict[str, Any] = {}

    def provider(**kw):
        seen.update(kw)
        return q, lse, None

    monkeypatch.setattr(fa, "_external_fa3_varlen", lambda: (provider, frozenset((flag,))))
    out, actual_lse = fa.flash_attn_3_varlen(q, k, v, **kwargs, return_softmax_lse=True)
    assert out is q and actual_lse is lse
    assert seen[flag] is True


@pytest.mark.cpu
def test_non_positive_softcap_is_forwarded_as_zero(monkeypatch):
    q, k, v, kwargs = inputs()
    seen: dict[str, Any] = {}

    def provider(*, softcap=None, **kw):
        seen["softcap"] = softcap
        return q

    set_provider(monkeypatch, provider)
    assert fa.flash_attn_3_varlen(q, k, v, **kwargs, softcap=-1.0) is q
    # MATE only keeps its default kernel for softcap == 0.0, so a disabled softcap must arrive as 0.0.
    assert seen == {"softcap": 0.0}


@pytest.mark.cpu
def test_softcap_forwarded_when_supported(monkeypatch):
    q, k, v, kwargs = inputs()
    seen: dict[str, Any] = {}

    def provider(*, softcap=0.0, **kw):
        seen["softcap"] = softcap
        return q

    set_provider(monkeypatch, provider)
    fa.flash_attn_3_varlen(q, k, v, **kwargs, softcap=2.0)
    assert seen == {"softcap": 2.0}


@pytest.mark.cpu
@pytest.mark.parametrize("option", [dict(softcap=1.0), dict(return_softmax_lse=True)])
def test_unsupported_features_fail_closed(monkeypatch, option):
    q, k, v, kwargs = inputs()

    def provider(**kw):
        pytest.fail("unsupported features must fail before calling the provider")

    set_provider(monkeypatch, provider)
    with pytest.raises(NotImplementedError):
        fa.flash_attn_3_varlen(q, k, v, **kwargs, **option)


@pytest.mark.cpu
@pytest.mark.parametrize(
    "result", [None, torch.zeros(5, 2, 64), (torch.zeros(5, 2, 64), None), (torch.zeros(5, 2, 64), torch.zeros(5, 2))]
)
def test_malformed_lse_results_are_rejected(monkeypatch, result):
    q, k, v, kwargs = inputs()

    def provider(*, return_attn_probs=False, **kw):
        return result

    set_provider(monkeypatch, provider)
    with pytest.raises(RuntimeError):
        fa.flash_attn_3_varlen(q, k, v, **kwargs, return_softmax_lse=True)


@pytest.mark.parametrize(
    "device_kind",
    [
        pytest.param("cuda", marks=hardware_marks(res={"cuda": "L4"}, num_cards=1)),
        pytest.param("musa", marks=hardware_marks(res={"musa": "S5000"}, num_cards=1)),
    ],
)
@pytest.mark.parametrize("causal", [False, True])
def test_standalone_gpu_lse_parity(device_kind, causal):
    device_module = getattr(torch, device_kind, None)
    if device_module is None or not device_module.is_available():
        pytest.skip(f"requires {device_kind}")
    if fa.flash_attn_3_varlen_unsupported(False, True) is not None:
        pytest.skip("standalone FA3 provider with LSE is not installed")
    torch.manual_seed(42)
    q, k, v, kwargs = inputs()
    q, k, v = [x.to(device=device_kind, dtype=torch.bfloat16) for x in (q, k, v)]
    kwargs = {key: value.to(device_kind) if isinstance(value, torch.Tensor) else value for key, value in kwargs.items()}
    out, lse = fa.flash_attn_3_varlen(q, k, v, **kwargs, causal=causal, return_softmax_lse=True)
    expected, expected_lse = [], []
    for qs, qe, ks, ke in ((0, 2, 0, 3), (2, 5, 3, 7)):
        qq = q[qs:qe].float()
        kk = k[ks:ke].float().expand(-1, 2, -1)
        vv = v[ks:ke].float().expand(-1, 2, -1)
        score = torch.einsum("qhd,khd->hqk", qq, kk) / 8
        if causal:
            qi = torch.arange(qe - qs, device=device_kind)[:, None]
            ki = torch.arange(ke - ks, device=device_kind)[None, :]
            score.masked_fill_(ki > qi + (ke - ks) - (qe - qs), float("-inf"))
        expected_lse.append(torch.logsumexp(score, -1))
        expected.append(torch.einsum("hqk,khd->qhd", torch.softmax(score, -1), vv))
    torch.testing.assert_close(out.float(), torch.cat(expected), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(lse.float(), torch.cat(expected_lse, -1), rtol=3e-3, atol=3e-3)
