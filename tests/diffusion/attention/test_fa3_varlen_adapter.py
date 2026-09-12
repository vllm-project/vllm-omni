# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import inspect
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch

from vllm_omni.diffusion.attention.backends.utils import fa

pytestmark = [pytest.mark.diffusion, pytest.mark.local_model]


@pytest.fixture(autouse=True)
def clear_provider_cache():
    fa._external_fa3_varlen.cache_clear()
    yield
    fa._external_fa3_varlen.cache_clear()


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


@pytest.mark.cpu
@pytest.mark.parametrize("layout", ["fa3_fwd_interface", "flash_attn_interface", "flash_attn_3.interface"])
def test_provider_layout_and_cached_resolution(monkeypatch, layout):
    calls = []

    def provider(q, **kwargs):
        return q

    def load(name):
        calls.append(name)
        if name == layout:
            return SimpleNamespace(flash_attn_varlen_func=provider)
        raise ImportError(name)

    monkeypatch.setattr(fa.importlib, "import_module", load)
    assert fa._external_fa3_varlen()[0] is provider
    first_calls = calls[:]
    assert fa._external_fa3_varlen()[0] is provider
    assert calls == first_calls


@pytest.mark.cpu
def test_missing_provider_has_actionable_error(monkeypatch):
    def missing(name):
        raise ImportError(name)

    monkeypatch.setattr(fa.importlib, "import_module", missing)
    with pytest.raises(ImportError, match="standalone FlashAttention-3"):
        fa._external_fa3_varlen()


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
def test_optional_features_forwarded_only_when_supported(monkeypatch):
    q, k, v, kwargs = inputs()
    sink = torch.randn(2)
    seen: dict[str, Any] = {}

    def provider(*, sinks=None, softcap=0.0, deterministic=False, **kw):
        seen.update(sinks=sinks, softcap=softcap, deterministic=deterministic)
        return q

    set_provider(monkeypatch, provider)
    fa.flash_attn_3_varlen(q, k, v, **kwargs, sinks=sink, softcap=2.0, deterministic=True)
    assert seen == dict(sinks=sink, softcap=2.0, deterministic=True)


@pytest.mark.cpu
@pytest.mark.parametrize(
    "option", [dict(sinks=torch.ones(2)), dict(softcap=1.0), dict(deterministic=True), dict(return_softmax_lse=True)]
)
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


@pytest.mark.cpu
def test_bundled_backend_keeps_priority(monkeypatch):
    q, k, v, kwargs = inputs()
    lse = torch.zeros(2, 5)
    module = ModuleType("vllm.vllm_flash_attn")

    def bundled(*args, **kw):
        assert kw["fa_version"] == 3 and kw["return_softmax_lse"] is True
        return q, lse

    setattr(module, "flash_attn_varlen_func", bundled)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(fa, "resolve_vllm_flash_attn_version", lambda requested: 3)

    def unused():
        pytest.fail("standalone provider must not replace the bundled backend")

    monkeypatch.setattr(fa, "_external_fa3_varlen", unused)
    out, actual_lse = fa.vllm_flash_attn_varlen_with_lse(q, k, v, **kwargs)
    assert out is q and actual_lse is lse


@pytest.mark.cpu
@pytest.mark.parametrize("requested", [None, 3])
def test_shared_lse_entry_uses_standalone_fallback(monkeypatch, requested):
    q, k, v, kwargs = inputs()
    lse = torch.zeros(2, 5)
    monkeypatch.setitem(sys.modules, "vllm.vllm_flash_attn", None)

    def provider(*, return_attn_probs=False, **kw):
        assert return_attn_probs
        return q, lse

    set_provider(monkeypatch, provider)
    out, actual_lse = fa.vllm_flash_attn_varlen_with_lse(q, k, v, **kwargs, fa_version=requested)
    assert out is q and actual_lse is lse


@pytest.mark.cpu
@pytest.mark.parametrize("requested", [2, 4])
def test_explicit_versions_are_not_redirected(monkeypatch, requested):
    q, k, v, kwargs = inputs()
    monkeypatch.setitem(sys.modules, "vllm.vllm_flash_attn", None)
    with pytest.raises(ImportError):
        fa.vllm_flash_attn_varlen_with_lse(q, k, v, **kwargs, fa_version=requested)


@pytest.mark.parametrize(
    "device_kind", [pytest.param("cuda", marks=pytest.mark.cuda), pytest.param("musa", marks=pytest.mark.musa)]
)
@pytest.mark.parametrize("causal", [False, True])
def test_standalone_gpu_lse_parity(device_kind, causal):
    device_module = getattr(torch, device_kind, None)
    if device_module is None or not device_module.is_available():
        pytest.skip(f"requires {device_kind}")
    try:
        fa._external_fa3_varlen()
    except ImportError:
        pytest.skip("standalone FA3 provider is not installed")
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
