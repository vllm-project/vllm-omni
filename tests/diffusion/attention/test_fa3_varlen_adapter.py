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
from vllm_omni.platforms import current_omni_platform

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


_NON_PAGED_KEYS = {"q", "k", "v", "cu_seqlens_q", "cu_seqlens_k", "max_seqlen_q", "max_seqlen_k", "softmax_scale"}


def paged_inputs():
    q = torch.randn(5, 2, 64)
    pages = torch.randn(3, 16, 1, 64)
    kwargs = dict(
        cu_seqlens_q=torch.tensor([0, 2, 5], dtype=torch.int32),
        max_seqlen_q=3,
        max_seqlen_k=32,
        seqused_k=torch.tensor([20, 7], dtype=torch.int32),
        block_table=torch.tensor([[2, 0], [1, 0]], dtype=torch.int32),
    )
    return q, pages, kwargs


def paged_provider(
    q,
    k,
    v,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    seqused_k=None,
    block_table=None,
    softmax_scale=None,
    causal=False,
    return_softmax_lse=False,
    out=None,
):
    raise AssertionError("replaced per test")


@pytest.mark.cpu
def test_non_paged_calls_pass_no_paged_options(monkeypatch):
    q, k, v, kwargs = inputs()
    seen: dict[str, Any] = {}

    def provider(**kw):
        seen.update(kw)
        return q

    monkeypatch.setattr(
        fa, "_external_fa3_varlen", lambda: (provider, frozenset(inspect.signature(paged_provider).parameters))
    )
    assert fa.flash_attn_3_varlen(q, k, v, **kwargs) is q
    # Providers that predate paged KV must keep receiving exactly the non-paged arguments.
    assert set(seen) == _NON_PAGED_KEYS | {"causal"}


@pytest.mark.cpu
@pytest.mark.parametrize("return_lse", [False, True])
def test_paged_call_forwards_block_table_seqused_k_and_out(monkeypatch, return_lse):
    q, pages, kwargs = paged_inputs()
    out = torch.empty_like(q)
    lse = torch.randn(2, 5)
    seen: dict[str, Any] = {}

    def provider(**kw):
        seen.update(kw)
        kw["out"].copy_(q)
        return (kw["out"], lse) if kw.get("return_softmax_lse") else kw["out"]

    monkeypatch.setattr(
        fa, "_external_fa3_varlen", lambda: (provider, frozenset(inspect.signature(paged_provider).parameters))
    )
    result = fa.flash_attn_3_varlen(
        q, pages, pages, **kwargs, softmax_scale=0.5, out=out, return_softmax_lse=return_lse
    )

    assert set(seen) == _NON_PAGED_KEYS | {"causal", "seqused_k", "block_table", "out"} | (
        {"return_softmax_lse"} if return_lse else set()
    )
    assert seen["cu_seqlens_k"] is None
    assert seen["block_table"] is kwargs["block_table"] and seen["seqused_k"] is kwargs["seqused_k"]
    assert seen["out"] is out and seen["k"] is pages and seen["max_seqlen_k"] == 32
    if return_lse:
        assert result[0] is out and result[1] is lse
    else:
        assert result is out
    assert torch.equal(out, q)


@pytest.mark.cpu
@pytest.mark.parametrize("return_lse", [False, True])
def test_out_holds_the_output_of_a_provider_that_returns_a_new_tensor(monkeypatch, return_lse):
    q, pages, kwargs = paged_inputs()
    out = torch.zeros_like(q)
    result_tensor = torch.randn_like(q)

    def provider(**kw):
        return (result_tensor, torch.zeros(2, 5)) if return_lse else result_tensor

    monkeypatch.setattr(
        fa, "_external_fa3_varlen", lambda: (provider, frozenset(inspect.signature(paged_provider).parameters))
    )
    result = fa.flash_attn_3_varlen(q, pages, pages, **kwargs, out=out, return_softmax_lse=return_lse)

    assert (result[0] if return_lse else result) is out
    assert torch.equal(out, result_tensor)


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("missing", "option"),
    [("block_table", "paged KV"), ("seqused_k", "seqused_k"), ("out", "out tensor")],
)
def test_paged_options_fail_closed_without_provider_support(monkeypatch, missing, option):
    q, pages, kwargs = paged_inputs()

    def provider(**kw):
        pytest.fail("unsupported options must fail before calling the provider")

    parameters = frozenset(inspect.signature(paged_provider).parameters) - {missing}
    monkeypatch.setattr(fa, "_external_fa3_varlen", lambda: (provider, parameters))
    with pytest.raises(NotImplementedError, match=option):
        fa.flash_attn_3_varlen(q, pages, pages, **kwargs, out=torch.empty_like(q))
    assert fa.flash_attn_3_varlen_unsupported(**{missing: True}) is not None


@pytest.mark.cpu
@pytest.mark.parametrize("case", ["no_seqused_k", "cu_seqlens_k", "no_keys"])
def test_paged_argument_contract(monkeypatch, case):
    q, pages, kwargs = paged_inputs()

    def provider(**kw):
        pytest.fail("invalid arguments must fail before calling the provider")

    set_provider(monkeypatch, provider)
    if case == "no_seqused_k":
        del kwargs["seqused_k"]
    elif case == "cu_seqlens_k":
        kwargs["cu_seqlens_k"] = torch.tensor([0, 20, 27], dtype=torch.int32)
    else:
        del kwargs["block_table"]
    with pytest.raises(ValueError, match="seqused_k|cu_seqlens_k"):
        fa.flash_attn_3_varlen(q, pages, pages, **kwargs)


_VLLM_KEYS = {
    "q",
    "k",
    "v",
    "cu_seqlens_q",
    "cu_seqlens_k",
    "max_seqlen_q",
    "max_seqlen_k",
    "dropout_p",
    "softmax_scale",
    "causal",
    "window_size",
    "softcap",
    "return_softmax_lse",
    "fa_version",
}


def install_vllm_flash_attn(monkeypatch, result, seen):
    def flash_attn_varlen_func(q, k, v, max_seqlen_q, cu_seqlens_q, max_seqlen_k, **kwargs):
        seen.update(
            kwargs, q=q, k=k, v=v, max_seqlen_q=max_seqlen_q, cu_seqlens_q=cu_seqlens_q, max_seqlen_k=max_seqlen_k
        )
        return result

    module = ModuleType("vllm.vllm_flash_attn")
    setattr(module, "flash_attn_varlen_func", flash_attn_varlen_func)
    monkeypatch.setitem(sys.modules, "vllm.vllm_flash_attn", module)
    requested: list[Any] = []

    def resolve(version=None):
        requested.append(version)
        return 3

    monkeypatch.setattr(fa, "resolve_vllm_flash_attn_version", resolve)
    return requested


@pytest.mark.cpu
@pytest.mark.parametrize("return_lse", [False, True])
def test_vllm_paged_varlen_passes_paged_keys_without_cu_seqlens_k(monkeypatch, return_lse):
    q, pages, kwargs = paged_inputs()
    out = torch.empty_like(q)
    lse = torch.randn(2, 5)
    seen: dict[str, Any] = {}
    requested = install_vllm_flash_attn(monkeypatch, (out, lse) if return_lse else out, seen)

    result = fa.vllm_flash_attn_paged_varlen(
        q, pages, pages, **kwargs, softmax_scale=0.5, out=out, return_softmax_lse=return_lse, fa_version=2
    )

    assert requested == [2]
    assert set(seen) == _VLLM_KEYS | {"seqused_k", "block_table", "out"}
    assert seen["fa_version"] == 3
    assert seen["cu_seqlens_k"] is None
    assert seen["seqused_k"] is kwargs["seqused_k"] and seen["block_table"] is kwargs["block_table"]
    assert seen["cu_seqlens_q"] is kwargs["cu_seqlens_q"]
    assert (seen["max_seqlen_q"], seen["max_seqlen_k"]) == (3, 32)
    assert seen["out"] is out and seen["return_softmax_lse"] is return_lse
    assert seen["softmax_scale"] == 0.5 and seen["causal"] is False and seen["dropout_p"] == 0.0
    if return_lse:
        assert result[0] is out and result[1] is lse
    else:
        assert result is out


@pytest.mark.cpu
def test_vllm_varlen_with_lse_forwards_its_arguments(monkeypatch):
    q, k, v, kwargs = inputs()
    out, lse = torch.empty_like(q), torch.randn(2, 5)
    seen: dict[str, Any] = {}
    requested = install_vllm_flash_attn(monkeypatch, (out, lse), seen)

    result = fa.vllm_flash_attn_varlen_with_lse(
        q, k, v, **kwargs, softmax_scale=0.5, softcap=-1.0, causal=True, deterministic=True, fa_version=2
    )

    assert result[0] is out and result[1] is lse
    assert requested == [2]
    # Non-paged calls pass no paged options.
    assert set(seen) == _VLLM_KEYS | {"deterministic"}
    assert seen["cu_seqlens_q"] is kwargs["cu_seqlens_q"] and seen["cu_seqlens_k"] is kwargs["cu_seqlens_k"]
    assert (seen["max_seqlen_q"], seen["max_seqlen_k"], seen["fa_version"]) == (3, 4, 3)
    assert seen["deterministic"] is True and seen["return_softmax_lse"] is True
    assert seen["softmax_scale"] == 0.5 and seen["causal"] is True and seen["softcap"] == 0.0
    assert seen["dropout_p"] == 0.0 and seen["window_size"] is None


@pytest.mark.cpu
def test_vllm_paged_varlen_rejects_a_transposed_lse(monkeypatch):
    q, pages, kwargs = paged_inputs()
    install_vllm_flash_attn(monkeypatch, (torch.empty_like(q), torch.zeros(5, 2)), {})
    with pytest.raises(RuntimeError, match=r"softmax_lse shape \(2, 5\)"):
        fa.vllm_flash_attn_paged_varlen(q, pages, pages, **kwargs, return_softmax_lse=True)


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


@pytest.mark.parametrize(
    "device_kind",
    [
        pytest.param("cuda", marks=hardware_marks(res={"cuda": "L4"}, num_cards=1)),
        pytest.param("musa", marks=hardware_marks(res={"musa": "S5000"}, num_cards=1)),
    ],
)
def test_paged_gpu_parity(device_kind):
    # ROCm also reports torch.cuda as available but has no vLLM paged FlashAttention here.
    on_platform = current_omni_platform.is_cuda() if device_kind == "cuda" else current_omni_platform.is_musa()
    device_module = getattr(torch, device_kind, None)
    if not on_platform or device_module is None or not device_module.is_available():
        pytest.skip(f"requires the {device_kind} platform")
    paged = dict(block_table=True, seqused_k=True, out=True)
    if device_kind == "musa" and fa.flash_attn_3_varlen_unsupported(False, True, **paged) is not None:
        pytest.skip("standalone FA3 provider with paged KV is not installed")
    torch.manual_seed(42)
    q, k_pages, kwargs = paged_inputs()
    v_pages = torch.randn_like(k_pages)
    q, k_pages, v_pages = [x.to(device=device_kind, dtype=torch.bfloat16) for x in (q, k_pages, v_pages)]
    kwargs = {key: value.to(device_kind) if isinstance(value, torch.Tensor) else value for key, value in kwargs.items()}
    out = torch.empty_like(q)
    run = fa.flash_attn_3_varlen if device_kind == "musa" else fa.vllm_flash_attn_paged_varlen
    result, lse = run(q, k_pages, v_pages, **kwargs, out=out, return_softmax_lse=True)
    assert result is out
    expected, expected_lse = [], []
    for row, (qs, qe) in enumerate(((0, 2), (2, 5))):
        used = int(kwargs["seqused_k"][row])
        pages = kwargs["block_table"][row, : -(-used // 16)].long()
        kk = k_pages[pages].flatten(0, 1)[:used].float().expand(-1, 2, -1)
        vv = v_pages[pages].flatten(0, 1)[:used].float().expand(-1, 2, -1)
        score = torch.einsum("qhd,khd->hqk", q[qs:qe].float(), kk) / 8
        expected_lse.append(torch.logsumexp(score, -1))
        expected.append(torch.einsum("hqk,khd->qhd", torch.softmax(score, -1), vv))
    torch.testing.assert_close(out.float(), torch.cat(expected), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(lse.float(), torch.cat(expected_lse, -1), rtol=3e-3, atol=3e-3)
