# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import inspect
import sys
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import torch

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.attention.backends.utils import fa
from vllm_omni.diffusion.models.magi2 import attention, window_attention
from vllm_omni.diffusion.models.magi2.window_attention import (
    Magi2WindowAttentionPlan,
    build_window_attention_plan,
    merge_attention_partials,
    prewarm_window_attention,
    torch_window_attention_reference,
)
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model]

# (11, 8, 12) ends with a partial 3-frame temporal block; 300 text tokens make three query chunks per key run.
_GRID = (11, 8, 12)


@pytest.fixture(autouse=True)
def clear_provider_cache():
    fa._external_fa3_varlen.cache_clear()
    fa.flash_attn_3_varlen_unsupported.cache_clear()
    yield
    fa._external_fa3_varlen.cache_clear()
    fa.flash_attn_3_varlen_unsupported.cache_clear()


def _platform(monkeypatch: pytest.MonkeyPatch, kind: str, device_type: str = "cpu") -> None:
    # CPU tensors stand in for the device; the paged providers below are emulated in PyTorch.
    monkeypatch.setattr(
        window_attention,
        "current_omni_platform",
        SimpleNamespace(is_cuda=lambda: kind == "cuda", is_musa=lambda: kind == "musa", device_type=device_type),
    )


def _paged_attention(q, k_pages, v_pages, cu_seqlens_q, seqused_k, block_table, softmax_scale):
    """fp32 paged varlen attention that, like the device kernels, reads whole pages and masks afterwards."""

    page_size = k_pages.shape[1]
    group = q.shape[1] // k_pages.shape[2]
    out = torch.empty(q.shape, dtype=torch.float32)
    lse = torch.empty(q.shape[1], q.shape[0])
    for row in range(block_table.shape[0]):
        start, end, used = int(cu_seqlens_q[row]), int(cu_seqlens_q[row + 1]), int(seqused_k[row])
        pages = block_table[row, : -(-used // page_size)].long()
        k = k_pages[pages].flatten(0, 1).float().repeat_interleave(group, dim=1)
        v = v_pages[pages].flatten(0, 1).float().repeat_interleave(group, dim=1)
        scores = torch.einsum("qhd,khd->hqk", q[start:end].float(), k) * softmax_scale
        scores[..., used:] = float("-inf")
        lse[:, start:end] = scores.logsumexp(dim=-1)
        # The masked tail still enters P @ V, so non-finite padding reaches the output.
        out[start:end] = torch.einsum("hqk,khd->qhd", scores.softmax(dim=-1), v)
    return out, lse


def _finish(q, result, lse, out, return_lse):
    if out is not None:
        out.copy_(result)
        result = out
    else:
        result = result.to(q.dtype)
    return (result, lse) if return_lse else result


def fa3_provider(
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
    softcap=0.0,
    return_softmax_lse=False,
    out=None,
):
    result, lse = _paged_attention(q, k, v, cu_seqlens_q, seqused_k, block_table, softmax_scale)
    return _finish(q, result, lse, out, return_softmax_lse)


def _install_fa3(monkeypatch: pytest.MonkeyPatch, drop: frozenset[str] = frozenset()) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def provider(**kwargs):
        calls.append(kwargs)
        return fa3_provider(**kwargs)

    parameters = frozenset(inspect.signature(fa3_provider).parameters) - drop
    monkeypatch.setattr(fa, "_external_fa3_varlen", lambda: (provider, parameters))
    return calls


def _install_vllm_fa(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def flash_attn_varlen_func(q, k, v, max_seqlen_q, cu_seqlens_q, max_seqlen_k, **kwargs):
        calls.append(
            dict(kwargs, q=q, k=k, max_seqlen_q=max_seqlen_q, cu_seqlens_q=cu_seqlens_q, max_seqlen_k=max_seqlen_k)
        )
        assert kwargs["cu_seqlens_k"] is None
        result, lse = _paged_attention(
            q, k, v, cu_seqlens_q, kwargs["seqused_k"], kwargs["block_table"], kwargs["softmax_scale"]
        )
        return _finish(q, result, lse, kwargs["out"], kwargs["return_softmax_lse"])

    module = ModuleType("vllm.vllm_flash_attn")
    setattr(module, "flash_attn_varlen_func", flash_attn_varlen_func)
    monkeypatch.setitem(sys.modules, "vllm.vllm_flash_attn", module)
    monkeypatch.setattr(fa, "resolve_vllm_flash_attn_version", lambda version=None: version)
    monkeypatch.setattr(window_attention, "resolve_magi2_flash_attn_version", lambda: 3)
    return calls


def _qkv(plan: Magi2WindowAttentionPlan, num_heads_q: int, num_heads_kv: int, dtype=torch.float32, head_dim=16):
    generator = torch.Generator().manual_seed(plan.text_len * 10 + num_heads_q)
    q = torch.randn(plan.num_tokens, num_heads_q, head_dim, generator=generator)
    k = torch.randn(plan.num_tokens, num_heads_kv, head_dim, generator=generator)
    v = torch.randn(plan.num_tokens, num_heads_kv, head_dim, generator=generator)
    return q.to(dtype), k.to(dtype), v.to(dtype)


def _assert_production_calls(
    calls: list[dict[str, Any]], plan: Magi2WindowAttentionPlan, num_heads_q: int, *, same_plan: bool = True
) -> None:
    """Check the paged calls of one window attention; ``same_plan`` also requires the plan's own tensors."""

    def passed(value: torch.Tensor, expected: torch.Tensor) -> bool:
        return value is expected if same_plan else torch.equal(value, expected)

    video, *text = calls
    assert len(text) == int(plan.text_len > 0)
    for call in calls:
        assert call["cu_seqlens_k"] is None
        assert call["max_seqlen_q"] == plan.query_chunk == 128
        assert call["k"].shape[:2] == (plan.num_pages, plan.page_size)
        assert call["block_table"].dtype == call["seqused_k"].dtype == call["cu_seqlens_q"].dtype == torch.int32
    assert passed(video["block_table"], plan.page_table) and passed(video["seqused_k"], plan.seqused_k)
    assert passed(video["cu_seqlens_q"], plan.cu_seqlens_q) and video["max_seqlen_k"] == plan.max_seqlen_k
    assert video["q"].shape == (plan.num_video_tokens, num_heads_q, video["q"].shape[-1])
    assert video["out"] is not None and not video.get("return_softmax_lse")
    if text:
        (text_call,) = text
        assert passed(text_call["block_table"], plan.text_split_page_table)
        assert passed(text_call["seqused_k"], plan.text_split_seqused_k)
        assert passed(text_call["cu_seqlens_q"], plan.text_split_cu_seqlens_q)
        assert text_call["max_seqlen_k"] == plan.text_split_max_seqlen_k
        assert text_call["q"].shape[0] == plan.text_key_splits * plan.text_len
        assert text_call["return_softmax_lse"] is True and text_call.get("out") is None


@pytest.mark.cpu
@pytest.mark.parametrize("text_len", [0, 1, 37, 128, 130, 300])
@pytest.mark.parametrize(("num_heads_q", "num_heads_kv"), [(4, 1), (8, 2)])
def test_musa_paged_path_matches_the_reference(
    monkeypatch: pytest.MonkeyPatch, text_len: int, num_heads_q: int, num_heads_kv: int
) -> None:
    _platform(monkeypatch, "musa")
    calls = _install_fa3(monkeypatch)
    plan = build_window_attention_plan(_GRID, text_len)
    q, k, v = _qkv(plan, num_heads_q, num_heads_kv)
    expected = torch_window_attention_reference(q, k, v, plan, softmax_scale=0.3)
    monkeypatch.setattr(
        window_attention,
        "torch_window_attention_reference",
        lambda *args, **kwargs: pytest.fail("the device path must not run the reference"),
    )

    out = window_attention.window_attention(q, k, v, plan, softmax_scale=0.3)

    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-6)
    _assert_production_calls(calls, plan, num_heads_q)
    assert calls[0]["out"].data_ptr() == out.data_ptr()


@pytest.mark.cpu
@pytest.mark.parametrize(("grid", "text_len"), [((9, 4, 8), 37), ((1, 4, 4), 1), ((1, 4, 4), 120)])
def test_musa_paged_path_matches_the_reference_on_small_grids(
    monkeypatch: pytest.MonkeyPatch, grid: tuple[int, int, int], text_len: int
) -> None:
    _platform(monkeypatch, "musa")
    _install_fa3(monkeypatch)
    plan = build_window_attention_plan(grid, text_len)
    q, k, v = _qkv(plan, 8, 2)

    out = window_attention.window_attention(q, k, v, plan)

    torch.testing.assert_close(out, torch_window_attention_reference(q, k, v, plan), rtol=1e-5, atol=1e-6)


@pytest.mark.cpu
def test_musa_paged_path_returns_bf16(monkeypatch: pytest.MonkeyPatch) -> None:
    _platform(monkeypatch, "musa")
    _install_fa3(monkeypatch)
    plan = build_window_attention_plan(_GRID, 130)
    q, k, v = _qkv(plan, 4, 1, dtype=torch.bfloat16)

    out = window_attention.window_attention(q, k, v, plan)

    assert out.dtype == torch.bfloat16
    expected = torch_window_attention_reference(q.float(), k.float(), v.float(), plan)
    torch.testing.assert_close(out.float(), expected, rtol=1.6e-2, atol=1e-2)


@pytest.mark.cpu
def test_cuda_paged_path_calls_vllm_flash_attn_without_cu_seqlens_k(monkeypatch: pytest.MonkeyPatch) -> None:
    _platform(monkeypatch, "cuda")
    calls = _install_vllm_fa(monkeypatch)
    monkeypatch.setattr(fa, "_external_fa3_varlen", lambda: pytest.fail("CUDA must not use the standalone FA3 path"))
    plan = build_window_attention_plan(_GRID, 130)
    q, k, v = _qkv(plan, 8, 2)

    out = window_attention.window_attention(q, k, v, plan)

    torch.testing.assert_close(out, torch_window_attention_reference(q, k, v, plan), rtol=1e-5, atol=1e-6)
    _assert_production_calls(calls, plan, 8)
    assert [call["fa_version"] for call in calls] == [3, 3]


@pytest.mark.cpu
def test_cuda_paged_path_resolves_the_magi2_flash_attn_version_once(monkeypatch: pytest.MonkeyPatch) -> None:
    _platform(monkeypatch, "cuda")
    calls = _install_vllm_fa(monkeypatch)
    monkeypatch.setattr(
        window_attention, "resolve_magi2_flash_attn_version", attention.resolve_magi2_flash_attn_version
    )
    requested: list[Any] = []

    def resolve(version: Any = None) -> int:
        requested.append(version)
        return 2

    monkeypatch.setattr(attention, "resolve_vllm_flash_attn_version", resolve)
    monkeypatch.setenv("MAGI2_FLASH_ATTN_VERSION", "2")
    info = Mock()
    monkeypatch.setattr(attention.logger, "info", info)
    plan = build_window_attention_plan((9, 4, 8), 37)
    q, k, v = _qkv(plan, 4, 1)

    attention.resolve_magi2_flash_attn_version.cache_clear()
    try:
        for _ in range(3):
            window_attention.window_attention(q, k, v, plan)
    finally:
        attention.resolve_magi2_flash_attn_version.cache_clear()

    # Every call takes the MAGI2_FLASH_ATTN_VERSION override, resolved and logged once per process.
    assert [call["fa_version"] for call in calls] == [2] * 6
    assert requested == ["2"]
    info.assert_called_once()


@pytest.mark.cpu
def test_musa_provider_without_paged_kv_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    _platform(monkeypatch, "musa")
    calls = _install_fa3(monkeypatch, drop=frozenset({"block_table"}))
    monkeypatch.setattr(
        window_attention,
        "torch_window_attention_reference",
        lambda *args, **kwargs: pytest.fail("a missing paged kernel must not fall back to the reference"),
    )
    plan = build_window_attention_plan((9, 4, 8), 3)
    q, k, v = _qkv(plan, 4, 1)

    with pytest.raises(NotImplementedError, match=r"paged KV \(block_table\)"):
        window_attention.window_attention(q, k, v, plan)
    assert calls == []


@pytest.mark.cpu
def test_page_padding_is_zeroed_in_a_reused_allocation(monkeypatch: pytest.MonkeyPatch) -> None:
    _platform(monkeypatch, "musa")
    _install_fa3(monkeypatch)
    plan = build_window_attention_plan(_GRID, 37)
    assert plan.num_padded_tokens > plan.num_tokens
    q, k, v = _qkv(plan, 4, 1)
    expected = torch_window_attention_reference(q, k, v, plan)
    new_empty = torch.Tensor.new_empty

    def poisoned_new_empty(self, *args, **kwargs):
        # A recycled allocation still holding non-finite values.
        return new_empty(self, *args, **kwargs).fill_(float("nan"))

    monkeypatch.setattr(torch.Tensor, "new_empty", poisoned_new_empty)
    pages = plan.pad_to_pages(k)
    out = window_attention.window_attention(q, k, v, plan)

    assert not pages.flatten(0, 1)[plan.num_tokens :].any()
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.cpu
def test_paged_path_refuses_a_plan_on_another_device(monkeypatch: pytest.MonkeyPatch) -> None:
    _platform(monkeypatch, "musa", device_type="meta")
    _install_fa3(monkeypatch)
    plan = build_window_attention_plan((9, 4, 8), 3)
    q = torch.empty(plan.num_tokens, 4, 16, device="meta")
    kv = torch.empty(plan.num_tokens, 1, 16, device="meta")

    with pytest.raises(ValueError, match="plan is on cpu"):
        window_attention.window_attention(q, kv, kv, plan)


def _split_attention(q, k, v, bounds, scale):
    partials, lse = [], []
    for start, end in zip(bounds, bounds[1:]):
        scores = torch.einsum("qhd,khd->qhk", q, k[start:end]) * scale
        lse.append(scores.logsumexp(dim=-1))
        partials.append(torch.einsum("qhk,khd->qhd", scores.softmax(dim=-1), v[start:end]))
    return torch.stack(partials), torch.stack(lse)


@pytest.mark.cpu
@pytest.mark.parametrize("scale", [0.25, 40.0])
def test_merge_attention_partials_equals_attention_over_the_union(scale: float) -> None:
    generator = torch.Generator().manual_seed(5)
    q = torch.randn(9, 3, 8, generator=generator)
    k = torch.randn(70, 3, 8, generator=generator)
    v = torch.randn(70, 3, 8, generator=generator)
    expected = torch.einsum("qhk,khd->qhd", (torch.einsum("qhd,khd->qhk", q, k) * scale).softmax(dim=-1), v)
    partials, lse = _split_attention(q, k, v, [0, 3, 40, 41, 70], scale)
    # At the large scale the LSEs are far beyond the fp32 exp range, so the merge must subtract their maximum.
    assert scale < 1 or lse.abs().max() > 100

    merged = merge_attention_partials(partials, lse)

    assert merged.dtype == torch.float32
    torch.testing.assert_close(merged, expected, rtol=1e-5, atol=1e-5)
    single = merge_attention_partials(*_split_attention(q, k, v, [0, 70], scale))
    torch.testing.assert_close(single, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.cpu
def test_merge_attention_partials_merges_bf16_partials_in_fp32() -> None:
    generator = torch.Generator().manual_seed(6)
    q, k, v = (torch.randn(shape, generator=generator) for shape in ((5, 2, 8), (40, 2, 8), (40, 2, 8)))
    partials, lse = _split_attention(q, k, v, [0, 13, 40], 0.5)
    merged = merge_attention_partials(partials.bfloat16(), lse)

    assert merged.dtype == torch.float32
    weights = torch.exp(lse - lse.amax(dim=0)).unsqueeze(-1)
    expected = (partials.bfloat16().float() * weights).sum(dim=0) / weights.sum(dim=0)
    assert torch.equal(merged, expected)


@pytest.mark.cpu
def test_prewarm_issues_the_production_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    _platform(monkeypatch, "musa")
    warm_calls = _install_fa3(monkeypatch)

    prewarm_window_attention("cpu", num_heads_q=4, num_heads_kv=1, head_dim=128, dtype=torch.bfloat16)

    warm_plan = build_window_attention_plan((1, 4, 4), 1)
    _assert_production_calls(warm_calls, warm_plan, 4, same_plan=False)
    assert len(warm_calls) == 2
    for call in warm_calls:
        assert call["q"].dtype == call["k"].dtype == torch.bfloat16
        assert call["q"].shape[1:] == (4, 128) and call["k"].shape[2:] == (1, 128)

    # The same keyword arguments and scalar settings as a production-shaped forward.
    plan = build_window_attention_plan(_GRID, 300)
    calls = _install_fa3(monkeypatch)
    q, k, v = _qkv(plan, 4, 1, dtype=torch.bfloat16, head_dim=128)
    window_attention.window_attention(q, k, v, plan)
    for warm, call in zip(warm_calls, calls, strict=True):
        assert warm.keys() == call.keys()
        for name, value in call.items():
            if not isinstance(value, torch.Tensor) and name != "max_seqlen_k":
                assert warm[name] == value, name


@pytest.mark.cpu
@pytest.mark.parametrize("platform", ["cpu", "musa_platform_host_tensor"])
def test_prewarm_is_a_noop_without_a_paged_kernel(monkeypatch: pytest.MonkeyPatch, platform: str) -> None:
    if platform != "cpu":
        _platform(monkeypatch, "musa", device_type="musa")
    monkeypatch.setattr(fa, "_external_fa3_varlen", lambda: pytest.fail("prewarm must not reach a provider"))
    monkeypatch.setattr(
        window_attention, "window_attention", lambda *args, **kwargs: pytest.fail("prewarm must not run attention")
    )

    prewarm_window_attention("cpu", num_heads_q=4, num_heads_kv=1, head_dim=128, dtype=torch.bfloat16)


@pytest.mark.parametrize(
    "device_kind",
    [
        pytest.param("cuda", marks=hardware_marks(res={"cuda": "L4"}, num_cards=1)),
        pytest.param("musa", marks=hardware_marks(res={"musa": "S5000"}, num_cards=1)),
    ],
)
@pytest.mark.parametrize("text_len", [0, 37, 130])
def test_device_paged_window_attention_matches_the_reference(device_kind: str, text_len: int) -> None:
    # ROCm also reports torch.cuda as available but has no paged kernel here.
    on_platform = current_omni_platform.is_cuda() if device_kind == "cuda" else current_omni_platform.is_musa()
    device_module = getattr(torch, device_kind, None)
    if not on_platform or device_module is None or not device_module.is_available():
        pytest.skip(f"requires the {device_kind} platform")
    paged = dict(block_table=True, seqused_k=True, out=True)
    if device_kind == "musa" and fa.flash_attn_3_varlen_unsupported(False, True, **paged) is not None:
        pytest.skip("standalone FA3 provider with paged KV is not installed")
    plan = build_window_attention_plan(_GRID, text_len)
    q, k, v = _qkv(plan, 4, 1, dtype=torch.bfloat16, head_dim=128)
    expected = torch_window_attention_reference(q.float(), k.float(), v.float(), plan)

    prewarm_window_attention(device_kind, 4, 1, 128, torch.bfloat16)
    out = window_attention.window_attention(
        q.to(device_kind), k.to(device_kind), v.to(device_kind), plan.to(device_kind)
    )

    assert out.dtype == torch.bfloat16 and torch.isfinite(out).all()
    torch.testing.assert_close(out.float().cpu(), expected, rtol=0.01, atol=0.015)
