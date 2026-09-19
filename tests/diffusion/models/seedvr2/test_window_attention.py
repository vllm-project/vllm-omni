# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Dispatch and isolation tests for SeedVR2 window attention.

The packed-varlen path is only correct on a backend that keeps N-document
packed boundaries isolated (``supports_multi_doc_packed_varlen``).  These tests
drive the *production* ``NaSwinAttention`` with stand-in backends, so the
dispatch decision itself - not a re-implementation of it - is what is exercised.
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.models.seedvr2 import nadit
from vllm_omni.diffusion.models.seedvr2.na_ops import LocalWindowContext, build_local_window_context
from vllm_omni.diffusion.models.seedvr2.window_sp import (
    RankWindowPlan,
    WindowLayoutKey,
    to_cu_seqlens,
)

pytestmark = [
    pytest.mark.diffusion,
    pytest.mark.parallel,
    pytest.mark.sp,
    pytest.mark.core_model,
    pytest.mark.cpu,
    *hardware_marks(res={"cuda": "L4"}, num_cards=1),
]

REGULAR = "720pwin_by_size_bysize"


class _Backend:
    """Minimal stand-in for an attention backend class."""

    _name = "STUB"
    _varlen = False

    @classmethod
    def get_name(cls) -> str:
        return cls._name

    @classmethod
    def supports_multi_doc_packed_varlen(cls) -> bool:
        return cls._varlen


class UnsupportedBackend(_Backend):
    _name = "STUB_NO_PACKED_VARLEN"
    _varlen = False


class SupportedBackend(_Backend):
    _name = "STUB_PACKED_VARLEN"
    _varlen = True


class StubAttention(torch.nn.Module):
    """Replaces the shared ``Attention`` layer; records how it was driven."""

    backend_cls: type[_Backend] = UnsupportedBackend
    instances: list[StubAttention] = []

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.kwargs = kwargs
        self.attn_backend = type(self).backend_cls
        self.calls: list[dict] = []
        StubAttention.instances.append(self)

    def forward(self, query, key, value, attn_metadata=None):
        if not self.attn_backend.supports_multi_doc_packed_varlen():
            raise AssertionError("packed-varlen forward must not run on an unsupported backend")
        if query.shape[0] != 1:
            raise AssertionError(f"packed attention expects a batch of one document set, got {tuple(query.shape)}")
        extra = dict(attn_metadata.extra)
        self.calls.append(extra)
        cu = extra["cu_seqlens_q"].tolist()
        out = torch.empty_like(query)
        batch = query.shape[0]
        for start, end in zip(cu, cu[1:]):
            # (batch, seq, heads, dim) -> (batch, heads, seq, dim): isolate the document.
            q = query[:, start:end].transpose(1, 2)
            k = key[:, start:end].transpose(1, 2)
            v = value[:, start:end].transpose(1, 2)
            attended = F.scaled_dot_product_attention(q, k, v, scale=self.kwargs["softmax_scale"])
            out[:, start:end] = attended.transpose(1, 2)
        assert out.shape[0] == batch
        return out


def _rank_plan(lengths: list[int]) -> RankWindowPlan:
    """A real ``RankWindowPlan`` whose windows have exactly these video lengths."""
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    total = offsets[-1]
    key = WindowLayoutKey(
        token_grid=(total, 1, 1),
        window_method=REGULAR,
        window_shape=(1, 1, 1),
        geometry_version=1,
        geometry_fingerprint="test-fixture",
    )
    return RankWindowPlan(
        layout_key=key,
        rank=0,
        window_ids=torch.arange(len(lengths), dtype=torch.int64),
        global_token_ids=torch.arange(total, dtype=torch.int64),
        video_cu_seqlens=to_cu_seqlens(torch.tensor(offsets, dtype=torch.int64), context="test fixture"),
        window_shapes=torch.tensor([[length, 1, 1] for length in lengths], dtype=torch.int64),
    )


def _make_context(lengths: list[int], text_len: int) -> LocalWindowContext:
    plan = _rank_plan(lengths)
    return build_local_window_context(plan, text_len=text_len, global_windows=len(lengths), device="cpu")


def _make_attention(monkeypatch, request_varlen: bool, backend_cls: type[_Backend]) -> nadit.NaSwinAttention:
    StubAttention.instances.clear()
    StubAttention.backend_cls = backend_cls
    monkeypatch.setattr(nadit, "Attention", StubAttention)
    return nadit.NaSwinAttention(
        vid_dim=8,
        txt_dim=8,
        heads=2,
        head_dim=6,
        qk_bias=False,
        qk_norm_eps=1e-5,
        rope_dim=6,
        shared_weights=False,
        use_varlen_kernel=request_varlen,
    )


def _input(ctx: LocalWindowContext, seed: int = 7723):
    generator = torch.Generator().manual_seed(seed)
    vid = torch.randn(ctx.num_video_tokens, 8, generator=generator)
    txt = torch.randn(ctx.text_len, 8, generator=generator)
    return vid, txt


def test_unsupported_backend_never_reaches_the_packed_path(monkeypatch):
    """R1 regression: default request + unsupported backend -> grouped SDPA."""
    module = _make_attention(monkeypatch, request_varlen=True, backend_cls=UnsupportedBackend)
    assert module.attention_path == "grouped_sdpa"
    assert module.varlen_fallback_reason is not None

    ctx = _make_context([3, 5, 2], text_len=2)
    vid, txt = _input(ctx)
    with torch.no_grad():
        vid_gated, txt_gated = module(vid, txt, ctx, None)

    stub = StubAttention.instances[0]
    assert stub.calls == [], "grouped SDPA must not call the shared packed path"
    assert module.attention_stats["grouped_sdpa_calls"] == 1

    # The same weights through the packed path (per-document kernel in the stub)
    # must produce the same video *and* text rows.
    packed = copy.deepcopy(module)
    packed.use_varlen_kernel = True
    packed.attention.attn_backend = SupportedBackend
    with torch.no_grad():
        vid_packed, txt_packed = packed(vid, txt, ctx, None)
    assert torch.allclose(vid_gated, vid_packed, atol=1e-6)
    assert torch.allclose(txt_gated, txt_packed, atol=1e-6)


def test_supported_backend_uses_packed_path_with_metadata(monkeypatch):
    module = _make_attention(monkeypatch, request_varlen=True, backend_cls=SupportedBackend)
    assert module.attention_path == "packed_varlen"
    assert module.varlen_fallback_reason is None

    ctx = _make_context([2, 3, 2], text_len=1)
    vid, txt = _input(ctx)
    with torch.no_grad():
        module(vid, txt, ctx, None)

    stub = StubAttention.instances[0]
    assert len(stub.calls) == 1
    extra = stub.calls[0]
    assert extra["cu_seqlens_q"] is ctx.joint_cu_seqlens
    assert extra["cu_seqlens_k"] is ctx.joint_cu_seqlens
    assert extra["max_seqlen_q"] == ctx.max_joint_len
    assert extra["max_seqlen_k"] == ctx.max_joint_len
    assert ctx.joint_cu_seqlens.tolist() == [0, 3, 7, 10]
    assert module.attention_stats["packed_varlen_calls"] == 1


def test_user_disabled_varlen_is_not_overridden(monkeypatch):
    module = _make_attention(monkeypatch, request_varlen=False, backend_cls=SupportedBackend)
    assert module.attention_path == "grouped_sdpa"
    assert module.varlen_fallback_reason is None, "no fallback warning when varlen was not requested"

    ctx = _make_context([2, 1, 1], text_len=1)
    vid, txt = _input(ctx)
    with torch.no_grad():
        module(vid, txt, ctx, None)
    assert StubAttention.instances[0].calls == []
    assert module.attention_stats["grouped_sdpa_calls"] == 1


def test_window_isolation_with_zero_scores():
    """Zero scores -> uniform means per window; windows stay isolated."""
    lengths = [2, 3, 2]
    ctx = _make_context(lengths, text_len=1)
    offsets = ctx.joint_cu_seqlens.tolist()
    assert offsets == [0, 3, 7, 10]

    q = torch.zeros(ctx.joint_len, 1, 1)
    k = torch.zeros(ctx.joint_len, 1, 1)
    v = torch.zeros(ctx.joint_len, 1, 1)
    for window, value in enumerate((1.0, 9.0, -4.0)):
        start = offsets[window]
        v[start : start + lengths[window]] = value

    scale = 1.0 / (1.0**0.5)
    out = _attention_oracle(q, k, v, ctx, scale)
    # Each window: [video rows, one zero text row] -> uniform mean.
    expected = [2 / 3, 27 / 4, -8 / 3]
    for window, length in enumerate(lengths):
        start = offsets[window]
        assert out[start].item() == pytest.approx(expected[window], rel=1e-6)

    # Only the second window's video values change: windows 1 and 3 must not move.
    changed = v.clone()
    changed[offsets[1] : offsets[1] + lengths[1]] = 27.0
    out_changed = _attention_oracle(q, k, changed, ctx, scale)
    assert torch.equal(out_changed[: offsets[1]], out[: offsets[1]])
    assert torch.equal(out_changed[offsets[2] :], out[offsets[2] :])


def _attention_oracle(q, k, v, ctx: LocalWindowContext, scale: float) -> torch.Tensor:
    cu = ctx.joint_cu_seqlens.tolist()
    out = torch.empty_like(q)
    for start, end in zip(cu, cu[1:]):
        qq = q[start:end].transpose(0, 1).unsqueeze(0)
        kk = k[start:end].transpose(0, 1).unsqueeze(0)
        vv = v[start:end].transpose(0, 1).unsqueeze(0)
        out[start:end] = F.scaled_dot_product_attention(qq, kk, vv, scale=scale).squeeze(0).transpose(0, 1)
    return out


def test_fallback_reason_is_stable_across_layers(monkeypatch):
    reasons = set()
    for _ in range(3):
        module = _make_attention(monkeypatch, request_varlen=True, backend_cls=UnsupportedBackend)
        reasons.add(module.varlen_fallback_reason)
    assert len(reasons) == 1
    assert not any(char.isdigit() for char in reasons.pop()), "warning must not carry a layer id"


def test_empty_rank_skips_packed_kernel_but_still_reduces_text(monkeypatch):
    module = _make_attention(monkeypatch, request_varlen=True, backend_cls=SupportedBackend)

    class _Runtime:
        def __init__(self) -> None:
            self.text_calls = 0

        def reduce_text(self, local_sum, global_windows):
            self.text_calls += 1
            return local_sum / global_windows

    ctx = _make_context([2, 2], text_len=2)
    empty = LocalWindowContext(
        layout_key=ctx.layout_key,
        window_shapes=torch.empty((0, 3), dtype=torch.int64),
        video_cu_seqlens=torch.zeros(1, dtype=torch.int32),
        joint_cu_seqlens=torch.zeros(1, dtype=torch.int32),
        joint_order=torch.empty(0, dtype=torch.int64),
        vid_src=torch.empty(0, dtype=torch.int64),
        txt_src=torch.empty(0, dtype=torch.int64),
        text_len=2,
        local_windows=0,
        global_windows=2,
    )
    runtime = _Runtime()
    with torch.no_grad():
        vid_out, txt_out = module(torch.zeros(0, 8), torch.randn(2, 8), empty, runtime)
    assert vid_out.shape[0] == 0
    assert txt_out.shape == (2, 8)
    assert runtime.text_calls == 1, "an empty rank must still take part in the text reduction"
    assert StubAttention.instances[0].calls == []
    assert module.attention_stats["no_local_windows_calls"] == 1
