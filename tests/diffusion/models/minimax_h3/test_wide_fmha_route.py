# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionImpl
from vllm_omni.diffusion.models.minimax_h3.attention import wide_fmha

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize(
    ("bounds", "expected"),
    [
        ([0, 37750, 37760], True),
        ([1, 37750, 37760], False),
        ([0, 10, 37760], False),
        ([0, 37750, 37759], False),
        ([0, 15, 37750, 37760], False),
    ],
)
def test_verify_actual_segment_lengths(bounds, expected):
    cu = torch.tensor(bounds, dtype=torch.int32)
    assert wide_fmha.verified_wide_cut(cu, 37750, 37760) is expected
    assert not wide_fmha.verified_wide_cut(cu, 37752, 37760)
    assert not wide_fmha.verified_wide_cut(cu.long(), 37750, 37760)


def _impl_and_metadata():
    impl = object.__new__(wide_fmha.MiniMaxH3WideFlashImpl)
    impl._kernel_variant = "aiter"
    impl.causal = impl.is_cross_attn = impl.fa_deterministic = False
    impl.softmax_scale = 128**-0.5
    cu = torch.tensor([0, 4, 5], dtype=torch.int32)
    extra = dict(
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        max_seqlen_q=4,
        max_seqlen_k=4,
        valid_kv_length=4,
        rocm_h3_wide_fmha_path="/test/kernel.hsaco",
    )
    return impl, AttentionMetadata(extra=extra)


@pytest.mark.parametrize("already_strided", [False, True])
def test_post_ulysses_route_passes_original_storage_to_runtime(monkeypatch, already_strided):
    # Runtime owns layout conversion after it has confirmed a workspace.
    monkeypatch.setattr(wide_fmha, "_SHAPE", (5, 2, 4))
    impl, metadata = _impl_and_metadata()
    slots = torch.arange(5 * 3 * 2 * 4, dtype=torch.bfloat16).reshape(1, 5, 3, 2, 4).unbind(2)
    inputs = slots if already_strided else tuple(t.contiguous() for t in slots)

    def check(q, k, v, cut, path):
        assert cut == 4 and path == "/test/kernel.hsaco"
        for view, ref in zip((q, k, v), inputs):
            assert view.stride() == ref.stride()[1:]
            assert torch.equal(view, ref[0])
            assert view.data_ptr() == ref.data_ptr()
        return (q + 2 * k + 3 * v).contiguous()

    monkeypatch.setattr(wide_fmha, "try_wide_fmha", check)
    with torch.no_grad():
        result = impl.forward_hip(*inputs, metadata)
    assert torch.equal(result, slots[0] + 2 * slots[1] + 3 * slots[2])


@pytest.mark.parametrize(
    "reason",
    ["off", "backend", "scale", "gradient", "quantized", "causal", "mask", "cross", "cuts", "sparse", "native"],
)
def test_ineligible_calls_fall_back_with_original_inputs(monkeypatch, reason):
    monkeypatch.setattr(wide_fmha, "_SHAPE", (5, 2, 4))
    impl, metadata = _impl_and_metadata()
    if reason == "off":
        metadata.extra.pop("rocm_h3_wide_fmha_path")
    elif reason == "backend":
        impl._kernel_variant = None
    elif reason == "scale":
        impl.softmax_scale = 1
    elif reason == "quantized":
        metadata.extra["kv_cache_dtype"] = "fp8"
    elif reason == "causal":
        impl.causal = True
    elif reason == "mask":
        metadata.attn_mask = torch.ones(5, dtype=torch.bool)
    elif reason == "cross":
        impl.is_cross_attn = True
    elif reason == "cuts":
        metadata.extra["cu_seqlens_k"] = metadata.extra["cu_seqlens_q"].clone()

    inputs = tuple(torch.ones(1, 5, 2, 4, dtype=torch.bfloat16) for _ in range(3))
    if reason == "sparse":
        inputs = (inputs[0], inputs[1].to_sparse(), inputs[2])
    calls = []

    def native(*args):
        calls.append(1)
        return None

    def fallback(self, q, k, v, metadata_in):
        assert q is inputs[0] and k is inputs[1] and v is inputs[2] and metadata_in is metadata
        return inputs[0]

    monkeypatch.setattr(wide_fmha, "try_wide_fmha", native)
    monkeypatch.setattr(FlashAttentionImpl, "forward_hip", fallback)
    with torch.set_grad_enabled(reason == "gradient"):
        assert impl.forward_hip(*inputs, metadata) is inputs[0]
    assert len(calls) == int(reason == "native")


def test_model_route_with_rocm_padding_mask_policy(monkeypatch):
    from types import SimpleNamespace

    from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionBackend
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_blocks import MiniMaxH3Attention

    impl, _ = _impl_and_metadata()
    monkeypatch.setattr(wide_fmha, "_SHAPE", (5, 2, 4))
    monkeypatch.setattr(FlashAttentionBackend, "supports_packed_mask_free", classmethod(lambda cls: False))
    monkeypatch.setattr(FlashAttentionBackend, "supports_prefix_kv_slicing", False)
    q = torch.ones((5, 2, 4), dtype=torch.bfloat16)
    cu = torch.tensor([0, 4, 5], dtype=torch.int32)
    calls = []

    class Ulysses(torch.nn.Module):
        def effective_attention(self):
            return impl, FlashAttentionBackend, None

        def _get_active_parallel_strategy(self):
            return SimpleNamespace(name="ulysses")

        def forward(self, query, key, value, metadata):
            calls.append(metadata)
            return impl.forward_hip(query, key, value, metadata)

    owner = object.__new__(MiniMaxH3Attention)
    torch.nn.Module.__init__(owner)
    owner.attention = Ulysses()
    monkeypatch.setattr(wide_fmha, "try_wide_fmha", lambda q, k, v, cut, path: 2 * q)
    monkeypatch.setattr(FlashAttentionImpl, "forward_hip", lambda self, q, k, v, md: q)
    with torch.no_grad():
        options = dict(cu_seqlens=cu, max_seqlen=4, packed_total=5)
        stock = owner._run_packed_attention(q, q, q, **options)
        wide = owner._run_packed_attention(q, q, q, wide_fmha_path="/test/kernel.hsaco", **options)
    assert torch.equal(stock, q) and torch.equal(wide, 2 * q)
    assert torch.equal(calls[0].attn_mask, torch.tensor([[True, True, True, True, False]]))
    assert calls[1].attn_mask is None
    assert calls[1].extra["cu_seqlens_q"] is calls[1].extra["cu_seqlens_k"] is cu
