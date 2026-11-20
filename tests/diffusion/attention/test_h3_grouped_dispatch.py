# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from unittest.mock import Mock

import pytest
import torch

from vllm_omni.diffusion.attention.backends import flash_attn
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.backends.utils import fa, rocm_h3_fmha
from vllm_omni.diffusion.attention.backends.utils.rocm_h3_bounds import has_packed_fmha_bounds, packed_cu_seqlens
from vllm_omni.diffusion.data import AttentionConfig, AttentionSpec
from vllm_omni.diffusion.models.minimax_h3.attention import flash_h3
from vllm_omni.diffusion.models.minimax_h3.attention.flash_h3 import MiniMaxH3FlashAttentionImpl

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_h3_grouped_fmha_typed_option():
    config = AttentionConfig(per_role={"self": {"backend": "FLASH_ATTN", "rocm_h3_grouped_fmha": True}})
    assert config.per_role["self"].backend_kwargs() == {"rocm_h3_grouped_fmha": True}
    assert AttentionSpec("FLASH_ATTN").backend_kwargs() is None
    with pytest.raises(ValueError, match="requires the FLASH_ATTN"):
        AttentionSpec("TORCH_SDPA", rocm_h3_grouped_fmha=True)
    with pytest.raises(TypeError, match="boolean"):
        AttentionSpec("FLASH_ATTN", rocm_h3_grouped_fmha="false")


@pytest.mark.parametrize("scenario", ["enabled", "disabled", "causal", "cross", "scale", "other_fa", "deterministic"])
def test_only_h3_override_enables_grouped_attention(monkeypatch, scenario):
    monkeypatch.setattr(fa, "IS_AITER", scenario != "other_fa")
    monkeypatch.setattr(fa, "IS_FLASH_ATTN_4", False)
    monkeypatch.setattr(flash_h3.current_omni_platform, "is_rocm", lambda: True)
    monkeypatch.setattr(
        flash_attn, "get_current_diffusion_config_or_none", lambda: Mock(fa_deterministic=scenario == "deterministic")
    )
    kwargs = dict(
        num_heads=56,
        head_size=128,
        softmax_scale=0.5 if scenario == "scale" else 128**-0.5,
        causal=scenario == "causal",
        role="cross" if scenario == "cross" else "self",
        backend_kwargs={"rocm_h3_grouped_fmha": scenario != "disabled"},
    )
    assert MiniMaxH3FlashAttentionImpl(**kwargs).rocm_h3_grouped_fmha is (scenario == "enabled")
    assert not hasattr(flash_attn.FlashAttentionImpl(**kwargs), "rocm_h3_grouped_fmha")


def test_h3_grouped_bound_provenance(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "test")
    cpu = torch.device("cpu")
    owned = [0, 37750, 37760]
    with torch.inference_mode():
        bound = packed_cu_seqlens(owned, cpu)
        assert has_packed_fmha_bounds(bound, 37750)
        assert not has_packed_fmha_bounds(bound, 37749)
        assert not has_packed_fmha_bounds(bound.clone(), 37750)
        owned[1] = 1
        assert bound[1] == 37750
        # Restoring identical values must not reinstate the original version.
        bound.add_(1).sub_(1)
        assert not has_packed_fmha_bounds(bound, 37750)
    assert not has_packed_fmha_bounds(packed_cu_seqlens([0, 37752, 37760], cpu), 37752)


@pytest.mark.parametrize(
    "scenario",
    ["grouped", "declined", "separate_k", "unmarked", "mutable", "stale", "wrong_max", "mask", "vdn", "bias"],
)
def test_h3_grouped_fmha_dispatch(monkeypatch, scenario):
    # Miniature strided receive views isolate the dispatcher from hardware.
    packet = torch.randn(1, 16, 3, 2, 4)
    q, k, v = packet.unbind(dim=2)
    assert not q.is_contiguous()
    monkeypatch.setattr(torch.version, "hip", "test")
    cut = 37750
    bounds = packed_cu_seqlens([0, cut, 37760], torch.device("cpu"))
    if scenario == "mutable":
        bounds = bounds.clone()
    if scenario == "stale":
        bounds.add_(1).sub_(1)
    if scenario == "wrong_max":
        cut -= 1
    extra = dict(cu_seqlens_q=bounds, cu_seqlens_k=bounds, max_seqlen_q=cut, max_seqlen_k=cut, valid_kv_length=cut)
    if scenario == "vdn":
        extra["vdn_window"] = object()
    if scenario == "bias":
        extra["attn_bias"] = torch.tensor(1)
    mask = torch.ones(1, 16, dtype=torch.bool) if scenario == "mask" else None
    if scenario != "unmarked":
        extra["minimax_h3_grouped_fmha"] = True
        extra["accepts_strided_ulysses_qkv"] = True
    if scenario == "separate_k":
        extra["cu_seqlens_k"] = bounds.clone()

    sentinel = torch.zeros_like(q)
    grouped = Mock(return_value=sentinel if scenario == "grouped" else None)
    monkeypatch.setattr(rocm_h3_fmha, "try_grouped_fmha", grouped)
    original = Mock(return_value=q.reshape(16, 2, 4))
    monkeypatch.setattr(fa, "HAS_FLASH_ATTN", True)
    monkeypatch.setattr(fa, "flash_attn_varlen_func", original)
    impl = MiniMaxH3FlashAttentionImpl(num_heads=2, head_size=4, softmax_scale=0.5, causal=False)
    impl.rocm_h3_grouped_fmha = True
    out = impl.forward_hip(q, k, v, AttentionMetadata(extra=extra, attn_mask=mask))
    if scenario == "grouped":
        assert out is sentinel
        original.assert_not_called()
    else:
        original.assert_called_once()
        torch.testing.assert_close(out, q)
        if scenario != "unmarked":
            assert all(original.call_args.kwargs[name].is_contiguous() for name in ("q", "k", "v"))
    assert grouped.call_count == int(scenario in ("grouped", "declined"))


@pytest.mark.parametrize("scenario", ["fp8", "piecewise", "incomplete"])
def test_h3_keeps_specialized_flash_semantics(monkeypatch, scenario):
    monkeypatch.setattr(torch.version, "hip", "test")
    cut = 37750
    cu = packed_cu_seqlens([0, cut, 37760], torch.device("cpu"))
    extra = dict(
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        max_seqlen_q=cut,
        max_seqlen_k=cut,
        valid_kv_length=cut,
        minimax_h3_grouped_fmha=True,
        accepts_strided_ulysses_qkv=True,
    )
    if scenario == "fp8":
        extra["kv_cache_dtype"] = "fp8"
    if scenario == "incomplete":
        del extra["cu_seqlens_k"]
    metadata = AttentionMetadata(extra=extra, full_attn_spans=[[(0, 16)]] if scenario == "piecewise" else None)
    q, k, v = torch.randn(1, 16, 3, 2, 4).unbind(dim=2)
    fallback = Mock(return_value=q)
    monkeypatch.setattr(flash_attn.FlashAttentionImpl, "forward_hip", fallback)
    grouped = Mock(side_effect=AssertionError("specialized FlashAttention semantics use the inherited implementation"))
    monkeypatch.setattr(rocm_h3_fmha, "try_grouped_fmha", grouped)
    impl = MiniMaxH3FlashAttentionImpl(num_heads=2, head_size=4, softmax_scale=0.5)
    impl.rocm_h3_grouped_fmha = True
    assert impl.forward_hip(q, k, v, metadata) is q
    fallback.assert_called_once()
    assert fallback.call_args.args[-1] is metadata
    assert all(t.is_contiguous() for t in fallback.call_args.args[:3])
    grouped.assert_not_called()
