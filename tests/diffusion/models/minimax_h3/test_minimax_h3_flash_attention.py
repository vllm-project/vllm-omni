# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata, QueryRange
from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionImpl
from vllm_omni.diffusion.models.minimax_h3.attention import flash_h3
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, *hardware_marks(res={"npu": "A3"}, num_cards=1)]


@pytest.mark.parametrize(
    "mode",
    ["fused", "default", "laser", "causal", "masked", "compact", "gqa", "unresolved", "short", "piecewise", "ranges"],
)
def test_h3_prefix_dispatch_preserves_unselected_paths(monkeypatch, mode):
    query = torch.arange(16384, dtype=torch.float32).to(torch.bfloat16).reshape(1, 4096, 2, 2)
    key, value = query + 10, query + 20
    calls = []

    def parent(self, q, k, v, attn_metadata=None):
        calls.append("parent")
        return torch.zeros_like(q)

    def prefix(q, k, v, scale):
        calls.append("prefix")
        assert torch.equal(q, query)
        assert torch.equal(k, key[:, :4093]) and torch.equal(v, value[:, :4093])
        assert scale == 0.17
        return torch.ones_like(q)

    monkeypatch.setattr(FlashAttentionImpl, "forward_fa_npu", parent)
    monkeypatch.setattr(flash_h3, "_prefix_bnsd", prefix)
    impl = object.__new__(flash_h3.MiniMaxH3FlashAttentionImpl)
    impl.causal, impl.softmax_scale = mode == "causal", 0.17
    result = None if mode == "unresolved" else ([4093, 4096], [4093, 4096])
    monkeypatch.setattr(impl, "_resolve_packed_seq_npu", lambda q, k, extra: result)
    extra: dict[str, bool | int] = {"npu_attn_varlen": True}
    if mode == "compact":
        extra["minimax_h3_compact_reference_rows"] = 2
    mask = torch.ones(1, 4096, dtype=torch.bool) if mode == "masked" else None
    metadata = AttentionMetadata(
        attn_mask=mask,
        extra=extra,
        full_attn_spans=[[(0, 4093)]] if mode == "piecewise" else None,
        query_ranges=(QueryRange(0, 4093, 0),) if mode == "ranges" else None,
    )
    if mode == "default":
        monkeypatch.delenv("MINDIE_SD_FA_TYPE", raising=False)
    else:
        monkeypatch.setenv("MINDIE_SD_FA_TYPE", "ascend_laser_attention" if mode == "laser" else "fused_attn_score")
    actual_query, actual_key, actual_value = query, key[:, :, :1] if mode == "gqa" else key, value
    if mode == "short":
        actual_query, actual_key, actual_value = (tensor[:, :128] for tensor in (query, key, value))
    output = impl.forward_fa_npu(actual_query, actual_key, actual_value, metadata)
    assert calls == ["prefix" if mode == "fused" else "parent"]
    assert torch.equal(output, torch.ones_like(actual_query) if mode == "fused" else torch.zeros_like(actual_query))


@pytest.mark.skipif(not current_omni_platform.is_npu(), reason="Requires native Ascend fused attention")
def test_native_prefix_bnsd_excludes_padding_and_preserves_inputs(monkeypatch):
    monkeypatch.setenv("MINDIE_SD_FA_TYPE", "fused_attn_score")
    torch.manual_seed(19)
    device = torch.device(current_omni_platform.device_type)
    query, key, value = [torch.randn(1, 4096, 2, 64, dtype=torch.bfloat16).to(device) for _ in range(3)]
    original = [tensor.clone() for tensor in (query, key, value)]
    used, scale = 4093, 0.17
    cu = torch.tensor([0, used, 4096], dtype=torch.int32, device=device)
    metadata = AttentionMetadata(
        extra={
            "npu_attn_varlen": True,
            "cu_seqlens_q": cu,
            "cu_seqlens_k": cu,
            "max_seqlen_q": used,
            "max_seqlen_k": used,
            "laser_input_scale": 256.0,
        }
    )
    impl = flash_h3.MiniMaxH3FlashAttentionImpl(num_heads=2, head_size=64, softmax_scale=scale, qkv_layout="BSND")
    with torch.inference_mode():
        output = impl.forward_fa_npu(query, key, value, metadata)
        dirty_key, dirty_value = key.clone(), value.clone()
        dirty_key[:, used:] = 1000
        dirty_value[:, used:] = -1000
        repeated = impl.forward_fa_npu(query, dirty_key, dirty_value, metadata)
    q, k, v = query[:, :used].float().cpu(), key[:, :used].float().cpu(), value[:, :used].float().cpu()
    scores = torch.einsum("bqhd,bkhd->bhqk", q, k) * scale
    expected = torch.einsum("bhqk,bkhd->bqhd", scores.softmax(dim=-1), v)
    assert output.is_contiguous() and output.dtype == torch.bfloat16
    assert output.shape == query.shape and torch.isfinite(output).all()
    torch.testing.assert_close(output[:, :used].float().cpu(), expected, rtol=1e-2, atol=2e-3)
    assert torch.equal(output[:, :used], repeated[:, :used])
    assert all(torch.equal(before, after) for before, after in zip(original, (query, key, value)))
