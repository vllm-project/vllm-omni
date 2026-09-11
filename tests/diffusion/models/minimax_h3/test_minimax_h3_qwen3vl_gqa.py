# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.npu]


@pytest.mark.skipif(not current_omni_platform.is_npu(), reason="requires Ascend NPU")
def test_minimax_h3_npu_sdpa_patch_is_independent_and_idempotent(monkeypatch) -> None:
    from vllm_omni.diffusion.models.minimax_h3 import encoder as encoder_module
    from vllm_omni.platforms.npu.models import minimax_h3 as npu_minimax_h3

    def sdpa_sentinel(query, key, value):
        del key, value
        return query

    def rope_sentinel(q, k, cos, sin):
        del cos, sin
        return q, k

    monkeypatch.setattr(npu_minimax_h3, "_ROPE_PATCHED", False)
    monkeypatch.setattr(npu_minimax_h3, "_SDPA_PATCHED", False)
    monkeypatch.setattr(encoder_module, "_apply_rotary_pos_emb", rope_sentinel)
    monkeypatch.setattr(encoder_module, "_scaled_dot_product_attention", sdpa_sentinel)

    npu_minimax_h3.apply_minimax_h3_qwen3vl_sdpa_patch()

    assert encoder_module._scaled_dot_product_attention is npu_minimax_h3._scaled_dot_product_attention_npu
    assert encoder_module._apply_rotary_pos_emb is rope_sentinel
    assert not npu_minimax_h3._ROPE_PATCHED
    assert npu_minimax_h3._SDPA_PATCHED

    monkeypatch.setattr(encoder_module, "_scaled_dot_product_attention", sdpa_sentinel)
    npu_minimax_h3.apply_minimax_h3_qwen3vl_sdpa_patch()

    assert encoder_module._scaled_dot_product_attention is sdpa_sentinel


@pytest.mark.skipif(not current_omni_platform.is_npu(), reason="requires Ascend NPU")
@pytest.mark.parametrize(
    ("num_heads", "num_kv_heads", "expected_enable_gqa"),
    [(2, 2, False), (4, 2, True)],
)
def test_minimax_h3_npu_sdpa_preserves_compressed_heads_and_selects_gqa(
    monkeypatch,
    num_heads: int,
    num_kv_heads: int,
    expected_enable_gqa: bool,
) -> None:
    from vllm_omni.platforms.npu.models import minimax_h3 as npu_minimax_h3

    captured: dict[str, object] = {}

    def fake_sdpa(query, key, value, **kwargs):
        captured.update(query=query, key=key, value=value, kwargs=kwargs)
        return query

    monkeypatch.setattr(F, "scaled_dot_product_attention", fake_sdpa)
    query = torch.randn(1, num_heads, 3, 8)
    key = torch.randn(1, num_kv_heads, 3, 8)
    value = torch.randn_like(key)

    output = npu_minimax_h3._scaled_dot_product_attention_npu(query, key, value)

    assert captured["query"] is query
    assert captured["key"] is key
    assert captured["value"] is value
    assert captured["kwargs"] == {
        "dropout_p": 0.0,
        "is_causal": True,
        "enable_gqa": expected_enable_gqa,
    }
    assert output is query


@pytest.mark.skipif(not current_omni_platform.is_npu(), reason="requires Ascend NPU")
@pytest.mark.parametrize(
    ("num_heads", "num_key_heads", "num_value_heads", "message"),
    [
        (4, 2, 1, "key and value to have the same number of heads"),
        (4, 0, 0, "at least one KV head"),
        (3, 2, 2, "query heads to be a multiple of KV heads"),
    ],
)
def test_minimax_h3_npu_sdpa_rejects_invalid_head_configurations(
    num_heads: int,
    num_key_heads: int,
    num_value_heads: int,
    message: str,
) -> None:
    from vllm_omni.platforms.npu.models import minimax_h3 as npu_minimax_h3

    query = torch.randn(1, num_heads, 3, 8)
    key = torch.randn(1, num_key_heads, 3, 8)
    value = torch.randn(1, num_value_heads, 3, 8)

    with pytest.raises(ValueError, match=message):
        npu_minimax_h3._scaled_dot_product_attention_npu(query, key, value)


@pytest.mark.skipif(not current_omni_platform.is_npu(), reason="requires Ascend NPU")
def test_minimax_h3_npu_causal_gqa_matches_expanded_kv() -> None:
    """Exercise the real torch-npu causal GQA operator."""
    from vllm_omni.platforms.npu.models import minimax_h3 as npu_minimax_h3

    torch.manual_seed(0)
    batch_size, seq_len = 1, 16
    num_heads, num_kv_heads, head_size = 8, 2, 128
    device = torch.device("npu")
    dtype = torch.bfloat16

    query = torch.randn(batch_size, num_heads, seq_len, head_size, device=device, dtype=dtype)
    key = torch.randn(batch_size, num_kv_heads, seq_len, head_size, device=device, dtype=dtype)
    value = torch.randn_like(key)

    actual = npu_minimax_h3._scaled_dot_product_attention_npu(query, key, value)

    repeat_num = num_heads // num_kv_heads
    expected = F.scaled_dot_product_attention(
        query,
        key.repeat_interleave(repeat_num, dim=1),
        value.repeat_interleave(repeat_num, dim=1),
        dropout_p=0.0,
        is_causal=True,
    )

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
