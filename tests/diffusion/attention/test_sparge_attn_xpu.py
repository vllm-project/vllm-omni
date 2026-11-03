# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
import torch.nn.functional as F

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.attention.backends.sparge_attn import SpargeAttentionBackend, SpargeAttentionImpl

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]

if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
    pytest.skip("XPU is unavailable", allow_module_level=True)

pytest.importorskip("auto_round_kernel", reason="auto-round-lib is unavailable")
try:
    SpargeAttentionBackend.validate_available()
except ImportError as exc:
    pytest.skip(str(exc), allow_module_level=True)


@hardware_test(res={"xpu": "B60"}, num_cards=1)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ("head_dim", "q_heads", "kv_heads", "q_len", "kv_len", "causal", "expected_q_tile", "extra_kwargs"),
    [
        (64, 4, 4, 1024, 256, False, 128, {}),
        (64, 12, 4, 512, 257, False, 128, {}),
        (128, 12, 4, 1024, 512, False, 256, {}),
        (128, 12, 4, 1024, 512, False, 0, {"sparge_q_block_tokens": 64}),
        (128, 4, 4, 512, 512, True, 256, {}),
        (128, 4, 4, 4000, 77, False, 256, {}),
    ],
)
def test_real_ark_topk_one_matches_sdpa(
    head_dim,
    q_heads,
    kv_heads,
    q_len,
    kv_len,
    causal,
    expected_q_tile,
    extra_kwargs,
    dtype,
):
    torch.manual_seed(0)
    torch.xpu.manual_seed(0)
    scale = head_dim**-0.5
    impl = SpargeAttentionImpl(
        q_heads,
        head_dim,
        scale,
        causal=causal,
        num_kv_heads=kv_heads,
        backend_kwargs={"sparge_topk": 1.0, **extra_kwargs},
    )
    assert impl.q_tile_override == expected_q_tile
    query = torch.randn(1, q_len, q_heads, head_dim, device="xpu", dtype=dtype)
    key = torch.randn(1, kv_len, kv_heads, head_dim, device="xpu", dtype=dtype)
    value = torch.randn(1, kv_len, kv_heads, head_dim, device="xpu", dtype=dtype)

    actual = impl.forward_xpu(query, key, value)
    query_ref = query.transpose(1, 2).float()
    key_ref = key.transpose(1, 2).float()
    value_ref = value.transpose(1, 2).float()
    if q_heads != kv_heads:
        repeat = q_heads // kv_heads
        key_ref = key_ref.repeat_interleave(repeat, dim=1)
        value_ref = value_ref.repeat_interleave(repeat, dim=1)
    expected = F.scaled_dot_product_attention(
        query_ref,
        key_ref,
        value_ref,
        dropout_p=0.0,
        is_causal=causal,
        scale=scale,
    ).transpose(1, 2)

    assert torch.isfinite(actual).all()
    cosine = F.cosine_similarity(actual.float().flatten(2), expected.flatten(2), dim=-1).mean().item()
    assert cosine > 0.999, f"cosine similarity {cosine:.6f} is below 0.999"
