# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ring SDPA kernel on XPU: the per-block LSE must be correct, or the ring merge is silently wrong.

XPU's ``aten._scaled_dot_product_efficient_attention`` returns an all-zero LSE, so
``pytorch_attn_forward`` swaps it for the flash op (fp16/bf16) or an fp32 math path.
"""

import pytest
import torch
import torch.nn.functional as F

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.attention.backends.ring import ring_kernels
from vllm_omni.diffusion.attention.backends.ring.ring_utils import update_out_and_lse

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("op_type", "dtype", "causal", "q_len", "kv_len", "expected"),
    [
        ("efficient", torch.bfloat16, False, 128, 192, "flash"),
        ("efficient", torch.float16, True, 128, 128, "flash"),
        # The XPU flash op aligns a causal mask bottom-right when q_len != kv_len; efficient is top-left.
        ("efficient", torch.bfloat16, True, 128, 192, "math"),
        # The XPU flash op supports fp16/bf16 only.
        ("efficient", torch.float32, False, 128, 128, "math"),
        # An explicit flash request is left alone (it rejects fp32 itself).
        ("flash", torch.bfloat16, True, 128, 192, "flash"),
    ],
)
def test_xpu_sdpa_op_type(op_type, dtype, causal, q_len, kv_len, expected):
    assert ring_kernels._xpu_sdpa_op_type(op_type, dtype, causal, q_len, kv_len) == expected


@pytest.mark.cpu
def test_xpu_sdpa_op_type_without_aten_flash():
    # Torch XPU builds without SYCL-TLA have no aten flash op; bf16 then takes the math path too.
    assert ring_kernels._xpu_sdpa_op_type("efficient", torch.bfloat16, False, 128, 128, has_flash=False) == "math"


@pytest.mark.cpu
@pytest.mark.parametrize(("causal", "q_len", "kv_len"), [(False, 48, 80), (True, 64, 64), (True, 48, 80)])
@pytest.mark.parametrize("q_chunk", [1024, 20])  # 20: several query chunks, the last one partial
def test_math_attn_with_lse_matches_reference(causal, q_len, kv_len, q_chunk):
    torch.manual_seed(0)
    q = torch.randn(1, 2, q_len, 32)
    k, v = torch.randn(1, 2, kv_len, 32), torch.randn(1, 2, kv_len, 32)
    scale = 32**-0.5

    out, lse = ring_kernels._math_attn_with_lse(q, k, v, scale, causal, q_chunk=q_chunk)

    scores = (q @ k.transpose(-2, -1)) * scale
    mask = torch.ones(q_len, kv_len, dtype=torch.bool).tril() if causal else None  # top-left, as F.sdpa is_causal
    torch.testing.assert_close(out, F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale))
    torch.testing.assert_close(
        lse, torch.logsumexp(scores if mask is None else scores.masked_fill(~mask, -torch.inf), -1)
    )


@hardware_test(res={"xpu": "B60"}, num_cards=1)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
def test_xpu_ring_kernel_lse(dtype, causal):
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 256, 4, 128).to("xpu", dtype) for _ in range(3))
    scale = 128**-0.5

    _, lse = ring_kernels.pytorch_attn_forward(q, k, v, softmax_scale=scale, causal=causal)

    scores = (q.transpose(1, 2).float() @ k.transpose(1, 2).float().transpose(-2, -1)) * scale
    if causal:
        scores = scores.masked_fill(~torch.ones(256, 256, dtype=torch.bool, device="xpu").tril(), -torch.inf)
    torch.testing.assert_close(lse, torch.logsumexp(scores, -1), atol=1e-4, rtol=1e-4)


@hardware_test(res={"xpu": "B60"}, num_cards=1)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_xpu_ring_kernel_merge_matches_full_attention(dtype):
    """Two K/V blocks merged through their LSE (one ring step each) must equal attention over all K/V."""
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 256, 4, 128).to("xpu", dtype) for _ in range(3))
    scale = 128**-0.5
    out = lse = None
    for block in (slice(0, 128), slice(128, 256)):
        block_out, block_lse = ring_kernels.pytorch_attn_forward(
            q, k[:, block], v[:, block], softmax_scale=scale, causal=False
        )
        out, lse = update_out_and_lse(out, lse, block_out, block_lse, lse_layout="bhs")

    ref = F.scaled_dot_product_attention(
        q.transpose(1, 2).float(), k.transpose(1, 2).float(), v.transpose(1, 2).float(), scale=scale
    ).transpose(1, 2)
    tol = 1e-4 if dtype == torch.float32 else 2e-2
    torch.testing.assert_close(out.float(), ref, atol=tol, rtol=tol)
