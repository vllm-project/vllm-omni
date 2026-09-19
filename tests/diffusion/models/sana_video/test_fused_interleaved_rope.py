# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from vllm.triton_utils import HAS_TRITON

NVIDIA_CUDA_AVAILABLE = torch.cuda.is_available() and torch.version.hip is None

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.cuda,
    pytest.mark.diffusion,
    pytest.mark.skipif(not NVIDIA_CUDA_AVAILABLE, reason="NVIDIA CUDA required"),
    pytest.mark.skipif(not HAS_TRITON, reason="Triton required"),
]


def _reference(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    output = torch.empty_like(hidden_states)
    output[..., 0::2] = x1 * cos[..., 0::2] - x2 * sin[..., 1::2]
    output[..., 1::2] = x1 * sin[..., 1::2] + x2 * cos[..., 0::2]
    return output


@pytest.mark.parametrize("table_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    ("batch", "seq_len", "num_heads", "head_dim"),
    [
        (1, 1, 1, 2),
        (1, 17, 3, 12),
        (2, 129, 20, 112),
    ],
)
def test_fused_interleaved_rope_is_bit_exact(
    table_dtype,
    batch,
    seq_len,
    num_heads,
    head_dim,
):
    from vllm_omni.diffusion.layers.fused_interleaved_rope import (
        fused_interleaved_rope,
    )

    generator = torch.Generator(device="cuda").manual_seed(42)
    q = torch.randn(
        batch,
        seq_len,
        num_heads,
        head_dim,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    k = torch.randn_like(q)
    cos = torch.randn(
        1,
        seq_len,
        1,
        head_dim,
        dtype=table_dtype,
        device="cuda",
        generator=generator,
    )
    sin = torch.randn_like(cos)

    q_out, k_out = fused_interleaved_rope(q, k, cos, sin)

    assert torch.equal(q_out, _reference(q, cos, sin))
    assert torch.equal(k_out, _reference(k, cos, sin))
    assert q_out.is_contiguous()
    assert k_out.is_contiguous()
    assert q_out.data_ptr() not in (q.data_ptr(), k.data_ptr())
    assert k_out.data_ptr() not in (q.data_ptr(), k.data_ptr())


def test_fused_interleaved_rope_preserves_bfloat16_intermediate_rounding():
    from vllm_omni.diffusion.layers.fused_interleaved_rope import (
        fused_interleaved_rope,
    )

    # These products land on opposite sides of a BF16 subtraction depending
    # on whether each multiplication is materialized in BF16 first.  This
    # locks down the Diffusers eager contract rather than only final casting.
    q = torch.tensor([[[[0.9375, 0.9375]]]], dtype=torch.bfloat16, device="cuda")
    k = -q
    cos = torch.tensor([[[[0.9375, 0.0]]]], dtype=torch.bfloat16, device="cuda")
    sin = torch.tensor([[[[0.0, 0.9453125]]]], dtype=torch.bfloat16, device="cuda")

    q_reference = _reference(q, cos, sin)
    without_intermediate_rounding = (
        q[..., 0].float() * cos[..., 0].float() - q[..., 1].float() * sin[..., 1].float()
    ).to(torch.bfloat16)
    assert not torch.equal(q_reference[..., 0], without_intermediate_rounding)

    q_out, k_out = fused_interleaved_rope(q, k, cos, sin)

    assert torch.equal(q_out, q_reference)
    assert torch.equal(k_out, _reference(k, cos, sin))


def test_fused_interleaved_rope_preserves_special_value_classes():
    from vllm_omni.diffusion.layers.fused_interleaved_rope import (
        fused_interleaved_rope,
    )

    values = [
        float("nan"),
        1.0,
        float("inf"),
        0.0,
        -float("inf"),
        -0.0,
        torch.finfo(torch.bfloat16).max,
        float.fromhex("0x1p-133"),
        -torch.finfo(torch.bfloat16).max,
        -float.fromhex("0x1p-133"),
        0.9375,
        0.9453125,
        -2.0,
        0.5,
        3.0,
        -4.0,
    ]
    q = torch.tensor(values, dtype=torch.bfloat16, device="cuda").reshape(1, 1, 1, -1)
    k = q.flip(-1).contiguous()
    cos = torch.zeros_like(q)
    sin = torch.zeros_like(q)
    cos[..., 0::2] = 1

    q_out, k_out = fused_interleaved_rope(q, k, cos, sin)
    for actual, reference in (
        (q_out, _reference(q, cos, sin)),
        (k_out, _reference(k, cos, sin)),
    ):
        assert torch.equal(torch.isnan(actual), torch.isnan(reference))
        non_nan = ~torch.isnan(reference)
        assert torch.equal(actual.view(torch.int16)[non_nan], reference.view(torch.int16)[non_nan])


def test_fused_interleaved_rope_predicate_rejects_unsupported_inputs():
    from vllm_omni.diffusion.layers.fused_interleaved_rope import (
        can_use_fused_interleaved_rope,
    )

    q = torch.empty(1, 17, 3, 12, dtype=torch.bfloat16, device="cuda")
    k = torch.empty_like(q)
    cos = torch.empty(1, 17, 1, 12, dtype=torch.bfloat16, device="cuda")
    sin = torch.empty_like(cos)

    assert can_use_fused_interleaved_rope(q, k, cos, sin)

    flat_q, flat_k = q.flatten(), k.flatten()
    assert not can_use_fused_interleaved_rope(flat_q, flat_k, cos, sin)

    bad_k_shape = torch.empty(1, 16, 3, 12, dtype=torch.bfloat16, device="cuda")
    assert not can_use_fused_interleaved_rope(q, bad_k_shape, cos, sin)

    assert not can_use_fused_interleaved_rope(q.float(), k.float(), cos, sin)
    assert not can_use_fused_interleaved_rope(q, k.float(), cos, sin)

    q_odd = torch.empty(1, 17, 3, 11, dtype=torch.bfloat16, device="cuda")
    k_odd = torch.empty_like(q_odd)
    cos_odd = torch.empty(1, 17, 1, 11, dtype=torch.bfloat16, device="cuda")
    sin_odd = torch.empty_like(cos_odd)
    assert not can_use_fused_interleaved_rope(q_odd, k_odd, cos_odd, sin_odd)

    q_noncontiguous = torch.empty(1, 17, 3, 24, dtype=torch.bfloat16, device="cuda")[..., ::2]
    k_noncontiguous = torch.empty(1, 17, 3, 24, dtype=torch.bfloat16, device="cuda")[..., ::2]
    assert q_noncontiguous.shape == q.shape and not q_noncontiguous.is_contiguous()
    assert k_noncontiguous.shape == k.shape and not k_noncontiguous.is_contiguous()
    assert not can_use_fused_interleaved_rope(q_noncontiguous, k, cos, sin)
    assert not can_use_fused_interleaved_rope(q, k_noncontiguous, cos, sin)

    assert not can_use_fused_interleaved_rope(q, k, cos.double(), sin.double())
    assert not can_use_fused_interleaved_rope(q, k, cos.float(), sin)

    bad_cos_shape = torch.empty(1, 17, 1, 10, dtype=torch.bfloat16, device="cuda")
    bad_sin_shape = torch.empty_like(bad_cos_shape)
    assert not can_use_fused_interleaved_rope(q, k, bad_cos_shape, bad_sin_shape)

    cos_noncontiguous = torch.empty(1, 17, 1, 24, dtype=torch.bfloat16, device="cuda")[..., ::2]
    sin_noncontiguous = torch.empty(1, 17, 1, 24, dtype=torch.bfloat16, device="cuda")[..., ::2]
    assert cos_noncontiguous.shape == cos.shape and not cos_noncontiguous.is_contiguous()
    assert sin_noncontiguous.shape == sin.shape and not sin_noncontiguous.is_contiguous()
    assert not can_use_fused_interleaved_rope(q, k, cos_noncontiguous, sin)
    assert not can_use_fused_interleaved_rope(q, k, cos, sin_noncontiguous)

    q_empty = torch.empty(1, 0, 3, 12, dtype=torch.bfloat16, device="cuda")
    k_empty = torch.empty_like(q_empty)
    cos_empty = torch.empty(1, 0, 1, 12, dtype=torch.bfloat16, device="cuda")
    sin_empty = torch.empty_like(cos_empty)
    assert not can_use_fused_interleaved_rope(q_empty, k_empty, cos_empty, sin_empty)

    assert not can_use_fused_interleaved_rope(q, k, cos.cpu(), sin.cpu())

    q_requires_grad = q.detach().requires_grad_()
    assert not can_use_fused_interleaved_rope(q_requires_grad, k, cos, sin)
