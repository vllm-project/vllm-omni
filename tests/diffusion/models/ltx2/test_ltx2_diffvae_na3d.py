# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""ROCm qualification for the bounded LTX-2.5 DiffVAE Triton fallback."""

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from tests.helpers.mark import hardware_test

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@hardware_test(res={"rocm": "MI325"}, num_cards=1)
@torch.inference_mode()
def test_triton_na3d_matches_explicit_inward_shift_mask_on_rocm():
    from vllm_omni.diffusion.models.ltx2.ops.na3d import na3d

    # Exercise inward shifting on every axis instead of the degenerate case
    # where the 11x11x11 window covers the complete token grid.
    shape = (13, 12, 17)
    generator = torch.Generator(device="cuda").manual_seed(43)
    query, key, value = [
        torch.randn(1, *shape, 4, 64, device="cuda", dtype=torch.bfloat16, generator=generator) for _ in range(3)
    ]
    query = query * 0.125
    coordinates = torch.stack(
        torch.meshgrid(*(torch.arange(n, device="cuda") for n in shape), indexing="ij"), dim=-1
    ).reshape(-1, 3)
    start = (coordinates - 5).clamp_min(0).minimum(torch.tensor(shape, device="cuda") - 11)
    mask = ((coordinates[None] >= start[:, None]) & (coordinates[None] < start[:, None] + 11)).all(-1)
    with sdpa_kernel(SDPBackend.MATH):
        expected = (
            torch.nn.functional.scaled_dot_product_attention(
                *(tensor.flatten(1, 3).transpose(1, 2).float() for tensor in (query, key, value)),
                attn_mask=mask,
                scale=1.0,
            )
            .transpose(1, 2)
            .reshape(query.shape)
        )
    actual = na3d(query, key, value, kernel_size=(11, 11, 11), scale=1.0).float()

    assert actual.isfinite().all()
    assert float((actual - expected).norm() / expected.norm()) < 0.005
    torch.testing.assert_close(actual, expected, atol=0.008, rtol=0.025)
