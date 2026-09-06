# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from vllm.triton_utils import HAS_TRITON

from vllm_omni.diffusion.attention.ops.minimax_h3_modulation import (
    indexed_gate_rms_norm_scale_shift,
    rms_norm_indexed_scale_shift,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cuda]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
def test_fused_modulation_preserves_bf16_residual_boundary() -> None:
    torch.manual_seed(17)
    rows, hidden_size, conditions = 11, 3072, 4
    tensors = [
        torch.randn(rows, hidden_size, device="cuda", dtype=torch.bfloat16),
        torch.randn(conditions, hidden_size, device="cuda", dtype=torch.bfloat16),
        torch.randn(rows, hidden_size, device="cuda", dtype=torch.bfloat16),
        torch.randn(hidden_size, device="cuda", dtype=torch.bfloat16),
        torch.randn(conditions, hidden_size, device="cuda", dtype=torch.bfloat16),
        torch.randn(conditions, hidden_size, device="cuda", dtype=torch.bfloat16),
    ]
    residual, gate, branch, weight, shift, scale = tensors
    indices = torch.arange(rows, device="cuda") % conditions
    eps = 1e-6

    residual_out, modulated_out = indexed_gate_rms_norm_scale_shift(
        residual,
        gate,
        branch,
        weight,
        shift,
        scale,
        indices,
        eps,
    )
    expected = rms_norm_indexed_scale_shift(
        residual_out,
        weight,
        shift,
        scale,
        indices,
        eps,
    )

    assert torch.equal(modulated_out, expected)
