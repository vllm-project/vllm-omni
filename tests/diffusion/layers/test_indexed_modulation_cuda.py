# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDA indexed-gate paths are bit-identical to the single-row kernels they replace."""

import pytest
import torch
from vllm.triton_utils import triton

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.layers import indexed_modulation as im

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.diffusion,
    *hardware_marks(res={"cuda": "L4"}, num_cards=1),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


def _inputs(rows: int, hidden: int, table_dtype: torch.dtype):
    torch.manual_seed(0)
    bf = dict(device="cuda", dtype=torch.bfloat16)
    residual, branch = torch.randn(rows, hidden, **bf), torch.randn(rows, hidden, **bf)
    tables = (0.3 * torch.randn(12, 3 * hidden, device="cuda")).to(table_dtype)
    gate, shift, scale = tables[:, :hidden], tables[:, hidden : 2 * hidden], tables[:, 2 * hidden :]
    weight = (1 + 0.1 * torch.randn(hidden, device="cuda")).to(torch.bfloat16)
    indices = torch.randint(0, 12, (rows,), device="cuda")
    return residual, branch, gate, shift, scale, weight, indices


@pytest.mark.parametrize("table_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(("rows", "hidden"), [(1, 5376), (777, 5376), (64, 3072), (33, 1000)])
def test_column_tiled_gate_matches_row_kernel(table_dtype: torch.dtype, rows: int, hidden: int) -> None:
    residual, branch, gate, _, _, _, indices = _inputs(rows, hidden, table_dtype)
    expected = torch.empty_like(residual)
    im._indexed_gate_kernel[(rows,)](
        expected,
        residual,
        gate,
        branch,
        indices,
        hidden,
        expected.stride(0),
        residual.stride(0),
        gate.stride(0),
        branch.stride(0),
        indices.stride(0),
        0,
        block_n=triton.next_power_of_2(hidden),
        num_warps=8,
    )
    assert torch.equal(im.indexed_gate(residual, gate, branch, indices), expected)


@pytest.mark.parametrize("table_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(("rows", "hidden"), [(1, 5376), (777, 5376), (64, 3072)])
def test_fused_gate_norm_matches_gate_then_norm(table_dtype: torch.dtype, rows: int, hidden: int) -> None:
    residual, branch, gate, shift, scale, weight, indices = _inputs(rows, hidden, table_dtype)
    expected_residual = im.indexed_gate(residual, gate, branch, indices)
    expected_modulated = im.rms_norm_indexed_scale_shift(expected_residual, weight, shift, scale, indices, 1e-6)
    actual_residual, actual_modulated = im.indexed_gate_rms_norm_scale_shift(
        residual, gate, branch, weight, shift, scale, indices, 1e-6
    )
    assert torch.equal(actual_residual, expected_residual)
    assert torch.equal(actual_modulated, expected_modulated)
