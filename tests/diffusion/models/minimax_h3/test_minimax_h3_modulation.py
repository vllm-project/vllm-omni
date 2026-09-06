# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from vllm.triton_utils import HAS_TRITON

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.attention.ops.minimax_h3_modulation import (
    _MAX_1D_GRID_SIZE,
    _iter_row_chunks,
    _launch_row_chunks,
    indexed_gate,
    indexed_gate_rms_norm_scale_shift,
    indexed_scale_shift_,
    rms_norm_indexed_scale_shift,
)
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        (0, []),
        (_MAX_1D_GRID_SIZE, [(0, _MAX_1D_GRID_SIZE)]),
        (_MAX_1D_GRID_SIZE + 1, [(0, _MAX_1D_GRID_SIZE), (_MAX_1D_GRID_SIZE, 1)]),
        (69888, [(0, _MAX_1D_GRID_SIZE), (_MAX_1D_GRID_SIZE, 4353)]),
    ],
)
def test_row_chunks_respect_ascend_grid_limit(rows, expected):
    assert list(_iter_row_chunks(rows)) == expected


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("device_type", "expected_grids_and_offsets"),
    [
        ("npu", [(_MAX_1D_GRID_SIZE, 0), (4353, _MAX_1D_GRID_SIZE)]),
        ("cuda", [(69888, 0)]),
    ],
)
def test_launch_row_chunks_is_npu_only(device_type, expected_grids_and_offsets):
    class RecordingKernel:
        def __init__(self):
            self.calls = []

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                self.calls.append((grid, args, kwargs))

            return launch

    kernel = RecordingKernel()
    first_arg = object()
    second_arg = object()

    _launch_row_chunks(kernel, 69888, device_type, first_arg, second_arg, block_n=16)

    assert kernel.calls == [
        ((grid,), (first_arg, second_arg, row_offset), {"block_n": 16})
        for grid, row_offset in expected_grids_and_offsets
    ]


@pytest.mark.skipif(not current_omni_platform.is_npu(), reason="requires Ascend NPU")
@pytest.mark.parametrize(
    "device",
    [pytest.param("npu", marks=hardware_marks(res={"npu": "A3"}, num_cards=1))],
)
def test_modulation_wrappers_match_reference_above_grid_limit(device):
    rows = 69888
    hidden_size = 16
    num_conditions = 7
    eps = 1e-6
    dtype = torch.bfloat16

    torch.manual_seed(0)
    x = torch.randn(rows, hidden_size, dtype=dtype)
    residual = torch.randn_like(x)
    branch = torch.randn_like(x)
    weight = torch.randn(hidden_size, dtype=dtype)
    shift = torch.randn(num_conditions, hidden_size, dtype=dtype)
    scale = torch.randn_like(shift)
    gate = torch.randn_like(shift)
    indices = torch.arange(rows) % num_conditions

    indexed_shift = shift.index_select(0, indices)
    indexed_scale = scale.index_select(0, indices)
    indexed_gate_values = gate.index_select(0, indices)

    x_float = x.float()
    branch_float = branch.float()
    indexed_shift_float = indexed_shift.float()
    indexed_scale_float = indexed_scale.float()
    indexed_gate_float = indexed_gate_values.float()

    expected_scale_shift = (x_float * (1.0 + indexed_scale_float) + indexed_shift_float).to(dtype)
    expected_gate = (x_float + indexed_gate_float * branch_float).to(dtype)

    normalized = x_float
    normalized = normalized * torch.rsqrt(normalized.pow(2).mean(-1, keepdim=True) + eps)
    normalized = normalized * weight.float()
    expected_rms = (normalized * (1.0 + indexed_scale_float) + indexed_shift_float).to(dtype)

    updated_residual = residual.float() + indexed_gate_float * branch_float
    expected_residual = updated_residual.to(dtype)
    normalized_residual = updated_residual
    normalized_residual = normalized_residual * torch.rsqrt(normalized_residual.pow(2).mean(-1, keepdim=True) + eps)
    normalized_residual = normalized_residual * weight.float()
    expected_modulated = (normalized_residual * (1.0 + indexed_scale_float) + indexed_shift_float).to(dtype)

    npu_x = x.to(device)
    npu_residual = residual.to(device)
    npu_branch = branch.to(device)
    npu_weight = weight.to(device)
    npu_shift = shift.to(device)
    npu_scale = scale.to(device)
    npu_gate = gate.to(device)
    npu_indices = indices.to(device)

    actual_scale_shift = indexed_scale_shift_(npu_x.clone(), npu_shift, npu_scale, npu_indices)
    actual_gate = indexed_gate(npu_x, npu_gate, npu_branch, npu_indices)
    actual_rms = rms_norm_indexed_scale_shift(
        npu_x,
        npu_weight,
        npu_shift,
        npu_scale,
        npu_indices,
        eps,
    )
    actual_residual, actual_modulated = indexed_gate_rms_norm_scale_shift(
        npu_residual,
        npu_gate,
        npu_branch,
        npu_weight,
        npu_shift,
        npu_scale,
        npu_indices,
        eps,
    )

    for actual, expected in (
        (actual_scale_shift, expected_scale_shift),
        (actual_gate, expected_gate),
        (actual_rms, expected_rms),
        (actual_residual, expected_residual),
        (actual_modulated, expected_modulated),
    ):
        torch.testing.assert_close(actual.cpu(), expected, atol=2e-2, rtol=2e-2)


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
