# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import sys
from types import ModuleType

import pytest
import torch
from vllm.triton_utils import HAS_TRITON

from tests.helpers.mark import hardware_marks, hardware_test
from vllm_omni.diffusion.layers.indexed_modulation import (
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


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("kind", "available", "rows"),
    [
        ("rms", True, 3),
        ("rms", False, 3),
        ("gate", True, 3),
        ("gate", False, 3),
        ("gate", True, 0),
    ],
)
def test_mindiesd_modulation_dispatch(monkeypatch, mocker, kind, available, rows):
    from vllm_omni.diffusion.layers import indexed_modulation as modulation

    x = mocker.Mock(is_cpu=False, device=mocker.Mock(type="npu"), shape=(rows, 128))
    gate, branch, weight, shift, scale, indices = (mocker.Mock() for _ in range(6))
    output = mocker.Mock()
    allocate = mocker.patch.object(modulation.torch, "empty_like", return_value=output)
    launch = mocker.patch.object(modulation, "_launch_row_chunks")
    mocker.patch.object(modulation, "triton", mocker.Mock(next_power_of_2=lambda width: 128))
    eps = 1e-5
    args: tuple[object, ...]
    sd_args: tuple[object, ...]
    if kind == "rms":
        call = rms_norm_indexed_scale_shift
        args = sd_args = (x, weight, shift, scale, indices, eps)
        module, symbol = "mindiesd", "rms_norm_indexed_scale_shift"
        sd_result = mocker.sentinel.normalized
    else:
        call = indexed_gate
        args = (x, gate, branch, indices)
        sd_args = (output, *args)
        module, symbol = "mindiesd.layers.index_modulation", "indexed_gate"
        sd_result = output
    op = mocker.Mock(return_value=sd_result)
    interface = ModuleType(module)
    if available:
        setattr(interface, symbol, op)
    monkeypatch.setitem(sys.modules, module, interface)

    actual = call(*args)

    if available and rows:
        op.assert_called_once_with(*sd_args)
        launch.assert_not_called()
        assert actual == sd_result
        if kind == "gate":
            allocate.assert_called_once_with(x)
        else:
            allocate.assert_not_called()
    else:
        op.assert_not_called()
        assert actual == output
        allocate.assert_called_once_with(x)
        if rows:
            launch.assert_called_once()
        else:
            launch.assert_not_called()


@pytest.mark.cpu
def test_npu_fused_modulation_uses_omni_triton_with_eagle_installed(monkeypatch, mocker):
    from vllm_omni.diffusion.layers import indexed_modulation as modulation

    interface = ModuleType("mindiesd.layers._custom_ops")
    eagle = mocker.Mock()
    interface.eagle_indexed_gate_norm_modulate = eagle
    monkeypatch.setitem(sys.modules, interface.__name__, interface)

    residual = mocker.Mock(is_cpu=False, device=mocker.Mock(type="npu"), shape=(3, 128))
    gate, branch, weight, shift, scale, indices = (mocker.Mock() for _ in range(6))
    outputs = (mocker.sentinel.residual, mocker.sentinel.modulated)
    mocker.patch.object(modulation.torch, "empty_like", side_effect=outputs)
    launch = mocker.patch.object(modulation, "_launch_row_chunks")
    mocker.patch.object(modulation, "triton", mocker.Mock(next_power_of_2=lambda width: 128))

    assert indexed_gate_rms_norm_scale_shift(residual, gate, branch, weight, shift, scale, indices, 1e-5) == outputs
    eagle.assert_not_called()
    launch.assert_called_once()
    assert launch.call_args.args[:5] == (modulation._indexed_gate_rms_norm_scale_shift_kernel, 3, "npu", *outputs)


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
    normalized_residual = expected_residual.float()
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


@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards=1)
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


@hardware_test(res={"npu": "A2"}, num_cards=1)
@pytest.mark.skipif(not current_omni_platform.is_npu(), reason="requires Ascend NPU")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
@pytest.mark.parametrize("hidden_size", [3072, 5376])
def test_npu_fused_modulation_preserves_bf16_residual_boundary(hidden_size):
    device = current_omni_platform.get_torch_device()
    residual = torch.ones(2, hidden_size, device=device, dtype=torch.bfloat16)
    gate = torch.ones_like(residual[:1])
    branch = torch.zeros_like(residual)
    branch[:, 0] = 2**-8
    weight = torch.ones_like(residual[0])
    shift = -gate
    scale = torch.zeros_like(gate)
    indices = torch.zeros(residual.shape[0], device=device, dtype=torch.int64)

    residual_out, modulated_out = indexed_gate_rms_norm_scale_shift(
        residual, gate, branch, weight, shift, scale, indices, 1e-6
    )

    # The first-column update is a BF16 halfway value and rounds back to one.
    # Subtracting one after RMSNorm exposes any use of the unrounded update.
    assert torch.equal(residual_out, residual)
    torch.testing.assert_close(modulated_out, torch.zeros_like(modulated_out), atol=1e-5, rtol=0)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
@pytest.mark.parametrize("fused_gate", [False, True])
def test_fused_modulation_keeps_fp32_affine_intermediates(fused_gate: bool) -> None:
    device = current_omni_platform.get_torch_device()
    rows, hidden_size = 2, 5376
    x = torch.ones(rows, hidden_size, device=device, dtype=torch.bfloat16)
    weight = torch.full((hidden_size,), 1.125, device=device, dtype=x.dtype)
    shift = torch.zeros(1, hidden_size, device=device, dtype=x.dtype)
    scale = torch.full_like(shift, 0.09375)
    indices = torch.zeros(rows, device=device, dtype=torch.int64)
    eps = 1e-6
    if fused_gate:
        residual, actual = indexed_gate_rms_norm_scale_shift(
            x, torch.zeros_like(shift), x, weight, shift, scale, indices, eps
        )
        assert torch.equal(residual, x)
    else:
        actual = rms_norm_indexed_scale_shift(x, weight, shift, scale, indices, eps)

    # Without the RMS epsilon, 1.125 * 1.09375 = 1.23046875 is exactly
    # halfway between two BF16 values. FP32 normalization keeps the epsilon
    # and rounds down to 1.2265625. Premature BF16 rounding drops the epsilon
    # and instead rounds the tie up to 1.234375, changing every output row.
    assert torch.equal(actual, torch.full_like(x, 1.2265625))
