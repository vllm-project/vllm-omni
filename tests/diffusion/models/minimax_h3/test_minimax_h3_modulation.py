# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import sys

import pytest
import torch
from vllm.triton_utils import HAS_TRITON

from tests.helpers.device import DEVICE_MODULE, DEVICE_TYPE, NVIDIA_CUDA_OR_XPU
from tests.helpers.mark import hardware_marks, hardware_test
from vllm_omni.diffusion.layers import indexed_modulation
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


@pytest.mark.skipif(not current_omni_platform.is_npu(), reason="requires Ascend NPU")
@pytest.mark.parametrize(
    "device",
    [pytest.param("npu", marks=hardware_marks(res={"npu": "A3"}, num_cards=1))],
)
def test_modulation_wrappers_match_reference_above_grid_limit(device, mocker):
    mocker.patch.object(indexed_modulation, "_supports_mindiesd_inputs", return_value=False)
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


@hardware_test(res={"cuda": "L4", "rocm": "MI325", "xpu": "B60"}, num_cards=1)
@pytest.mark.skipif(not DEVICE_MODULE.is_available(), reason="GPU required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
def test_fused_modulation_preserves_bf16_residual_boundary() -> None:
    torch.manual_seed(17)
    rows, hidden_size, conditions = 11, 3072, 4
    tensors = [
        torch.randn(rows, hidden_size, device=DEVICE_TYPE, dtype=torch.bfloat16),
        torch.randn(conditions, hidden_size, device=DEVICE_TYPE, dtype=torch.bfloat16),
        torch.randn(rows, hidden_size, device=DEVICE_TYPE, dtype=torch.bfloat16),
        torch.randn(hidden_size, device=DEVICE_TYPE, dtype=torch.bfloat16),
        torch.randn(conditions, hidden_size, device=DEVICE_TYPE, dtype=torch.bfloat16),
        torch.randn(conditions, hidden_size, device=DEVICE_TYPE, dtype=torch.bfloat16),
    ]
    residual, gate, branch, weight, shift, scale = tensors
    indices = torch.arange(rows, device=DEVICE_TYPE) % conditions
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
def test_npu_fused_modulation_preserves_bf16_residual_boundary():
    device = current_omni_platform.get_torch_device()
    residual = torch.ones(2, 5376, device=device, dtype=torch.bfloat16)
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


@hardware_test(res={"npu": "A2"}, num_cards=1)
@pytest.mark.skipif(not current_omni_platform.is_npu(), reason="requires Ascend NPU")
@pytest.mark.parametrize("width", [8192, 8193])
def test_mindiesd_rms_width_limit(mocker, width):
    sdk = pytest.importorskip("mindiesd.layers.index_modulation")
    device = current_omni_platform.get_torch_device()
    x = torch.empty(1, width, dtype=torch.bfloat16, device=device)
    weight = x[0]
    indices = torch.zeros(1, dtype=torch.long, device=device)
    sdk_rms = mocker.patch.object(sdk, "rms_norm_indexed_scale_shift", return_value=x)
    # Check dispatch independently of the fallback kernel's on-chip buffer limit.
    triton_launch = mocker.patch.object(indexed_modulation, "_launch_row_chunks")

    result = rms_norm_indexed_scale_shift(x, weight, x, x, indices, 1e-6)

    if width == 8192:
        assert result is x
        sdk_rms.assert_called_once_with(x, weight, x, x, indices, 1e-6)
        triton_launch.assert_not_called()
    else:
        sdk_rms.assert_not_called()
        triton_launch.assert_called_once()


@hardware_test(res={"npu": "A2"}, num_cards=1)
@pytest.mark.skipif(not current_omni_platform.is_npu(), reason="requires Ascend NPU")
@pytest.mark.parametrize(
    ("dtype", "eps", "col_stride", "expected_sdk_calls"),
    [
        pytest.param(torch.bfloat16, 1e-6, 1, (1, 1), id="h3-bf16"),
        pytest.param(torch.float16, 1e-6, 1, (0, 0), id="fp16"),
        pytest.param(torch.bfloat16, 0.0, 1, (0, 1), id="zero-eps"),
        pytest.param(torch.bfloat16, 1e-50, 1, (0, 1), id="eps-rounds-to-zero"),
        pytest.param(torch.bfloat16, 1e-6, 2, (0, 0), id="strided-columns"),
        pytest.param(torch.bfloat16, 1e-6, 1, None, id="missing-sdk"),
    ],
)
def test_npu_modulation_dispatch_and_outputs(monkeypatch, mocker, dtype, eps, col_stride, expected_sdk_calls):
    if expected_sdk_calls is None:
        monkeypatch.setitem(sys.modules, "mindiesd.layers.index_modulation", None)
        triton_launch = mocker.spy(indexed_modulation, "_launch_row_chunks")
    else:
        sdk = pytest.importorskip("mindiesd.layers.index_modulation")
        sdk_rms = mocker.spy(sdk, "rms_norm_indexed_scale_shift")
        sdk_gate = mocker.spy(sdk, "indexed_gate")

    width = 5376
    device = current_omni_platform.get_torch_device()
    generator = torch.Generator().manual_seed(1234)
    x, other, gate, shift, scale = [
        torch.randn(rows, width * col_stride, generator=generator).to(dtype) for rows in (5, 5, 3, 3, 3)
    ]
    weight = torch.randn(width, generator=generator).to(dtype)
    indices = torch.tensor([2, 0, 2, 1, 0])
    npu_x, npu_other, npu_gate, npu_shift, npu_scale = (
        t.to(device)[:, ::col_stride] for t in (x, other, gate, shift, scale)
    )
    npu_indices = indices.to(device)
    result_rms = rms_norm_indexed_scale_shift(npu_x, weight.to(device), npu_shift, npu_scale, npu_indices, eps)
    result_gate = indexed_gate(npu_x, npu_gate, npu_other, npu_indices)
    if expected_sdk_calls is None:
        assert triton_launch.call_count == 2
    else:
        assert (sdk_rms.call_count, sdk_gate.call_count) == expected_sdk_calls

    x, other, gate, shift, scale = (t[:, ::col_stride].float() for t in (x, other, gate, shift, scale))
    norm = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * weight.float()
    torch.testing.assert_close(result_rms.cpu(), (norm * (1 + scale[indices]) + shift[indices]).to(dtype))
    torch.testing.assert_close(result_gate.cpu(), (x + gate[indices] * other).to(dtype))


@hardware_test(res={"cuda": "L4", "xpu": "B60"}, num_cards=1)
@pytest.mark.skipif(not NVIDIA_CUDA_OR_XPU, reason="NVIDIA CUDA or XPU required")
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
