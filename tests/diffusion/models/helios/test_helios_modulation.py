# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
import torch.nn.functional as F
from vllm.triton_utils import HAS_TRITON

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.layers.indexed_modulation import (
    indexed_gate,
    indexed_scale_shift_,
)
from vllm_omni.diffusion.models.helios.helios_transformer import (
    _make_compact_modulation,
)
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]

_HELIOS_HIDDEN_SIZE = 5120


def _make_modulation_case(
    batch_size: int,
    sequence_length: int,
    history_context_length: int,
    hidden_size: int,
    has_history_condition: bool,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the production compact rows and indices plus a per-token reference."""
    current = torch.randn(batch_size, 6, hidden_size, dtype=dtype)
    history = None
    if has_history_condition:
        history = torch.randn(1, 6, hidden_size, dtype=dtype).expand(batch_size, -1, -1)
    compact, indices = _make_compact_modulation(current, history, sequence_length, history_context_length)

    if history is None:
        expanded = current.unsqueeze(1).expand(-1, sequence_length, -1, -1)
    else:
        expanded = torch.cat(
            (
                history.unsqueeze(1).expand(-1, history_context_length, -1, -1),
                current.unsqueeze(1).expand(-1, sequence_length - history_context_length, -1, -1),
            ),
            dim=1,
        )
    return compact, indices, expanded


def _apply_block_like_modulation(
    hidden_states: torch.Tensor,
    branch: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    gate: torch.Tensor,
    indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    normalized = F.layer_norm(hidden_states.float(), (_HELIOS_HIDDEN_SIZE,), eps=1e-6)
    normalized = normalized.flatten(0, 1).contiguous()
    modulated = indexed_scale_shift_(normalized, shift, scale, indices).view_as(hidden_states)
    modulated = modulated.type_as(hidden_states)

    residual = indexed_gate(
        hidden_states.float().flatten(0, 1),
        gate,
        branch.flatten(0, 1),
        indices,
    ).view_as(hidden_states)
    return modulated, residual.type_as(hidden_states)


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("history_context_length", "has_history_condition"),
    [(0, False), (3, True), (3, False)],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_compact_modulation_matches_sequence_expansion(
    history_context_length: int,
    has_history_condition: bool,
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(7)
    batch_size, sequence_length, hidden_size = 2, 7, 5120
    compact, indices, expanded = _make_modulation_case(
        batch_size,
        sequence_length,
        history_context_length,
        hidden_size,
        has_history_condition,
        dtype,
    )

    selected = compact.index_select(0, indices).view_as(expanded)
    assert torch.equal(selected, expanded)

    shift, scale, gate, *_ = compact.float().chunk(6, dim=1)
    shift, scale, gate = shift.squeeze(1), scale.squeeze(1), gate.squeeze(1)
    x = torch.randn(batch_size, sequence_length, hidden_size)
    branch = torch.randn(batch_size, sequence_length, hidden_size, dtype=dtype)
    expanded = expanded.float()

    expected_affine = x * (1.0 + expanded[:, :, 1]) + expanded[:, :, 0]
    actual_affine = indexed_scale_shift_(x.flatten(0, 1).clone(), shift, scale, indices).view_as(x)
    assert torch.equal(actual_affine, expected_affine)

    expected_gate = x + expanded[:, :, 2] * branch
    actual_gate = indexed_gate(x.flatten(0, 1), gate, branch.flatten(0, 1), indices).view_as(x)
    assert torch.equal(actual_gate, expected_gate)


@pytest.mark.skipif(not current_omni_platform.is_npu(), reason="requires Ascend NPU")
@pytest.mark.parametrize(
    "device",
    [pytest.param("npu", marks=hardware_marks(res={"npu": "A3"}, num_cards=1))],
)
@pytest.mark.parametrize("has_history_condition", [False, True])
def test_helios_width_modulation_kernels_match_reference(device: str, has_history_condition: bool) -> None:
    # The shared operator suite separately covers Ascend's >65,535-row
    # chunked-launch path; keep this test focused on Helios's 5,120 width.
    torch.manual_seed(11)
    batch_size, sequence_length, history_context_length, hidden_size = 2, 33, 17, 5120
    compact, indices, _ = _make_modulation_case(
        batch_size,
        sequence_length,
        history_context_length,
        hidden_size,
        has_history_condition,
        torch.bfloat16,
    )

    shift, scale, gate, *_ = compact.float().chunk(6, dim=1)
    shift, scale, gate = shift.squeeze(1), scale.squeeze(1), gate.squeeze(1)
    x = torch.randn(batch_size * sequence_length, hidden_size)
    bf16_branch = torch.randn(batch_size * sequence_length, hidden_size, dtype=torch.bfloat16)
    fp32_branch = torch.randn_like(x)

    npu_x = x.to(device)
    npu_shift = shift.to(device)
    npu_scale = scale.to(device)
    npu_gate = gate.to(device)
    npu_indices = indices.to(device)
    npu_bf16_branch = bf16_branch.to(device)
    npu_fp32_branch = fp32_branch.to(device)

    indexed_shift = npu_shift.index_select(0, npu_indices)
    indexed_scale = npu_scale.index_select(0, npu_indices)
    indexed_gate_values = npu_gate.index_select(0, npu_indices)
    expected_affine = npu_x * (1.0 + indexed_scale) + indexed_shift
    expected_bf16_gate = npu_x + indexed_gate_values * npu_bf16_branch
    expected_fp32_gate = npu_x + indexed_gate_values * npu_fp32_branch

    actual_affine = indexed_scale_shift_(npu_x.clone(), npu_shift, npu_scale, npu_indices)
    actual_bf16_gate = indexed_gate(npu_x, npu_gate, npu_bf16_branch, npu_indices)
    actual_fp32_gate = indexed_gate(npu_x, npu_gate, npu_fp32_branch, npu_indices)

    assert torch.equal(actual_affine, expected_affine)
    assert torch.equal(actual_bf16_gate, expected_bf16_gate)
    assert torch.equal(actual_fp32_gate, expected_fp32_gate)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton required")
@pytest.mark.parametrize("has_history_condition", [False, True])
def test_helios_width_modulation_kernels_cuda(has_history_condition: bool) -> None:
    torch.manual_seed(13)
    batch_size, hidden_size = 2, _HELIOS_HIDDEN_SIZE
    compiled_block = torch.compile(_apply_block_like_modulation, dynamic=True, fullgraph=True)

    for sequence_length in (17, 33):
        history_context_length = sequence_length // 2
        compact, indices, _ = _make_modulation_case(
            batch_size,
            sequence_length,
            history_context_length,
            hidden_size,
            has_history_condition,
            torch.bfloat16,
        )

        # Production unbinds the six rows after moving the complete modulation
        # tensor to the accelerator, so each row has stride 6 * hidden_size.
        modulation = compact.to(device="cuda", dtype=torch.float32)
        shift, scale, gate, *_ = modulation.unbind(1)
        indices = indices.to("cuda")
        assert shift.stride() == scale.stride() == gate.stride() == (6 * hidden_size, 1)

        rows = batch_size * sequence_length
        x = torch.randn(rows, hidden_size, device="cuda")
        bf16_branch = torch.randn(rows, hidden_size, device="cuda", dtype=torch.bfloat16)
        fp32_branch = torch.randn_like(x)

        indexed_shift = shift.index_select(0, indices)
        indexed_scale = scale.index_select(0, indices)
        indexed_gate_values = gate.index_select(0, indices)
        expected_affine = x * (1.0 + indexed_scale) + indexed_shift
        expected_bf16_gate = x + indexed_gate_values * bf16_branch
        expected_fp32_gate = x + indexed_gate_values * fp32_branch

        actual_affine = indexed_scale_shift_(x.clone(), shift, scale, indices)
        actual_bf16_gate = indexed_gate(x, gate, bf16_branch, indices)
        actual_fp32_gate = indexed_gate(x, gate, fp32_branch, indices)

        for actual, expected in (
            (actual_affine, expected_affine),
            (actual_bf16_gate, expected_bf16_gate),
            (actual_fp32_gate, expected_fp32_gate),
        ):
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

        hidden_states = x.to(torch.bfloat16).view(batch_size, sequence_length, hidden_size)
        branch = bf16_branch.view_as(hidden_states)
        eager_outputs = _apply_block_like_modulation(hidden_states, branch, shift, scale, gate, indices)
        compiled_outputs = compiled_block(hidden_states, branch, shift, scale, gate, indices)
        for actual, expected in zip(compiled_outputs, eager_outputs, strict=True):
            # Both paths return BF16. Inductor can change FP32 reduction and FMA
            # rounding before that cast, so use assert_close's BF16 tolerance.
            torch.testing.assert_close(actual, expected)
