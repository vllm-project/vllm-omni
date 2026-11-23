# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU references for the Drama 3 tower and sampling kernels used on NPU."""

import pytest
import torch

from vllm_omni.model_executor.models.fish_omni.kernels import (
    draw_top_k_top_p,
    gemma_rms_norm,
    gemma_rms_norm_table,
    sigmoid_gate,
    tower_mix_norm,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

EPS = 1e-6


def _gemma(values: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """vLLM's zero-centered Gemma RMSNorm, the reference these kernels follow."""
    return values * torch.rsqrt(values.pow(2).mean(-1, keepdim=True) + EPS) * (weight + 1)


def test_gemma_rms_norm_matches_zero_centered_reference():
    x = torch.tensor([[1.0, 3.0, 0.0, 2.0], [0.5, -0.5, 0.25, 0.0]])
    weight = torch.tensor([0.0, 1.0, -0.5, 0.25])
    residual = torch.tensor([[0.5, 0.0, 1.0, -1.0], [0.0, 0.25, 0.0, 0.5]])
    out, summed = gemma_rms_norm(x, weight, EPS, residual)
    total = x + residual
    expected = _gemma(total, weight)
    torch.testing.assert_close(summed, total)
    torch.testing.assert_close(out, expected)


def test_gemma_rms_norm_without_residual_returns_only_the_norm():
    x = torch.tensor([[1.0, 3.0, 0.0, 2.0]])
    weight = torch.tensor([0.0, 1.0, -0.5, 0.25])
    torch.testing.assert_close(gemma_rms_norm(x, weight, EPS), _gemma(x, weight))


@pytest.mark.parametrize("with_extra", [False, True])
def test_tower_mix_norm_concatenates_code_and_hidden(with_extra):
    hidden = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    code = torch.tensor([1, 0])
    mask = torch.tensor([1.0, 0.0])
    table = torch.tensor([[3.0, 1.0], [0.0, 4.0]])
    extra = torch.tensor([[0.5, -1.5], [2.0, 0.25]]) if with_extra else None
    enorm = torch.tensor([0.5, -0.25])
    hnorm = torch.tensor([0.0, 1.0])

    out = tower_mix_norm(hidden, code, mask, table, extra, enorm, hnorm, EPS)

    embedded = table[code] * mask.unsqueeze(-1)
    if extra is not None:
        embedded = embedded + extra
    torch.testing.assert_close(out, torch.cat([_gemma(embedded, enorm), _gemma(hidden, hnorm)], dim=-1))


def test_gemma_rms_norm_table_adds_the_code_embedding_to_the_residual():
    x = torch.tensor([[1.0, 3.0, 0.0, 2.0], [0.5, -0.5, 0.25, 0.0]])
    code = torch.tensor([1, 0])
    mask = torch.tensor([1.0, 0.0])
    table = torch.tensor([[3.0, 1.0, 0.0, 2.0], [0.0, 4.0, -1.0, 0.5]])
    weight = torch.tensor([0.0, 1.0, -0.5, 0.25])

    out, summed = gemma_rms_norm_table(x, code, mask, table, weight, EPS)

    total = x + table[code] * mask.unsqueeze(-1)
    torch.testing.assert_close(summed, total)
    torch.testing.assert_close(out, _gemma(total, weight))


def test_sigmoid_gate_matches_fp32_product():
    x = torch.tensor([0.0, 1.0, -2.0])
    gate = torch.tensor([0.0, 2.0, -1.0])
    torch.testing.assert_close(sigmoid_gate(x, gate), x * torch.sigmoid(gate))


def test_draw_keeps_rank_zero_when_temperature_collapses():
    logits = torch.tensor([[0.0, 5.0, 1.0, -3.0], [2.0, 2.0, 9.0, 0.0]])
    rows = logits.shape[0]
    drawn = draw_top_k_top_p(
        logits,
        temperature=torch.zeros(rows),
        top_k=torch.ones(rows, dtype=torch.int64),
        top_p=torch.ones(rows),
        uniform=torch.rand(rows, 2),
        max_top_k=2,
    )
    assert drawn.tolist() == [1, 2]


def test_draw_maps_column_indices_through_id_map():
    logits = torch.tensor([[0.0, 5.0, 1.0, -3.0], [2.0, 2.0, 9.0, 0.0]])
    rows = logits.shape[0]
    id_map = torch.tensor([70, 71, 72, 73], dtype=torch.int64)
    drawn = draw_top_k_top_p(
        logits,
        temperature=torch.zeros(rows),
        top_k=torch.ones(rows, dtype=torch.int64),
        top_p=torch.ones(rows),
        uniform=torch.rand(rows, 2),
        max_top_k=2,
        id_map=id_map,
    )
    assert drawn.tolist() == [71, 72]


def test_draw_reuses_a_shared_topk_result():
    logits = torch.tensor([[0.0, 5.0, 1.0, -3.0], [2.0, 2.0, 9.0, 0.0]])
    rows = logits.shape[0]
    shared = torch.topk(logits, 2, dim=-1)
    drawn = draw_top_k_top_p(
        logits,
        temperature=torch.zeros(rows),
        top_k=torch.ones(rows, dtype=torch.int64),
        top_p=torch.ones(rows),
        uniform=torch.rand(rows, 2),
        max_top_k=2,
        top=shared,
    )
    assert drawn.tolist() == [1, 2]


def test_eager_qk_rope_splits_gate_and_rotates_the_first_half():
    from vllm_omni.model_executor.models.fish_omni.kernels import eager_qk_rmsnorm_rope_gate

    head_dim, rotary_dim, heads = 4, 4, 1
    q_gate = torch.tensor([[1.0, 0.0, 0.0, 0.0, 3.0, 4.0, 5.0, 6.0]])
    k = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    weight = torch.zeros(head_dim)
    cos_sin = torch.tensor([[1.0, 0.0, 0.0, 1.0]])
    positions = torch.zeros(1, dtype=torch.long)
    q, k_out, gate = eager_qk_rmsnorm_rope_gate(
        q_gate,
        k,
        weight,
        weight,
        cos_sin,
        positions,
        EPS,
        heads,
        heads,
        head_dim,
        rotary_dim,
        norm_beta=1.0,
    )
    torch.testing.assert_close(gate, torch.tensor([[3.0, 4.0, 5.0, 6.0]]))
    assert q.shape == (1, head_dim)
    assert k_out.shape == (1, head_dim)
    # Identity rotation (cos=1, sin=0) leaves the normalized q direction.
    assert q[0, 0] > 0 and torch.count_nonzero(q) >= 1


def test_draw_restricts_to_top_p_mass():
    # Column 0 holds almost all the mass, so top_p 0.5 keeps rank 0 only and
    # every draw returns it whatever the noise says.
    logits = torch.tensor([[20.0, 1.0, 0.0, -1.0]])
    for _ in range(8):
        drawn = draw_top_k_top_p(
            logits,
            temperature=torch.ones(1),
            top_k=torch.full((1,), 4, dtype=torch.int64),
            top_p=torch.full((1,), 0.5),
            uniform=torch.rand(1, 4),
            max_top_k=4,
        )
        assert drawn.tolist() == [0]
