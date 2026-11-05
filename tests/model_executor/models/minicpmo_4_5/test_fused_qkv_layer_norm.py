# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# ruff: noqa: N803

"""``qkv_head_layer_norm`` slot-pool indexing matches the direct ``(2, N, H, T, D)`` layout."""

import pytest
import torch

from tests.helpers.mark import hardware_marks
from vllm_omni.model_executor.models.minicpmo_4_5.ops import qkv_head_layer_norm

pytestmark = [
    pytest.mark.core_model,
    # The bug lives on the CPU fallback, so the CPU job must collect this file too.
    pytest.mark.cpu,
    *hardware_marks(res={"cuda": "L4", "rocm": "MI325"}, num_cards=1),
]

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _inputs(device: str, *, batch: int, frames: int, heads: int, head_dim: int) -> tuple:
    torch.manual_seed(0)
    inner = heads * head_dim
    qkv = torch.randn(batch, frames, 3 * inner, device=device)

    def _norm() -> torch.nn.LayerNorm:
        norm = torch.nn.LayerNorm(head_dim, eps=1e-6, device=device)
        with torch.no_grad():
            norm.weight.uniform_(0.5, 1.5)
            norm.bias.uniform_(-0.5, 0.5)
        return norm

    return qkv, _norm(), _norm()


def _split(qkv: torch.Tensor, heads: int, head_dim: int, k_norm: torch.nn.LayerNorm) -> tuple:
    """Expected K (LayerNormed) and V (raw) written into ``kv``."""
    batch, frames, _ = qkv.shape
    packed = qkv.view(batch, frames, 3, heads, head_dim)
    return k_norm(packed[:, :, 1].transpose(1, 2)), packed[:, :, 2].transpose(1, 2)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("affine", [True, False])
def test_slot_rows_match_direct_kv(device: str, affine: bool) -> None:
    """``rows`` only remaps the batch row; per-frame K/V values stay identical."""
    batch, frames, heads, head_dim, pool = 3, 5, 3, 4, 6
    qkv, q_norm, k_norm = _inputs(device, batch=batch, frames=frames, heads=heads, head_dim=head_dim)
    if not affine:
        q_norm = torch.nn.LayerNorm(head_dim, eps=1e-6, elementwise_affine=False, device=device)
        k_norm = torch.nn.LayerNorm(head_dim, eps=1e-6, elementwise_affine=False, device=device)
    rows = torch.tensor([4, 1, 5], device=device)

    direct = torch.zeros(2, batch, heads, frames, head_dim, device=device)
    pooled = torch.zeros(2, pool, heads, frames, head_dim, device=device)
    q_direct = qkv_head_layer_norm(qkv, direct, num_heads=heads, head_dim=head_dim, q_norm=q_norm, k_norm=k_norm)
    q_pooled = qkv_head_layer_norm(
        qkv, pooled, num_heads=heads, head_dim=head_dim, q_norm=q_norm, k_norm=k_norm, rows=rows
    )

    torch.testing.assert_close(q_pooled, q_direct)
    expected_k, expected_v = _split(qkv, heads, head_dim, k_norm)
    torch.testing.assert_close(direct[0], expected_k)
    torch.testing.assert_close(direct[1], expected_v)
    for n, pool_row in enumerate(rows.tolist()):
        torch.testing.assert_close(pooled[0, pool_row], expected_k[n])
        torch.testing.assert_close(pooled[1, pool_row], expected_v[n])
    written_rows = set(rows.tolist())
    for row in range(pool):
        if row not in written_rows:
            assert torch.count_nonzero(pooled[:, row]) == 0


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("affine", [True, False])
def test_slot_positions_scatter_kv(device: str, affine: bool) -> None:
    """``positions`` scatter each frame onto ``kv`` dim 3; negative entries are skipped."""
    batch, frames, heads, head_dim, pool, pool_frames = 2, 3, 2, 4, 4, 5
    qkv, q_norm, k_norm = _inputs(device, batch=batch, frames=frames, heads=heads, head_dim=head_dim)
    if not affine:
        q_norm = torch.nn.LayerNorm(head_dim, eps=1e-6, elementwise_affine=False, device=device)
        k_norm = torch.nn.LayerNorm(head_dim, eps=1e-6, elementwise_affine=False, device=device)
    # Row 0 writes frames 0/1/2 onto slots 2/-1/0; row 1 onto slots 4/1/-1 of another pool row.
    rows = torch.tensor([3, 0], device=device)
    positions = torch.tensor([[2, -1, 0], [4, 1, -1]], device=device)

    pooled = torch.zeros(2, pool, heads, pool_frames, head_dim, device=device)
    q_out = qkv_head_layer_norm(
        qkv,
        pooled,
        num_heads=heads,
        head_dim=head_dim,
        q_norm=q_norm,
        k_norm=k_norm,
        rows=rows,
        positions=positions,
    )
    assert tuple(q_out.shape) == (batch, heads, frames, head_dim)

    expected_k, expected_v = _split(qkv, heads, head_dim, k_norm)
    seen = {row: set() for row in rows.tolist()}
    for n, pool_row in enumerate(rows.tolist()):
        for t in range(frames):
            slot = int(positions[n, t])
            if slot >= 0:
                torch.testing.assert_close(pooled[0, pool_row, :, slot], expected_k[n, :, t])
                torch.testing.assert_close(pooled[1, pool_row, :, slot], expected_v[n, :, t])
                seen[pool_row].add(slot)
    # Frames mapped to -1 (and slots never targeted) stay untouched.
    for n, pool_row in enumerate(rows.tolist()):
        for slot in range(pool_frames):
            if slot not in seen[pool_row]:
                assert torch.count_nonzero(pooled[:, pool_row, :, slot]) == 0
