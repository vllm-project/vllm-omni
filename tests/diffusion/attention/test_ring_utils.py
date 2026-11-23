# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_omni.diffusion.attention.backends.ring.ring_utils import update_out_and_lse

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.mark.parametrize("layout", ["bhs", "bsh", "bhs_padded", "bhs1"])
def test_ring_lse_known_layouts_are_normalized(layout):
    block_out = torch.randn(2, 5, 3, 4)
    canonical = torch.randn(2, 5, 3)
    if layout == "bhs":
        block_lse = canonical.transpose(1, 2)
        lse_layout = "bhs"
    elif layout == "bsh":
        block_lse = canonical
        lse_layout = "bsh"
    elif layout == "bhs_padded":
        block_lse = torch.cat([canonical.transpose(1, 2), torch.randn(2, 3, 2)], dim=2)
        lse_layout = "bhs"
    else:
        block_lse = canonical.transpose(1, 2).unsqueeze(-1)
        lse_layout = "bhs"

    out, lse = update_out_and_lse(None, None, block_out, block_lse, lse_layout=lse_layout)

    assert out.shape == block_out.shape
    assert out.dtype == torch.float32
    assert lse.shape == (2, 5, 3, 1)
    assert torch.equal(lse.squeeze(-1), canonical)


def test_ring_lse_seq_len_equals_num_heads_keeps_asymmetric_values():
    batch, seq_len, num_heads, head_dim = 2, 4, 4, 8
    block_out = torch.randn(batch, seq_len, num_heads, head_dim)
    canonical = torch.arange(batch * seq_len * num_heads, dtype=torch.float32).reshape(batch, seq_len, num_heads)
    bhs = canonical.transpose(1, 2)

    assert bhs.shape == canonical.shape
    assert not torch.equal(bhs, canonical)

    _, lse_from_bhs = update_out_and_lse(None, None, block_out, bhs, lse_layout="bhs")
    _, lse_from_bsh = update_out_and_lse(None, None, block_out, canonical, lse_layout="bsh")

    assert torch.equal(lse_from_bhs.squeeze(-1), canonical)
    assert torch.equal(lse_from_bsh.squeeze(-1), canonical)

    _, lse_if_bsh_read_as_bhs = update_out_and_lse(None, None, block_out, canonical, lse_layout="bhs")
    assert not torch.equal(lse_if_bsh_read_as_bhs.squeeze(-1), canonical)
    assert torch.equal(lse_if_bsh_read_as_bhs.squeeze(-1), bhs)


@pytest.mark.parametrize(
    "invalid_lse",
    [
        torch.randn(2, 5),
        torch.randn(2, 4, 4),
        torch.randn(2, 3, 5, 2),
    ],
)
def test_ring_lse_unknown_layout_is_rejected(invalid_lse):
    block_out = torch.randn(2, 5, 3, 4)

    with pytest.raises(ValueError, match="Ring LSE"):
        update_out_and_lse(None, None, block_out, invalid_lse, lse_layout="bhs")


def test_ring_accumulation_rejects_mismatched_block_output():
    out = torch.randn(2, 5, 3, 4)
    lse = torch.randn(2, 5, 3, 1)
    block_out = torch.randn(2, 6, 3, 4)
    block_lse = torch.randn(2, 3, 6)

    with pytest.raises(ValueError, match="block output shape"):
        update_out_and_lse(out, lse, block_out, block_lse, lse_layout="bhs")


def test_ring_accumulation_matches_logsumexp_weighting():
    first_out = torch.zeros(1, 2, 1, 1)
    second_out = torch.ones(1, 2, 1, 1)
    first_lse = torch.zeros(1, 1, 2)
    second_lse = torch.zeros(1, 1, 2)

    out, lse = update_out_and_lse(None, None, first_out, first_lse, lse_layout="bhs")
    out, lse = update_out_and_lse(out, lse, second_out, second_lse, lse_layout="bhs")

    assert torch.allclose(out, torch.full_like(out, 0.5))
    assert torch.allclose(lse, torch.full_like(lse, torch.log(torch.tensor(2.0))))
