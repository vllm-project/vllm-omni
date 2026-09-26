# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for Pi-family shared attention primitives."""

import pytest
import torch

from vllm_omni.diffusion.models.pi.common import attention
from vllm_omni.diffusion.models.pi.pi0 import modeling_pi0
from vllm_omni.diffusion.models.pi.pi05 import modeling_pi05

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_variant_mask_exports_match_common_implementation():
    pad_masks = torch.tensor([[True, True, False, True]])
    block_masks = torch.tensor([[False, True, False, True]])

    expected_2d = attention.make_att_2d_masks(pad_masks, block_masks)
    expected_4d = attention.prepare_attention_masks_4d(expected_2d)

    for variant in (modeling_pi0, modeling_pi05):
        actual_2d = variant.make_att_2d_masks(pad_masks, block_masks)
        actual_4d = variant.prepare_attention_masks_4d(actual_2d)
        assert torch.equal(actual_2d, expected_2d)
        assert torch.equal(actual_4d, expected_4d)
        assert variant.OPENPI_ATTENTION_MASK_VALUE == attention.OPENPI_ATTENTION_MASK_VALUE


def test_repeat_kv_preserves_head_group_order():
    hidden_states = torch.tensor([[[[1.0], [2.0]], [[3.0], [4.0]]]])

    repeated = attention.repeat_kv(hidden_states, n_rep=2)

    assert repeated.shape == (1, 4, 2, 1)
    assert torch.equal(repeated[:, 0], hidden_states[:, 0])
    assert torch.equal(repeated[:, 1], hidden_states[:, 0])
    assert torch.equal(repeated[:, 2], hidden_states[:, 1])
    assert torch.equal(repeated[:, 3], hidden_states[:, 1])


def test_eager_attention_slices_reusable_mask_to_key_length():
    query = torch.tensor([[[[1.0], [1.0]]]])
    key = torch.tensor([[[[1.0], [0.0]]]])
    value = torch.tensor([[[[2.0], [6.0]]]])
    reusable_mask = torch.tensor([[[[0.0, attention.OPENPI_ATTENTION_MASK_VALUE, 0.0]]]])

    actual = attention.eager_attention(query, key, value, reusable_mask, num_kv_groups=1, scaling=1.0)

    assert torch.equal(actual, torch.full_like(actual, 2.0))


@pytest.mark.parametrize("name", ["pad_masks", "att_masks"])
def test_make_att_2d_masks_rejects_non_matrix_inputs(name):
    masks = {
        "pad_masks": torch.ones(1, 2, dtype=torch.bool),
        "att_masks": torch.zeros(1, 2, dtype=torch.bool),
    }
    masks[name] = masks[name][0]

    with pytest.raises(ValueError, match=rf"{name} must be 2-D"):
        attention.make_att_2d_masks(**masks)
