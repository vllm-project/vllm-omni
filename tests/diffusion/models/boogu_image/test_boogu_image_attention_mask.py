# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.diffusion.models.boogu_image.boogu_image_transformer import _make_attention_mask

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize("seq_lengths", [[16], [16, 16]])
def test_attention_mask_is_skipped_for_dense_batch(seq_lengths):
    hidden_states = torch.empty(len(seq_lengths), 16, 64)

    assert _make_attention_mask(hidden_states, seq_lengths) is None


def test_attention_mask_preserves_variable_length_padding():
    hidden_states = torch.empty(2, 16, 64)

    attention_mask = _make_attention_mask(hidden_states, [16, 12])

    expected = torch.tensor(
        [
            [True] * 16,
            [True] * 12 + [False] * 4,
        ]
    )
    torch.testing.assert_close(attention_mask, expected)
