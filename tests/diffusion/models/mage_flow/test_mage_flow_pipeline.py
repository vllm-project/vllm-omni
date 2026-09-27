# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Request-level padded batching, Mage-Flow's distinguishing capability."""

import pytest
import torch

from vllm_omni.diffusion.models.mage_flow.pipeline_mage_flow import MageFlowPipeline

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_pad_token_sequences_returns_padding_mask_and_lengths() -> None:
    short = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)
    long = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)

    padded, mask, lengths = MageFlowPipeline._pad_token_sequences([short, long])

    assert padded.shape == (2, 4, 3)
    assert lengths == [2, 4]
    torch.testing.assert_close(padded[0, :2], short[0])
    # Filler must be exactly zero: the transformer multiplies by this mask
    # rather than masking attention, so non-zero filler would leak into output.
    assert not padded[0, 2:].count_nonzero()
    assert mask.tolist() == [
        [True, True, False, False],
        [True, True, True, True],
    ]
