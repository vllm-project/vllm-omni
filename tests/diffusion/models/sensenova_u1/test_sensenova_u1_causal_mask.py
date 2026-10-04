# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for the SenseNova-U1 block causal mask construction.

``create_block_causal_mask`` is the shared mask producer for the model's
attention. On NPU it must return a bool mask with True=attend — the dtype the
platform-default FLASH_ATTN (mindiesd) backend consumes; an additive float
mask passed through unconverted silently corrupts the output. Non-NPU
platforms keep the historical additive 0.0/-inf float mask. The inline mask
built in ``SenseNovaU1Model.forward`` calls ``create_prefix_causal_mask`` on
NPU, which mirrors the same contract.
"""

import pytest
import torch

from vllm_omni.diffusion.models.sensenova_u1.sensenova_u1_transformer import (
    create_block_causal_mask,
    create_prefix_causal_mask,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _attend_set(mask: torch.Tensor) -> torch.Tensor:
    """Normalize either mask convention to a bool True=attend tensor."""
    return mask == 0.0 if mask.is_floating_point() else mask


def test_block_causal_mask_attend_set_is_lower_triangular():
    """With distinct time indexes the mask reduces to plain causal: a token
    attends to itself and every earlier position, nothing later."""
    index = torch.arange(8)
    attend = _attend_set(create_block_causal_mask(index))
    assert attend.shape == (1, 1, 8, 8)
    assert torch.equal(attend[0, 0], torch.ones(8, 8, dtype=torch.bool).tril())


def test_block_causal_mask_groups_by_time_index():
    """Tokens sharing a time index attend bidirectionally within their group;
    across groups only strictly earlier positions are attendable."""
    index = torch.tensor([0, 0, 1, 1])
    attend = _attend_set(create_block_causal_mask(index))[0, 0]
    assert attend[0, 1].item()  # same group: bidirectional
    assert attend[2, 0].item()  # other group, strictly earlier: prefix
    assert not attend[1, 2].item()  # other group, later: blocked
    assert not attend[0, 3].item()


def test_block_causal_mask_platform_dtype_contract():
    """NPU: bool True=attend (the mindiesd contract). Other platforms keep the
    additive float mask the SDPA fallback consumes."""
    from vllm_omni.platforms import current_omni_platform

    mask = create_block_causal_mask(torch.arange(4))
    if current_omni_platform.is_npu():
        assert mask.dtype == torch.bool
        assert mask[0, 0, 0, 0].item()  # True=attend, not True=masked
    else:
        assert mask.dtype == torch.float32
        assert mask[0, 0, 0, 0].item() == 0.0  # additive 0.0/-inf form
        assert mask.min().item() == float("-inf")


def test_prefix_causal_mask_kv_layout():
    """The KV-cache path mask is (seq_len, total_len): prefix keys stay
    attendable, current keys are sequence-causal, and the mask is bool
    True=attend on every platform (this helper only serves the NPU branch)."""
    seq_len, past_len = 4, 6
    total_len = past_len + seq_len

    mask = create_prefix_causal_mask(seq_len, total_len, past_len, torch.device("cpu"))

    attend = mask[0, 0]
    assert attend.shape == (seq_len, total_len)
    assert attend.dtype == torch.bool
    assert attend[0, 0].item()  # True=attend, not True=masked
    assert torch.equal(attend[:, :past_len], torch.ones(seq_len, past_len, dtype=torch.bool))
    assert torch.equal(attend[:, past_len:], torch.ones(seq_len, seq_len, dtype=torch.bool).tril())
