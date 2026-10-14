# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.diffusion.models.qwen_image_21.qkv_norm_rope import concat_prefix_bnsd

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize("batch,prefix,tokens", [(1, 0, 9), (1, 17, 32), (2, 13, 7)])
@pytest.mark.parametrize("head_major", [False, True])
def test_prefix_concat_preserves_values_and_returns_head_major(batch, prefix, tokens, head_major):
    def tensor(length):
        if head_major:
            return torch.randn(batch, 4, length, 8).transpose(1, 2)
        return torch.randn(batch, length, 4, 8)

    cached = tensor(prefix)
    current = tensor(tokens)
    cached_before, current_before = cached.clone(), current.clone()
    result = concat_prefix_bnsd(cached, current)
    torch.testing.assert_close(result, torch.cat((cached, current), dim=1), rtol=0, atol=0)
    torch.testing.assert_close(cached, cached_before, rtol=0, atol=0)
    torch.testing.assert_close(current, current_before, rtol=0, atol=0)
    assert result.transpose(1, 2).is_contiguous()


def test_prefix_concat_does_not_alias_live_requests():
    cached = torch.randn(1, 7, 4, 8)
    first = concat_prefix_bnsd(cached, torch.zeros(1, 9, 4, 8))
    second = concat_prefix_bnsd(cached, torch.ones(1, 9, 4, 8))
    assert first.untyped_storage().data_ptr() != second.untyped_storage().data_ptr()
    assert torch.count_nonzero(first[:, 7:]) == 0
    assert torch.all(second[:, 7:] == 1)
