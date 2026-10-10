# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compiled CUDA sparse attention contracts for the LiDAR VAE."""

from __future__ import annotations

import pytest
import torch

from tests.diffusion.models.cosmos3.test_lidar_neighborhood_attention import (
    OPERATOR_CASES,
    check_operator_matches_fp64_neighborhoods,
)
from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.models.cosmos3.lidar_encoder import neighborhood_attention as attention
from vllm_omni.platforms import current_omni_platform

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.diffusion,
    pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="requires CUDA"),
]


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("circular", [False, True])
@pytest.mark.parametrize("hw,kernel,dilation", OPERATOR_CASES)
def test_compiled_operator_matches_fp64_neighborhoods(hw, kernel, dilation, circular):
    check_operator_matches_fp64_neighborhoods(hw, kernel, dilation, circular, "cuda")


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_cuda_compile_failure_propagates_without_fallback(monkeypatch):
    def fail(*args):
        raise RuntimeError("injected compilation failure")

    monkeypatch.setattr(attention, "_get_compiled_runner", fail)
    q = torch.ones(1, 3, 7, 1, 8, device="cuda")
    with pytest.raises(RuntimeError, match="compiled sparse attention is required") as error:
        attention.neighborhood_attention_2d(q, q, q, kernel_size=3, dilation=1, scale=1.0)
    assert "injected compilation failure" in str(error.value.__cause__)


# No PR-time lane collects cards_2 tests under tests/diffusion/models; run manually on two GPUs.
@hardware_test(res={"cuda": "L4"}, num_cards=2)
def test_masks_follow_device_after_offload_reload():
    masks = []
    for index in [0, 1, 0]:
        q = torch.ones(1, 3, 7, 1, 8).to(torch.device("cuda", index))
        attention.neighborhood_attention_2d(q, q, q, kernel_size=3, dilation=1, scale=1.0)
        mask = attention._get_block_mask(q.device, 3, 7, (3, 3), (1, 1))
        assert mask.kv_indices.device == q.device
        masks.append(mask)
    assert masks[0] is masks[2] and masks[0] is not masks[1]
