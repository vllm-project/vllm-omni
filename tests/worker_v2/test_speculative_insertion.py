# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real-kernel coverage for the vLLM 0.33 resampled-token insertion backport."""

import pytest
import torch
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.spec_decode import rejection_sampler_utils

from tests.helpers.mark import hardware_marks
from vllm_omni.worker_v2.omni_sampler import patch_resampled_token_insertion

pytestmark = [pytest.mark.core_model, *hardware_marks(res={"cuda": "L4"}, num_cards=1)]


@pytest.mark.parametrize("temperature", [0.0, 0.8])
def test_resampled_tokens_and_counts_with_one_warp(monkeypatch, temperature):
    if not current_platform.is_cuda():
        pytest.skip("CUDA required")

    kernel = rejection_sampler_utils._insert_resampled_kernel
    original_run = kernel.run
    launched_warps = []

    def record_launch(*args, **kwargs):
        compiled = original_run(*args, **kwargs)
        launched_warps.append(compiled.metadata.num_warps)
        return compiled

    monkeypatch.setattr(kernel, "run", record_launch)
    patch_resampled_token_insertion()
    device = current_platform.device_type
    # Unequal request lengths and nonidentity state mapping cover all-accepted
    # bonus sampling, a rejected draft, and a request with no drafts. The vocab
    # matches MiniCPM-o and requires a padded reduction over 149 blocks.
    vocab_size = 151748
    logits = torch.full((13, vocab_size), -torch.inf, device=device)
    target_ids = torch.tensor([11, 12, 13, 14, 15, 16, 17, 91, 21, 22, 92, 24, 93], device=device)
    logits[torch.arange(13, device=device), target_ids] = 0.0
    draft_ids = torch.tensor([10, 11, 12, 13, 14, 15, 16, 17, 20, 21, 22, 23, 30], device=device)
    cu_num_logits = torch.tensor([0, 8, 12, 13], dtype=torch.int32, device=device)
    idx_mapping = torch.tensor([3, 1, 4], dtype=torch.int64, device=device)
    expanded_idx_mapping = torch.tensor([3] * 8 + [1] * 4 + [4], dtype=torch.int64, device=device)
    local_pos = torch.tensor(list(range(8)) + list(range(4)) + [0], dtype=torch.int32, device=device)
    positions = local_pos.to(torch.int64) + 267
    temperatures = torch.full((5,), temperature, device=device)
    seeds = torch.full((5,), 42, dtype=torch.int64, device=device)

    def sample():
        return rejection_sampler_utils.rejection_sample(
            logits,
            None,
            draft_ids,
            cu_num_logits,
            positions,
            idx_mapping,
            expanded_idx_mapping,
            local_pos,
            temperatures,
            seeds,
            7,
        )

    def check_output(sampled, counts):
        assert counts.tolist() == [8, 3, 1]
        assert sampled[0, :8].tolist() == [11, 12, 13, 14, 15, 16, 17, 91]
        assert sampled[1, :3].tolist() == [21, 22, 92]
        assert sampled[2, :1].tolist() == [93]

    check_output(*sample())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        sampled, counts = sample()
    for _ in range(32):
        graph.replay()
    check_output(sampled, counts)
    # A token-only stress assertion can miss this intermittent race. Verify the
    # real compiled launch too, including the one retained by CUDA graph capture.
    assert launched_warps == [1, 1]
