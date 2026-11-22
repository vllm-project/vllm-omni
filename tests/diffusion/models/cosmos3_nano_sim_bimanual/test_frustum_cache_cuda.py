# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exact page-boundary reads remain live through CUDA graph replay."""

import pytest
import torch

from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.dense_attention import CosmosSimDenseAttentionCache
from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.frustum_cache import FrustumKVArchive
from vllm_omni.experimental.ar_diffusion.capability import ARDiffusionKVBranchSpec
from vllm_omni.experimental.ar_diffusion.kv_cache import ARDiffusionKVCache
from vllm_omni.experimental.ar_diffusion.kv_cache.config import ARDiffusionKVConfig
from vllm_omni.experimental.ar_diffusion.kv_cache.state import ARDiffusionKVState

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]


@pytest.mark.parametrize("paged", [False, True])
@pytest.mark.parametrize("tokens", [394, 924])
@torch.no_grad()
def test_selected_history_changes_inside_captured_graph(paged, tokens):
    class Owner:
        pass

    owner = Owner()
    device = torch.device("cuda")
    text = [(torch.zeros(1, 2, 1, 8, device=device), torch.zeros(1, 2, 1, 8, device=device))]
    state = None
    if paged:
        cache = ARDiffusionKVCache(
            ARDiffusionKVConfig(enable=True, chunk_size=tokens, window_chunks=16, sink_chunks=0),
            num_layers=1,
            num_kv_heads=1,
            head_size=8,
            dtype=torch.float32,
            block_size=16,
            max_model_len=16 * tokens,
            available_bytes=1 << 25,
            kv_branches=(ARDiffusionKVBranchSpec("main", 0),),
            session_capacity=1,
            frames_per_block=1,
            max_scratch_tokens_per_branch=tokens,
            device=device,
        )
        state = ARDiffusionKVState(cache, "test", {"main": cache.begin_request("test")}, num_layers=1)
    stage = CosmosSimDenseAttentionCache()
    archive = FrustumKVArchive(owner, text, 2, tokens, 16, 2, 2, stage, state)
    try:
        for frame in range(8):
            archive.select(frame, [])
            archive.prepare(frame)
            for key, value in stage.kv:
                key[:, 2 : 2 + tokens].fill_(frame)
                value[:, 2 : 2 + tokens].fill_(-frame)
            archive.commit(1)
        archive.select(8, [0, 7])
        archive.prepare(8)
        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = stage.kv[0][0][:, 2 : 2 + 2 * tokens].sum()
        for selected in [[1, 3], [0, 7], [2, 5]]:
            archive.select(8, selected)
            archive.prepare(8)
            graph.replay()
            assert result.item() == sum(selected) * tokens * 8
        if state is not None:
            assert state.adapter("main").num_computed_tokens == 8 * tokens
            assert state.adapter("main").compacted_tokens == 0
    finally:
        if state is not None:
            state.close()
