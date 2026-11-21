# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Transfer joint storage preserves control/generated history across eviction."""

import pytest
import torch

from tests.diffusion.models.cosmos3_nano_sim_bimanual.test_batched_clean_commit import (
    BLOCK,
    HEAD,
    WIDTH,
    layers,
    run_forward,
)
from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.dense_attention import CosmosSimDenseAttentionCache
from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.pipeline_cosmos3_nano_sim_transfer import (
    _trim_transfer_history,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    "chunk,window,sinks", [(1, 1, 0), (1, 3, 0), (1, 3, 2), (1, 4, 1), (1, 30, 3), (2, 30, 0), (4, 30, 0)]
)
@torch.no_grad()
def test_joint_storage_matches_attention_and_clean_history(chunk, window, sinks):
    torch.manual_seed(42)
    net = layers(torch.device("cpu"), torch.float32)
    text = [(torch.randn(1, 5, 2, HEAD), torch.randn(1, 5, 2, HEAD)) for _ in net]
    cache = CosmosSimDenseAttentionCache()
    # ModuleList also supplies a weak-referenceable session owner.
    cache.activate(net, text, 5, 5 + 2 * window * BLOCK)
    original = staged = None
    addresses = [x.data_ptr() for pair in cache.kv for x in pair]
    for frame in range(0, 12, chunk):
        if chunk == 1 and original is not None:
            settings = dict(tokens_per_frame=BLOCK, window_frames=window, sink_frames=sinks)
            _trim_transfer_history(original, **settings)
            _trim_transfer_history(staged, dense_cache=cache, **settings)
            expected_frames = [i for i in range(frame) if i < sinks or i >= frame - (window - sinks - 1)]
            assert cache.history_length == 2 * len(expected_frames) * BLOCK
            if not cache.history_length:
                original = None

        def forward(frames, commit):
            nonlocal original, staged
            hidden = torch.randn(1, frames * BLOCK, WIDTH)
            before, original = run_forward(net, hidden, text, frame, dense=original, window=None, commit=commit)
            after, staged = run_forward(
                net, hidden, text, frame, dense=staged, window=None, commit=commit, dense_cache=cache
            )
            assert torch.equal(before, after)
            if original is not None:
                assert all(torch.equal(a, b) for old, new in zip(original, staged) for a, b in zip(old, new))
            assert all(
                torch.equal(buf[:, :5], src) for bufs, srcs in zip(cache.kv, text) for buf, src in zip(bufs, srcs)
            )

        forward(chunk, True)  # Commit all control latents before denoising.
        for _ in range(4):
            forward(chunk, False)
        for _ in range(chunk):
            forward(1, True)  # Clean generated latents replace scratch in order.
    assert [x.data_ptr() for pair in cache.kv for x in pair] == addresses
