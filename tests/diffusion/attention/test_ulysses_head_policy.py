# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Selection and fallback on supported diffusion layouts."""

from dataclasses import replace
from types import MethodType, SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.attention.parallel.head_overlap.head_policy import candidate_index, choose
from vllm_omni.diffusion.attention.parallel.head_overlap.query_tail import candidate, eligible

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_video_query_tail():
    from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata, VideoTokenLayout
    from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionBackend

    owner = SimpleNamespace(causal=False, effective_attention=lambda: (None, FlashAttentionBackend, None))
    kernel = MethodType(lambda self: None, owner)
    grid = (9, 30, 52)
    rows = 14040
    layout = VideoTokenLayout(prefix_len=0, latent_grid=grid, used_len=rows)
    md = AttentionMetadata(video_layout=layout, extra={"vsa_dit_seq_shape": grid})
    assert eligible(kernel, md, rows)
    assert eligible(kernel, AttentionMetadata(), rows)
    assert not eligible(kernel, md, rows + 4)
    for unsupported in (
        replace(md, attn_mask=torch.ones(1, rows, dtype=torch.bool)),
        replace(md, extra={**md.extra, "kv_cache_dtype": "fp8"}),
        replace(md, extra={"vsa_dit_seq_shape": (9, 30, 51)}),
        replace(md, video_layout=replace(layout, prefix_len=1)),
        replace(md, video_layout=replace(layout, used_len=rows - 1)),
    ):
        assert not eligible(kernel, unsupported, rows)
    owner.causal = True
    assert not eligible(kernel, md, rows)
    owner.causal = False
    owner.effective_attention = lambda: (None, None, None)
    assert not eligible(kernel, md, rows)


def test_candidate_selection_and_fallback():
    assert choose([[100] * 10, [97] * 10]) == 0
    assert choose([[100] * 10, [90] * 10]) == 1
    assert choose([[100] * 5, [80] * 5]) == 0
    assert [candidate_index(i) for i in range(8)] == [0, 1, 1, 0, 1, 0, 0, 1]
    for heads in (1, 5, 8, 10, 16):
        for rows in (512, 1024, 4096, 14040):
            choices, tail = candidate(heads, rows, 4, 120)
            assert choices[0] == (heads,)
            assert len(choices) <= 2 and all(sum(parts) == heads and min(parts) > 0 for parts in choices)
            assert 0 <= tail <= rows // 4


def test_mismatched_layer_order_skips_collective_preparation(monkeypatch):
    from vllm_omni.diffusion.attention.parallel.head_overlap import head_policy

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(head_policy.dist, "get_world_size", lambda group: 2)

    def gather(peers, local, group):
        peers[:] = [local, list(reversed(local))]

    monkeypatch.setattr(head_policy.dist, "all_gather_object", gather)
    policy = head_policy.HeadPolicy(None, 120)
    policy.component_key = "shape"
    policy.discovered["layout"] = head_policy.Trial((3, 5), 0, example=("cuda", "pg"))
    policy.discovery = [("first", "layout"), ("second", "layout")]
    policy.finish_component(None, (), None)
    assert policy.visits["shape"] == {}
    assert not policy.layouts and not policy.pending and not policy.discovered
