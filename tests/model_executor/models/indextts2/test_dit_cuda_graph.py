# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm_omni.model_executor.models.indextts2.dit_cuda_graph import (
    CUDAGraphDiTRunner,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _CopySpy:
    def __init__(self):
        self.sources = []

    def copy_(self, source):
        self.sources.append(source)
        return self


class _GraphSpy:
    def __init__(self):
        self.replays = 0

    def replay(self):
        self.replays += 1


def test_replay_only_updates_dynamic_inputs_and_reuses_static_output():
    runner = CUDAGraphDiTRunner(transformer=object())
    graph = _GraphSpy()
    static_x = _CopySpy()
    static_c = _CopySpy()
    static_pos = _CopySpy()
    static_mask = _CopySpy()
    static_out = object()
    entry = SimpleNamespace(
        graph=graph,
        static_x=static_x,
        static_c=static_c,
        static_pos=static_pos,
        static_mask=static_mask,
        static_out=static_out,
    )
    x_in = object()
    c = object()

    output = runner._replay(entry, x_in=x_in, c=c)

    assert static_x.sources == [x_in]
    assert static_c.sources == [c]
    assert static_pos.sources == []
    assert static_mask.sources == []
    assert graph.replays == 1
    assert output is static_out


def test_graph_stats_start_empty_and_remain_bounded():
    runner = CUDAGraphDiTRunner(transformer=object(), max_graphs=4)

    stats = runner.stats_snapshot()
    stats["hits"] = 99

    assert runner.stats_snapshot() == {
        "calls": 0,
        "hits": 0,
        "captures": 0,
        "capture_failures": 0,
        "evictions": 0,
        "eager": 0,
        "cache_size": 0,
    }
