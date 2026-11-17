# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.flow_prefix_cache import FlowPrefixCache
from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.packed_dit import FlowGraphRunner
from vllm_omni.model_executor.models.cosyvoice3.runtime import cosyvoice3_flow_graph_max

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Graph:
    def __init__(self, log):
        self.log = log

    def reset(self):
        self.log.append(self)


@pytest.mark.parametrize("owner", [FlowGraphRunner, FlowPrefixCache])
def test_flow_graph_caches_retire_a_whole_generation(monkeypatch, owner):
    """A flush resets every graph after one sync, so no graph outlives the
    pool blocks of a peer from its generation."""
    events = []
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: events.append("sync"))
    cache = object.__new__(owner)
    cache.graphs = {key: SimpleNamespace(graph=_Graph(events)) for key in ("a", "b", "c")}
    graphs = [state.graph for state in cache.graphs.values()]
    cache.max_graphs, cache.flushes = 3, 0

    (cache.flush if owner is FlowGraphRunner else cache.flush_graphs)()

    assert events == ["sync", *graphs]
    assert cache.graphs == {}
    assert cache.flushes == 1


def test_flow_graph_limit_defaults_and_validation(monkeypatch):
    monkeypatch.delenv("COSYVOICE3_FLOW_GRAPH_MAX", raising=False)
    assert cosyvoice3_flow_graph_max() == 128
    monkeypatch.setenv("COSYVOICE3_FLOW_GRAPH_MAX", "0")
    with pytest.raises(ValueError, match="must be a positive integer"):
        cosyvoice3_flow_graph_max()
