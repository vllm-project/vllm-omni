# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Small CUDA correctness checks for shared stage output contracts."""

from types import SimpleNamespace

import pytest
import torch

from tests.helpers.mark import hardware_marks
from vllm_omni.worker_v2.omni_model_runner import OmniGPUModelRunner

pytestmark = [pytest.mark.core_model, *hardware_marks(res={"cuda": "L4"}, num_cards=1)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_aux_pytree_tracks_replayed_graph_values():
    runner = object.__new__(OmniGPUModelRunner)
    runner.model = SimpleNamespace(_returns_tuple=True, supports_mrv2_full_graph_aux_outputs=True)
    runner._configure_cudagraph_output_contract()
    source = torch.ones((4, 3), device="cuda")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = runner._flatten_capture_aux_output((source * 2, {"layers": [source + 1, source + 2]}))
    source.fill_(7)
    graph.replay()
    hidden, aux = runner._split_fullgraph_output(captured)
    assert torch.equal(hidden, torch.full_like(source, 14))
    assert torch.equal(aux["layers"][0], torch.full_like(source, 8))
    assert torch.equal(aux["layers"][1], torch.full_like(source, 9))
