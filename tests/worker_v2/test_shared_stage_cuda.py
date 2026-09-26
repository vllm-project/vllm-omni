# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Small CUDA correctness checks for shared stage output contracts."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.worker.gpu_generation_model_runner import _HostCopyBatch
from vllm_omni.worker_v2.omni_model_runner import OmniGPUModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cuda]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_host_copy_batch_preserves_noncontiguous_outputs_on_current_stream():
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        source = torch.arange(24, device="cuda", dtype=torch.float32).reshape(4, 6)
        copies = _HostCopyBatch(pin_memory=True)
        first = copies.copy(source.t())
        second = copies.copy(source + 10)
        copies.wait()
        assert first.is_contiguous() and first.is_pinned()
        assert torch.equal(first, torch.arange(24, dtype=torch.float32).reshape(4, 6).t())
        assert torch.equal(second, torch.arange(24, dtype=torch.float32).reshape(4, 6) + 10)
        copies.wait()  # no outstanding copies


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
