# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import pytest
import torch

from tests.helpers.mark import hardware_marks
from vllm_omni.utils.device_copy import index_to_device

pytestmark = [pytest.mark.core_model, *hardware_marks(res={"cuda": "L4"}, num_cards=1)]


def test_index_staging_survives_later_copies_and_graph_replay():
    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph, stream=stream):
            captured = index_to_device([17] * 6, "cuda")
    for _ in range(32):
        index_to_device([99] * 6, "cuda")
    graph.replay()
    torch.accelerator.synchronize()
    assert captured.tolist() == [17] * 6
