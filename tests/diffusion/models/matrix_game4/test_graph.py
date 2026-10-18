# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Graph state, stream ownership and output lifetime without model weights."""

import pytest
import torch
from vllm.config import VllmConfig, set_current_vllm_config

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.models.matrix_game4.graph import MatrixGraph

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, *hardware_marks(res={"cuda": "L4"}, num_cards=1)]


@torch.inference_mode()
@pytest.mark.parametrize("num_tokens", [405, 920])
def test_graph_preserves_caller_stream_state_and_owned_outputs(num_tokens):
    with set_current_vllm_config(VllmConfig()):
        caller = torch.accelerator.current_stream()
        state = torch.zeros(16, device="cuda")

        def advance(value):
            output = state + value
            state.copy_(output)
            return output

        graph = MatrixGraph(advance, (state,), num_tokens=num_tokens)
        # Capturing on vLLM's non-default stream must not leak that stream
        # into the caller/runner, whose output can cross Python threads.
        assert torch.accelerator.current_stream() == caller
        outputs = []
        for value in (1, 2, 3):
            outputs.append(graph(torch.full_like(state, value)))
            assert torch.accelerator.current_stream() == caller
        # Check only after every asynchronous replay: a later graph must not
        # overwrite an earlier output, and warmup must not advance history.
        assert [output.cpu().tolist() for output in outputs] == [[1.0] * 16, [3.0] * 16, [6.0] * 16]
        assert len(graph.wrapper.concrete_cudagraph_entries) == 1
        graph.close()
        assert not graph.wrapper.concrete_cudagraph_entries
