# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Graph cache ownership and replay buffer lifetimes without CUDA capture."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.voxcpm2.cuda_graph_wrapper import VoxCPM2CudaGraphWrapper
from vllm_omni.model_executor.models.voxcpm2.runtime_config import _VoxCPM2RuntimeConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _model_and_wrapper():
    model = torch.nn.Module()
    model.model = torch.nn.Identity()
    model.residual_model = torch.nn.Identity()
    model._mrv2_model_state = SimpleNamespace(per_patch_audio=False)
    wrapper = VoxCPM2CudaGraphWrapper(model, _VoxCPM2RuntimeConfig(), 8)
    return model, wrapper


@pytest.mark.parametrize("run_method", ["run_scaffold", "run_residual"])
def test_transformer_replay_refreshes_inputs_and_preserves_previous_output(monkeypatch, run_method):
    model, wrapper = _model_and_wrapper()
    captures = []

    def capture(module, batch_size, label, **kwargs):
        captures.append((module, batch_size, label))
        graph = SimpleNamespace(
            input_embeds=torch.empty(batch_size, 3),
            positions=torch.empty(batch_size, dtype=torch.long),
            output=torch.empty(batch_size, 3),
        )
        graph.graph = SimpleNamespace(replay=lambda: graph.output.copy_(graph.input_embeds + graph.positions[:, None]))
        return graph

    monkeypatch.setattr(wrapper, "_capture_graph", capture)
    run = getattr(wrapper, run_method)
    first = run(torch.ones(2, 3), torch.tensor([2, 3]), 2)
    second = run(torch.full((2, 3), 5.0), torch.tensor([7, 11]), 2)

    assert len(captures) == 1
    assert captures[0][0] is (model.model if run_method == "run_scaffold" else model.residual_model)
    torch.testing.assert_close(first, torch.tensor([[3.0] * 3, [4.0] * 3]))
    torch.testing.assert_close(second, torch.tensor([[12.0] * 3, [16.0] * 3]))


def test_unified_cache_distinguishes_layout_and_patch_count_and_is_model_local(monkeypatch):
    first_model, first_wrapper = _model_and_wrapper()
    second_model, second_wrapper = _model_and_wrapper()
    captures = []

    def capture(bucket, token_capacity=None, patch_count=None):
        graph = SimpleNamespace(batch_size=bucket, token_capacity=token_capacity, patch_count=patch_count)
        captures.append(graph)
        return graph

    for wrapper in (first_wrapper, second_wrapper):
        monkeypatch.setattr(wrapper, "_capture_unified_decode_graph", capture)

    mixed = first_wrapper.get_unified_graph(3, 16, 0)
    assert first_wrapper.get_unified_graph(4, 16, 0) is mixed
    assert first_wrapper.get_unified_graph(3, 16, 1) is not mixed
    assert first_wrapper.get_unified_graph(3, 8, 0) is not mixed
    assert first_wrapper.get_unified_graph(3, None, 0) is not mixed
    assert second_wrapper.get_unified_graph(3, 16, 0) is not mixed
    assert len(captures) == 5
    # Keep both owners alive throughout the cache operations above.
    assert first_wrapper.model._mrv2_model_state is first_model._mrv2_model_state
    assert second_wrapper.model._mrv2_model_state is second_model._mrv2_model_state
