# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The compiled Thinker forward must not emit `update` in its bytecode.

vLLM's cudagraph-in-compiler guard rejects any `update` name in the
Dynamo-transformed forward as a possible nn.Module buffer mutation
(vllm/compilation/wrapper.py). The check is name-based, so the local
plain-dict merge that fed IntermediateTensors tripped it too, and every
Thinker deployment with cudagraphs enabled hard-failed at compile time
(#8818). Pin the contract without model weights: no `.update` call in
the source, and the real forward passes the guard hook under Dynamo.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch._dynamo.convert_frame import register_bytecode_hook
from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
from vllm.config import CUDAGraphMode
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.qwen3_omni import qwen3_omni_moe_thinker as thinker_module

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.parallel]


class _Layer(nn.Module):
    def forward(self, positions, hidden_states, residual):
        total = hidden_states if residual is None else hidden_states + residual
        return total * 0.01 + 0.125, total


def _make_model(monkeypatch, start, end):
    rank = SimpleNamespace(is_first_rank=start == 0, is_last_rank=end == 48)
    monkeypatch.setattr(thinker_module, "get_pp_group", lambda: rank)
    model = object.__new__(thinker_module.Qwen3MoeLLMModel)
    nn.Module.__init__(model)
    model.start_layer = start
    model.end_layer = end
    model.layers = nn.ModuleList([_Layer() for _ in range(48)])
    return model


def test_forward_source_has_no_update_call():
    # Source-level proxy for the guard: without a `.update` call in the
    # forward, Dynamo cannot leave `update` in the transformed co_names.
    assert "update" not in thinker_module.Qwen3MoeLLMModel.forward.__code__.co_names


def test_compiled_forward_passes_cudagraph_guard(monkeypatch):
    model = _make_model(monkeypatch, 0, 24)
    model.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE),
        compile_debug_dump_path=lambda: None,
    )
    model._compiled_bytecode = None
    handle = register_bytecode_hook(
        lambda old_code, new_code: TorchCompileWithNoGuardsWrapper.bytecode_hook(model, old_code, new_code)
    )
    try:
        compiled = torch.compile(model.forward, backend="eager", fullgraph=True, dynamic=False)
        result = compiled(
            input_ids=None,
            positions=torch.arange(4),
            inputs_embeds=torch.ones(4, 8),
            intermediate_tensors=None,
            capture_layer_indices=[0, 24],
            return_hidden_states=True,
        )
    finally:
        handle.remove()
        torch._dynamo.reset()

    # The non-last-rank branch still returns the same IntermediateTensors
    # schema the dict.update version produced.
    assert isinstance(result, IntermediateTensors)
    assert set(result.tensors) == {"hidden_states", "residual", "capture_0"}
    # The guard hook accepted the transformed bytecode under cudagraph mode.
    assert model._compiled_bytecode is not None
    assert "update" not in model._compiled_bytecode.co_names
