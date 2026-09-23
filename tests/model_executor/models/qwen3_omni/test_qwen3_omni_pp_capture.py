# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Required Thinker snapshots must survive consecutive pipeline ranks.

Actual forward methods with deterministic decoder doubles test the contract
without model weights. Distributed serving remains a separate integration gate.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm.model_executor.models.utils import make_empty_intermediate_tensors_factory
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.qwen3_omni import qwen3_omni as combined_module
from vllm_omni.model_executor.models.qwen3_omni import qwen3_omni_moe_thinker as thinker_module

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.parallel]


class _Layer(nn.Module):
    def forward(self, positions, hidden_states, residual):
        total = hidden_states if residual is None else hidden_states + residual
        return total * 0.01 + 0.125, total


class _Norm(nn.Module):
    def forward(self, hidden_states, residual):
        return hidden_states + residual, None


def _forward(monkeypatch, start, end, embeddings, intermediate=None, *, capture=True):
    rank = SimpleNamespace(is_first_rank=start == 0, is_last_rank=end == 48)
    monkeypatch.setattr(thinker_module, "get_pp_group", lambda: rank)
    model = object.__new__(thinker_module.Qwen3MoeLLMModel)
    nn.Module.__init__(model)
    model.start_layer = start
    model.end_layer = end
    model.layers = nn.ModuleList([_Layer() for _ in range(48)])
    model.norm = _Norm()
    return model.forward(
        input_ids=None,
        positions=torch.arange(embeddings.shape[0]),
        inputs_embeds=embeddings,
        intermediate_tensors=intermediate,
        capture_layer_indices=[0, 24] if capture else None,
        return_hidden_states=capture,
    )


@pytest.mark.parametrize("tokens", [1, 4])
@pytest.mark.parametrize("boundaries", [(0, 12, 48), (0, 24, 48), (0, 36, 48), (0, 16, 32, 48)])
def test_required_captures_match_unsplit_forward(monkeypatch, tokens, boundaries):
    embeddings = torch.arange(tokens * 8, dtype=torch.float32).reshape(tokens, 8) / 8
    reference, captures = _forward(monkeypatch, 0, 48, embeddings)
    intermediate = None
    for start, end in zip(boundaries, boundaries[1:]):
        result = _forward(monkeypatch, start, end, embeddings, intermediate)
        if end < 48:
            assert isinstance(result, IntermediateTensors)
            assert all(isinstance(value, torch.Tensor) for value in result.tensors.values())
            intermediate = result
    actual, actual_captures = result
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    actual_layers = actual_captures["hidden_states"]["layers"]
    assert set(actual_layers) == {0, 24}
    for key in (0, 24):
        torch.testing.assert_close(actual_layers[key], captures["hidden_states"]["layers"][key], rtol=0, atol=0)
    # A later receive into the rank's persistent buffers cannot rewrite snapshots.
    assert intermediate is not None
    for value in intermediate.tensors.values():
        value.zero_()
    for key in (0, 24):
        torch.testing.assert_close(actual_layers[key], captures["hidden_states"]["layers"][key], rtol=0, atol=0)


def test_capture_disabled_keeps_original_intermediate_schema(monkeypatch):
    embeddings = torch.ones(4, 8)
    first = _forward(monkeypatch, 0, 24, embeddings, capture=False)
    assert set(first.tensors) == {"hidden_states", "residual"}
    actual, captures = _forward(monkeypatch, 24, 48, embeddings, first, capture=False)
    reference, _ = _forward(monkeypatch, 0, 48, embeddings, capture=False)
    assert captures is None
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@pytest.mark.parametrize(
    "start,expected", [(0, set()), (16, {"capture_0"}), (24, {"capture_0"}), (32, {"capture_0", "capture_24"})]
)
@pytest.mark.parametrize("staged", [False, True])
def test_factory_declares_only_captures_from_earlier_ranks(monkeypatch, start, expected, staged):
    thinker_config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=8))
    config = SimpleNamespace(
        thinker_config=thinker_config,
        talker_config=SimpleNamespace(accept_hidden_layer=24),
        code2wav_config=SimpleNamespace(),
        tts_bos_token_id=1,
        tts_eos_token_id=2,
        tts_pad_token_id=3,
    )
    model_config = SimpleNamespace(hf_config=config, multimodal_config=None)
    if staged:
        model_config.model_stage = "thinker"
    vllm_config = SimpleNamespace(model_config=model_config, with_hf_config=lambda *args, **kwargs: None)
    thinker = nn.Module()
    thinker.weight = nn.Parameter(torch.ones(1))
    thinker.language_model = SimpleNamespace(model=SimpleNamespace(start_layer=start))
    thinker.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(["hidden_states", "residual"], 8)
    monkeypatch.setattr(combined_module, "init_vllm_registered_model", lambda **kwargs: thinker)
    model = combined_module.Qwen3OmniMoeForConditionalGeneration(vllm_config=vllm_config)
    buffers = model.make_empty_intermediate_tensors(4, dtype=torch.float32, device="cpu")
    assert set(buffers.tensors) == {"hidden_states", "residual"} | (expected if staged else set())
    assert all(value.shape == (4, 8) for value in buffers.tensors.values())
