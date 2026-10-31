# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU checks against enumerated RAS draws and explicit vocabulary mappings."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

spec = importlib.util.spec_from_file_location(
    "cosyvoice3_dspark_math",
    Path(__file__).resolve().parents[4] / "vllm_omni/model_executor/models/cosyvoice3/dspark.py",
)
assert spec is not None and spec.loader is not None
dspark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dspark)


def test_verification_history_uses_drafts_and_never_crosses_request_boundaries():
    # Packed batch order differs from persistent request-state slots.
    table = torch.tensor([[90, 91, 2, 3, 0, 0], [80, 4, 5, 0, 0, 0]])
    rows = torch.tensor([1, 1, 1, 0, 0])
    positions = torch.tensor([2, 3, 4, 3, 4])
    inputs = torch.tensor([5, 6, 7, 3, 8])
    local = torch.tensor([0, 1, 2, 0, 1])
    history = dspark.verification_history(table, torch.tensor([2, 1]), rows, positions, inputs, local, 4)
    assert history.tolist() == [
        [5, 4, -1, -1],
        [6, 5, 4, -1],
        [7, 6, 5, 4],
        [3, 2, -1, -1],
        [8, 3, 2, -1],
    ]


def test_verification_history_excludes_entire_reference_prompt():
    history = dspark.verification_history(
        torch.tensor([[1, 1, 1, 0]]),
        torch.tensor([3]),
        torch.tensor([0]),
        torch.tensor([2]),
        torch.tensor([1]),
        torch.tensor([0]),
        10,
    )
    assert history.tolist() == [[-1] * 10]


def test_verification_logits_apply_constraints_and_ras_before_native_rejection(monkeypatch):
    class Penalties:
        def apply(self, *_):
            raise AssertionError("RAS must not apply ordinary repetition penalties")

    contexts = []

    class StopMask:
        def apply(self, logits, ctx):
            contexts.append(ctx)
            logits[:, 2] = float("-inf")
            return logits

    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.worker.gpu.sample.logits_processor.interface",
        SimpleNamespace(LogitsContext=SimpleNamespace),
    )
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.gpu.sample.penalties", SimpleNamespace(PenaltiesState=Penalties))
    sampler = SimpleNamespace(
        logits_processors=[StopMask(), Penalties()],
        sampling_states=SimpleNamespace(
            get_top_k_top_p=lambda *_: (None, None),
            temperature=SimpleNamespace(gpu=torch.ones(1)),
        ),
        req_states=SimpleNamespace(
            all_token_ids=SimpleNamespace(gpu=torch.tensor([[9, 0, 0, 0]])),
            prompt_len=SimpleNamespace(gpu=torch.tensor([1])),
        ),
        _cosyvoice3_ras_probabilities=dspark.ras_probabilities,
    )
    result = dspark.ras_verification_logits(
        sampler,
        torch.tensor([[4.0, 1.0, 0.0]]),
        torch.tensor([0]),
        torch.tensor([0]),
        np.array([0]),
        torch.tensor([1]),
        torch.tensor([0]),
        torch.tensor([0]),
        np.array([2]),
        default_top_p=0.8,
        default_top_k=1,
        win_size=10,
        tau_r=0.1,
    )
    # Token 0 repeats and is redrawn; token 2 is forbidden, so only 1 remains.
    torch.testing.assert_close(result.exp(), torch.tensor([[0.0, 1.0, 0.0]]))
    assert len(contexts) == 1
    assert contexts[0].seq_lens_upper_bound_np.tolist() == [2]


def _enumerate_ras(logits, temperature, top_k, top_p, history, win_size, tau_r):
    if temperature < 1e-5:
        return torch.nn.functional.one_hot(logits.argmax(), logits.numel()).double()
    full = (logits.double() / temperature).softmax(dim=-1)
    order = full.argsort(descending=True, stable=True)
    nucleus = torch.zeros_like(full)
    mass = 0.0
    for rank, token in enumerate(order.tolist()):
        if mass < top_p and (top_k <= 0 or rank < top_k):
            nucleus[token] = full[token]
        mass += float(full[token])
    nucleus /= nucleus.sum()
    result = torch.zeros_like(full)
    for token in range(full.numel()):
        repetitions = sum(h == token for h in history[-win_size:]) if win_size else 0
        if win_size and repetitions >= win_size * tau_r and any(h >= 0 for h in history):
            alternatives = full.clone()
            alternatives[token] = 0
            if alternatives.sum() > 0:
                result += nucleus[token] * alternatives / alternatives.sum()
            else:
                result[token] += nucleus[token]
        else:
            result[token] += nucleus[token]
    return result


@pytest.mark.parametrize("temperature", [0.0, 0.7, 1.0, 1.8])
@pytest.mark.parametrize("top_k,top_p", [(1, 0.8), (3, 0.6), (0, 1.0), (4, 0.95)])
@pytest.mark.parametrize("history", [[-1] * 10, [0, 1, 1, 3, -1], [2] * 10])
def test_ras_probabilities_match_enumerated_sample_then_redraw(temperature, top_k, top_p, history):
    logits = torch.tensor([1.2, 0.4, -0.3, -1.1, -2.0])
    result = dspark.ras_probabilities(
        logits.unsqueeze(0),
        torch.tensor([temperature]),
        torch.tensor([top_k]),
        torch.tensor([top_p]),
        torch.tensor([history]),
    )[0]
    expected = _enumerate_ras(logits, temperature, top_k, top_p, history, 10, 0.1)
    torch.testing.assert_close(result.double(), expected, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(result.sum(), torch.tensor(1.0))


def test_ras_top_k_one_still_redraws_repeated_token():
    logits = torch.tensor([[4.0, 1.0, 0.0]])
    result = dspark.ras_probabilities(
        logits, torch.ones(1), torch.ones(1), torch.tensor([0.8]), torch.tensor([[0] * 10])
    )
    assert result[0, 0] == 0
    assert result[0, 1] > result[0, 2] > 0


@pytest.mark.parametrize("gap", [20.0, 35.0, 80.0])
def test_ras_redraw_retains_mass_when_initial_float_probability_rounds_to_one(gap):
    logits = torch.tensor([[gap, 0.0, -1.0]])
    result = dspark.ras_probabilities(
        logits, torch.ones(1), torch.ones(1), torch.tensor([0.8]), torch.tensor([[0] * 10])
    )
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result[0, 1:], torch.tensor([0.7310586, 0.2689414]), atol=2e-6, rtol=2e-5)
    assert result[0, 0] == 0


def test_ras_single_finite_token_falls_back_to_same_token():
    result = dspark.ras_probabilities(
        torch.tensor([[float("-inf"), 2.0, float("-inf")]]),
        torch.ones(1),
        torch.tensor([3]),
        torch.tensor([0.8]),
        torch.tensor([[1] * 10]),
    )
    torch.testing.assert_close(result, torch.tensor([[0.0, 1.0, 0.0]]))


def test_vocabulary_remap_preserves_reachable_columns_and_noise_embedding(monkeypatch):
    monkeypatch.setattr(dspark, "UNIFIED_SPEECH_OFFSET", 4)
    monkeypatch.setattr(dspark, "STAGE_VOCAB_SIZE", 5)
    monkeypatch.setattr(dspark, "MASK_TOKEN_ID", 4)
    monkeypatch.setattr(dspark, "UNIFIED_MASK_TOKEN_ID", 9)
    original_ids = torch.tensor([0, 4, 7, 8, 9])
    weights = {
        "d2t": original_ids - torch.arange(5),
        "embed_tokens.weight": torch.arange(20).reshape(10, 2),
        "lm_head.weight": torch.arange(10).reshape(5, 2),
        "markov_head.markov_w1.weight": torch.arange(30).reshape(10, 3),
        "markov_head.markov_w2.weight": torch.arange(15).reshape(5, 3),
    }
    converted, selected = dspark.remap_vocabulary_weights(weights)
    assert selected.tolist() == [0, 3]
    assert (converted["d2t"] + torch.arange(2)).tolist() == [0, 3]
    torch.testing.assert_close(converted["embed_tokens.weight"][:4], weights["embed_tokens.weight"][4:8])
    torch.testing.assert_close(converted["embed_tokens.weight"][4], weights["embed_tokens.weight"][9])
    torch.testing.assert_close(converted["lm_head.weight"], weights["lm_head.weight"][[1, 2]])
    torch.testing.assert_close(
        converted["markov_head.markov_w2.weight"], weights["markov_head.markov_w2.weight"][[1, 2]]
    )


def test_config_keeps_trained_auxiliary_layers_not_draft_layer_count():
    config = dspark.converted_config(
        {
            "speculators_model_type": "dspark",
            "markov_head_type": "vanilla",
            "mask_token_id": dspark.UNIFIED_MASK_TOKEN_ID,
            "aux_hidden_state_layer_ids": [2, 12, 21],
            "transformer_layer_config": {"hidden_size": 896, "num_hidden_layers": 3, "model_type": "qwen3"},
            "markov_rank": 256,
            "block_size": 8,
        },
        6500,
    )
    assert config["dflash_config"]["target_layer_ids"] == [1, 11, 20]
    assert config["eagle_aux_hidden_state_layer_ids"] == [2, 12, 21]
    assert config["vocab_size"] == 6761
    assert config["draft_vocab_size"] == 6500
    assert config["sample_from_anchor"] is False and config["block_size"] == 7
