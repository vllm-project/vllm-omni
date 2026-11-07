# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import Qwen2Config
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.mammoth_moda2 import mammoth_moda2
from vllm_omni.model_executor.models.mammoth_moda2.mammoth_moda2 import (
    Mammoth2DecoderLayer,
    Mammoth3DecoderLayer,
    MammothModa2Qwen2ForCausalLM,
    MammothModa2Qwen3ForCausalLM,
    _build_token_routing,
    moe_forward,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _understanding_expert(hidden_states: torch.Tensor) -> torch.Tensor:
    return hidden_states + 10


def _generation_expert(hidden_states: torch.Tensor) -> torch.Tensor:
    return hidden_states * 2


@pytest.mark.parametrize("shape", [(6, 4), (2, 3, 4)])
def test_moe_forward_precomputed_routing_matches_tokenwise_reference(shape):
    hidden_states = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(shape)
    mask = torch.tensor([False, True, False, True, True, False]).reshape(shape[:-1])
    routing = _build_token_routing(mask)

    output = moe_forward(
        hidden_states,
        _understanding_expert,
        _generation_expert,
        token_routing=routing,
    )

    expected = torch.where(
        mask.unsqueeze(-1),
        _generation_expert(hidden_states),
        _understanding_expert(hidden_states),
    )
    torch.testing.assert_close(output, expected)
    assert output.is_contiguous()


@pytest.mark.parametrize(
    ("mask_value", "expected_expert"),
    [(False, "understanding"), (True, "generation")],
)
def test_moe_forward_precomputed_routing_skips_unused_expert(mask_value, expected_expert):
    calls = {"understanding": 0, "generation": 0}

    def understanding(hidden_states):
        calls["understanding"] += 1
        return hidden_states + 1

    def generation(hidden_states):
        calls["generation"] += 1
        return hidden_states + 2

    hidden_states = torch.ones(8, 4)
    routing = _build_token_routing(torch.full((8,), mask_value))

    output = moe_forward(
        hidden_states,
        understanding,
        generation,
        token_routing=routing,
    )

    assert calls[expected_expert] == 1
    assert calls["generation" if expected_expert == "understanding" else "understanding"] == 0
    expected_offset = 1 if expected_expert == "understanding" else 2
    torch.testing.assert_close(output, hidden_states + expected_offset)


def test_moe_forward_mask_fallback_preserves_existing_api():
    hidden_states = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    mask = torch.tensor([False, True, False, True, True, False])

    output = moe_forward(
        hidden_states,
        _understanding_expert,
        _generation_expert,
        mask,
    )

    expected = torch.where(
        mask.unsqueeze(-1),
        _generation_expert(hidden_states),
        _understanding_expert(hidden_states),
    )
    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize("shape", [(6, 4), (2, 3, 4)])
@pytest.mark.parametrize("routing_mode", ["mask", "precomputed", "both"])
@pytest.mark.parametrize("und_dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("gen_dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_moe_forward_preserves_expert_output_dtype_promotion(shape, routing_mode, und_dtype, gen_dtype):
    hidden_states = torch.arange(24, dtype=torch.float32).reshape(shape) / 7
    mask = torch.tensor([False, True, False, True, True, False]).reshape(shape[:-1])

    def understanding(states):
        return (states[..., :3] + 10).to(und_dtype)

    def generation(states):
        return (states[..., :3] * 2).to(gen_dtype)

    output = moe_forward(
        hidden_states,
        understanding,
        generation,
        mask if routing_mode != "precomputed" else None,
        token_routing=_build_token_routing(mask) if routing_mode != "mask" else None,
    )

    flat_hidden_states = hidden_states.reshape(-1, shape[-1])
    flat_mask = mask.reshape(-1)
    gen_positions = torch.where(flat_mask)[0]
    und_positions = torch.where(~flat_mask)[0]
    inverse_order = torch.argsort(torch.cat([gen_positions, und_positions]))
    expected = torch.cat([generation(flat_hidden_states[flat_mask]), understanding(flat_hidden_states[~flat_mask])])[
        inverse_order
    ].reshape(*shape[:-1], 3)

    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert output.is_contiguous()


def test_moe_forward_rejects_mask_shape_mismatch():
    with pytest.raises(ValueError, match="gen_token_mask shape mismatch"):
        moe_forward(
            torch.ones(4, 8),
            _understanding_expert,
            _generation_expert,
            torch.tensor([True, False]),
        )


def test_moe_forward_rejects_routing_size_mismatch():
    with pytest.raises(ValueError, match="token_routing size mismatch"):
        moe_forward(
            torch.ones(4, 8),
            _understanding_expert,
            _generation_expert,
            token_routing=(torch.tensor([0]), torch.tensor([1])),
        )


@pytest.fixture
def embedding_model():
    model = MammothModa2Qwen2ForCausalLM.__new__(MammothModa2Qwen2ForCausalLM)
    nn.Module.__init__(model)
    model.extra_gen_vocab = True
    model.gen_vocab_start_index = 4
    model.base_vocab_size = 4
    model.config = Qwen2Config(hidden_size=3, num_attention_heads=1, num_key_value_heads=1)
    model.embed_tokens = nn.Embedding.from_pretrained(torch.arange(12.0).reshape(4, 3))
    model.gen_embed_tokens = nn.Embedding.from_pretrained(torch.arange(9.0).reshape(3, 3) + 100)
    return model


@pytest.mark.parametrize("shape", [(6,), (2, 3)])
@pytest.mark.parametrize("routing_mode", ["inferred", "precomputed"])
@pytest.mark.parametrize("ids", [[0, 4, 6, 2, 5, 1], [0, 1, 3, 2, 1, 0], [4, 6, 5, 4, 5, 6]])
def test_get_input_embeddings_matches_table_reference(embedding_model, shape, routing_mode, ids):
    input_ids = torch.tensor(ids).reshape(shape)
    routing = _build_token_routing(input_ids >= 4) if routing_mode == "precomputed" else None
    calls = {"base": 0, "generation": 0}

    def record_base(module, inputs, output):
        calls["base"] += 1

    def record_generation(module, inputs, output):
        calls["generation"] += 1

    embedding_model.embed_tokens.register_forward_hook(record_base)
    embedding_model.gen_embed_tokens.register_forward_hook(record_generation)

    output = embedding_model.get_input_embeddings(input_ids, token_routing=routing)

    expected = torch.stack(
        [
            embedding_model.embed_tokens.weight[token_id]
            if token_id < 4
            else embedding_model.gen_embed_tokens.weight[token_id - 4]
            for token_id in ids
        ]
    ).reshape(*shape, 3)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert output.is_contiguous()
    assert calls == {
        "base": int(any(token_id < 4 for token_id in ids)),
        "generation": int(any(token_id >= 4 for token_id in ids)),
    }


@pytest.mark.parametrize("disabled", ["extra_vocab", "generation_embedding"])
def test_get_input_embeddings_without_generation_table(embedding_model, disabled):
    if disabled == "extra_vocab":
        embedding_model.extra_gen_vocab = False
    else:
        embedding_model.gen_embed_tokens = None
    input_ids = torch.tensor([0, 3, 1])

    output = embedding_model.get_input_embeddings(input_ids)

    torch.testing.assert_close(output, embedding_model.embed_tokens.weight[input_ids], rtol=0, atol=0)


@pytest.mark.parametrize(
    ("input_ids", "generation_positions", "base_positions"),
    [
        ([0, 4, 6, 2], [1], [0]),
        ([0, 1, 2, 3], [], [0]),
        ([4, 5, 6, 4], [0], []),
    ],
)
def test_get_input_embeddings_rejects_routing_size_mismatch(
    embedding_model, input_ids, generation_positions, base_positions
):
    routing = (torch.tensor(generation_positions, dtype=torch.long), torch.tensor(base_positions, dtype=torch.long))
    with pytest.raises(ValueError, match="token_routing size mismatch"):
        embedding_model.get_input_embeddings(torch.tensor(input_ids), token_routing=routing)


@pytest.mark.parametrize("mask_values", [[], [False, False], [True, True], [False, True, False, True]])
def test_token_routing_builder_produces_complete_disjoint_partition(mask_values):
    mask = torch.tensor(mask_values, dtype=torch.bool)
    generation_positions, base_positions = _build_token_routing(mask)
    assert generation_positions.tolist() == [index for index, value in enumerate(mask_values) if value]
    assert base_positions.tolist() == [index for index, value in enumerate(mask_values) if not value]
    assert sorted(generation_positions.tolist() + base_positions.tolist()) == list(range(len(mask_values)))


@pytest.mark.parametrize("mask_values", [[], [False, False], [True, True], [False, True, False, True]])
def test_token_routing_builder_defers_to_mask_during_capture(monkeypatch, mask_values):
    monkeypatch.setattr(mammoth_moda2, "_is_graph_capturing", lambda device: True)

    assert _build_token_routing(torch.tensor(mask_values, dtype=torch.bool)) is None


class _PassthroughNorm(nn.Module):
    def forward(self, hidden_states, residual=None):
        if residual is None:
            return hidden_states
        return hidden_states, residual


class _PassthroughAttention(nn.Module):
    def forward(self, *, positions, hidden_states):
        return hidden_states


@pytest.mark.parametrize(
    ("model_class", "layer_class"),
    [(MammothModa2Qwen2ForCausalLM, Mammoth2DecoderLayer), (MammothModa2Qwen3ForCausalLM, Mammoth3DecoderLayer)],
)
@pytest.mark.parametrize("input_mode", ["ids", "ids_and_embeds", "embeds_only"])
@pytest.mark.parametrize("capturing", [False, True])
@pytest.mark.parametrize("first_rank", [True, False])
def test_forward_reuses_routing_and_preserves_inputs_embeds_contract(
    embedding_model, monkeypatch, model_class, layer_class, input_mode, capturing, first_rank
):
    model = model_class.__new__(model_class)
    nn.Module.__init__(model)
    model.extra_gen_vocab = embedding_model.extra_gen_vocab
    model.gen_vocab_start_index = embedding_model.gen_vocab_start_index
    model.base_vocab_size = embedding_model.base_vocab_size
    model.config = embedding_model.config
    model.embed_tokens = embedding_model.embed_tokens
    model.gen_embed_tokens = embedding_model.gen_embed_tokens
    layers = []
    for _ in range(2):
        layer = layer_class.__new__(layer_class)
        nn.Module.__init__(layer)
        layer.input_layernorm = _PassthroughNorm()
        layer.post_attention_layernorm = _PassthroughNorm()
        layer.self_attn = _PassthroughAttention()
        layer.moe_enable = True
        layer.mlp = nn.Linear(3, 3)
        layer.gen_mlp = nn.Linear(3, 3)
        with torch.no_grad():
            layer.mlp.weight.copy_(torch.eye(3))
            layer.mlp.bias.fill_(10)
            layer.gen_mlp.weight.copy_(torch.eye(3) * 2)
            layer.gen_mlp.bias.zero_()
        layers.append(layer)
    model.layers = nn.ModuleList(layers)
    model.start_layer, model.end_layer = 0, len(layers)
    model.norm = _PassthroughNorm()
    monkeypatch.setattr(
        mammoth_moda2, "get_pp_group", lambda: SimpleNamespace(is_first_rank=first_rank, is_last_rank=True)
    )
    monkeypatch.setattr(mammoth_moda2, "_is_graph_capturing", lambda device: capturing)
    routing_calls = []

    def record_routing(mask):
        routing_calls.append(mask)
        routing = _build_token_routing(mask)
        if capturing:
            assert routing is None
        return routing

    monkeypatch.setattr(mammoth_moda2, "_build_token_routing", record_routing)
    input_ids = torch.tensor([0, 4, 6, 2])
    supplied_embeds = torch.arange(12.0).reshape(4, 3) + 200
    initial = (
        torch.stack(
            [
                model.embed_tokens.weight[0],
                model.gen_embed_tokens.weight[0],
                model.gen_embed_tokens.weight[2],
                model.embed_tokens.weight[2],
            ]
        )
        if input_mode == "ids"
        else supplied_embeds
    )

    intermediate_tensors = None
    if not first_rank:
        tensors = {"hidden_states": initial, "residual": None}
        if input_mode != "embeds_only":
            tensors["gen_token_mask"] = input_ids >= 4
        intermediate_tensors = IntermediateTensors(tensors)

    if capturing and input_mode == "embeds_only":
        with pytest.raises(RuntimeError, match="requires gen_token_mask"):
            model(
                input_ids=None,
                positions=torch.arange(4),
                inputs_embeds=supplied_embeds,
                intermediate_tensors=intermediate_tensors,
            )
        return

    output = model(
        input_ids=input_ids if first_rank and input_mode != "embeds_only" else None,
        positions=torch.arange(4),
        inputs_embeds=supplied_embeds if input_mode != "ids" else None,
        intermediate_tensors=intermediate_tensors,
    )

    expected = initial
    for _ in layers:
        expected = (
            expected + 10
            if input_mode == "embeds_only"
            else torch.where((input_ids >= 4).unsqueeze(-1), expected * 2, expected + 10)
        )
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    if input_mode == "embeds_only":
        assert routing_calls == [None] * (len(layers) + 1)
    else:
        assert len(routing_calls) == 1
        torch.testing.assert_close(routing_calls[0], input_ids >= 4)
