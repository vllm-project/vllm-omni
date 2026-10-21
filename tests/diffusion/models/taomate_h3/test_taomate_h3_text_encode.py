# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests of the graph-safe text-only prompt encode."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from vllm_omni.diffusion.models.minimax_h3.encoder import (
    MiniMaxH3Qwen3VLTextModel,
    MiniMaxH3Qwen3VLTextRotaryEmbedding,
    MiniMaxH3Qwen3VLVocabParallelEmbedding,
)
from vllm_omni.diffusion.models.taomate_h3.text_encode import (
    GraphSafeTextEncode,
    text_only_positions,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _Group:
    def __init__(self, rank: int = 0, world_size: int = 1) -> None:
        self.calls = 0
        self.rank_in_group = rank
        self.world_size = world_size

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return tensor


def _upstream_embedding(embed: MiniMaxH3Qwen3VLVocabParallelEmbedding, ids: torch.Tensor) -> torch.Tensor:
    """The encoder's vocab-parallel embedding with its boolean-mask writes (tensor-parallel path)."""
    start_idx = embed._tp_rank * embed.num_embeddings_per_partition
    end_idx = start_idx + embed.num_embeddings_per_partition
    input_mask = (ids < start_idx) | (ids >= end_idx)
    masked_input = ids.clone() - start_idx
    masked_input[input_mask] = 0
    output = F.embedding(masked_input, embed.weight)
    output[input_mask, :] = 0.0
    embed.group.all_reduce(output)
    return output


def test_graph_safe_embedding_matches_the_upstream_mask_writes() -> None:
    torch.manual_seed(0)
    vocab, dim = 16, 4
    for tp_rank in (0, 1):
        embed = MiniMaxH3Qwen3VLVocabParallelEmbedding(_Group(tp_rank, 2), vocab, dim, torch.float32)
        embed.weight.data.copy_(torch.randn(vocab // 2, dim))
        ids = torch.tensor([[0, 3, 7, 8, 12, 15, 9]])
        assert torch.equal(embed(ids), _upstream_embedding(embed, ids))
        assert embed.group.calls == 2
    single = MiniMaxH3Qwen3VLVocabParallelEmbedding(_Group(), vocab, dim, torch.float32)
    single.weight.data.copy_(torch.randn(vocab, dim))
    assert torch.equal(single(ids), F.embedding(ids, single.weight))


def test_text_only_positions_are_arange_on_three_axes() -> None:
    ids = torch.zeros(1, 9, dtype=torch.long)
    positions = text_only_positions(ids)
    assert tuple(positions.shape) == (3, 1, 9) and positions.dtype == torch.long
    for axis in range(3):
        assert torch.equal(positions[axis, 0], torch.arange(9))


def test_graph_safe_text_encode_runs_the_text_model_on_the_embedding() -> None:
    seen: dict[str, torch.Tensor] = {}

    def text_model(inputs_embeds: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        seen["embeds"] = inputs_embeds
        seen["positions"] = positions
        return inputs_embeds * 2.0

    text_model.embed_tokens = torch.nn.Embedding.from_pretrained(torch.arange(8.0).view(4, 2))  # type: ignore[attr-defined]
    module = GraphSafeTextEncode(SimpleNamespace(text_model=text_model))  # no rotary module: whole-model path
    assert module.inv_freq is None
    hidden = module(input_ids=torch.tensor([3, 1, 0]))
    assert hidden.dtype == torch.bfloat16 and tuple(hidden.shape) == (3, 2)
    assert torch.equal(hidden.float(), torch.tensor([[12.0, 14.0], [4.0, 6.0], [0.0, 2.0]]))
    assert tuple(seen["positions"].shape) == (3, 1, 3)


def test_prepared_rotary_frequencies_match_the_encoder_module(monkeypatch) -> None:
    """Prepared frequencies preserve the CPU rounding without regenerating during capture."""
    rotary = MiniMaxH3Qwen3VLTextRotaryEmbedding.__new__(MiniMaxH3Qwen3VLTextRotaryEmbedding)
    torch.nn.Module.__init__(rotary)
    rotary.dim = 16
    rotary.base = 10000.0
    rotary.mrope_section = [4, 2, 2]
    positions = text_only_positions(torch.zeros(1, 7, dtype=torch.long))
    x = torch.zeros(1, 7, 16, dtype=torch.bfloat16)
    cos_ref, sin_ref = rotary(x, positions)
    inv_freq = rotary._get_inv_freq(torch.device("cpu"))

    def unexpected_frequency_construction(*args):
        raise AssertionError("prepared rotary path must not rebuild inverse frequencies")

    monkeypatch.setattr(rotary, "_get_inv_freq", unexpected_frequency_construction)
    cos, sin = rotary(x, positions, inv_freq=inv_freq)
    assert torch.equal(cos, cos_ref) and torch.equal(sin, sin_ref)
    with pytest.raises(ValueError, match="must be float32"):
        rotary(x, positions, inv_freq=inv_freq.to(torch.bfloat16))


def test_graph_safe_text_encode_reuses_text_model_forward_after_dtype_cast(monkeypatch) -> None:
    class _Layer(torch.nn.Module):
        def forward(self, hidden, *, position_embeddings):
            cos, sin = position_embeddings
            return hidden + cos + sin

    model = MiniMaxH3Qwen3VLTextModel.__new__(MiniMaxH3Qwen3VLTextModel)
    torch.nn.Module.__init__(model)
    model.embed_tokens = MiniMaxH3Qwen3VLVocabParallelEmbedding(_Group(), 8, 16, torch.float32)
    model.embed_tokens.weight.data.copy_(torch.arange(128).reshape(8, 16) / 128)
    model.rotary_emb = MiniMaxH3Qwen3VLTextRotaryEmbedding(
        SimpleNamespace(hidden_size=16, num_attention_heads=1, rope_parameters={"rope_theta": 10000.0})
    )
    model.rotary_emb.mrope_section = [4, 2, 2]
    model.layers = torch.nn.ModuleList([_Layer(), _Layer()])
    encoder = torch.nn.Module()
    encoder.text_model = model
    encoder.device_target = torch.device("cpu")
    wrapper = GraphSafeTextEncode(encoder).to(torch.bfloat16)
    assert wrapper.inv_freq is not None and wrapper.inv_freq.dtype == torch.float32
    ids = torch.tensor([3, 1, 7])
    expected = model(model.embed_tokens(ids[None]), text_only_positions(ids[None]))[0]

    def unexpected_frequency_construction(*args):
        raise AssertionError("prepared rotary path must not rebuild inverse frequencies")

    monkeypatch.setattr(model.rotary_emb, "_get_inv_freq", unexpected_frequency_construction)
    calls = []
    hook = model.register_forward_hook(lambda *args: calls.append(1))
    actual = wrapper(input_ids=ids)
    hook.remove()
    assert calls == [1]
    assert torch.equal(actual, expected)
