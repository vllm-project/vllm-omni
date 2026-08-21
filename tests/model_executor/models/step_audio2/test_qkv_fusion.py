# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import pytest
import torch
import torch.nn as nn

from vllm_omni.model_executor.models.step_audio2.cosyvoice2.dit import (
    DiTAttention,
    MultiHeadedAttention,
    TimestepEmbedder,
)
from vllm_omni.model_executor.models.step_audio2.step_audio2_token2wav import (
    _load_flow_weights_strict,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def _force_default_gemm(mocker):
    from vllm.model_executor.layers.utils import default_unquantized_gemm

    mocker.patch(
        "vllm.model_executor.layers.linear.dispatch_unquantized_gemm",
        return_value=default_unquantized_gemm,
    )


def _reference_projections(dim: int) -> dict[str, nn.Linear]:
    return {shard_id: nn.Linear(dim, dim, bias=True) for shard_id in ("q", "k", "v")}


def _load_qkv_shards(qkv: nn.Module, references: dict[str, nn.Linear]) -> None:
    with torch.no_grad():
        for shard_id, linear in references.items():
            qkv.weight.weight_loader(qkv.weight, linear.weight, shard_id)
            qkv.bias.weight_loader(qkv.bias, linear.bias, shard_id)


def test_dit_qkv_matches_three_linears(init_fake_tp_group) -> None:
    torch.manual_seed(0)
    dim, num_heads, head_dim = 16, 4, 4
    attention = DiTAttention(
        dim,
        num_heads=num_heads,
        head_dim=head_dim,
        qkv_bias=True,
    )
    references = _reference_projections(dim)
    _load_qkv_shards(attention.to_qkv, references)

    hidden_states = torch.randn(2, 7, dim)
    expected = torch.cat(
        [references[name](hidden_states) for name in ("q", "k", "v")],
        dim=-1,
    )
    actual = attention.to_qkv(hidden_states)

    torch.testing.assert_close(actual, expected)


def test_conformer_qkv_matches_three_linears(init_fake_tp_group) -> None:
    torch.manual_seed(1)
    dim, num_heads = 16, 4
    attention = MultiHeadedAttention(num_heads, dim, dropout_rate=0.0, key_bias=True)
    references = _reference_projections(dim)
    _load_qkv_shards(attention.linear_qkv, references)

    hidden_states = torch.randn(2, 7, dim)
    actual = attention.forward_qkv(hidden_states, hidden_states, hidden_states)
    expected = tuple(
        references[name](hidden_states).view(2, 7, num_heads, dim // num_heads).transpose(1, 2)
        for name in ("q", "k", "v")
    )

    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor)


def test_timestep_frequency_buffer_matches_original_formula(init_fake_tp_group) -> None:
    torch.manual_seed(2)
    embedder = TimestepEmbedder(hidden_size=16, frequency_embedding_size=8)
    timestep = torch.tensor([0.0, 0.25, 0.75])
    scaled = timestep * embedder.scale
    half = embedder.frequency_embedding_size // 2
    frequencies = torch.exp(-math.log(10000) * torch.arange(start=0, end=half) / half).to(timestep)
    arguments = scaled[:, None] * frequencies[None]
    original_embedding = torch.cat(
        [torch.cos(arguments), torch.sin(arguments)],
        dim=-1,
    )
    expected = embedder.mlp(original_embedding)

    torch.testing.assert_close(embedder(timestep), expected, rtol=0, atol=0)
    assert "freqs" not in embedder.state_dict()


class _LoaderFixture(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = nn.Module()
        self.block.attn = nn.Module()
        self.block.attn.to_qkv = DiTAttention(
            16,
            num_heads=4,
            head_dim=4,
            qkv_bias=True,
        ).to_qkv
        self.block.linear_qkv = MultiHeadedAttention(
            4,
            16,
            dropout_rate=0.0,
            key_bias=True,
        ).linear_qkv
        self.other = nn.Linear(16, 16)


def _separate_checkpoint() -> dict[str, torch.Tensor]:
    checkpoint: dict[str, torch.Tensor] = {}
    for prefix in ("block.attn.to", "block.linear"):
        for index, shard_id in enumerate(("q", "k", "v"), start=1):
            checkpoint[f"{prefix}_{shard_id}.weight"] = torch.full(
                (16, 16),
                float(index),
            )
            checkpoint[f"{prefix}_{shard_id}.bias"] = torch.full(
                (16,),
                float(index + 3),
            )
    checkpoint["other.weight"] = torch.randn(16, 16)
    checkpoint["other.bias"] = torch.randn(16)
    return checkpoint


def test_strict_loader_packs_qkv_and_loads_all_parameters(
    init_fake_tp_group,
    tmp_path,
) -> None:
    model = _LoaderFixture()
    checkpoint = _separate_checkpoint()
    checkpoint_path = tmp_path / "flow.pt"
    torch.save(checkpoint, checkpoint_path)

    _load_flow_weights_strict(model, str(checkpoint_path))

    expected_dit_weight = torch.cat([checkpoint[f"block.attn.to_{name}.weight"] for name in ("q", "k", "v")])
    expected_dit_bias = torch.cat([checkpoint[f"block.attn.to_{name}.bias"] for name in ("q", "k", "v")])
    expected_conformer_weight = torch.cat([checkpoint[f"block.linear_{name}.weight"] for name in ("q", "k", "v")])
    expected_conformer_bias = torch.cat([checkpoint[f"block.linear_{name}.bias"] for name in ("q", "k", "v")])
    torch.testing.assert_close(model.block.attn.to_qkv.weight, expected_dit_weight)
    torch.testing.assert_close(model.block.attn.to_qkv.bias, expected_dit_bias)
    torch.testing.assert_close(model.block.linear_qkv.weight, expected_conformer_weight)
    torch.testing.assert_close(model.block.linear_qkv.bias, expected_conformer_bias)
    torch.testing.assert_close(model.other.weight, checkpoint["other.weight"])
    torch.testing.assert_close(model.other.bias, checkpoint["other.bias"])


@pytest.mark.parametrize("failure", ["missing", "unexpected"])
def test_strict_loader_rejects_incomplete_or_unexpected_checkpoint(
    init_fake_tp_group,
    tmp_path,
    failure: str,
) -> None:
    model = _LoaderFixture()
    checkpoint = _separate_checkpoint()
    if failure == "missing":
        checkpoint.pop("block.attn.to_q.weight")
        match = "Missing Flow checkpoint QKV shards"
    else:
        checkpoint["unexpected.weight"] = torch.randn(1)
        match = "Unexpected Flow checkpoint parameter"
    checkpoint_path = tmp_path / f"{failure}.pt"
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(RuntimeError, match=match):
        _load_flow_weights_strict(model, str(checkpoint_path))
