# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Regression of single-frame cache lifetime and request-local gate selection."""

import pytest
import torch

from vllm_omni.diffusion.models.qwen_image_21.qwen_image_21_transformer import (
    QwenImage21Rope,
    QwenImage21SequencePrepare,
    QwenImage21Transformer2DModel,
    QwenImage21TransformerBlock,
    _select_modulation_rows,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize("batch", [1, 2, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("mask_kind", ["none", "prefill", "decode", "mixed"])
def test_shared_activated_gate_preserves_batch_and_condition_rows(batch, dtype, mask_kind):
    torch.manual_seed(41)
    seq, width = 17, 128
    mask = None
    if mask_kind != "none":
        mask = torch.ones(seq, dtype=torch.bool)
        if mask_kind == "prefill":
            mask[:7] = False
        elif mask_kind == "mixed":
            mask[::2] = False
    table = torch.randn(batch + (mask is not None), width, dtype=dtype)
    original = table.clone()
    residual = torch.randn(batch, seq, width, dtype=dtype)
    output = torch.randn_like(residual)
    native = residual + _select_modulation_rows(table, mask).tanh() * output
    shared = residual + _select_modulation_rows(table.tanh(), mask) * output
    torch.testing.assert_close(shared, native, rtol=0, atol=0)
    assert torch.equal(table, original)


class _Attention(torch.nn.Module):
    def forward(self, hidden_states, freqs, **kwargs):
        return hidden_states.sin()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("masked", [False, True])
def test_shared_gates_reach_both_residuals_without_leaking_between_forwards(dtype, masked):
    torch.manual_seed(9)
    block = QwenImage21TransformerBlock.__new__(QwenImage21TransformerBlock)
    torch.nn.Module.__init__(block)
    block.img_norm1 = torch.nn.LayerNorm(8, elementwise_affine=False).to(dtype)
    block.img_norm2 = torch.nn.LayerNorm(8, elementwise_affine=False).to(dtype)
    block.attn = _Attention()
    block.img_mlp = torch.nn.SiLU()
    mask = torch.tensor([False, False, True, True, True]) if masked else None
    hidden = torch.randn(2, 5, 8, dtype=dtype)
    freqs = torch.empty(0)
    for _ in range(2):
        modulation = torch.randn(2 + masked, 32, dtype=dtype)
        _, gate1, _, gate2 = modulation.chunk(4, -1)
        shared = tuple(_select_modulation_rows(gate.tanh(), mask) for gate in (gate1, gate2))
        expected = block(hidden, modulation, freqs, mask)
        actual = block(hidden, modulation, freqs, mask, shared_gates=shared)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


class _TimeEmbed(torch.nn.Module):
    def forward(self, timestep, hidden_states):
        return timestep[:, None].expand(-1, 8)


class _OutputNorm(torch.nn.Module):
    def forward(self, hidden_states, temb, target_token_mask):
        return hidden_states


def _forward_model():
    # Keep the production forward, sequence preparation, blocks and graph body;
    # replace only learned projections/attention with tiny deterministic layers.
    model = QwenImage21Transformer2DModel.__new__(QwenImage21Transformer2DModel)
    torch.nn.Module.__init__(model)
    model.parallel_config = None
    model.causal_condition = True
    model.causal_block = False
    model._decode_graph_manager = None
    model.img_in = torch.nn.Identity()
    model.txt_in = torch.nn.Identity()
    model.sequence_prepare = QwenImage21SequencePrepare(model.img_in, QwenImage21Rope(10000, [2, 2, 4]))
    model.time_text_embed = _TimeEmbed()
    model.modulation = torch.nn.Linear(8, 32)
    model.norm_out = _OutputNorm()
    model.proj_out = torch.nn.Identity()
    blocks = []
    for _ in range(3):
        block = QwenImage21TransformerBlock.__new__(QwenImage21TransformerBlock)
        torch.nn.Module.__init__(block)
        block.img_norm1 = torch.nn.LayerNorm(8, elementwise_affine=False)
        block.img_norm2 = torch.nn.LayerNorm(8, elementwise_affine=False)
        block.attn = _Attention()
        block.img_mlp = torch.nn.SiLU()
        blocks.append(block)
    model.transformer_blocks = torch.nn.ModuleList(blocks)
    return model


@pytest.mark.parametrize("path", ["prefill", "decode", "graph_body"])
def test_model_entries_share_gates_once_and_refresh_each_step(path, monkeypatch):
    from vllm_omni.diffusion.models.qwen_image_21.decode_graph import QwenImage21DecodeGraphEntry

    torch.manual_seed(17)
    model = _forward_model().eval()
    hidden = torch.randn(1, 4, 8)
    text = torch.randn(1, 1, 8)
    entry = QwenImage21DecodeGraphEntry(
        branch="cond",
        batch_size=1,
        prefix_len=1,
        target_tokens=4,
        in_channels=8,
        dtype=torch.float32,
        device=torch.device("cpu"),
        kv_shapes=[torch.Size([1, 1, 1, 8])] * 3,
        kv_dtype=torch.float32,
        freqs=torch.ones(4, 4, dtype=torch.complex64),
    )
    entry.hidden.copy_(hidden)
    cache = entry.block_caches if path == "decode" else [{} for _ in range(3)]
    seen = []
    for block in model.transformer_blocks:
        block.register_forward_pre_hook(
            lambda module, args, kwargs: seen.append(kwargs.get("shared_gates")), with_kwargs=True
        )
    prepared = []
    prepare = model._prepare_shared_gates

    def observe_prepare(modulation, mask):
        result = prepare(modulation, mask)
        prepared.append(result)
        return result

    monkeypatch.setattr(model, "_prepare_shared_gates", observe_prepare)

    def run(timestep):
        if path == "graph_body":
            entry.timestep.fill_(timestep)
            return model._decode_graph_forward(entry)
        return model(
            hidden_states=hidden,
            encoder_hidden_states=text,
            timestep=torch.tensor([timestep]),
            img_shapes=[[(1, 2, 2)]],
            img_mask=torch.tensor([[False, True]]),
            kv_cache=cache,
            return_dict=False,
        )[0]

    previous = None
    with torch.no_grad():
        for timestep in (0.8, 0.2):
            model.share_residual_gates = False
            expected = run(timestep)
            assert all(gates is None for gates in seen[-3:])
            model.share_residual_gates = True
            count = len(prepared)
            actual = run(timestep)
            assert len(prepared) == count + 1
            gates = prepared[-1]
            assert gates is not None and all(value is gates for value in seen[-3:])
            if previous is not None:
                assert gates is not previous
                assert not torch.equal(gates[0], previous[0])
            previous = gates
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
