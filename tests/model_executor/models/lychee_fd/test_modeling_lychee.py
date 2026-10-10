# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import pytest
import torch

from vllm_omni.model_executor.models.lychee_fd.modeling_lychee import (
    fuse_channel_embeddings,
    merge_conditioning,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_channel_fusion_is_additive_and_request_local() -> None:
    text = torch.ones(2, 4)
    stoken = torch.full((2, 4), 2.0)
    control = torch.full((2, 4), 3.0)
    audio = torch.full((2, 4), 4.0)

    fused = fuse_channel_embeddings(
        text,
        stoken_embeddings=stoken,
        control_embeddings=control,
        audio_embeddings=audio,
    )

    torch.testing.assert_close(fused, torch.full((2, 4), 10.0))
    torch.testing.assert_close(text, torch.ones(2, 4))


def test_channel_fusion_rejects_implicit_broadcasting() -> None:
    with pytest.raises(ValueError, match="must match"):
        fuse_channel_embeddings(torch.zeros(2, 4), audio_embeddings=torch.zeros(1, 4))


def test_channel_fusion_keeps_decoder_dtype() -> None:
    fused = fuse_channel_embeddings(
        torch.ones(2, 4, dtype=torch.bfloat16),
        audio_embeddings=torch.ones(2, 4, dtype=torch.float32),
    )
    assert fused.dtype == torch.bfloat16


def test_merge_uses_supplied_same_step_text_embedding() -> None:
    speech = torch.tensor([[1.0, 2.0]])
    sampled_text = torch.tensor([[3.0, 5.0]])

    merged = merge_conditioning(speech, sampled_text)

    torch.testing.assert_close(merged, torch.tensor([[4.0, 7.0]]))


def test_merge_rejects_position_mismatch() -> None:
    with pytest.raises(ValueError, match="shapes must match"):
        merge_conditioning(torch.zeros(2, 4), torch.zeros(1, 4))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_checkpoint_rmsnorm_preserves_released_composition_and_residual_interface(dtype, monkeypatch):
    from vllm_omni.model_executor.models.lychee_fd.modeling_lychee import LycheeRMSNorm

    norm = LycheeRMSNorm(16, eps=1e-6).to(dtype=dtype)
    hidden = torch.randn(3, 16, dtype=dtype)
    residual = torch.randn(3, 16, dtype=dtype)
    actual, updated = norm(hidden, residual)
    expected_residual = hidden + residual

    def released(inputs):
        upcast = inputs.float()
        inverse = (upcast.pow(2).mean(-1, keepdim=True) + 1e-6).rsqrt()
        return (upcast * inverse * norm.weight).to(dtype)

    expected = released(expected_residual)
    torch.testing.assert_close(updated, expected_residual, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def fused_operator_is_not_the_released_formula(*args, **kwargs):
        raise AssertionError("Torch 2.13 fused RMSNorm must not replace the released composition")

    monkeypatch.setattr(torch.nn.functional, "rms_norm", fused_operator_is_not_the_released_formula)
    torch.testing.assert_close(norm(hidden), released(hidden), rtol=0, atol=0)


def test_nullable_side_channel_masks_reproduce_initial_singleton_fusion():
    from types import SimpleNamespace

    from vllm_omni.model_executor.models.lychee_fd.modeling_lychee import LycheeFullDuplexForConditionalGeneration

    captured = []

    def branches(positions, fused):
        captured.append(fused)
        return SimpleNamespace(text_hidden=fused, stoken_hidden=fused, control_hidden=fused)

    model = SimpleNamespace(embed_input_ids=lambda ids: ids.float()[:, None].expand(-1, 2), forward_branches=branches)
    LycheeFullDuplexForConditionalGeneration.forward(
        model,
        input_ids=torch.tensor([1, 2, 3]),
        positions=torch.arange(3),
        stoken_input_ids=torch.tensor([4, 4, 4]),
        control_input_ids=torch.tensor([5, 5, 5]),
        stoken_input_mask=torch.tensor([False, False, True]),
        control_input_mask=torch.tensor([False, False, True]),
        audio_embeddings=torch.tensor([[0, 0], [0, 0], [7, 7]]),
    )
    assert captured[0].tolist() == [[1, 1], [2, 2], [19, 19]]


def test_checkpoint_rotary_matches_frozen_old_cuda_outputs():
    import json
    from pathlib import Path
    from types import SimpleNamespace

    from vllm_omni.model_executor.models.lychee_fd.modeling_lychee import LycheeRotaryEmbedding

    fixture = json.loads((Path(__file__).parent / "fixtures/primitive_bf16_reference.json").read_text())

    def bf16(key):
        return torch.tensor(fixture[key], dtype=torch.bfloat16)

    rotary = LycheeRotaryEmbedding(
        SimpleNamespace(head_size=128, rotary_dim=128, is_neox_style=True, cos_sin_cache=bf16("rope_cache"))
    )
    query, key = rotary(torch.arange(2), bf16("rope_query"), bf16("rope_key"))
    torch.testing.assert_close(query, bf16("rope_expected_query"), rtol=0, atol=0)
    torch.testing.assert_close(key, bf16("rope_expected_key"), rtol=0, atol=0)
    torch.testing.assert_close(rotary(torch.arange(2), bf16("rope_query"))[0], query, rtol=0, atol=0)


def test_checkpoint_silu_mul_matches_frozen_old_cuda_rounding():
    import json
    from pathlib import Path

    from vllm_omni.model_executor.models.lychee_fd.modeling_lychee import LycheeSiluAndMul

    fixture = json.loads((Path(__file__).parent / "fixtures/primitive_bf16_reference.json").read_text())
    gate = torch.tensor(fixture["mlp_gate"], dtype=torch.bfloat16)
    up = torch.tensor(fixture["mlp_up"], dtype=torch.bfloat16)
    expected = torch.tensor(fixture["mlp_expected_product"], dtype=torch.bfloat16)
    actual = LycheeSiluAndMul()(torch.cat((gate, up)))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    float32_fused = (torch.nn.functional.silu(gate.float()) * up.float()).to(gate.dtype)
    assert bool((float32_fused != expected).all())


def test_checkpoint_gate_up_preserves_separate_gemm_shapes(monkeypatch):
    from types import SimpleNamespace

    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    from vllm_omni.model_executor.models.lychee_fd.modeling_lychee import LycheeGateUpLinear

    hidden = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
    weight = torch.arange(64, dtype=torch.bfloat16).reshape(16, 4)
    linear = SimpleNamespace(weight=weight, bias=None, skip_bias_add=False, quant_method=UnquantizedLinearMethod())
    projection = torch.nn.functional.linear
    shapes = []

    def record(inputs, matrix, bias=None):
        shapes.append(tuple(matrix.shape))
        return projection(inputs, matrix, bias)

    monkeypatch.setattr(torch.nn.functional, "linear", record)
    result, bias = LycheeGateUpLinear.forward(linear, hidden)
    assert shapes == [(8, 4), (8, 4)]
    assert bias is None
    expected = torch.cat((projection(hidden, weight[:8]), projection(hidden, weight[8:])), dim=-1)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_released_head_projection_uses_real_vocab_without_replacing_native_storage(monkeypatch):
    from types import SimpleNamespace

    from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod

    from vllm_omni.model_executor.models.lychee_fd.modeling_lychee import LycheeLogitsProcessor

    processor = LycheeLogitsProcessor.__new__(LycheeLogitsProcessor)
    torch.nn.Module.__init__(processor)
    processor.vocab_size = 5
    processor.head_dtype = None
    weight = torch.arange(32, dtype=torch.bfloat16).reshape(8, 4)
    head = SimpleNamespace(tp_size=1, quant_method=UnquantizedEmbeddingMethod(), weight=weight)
    hidden = torch.arange(4, dtype=torch.bfloat16)[None, :]
    bias = torch.arange(8, dtype=torch.bfloat16)
    projection = torch.nn.functional.linear
    recorded = []

    def record(inputs, matrix, selected_bias=None):
        recorded.append((tuple(matrix.shape), tuple(selected_bias.shape)))
        return projection(inputs, matrix, selected_bias)

    monkeypatch.setattr(torch.nn.functional, "linear", record)
    result = processor._apply_head(head, hidden, bias)
    assert recorded == [((5, 4), (5,))]
    assert head.weight is weight
    assert tuple(head.weight.shape) == (8, 4)
    torch.testing.assert_close(result, projection(hidden, weight[:5], bias[:5]), rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["quantized", "tp", "head_dtype"])
def test_released_head_projection_preserves_native_fallbacks(mode, monkeypatch):
    from types import SimpleNamespace

    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod

    from vllm_omni.model_executor.models.lychee_fd.modeling_lychee import LycheeLogitsProcessor

    processor = LycheeLogitsProcessor.__new__(LycheeLogitsProcessor)
    torch.nn.Module.__init__(processor)
    processor.vocab_size = 5
    processor.head_dtype = torch.float32 if mode == "head_dtype" else None
    head = SimpleNamespace(
        tp_size=2 if mode == "tp" else 1, quant_method=object() if mode == "quantized" else UnquantizedEmbeddingMethod()
    )
    hidden = torch.zeros(1, 4, dtype=torch.bfloat16)
    sentinel = torch.ones(1, 5)
    calls = []

    def native(self, actual_head, actual_hidden, bias):
        calls.append((actual_head, actual_hidden, bias))
        return sentinel

    monkeypatch.setattr(LogitsProcessor, "_apply_head", native)
    assert processor._apply_head(head, hidden, None) is sentinel
    assert len(calls) == 1
    assert calls[0][0] is head and calls[0][1] is hidden and calls[0][2] is None
