# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for Flux3TextEncoder.

These tests use lightweight Qwen-compatible fakes without loading checkpoint
weights. CPU tests cover hidden-state selection, output construction, and
post-load layer truncation. The compilation parity test requires one CUDA GPU
and compares eager and compiled BF16 outputs.
"""

from typing import NamedTuple
from unittest import mock

import pytest
import torch
import torch.nn as nn

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.models.flux3.flux3_text_encoder import (
    _DEFAULT_OUTPUT_LAYER,
    DEFAULT_PAD_MULTIPLE,
    DEFAULT_TEXT_TOKENS,
    Flux3TextEncoder,
)

CONTEXT_IN_DIM = 32
VEC_IN_DIM = 16
N_TOKENS = 8
QWEN_HIDDEN = 8  # 4 layers × 8 hidden = CONTEXT_IN_DIM

# Use 4 compact indices so tests don't need a 33-layer fake model.
# 4 × QWEN_HIDDEN (8) = CONTEXT_IN_DIM (32).
TEST_OUTPUT_LAYER = [0, 1, 2, 3]


class _FakeLanguageModelOutput(NamedTuple):
    hidden_states: tuple[torch.Tensor, ...]


class _FakeDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(QWEN_HIDDEN, QWEN_HIDDEN, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + torch.nn.functional.silu(self.proj(hidden_states))


class _FakeLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_FakeDecoderLayer() for _ in range(5)])
        self.norm = nn.LayerNorm(QWEN_HIDDEN)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        **_kwargs,
    ) -> _FakeLanguageModelOutput:
        hidden_states = torch.nn.functional.one_hot(
            input_ids.remainder(QWEN_HIDDEN),
            num_classes=QWEN_HIDDEN,
        ).to(self.layers[0].proj.weight.dtype)
        hidden_states = hidden_states + position_ids.unsqueeze(-1).to(hidden_states.dtype) / 100
        hidden_states = hidden_states * attention_mask.unsqueeze(-1)

        all_hidden_states = []
        for layer in self.layers:
            all_hidden_states.append(hidden_states)
            hidden_states = layer(hidden_states)
        all_hidden_states.append(self.norm(hidden_states))
        return _FakeLanguageModelOutput(tuple(all_hidden_states))


class _FakeQwen3(nn.Module):
    """Minimal stand-in exposing the Qwen module layout and device properties."""

    def __init__(self, device: str = "cpu", dtype: torch.dtype = torch.float32):
        super().__init__()
        self._device = torch.device(device)
        self._dtype = dtype
        self.model = nn.Module()
        self.model.language_model = _FakeLanguageModel()
        self.dummy = nn.Parameter(torch.zeros(1, dtype=dtype))

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype


def _make_encoder(**overrides) -> Flux3TextEncoder:
    kwargs = dict(
        context_in_dim=CONTEXT_IN_DIM,
        vec_in_dim=VEC_IN_DIM,
        max_sequence_length=N_TOKENS,
        output_layer=TEST_OUTPUT_LAYER,
    )
    kwargs.update(overrides)
    return Flux3TextEncoder(_FakeQwen3(), processor=mock.MagicMock(), **kwargs)


def _layers(n_layers: int, *, hidden: int = QWEN_HIDDEN, batch: int = 1, seq: int = N_TOKENS):
    """A fake Qwen ``output.hidden_states`` tuple: ``n_layers`` × (batch, seq, hidden)."""
    return tuple(torch.randn(batch, seq, hidden) for _ in range(n_layers))


def _token_inputs(device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids = torch.arange(2 * N_TOKENS, device=device).reshape(2, N_TOKENS)
    attention_mask = torch.ones_like(input_ids)
    attention_mask[:, -2:] = 0
    position_ids = attention_mask.cumsum(dim=-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    return input_ids, attention_mask, position_ids


class TestMockHead:
    """Default path: (ctx, vector) built from real Qwen hidden states."""

    def test_build_outputs_shapes_and_dtype(self):
        enc = _make_encoder()
        ctx, vector = enc._build_outputs(_layers(5))
        assert ctx.shape == (1, N_TOKENS, CONTEXT_IN_DIM)
        assert vector.shape == (1, VEC_IN_DIM)
        assert ctx.dtype == torch.bfloat16
        assert vector.dtype == torch.bfloat16

    def test_vector_is_zero(self):
        enc = _make_encoder()
        _, vector = enc._build_outputs(_layers(5))
        assert torch.all(vector == 0)

    def test_vec_in_dim_none_returns_empty_vector(self):
        enc = _make_encoder(vec_in_dim=None)
        _, vector = enc._build_outputs(_layers(5))
        assert vector.shape == (1, 0)
        assert vector.dtype == torch.bfloat16

    def test_ctx_uses_fixed_output_layers(self):
        enc = _make_encoder()
        layers = _layers(5)  # indices 0–4, TEST_OUTPUT_LAYER = [0,1,2,3]
        ctx, _ = enc._build_outputs(layers)
        expected = torch.cat([layers[k] for k in TEST_OUTPUT_LAYER], dim=-1).to(torch.bfloat16)
        assert torch.equal(ctx, expected)

    def test_raises_when_layer_count_mismatches_context_dim(self):
        # 4 output layers × hidden 8 = 32 ≠ 30 → should raise
        enc = _make_encoder(context_in_dim=30)
        with pytest.raises(ValueError, match="context_in_dim"):
            enc._build_outputs(_layers(5))

    def test_raises_when_index_out_of_range(self):
        # output_layer=[0,1,2,3] but only 3 hidden-state tensors (indices 0–2)
        enc = _make_encoder()
        with pytest.raises(ValueError, match="out of range"):
            enc._build_outputs(_layers(3))

    def test_encode_routes_through_qwen_then_head(self):
        enc = _make_encoder()
        enc._encode_qwen3 = mock.MagicMock(return_value=(_layers(5), torch.ones(1, N_TOKENS)))
        ctx, vector = enc.encode("a prompt")
        enc._encode_qwen3.assert_called_once()
        assert ctx.shape == (1, N_TOKENS, CONTEXT_IN_DIM)
        assert vector.shape == (1, VEC_IN_DIM)
        assert torch.all(vector == 0)


class TestDefaultOutputLayer:
    """The default output_layer uses a fixed set of hidden-state indices."""

    def test_default_is_fixed_layers(self):
        assert _DEFAULT_OUTPUT_LAYER == [4, 8, 12, 16, 20, 24, 28, 32]
        assert DEFAULT_TEXT_TOKENS == 8192
        assert DEFAULT_PAD_MULTIPLE == 80

    def test_encoder_stores_output_layer(self):
        enc = _make_encoder()
        assert enc.output_layer == TEST_OUTPUT_LAYER

    def test_encoder_keeps_all_layers_until_weights_are_loaded(self):
        enc = _make_encoder()
        language_model = enc.text_encoder.model.language_model

        assert len(language_model.layers) == 5
        assert isinstance(language_model.norm, nn.LayerNorm)

    def test_finalize_after_loading_truncates_layers_and_removes_final_norm(self):
        enc = _make_encoder()
        language_model = enc.text_encoder.model.language_model

        enc.finalize_after_loading()

        assert len(language_model.layers) == max(TEST_OUTPUT_LAYER)
        assert isinstance(language_model.norm, nn.Identity)

    def test_finalize_after_loading_is_idempotent(self):
        enc = _make_encoder()

        enc.finalize_after_loading()
        enc.finalize_after_loading()

        language_model = enc.text_encoder.model.language_model
        assert len(language_model.layers) == max(TEST_OUTPUT_LAYER)
        assert isinstance(language_model.norm, nn.Identity)

    def test_truncated_encoder_preserves_selected_hidden_states(self):
        torch.manual_seed(0)
        enc = _make_encoder().eval()
        inputs = _token_inputs()

        full_hidden_states = enc._language_model_forward(*inputs)
        expected = enc._build_outputs(full_hidden_states)

        enc.finalize_after_loading()
        truncated_hidden_states = enc._language_model_forward(*inputs)
        actual = enc._build_outputs(truncated_hidden_states)

        for layer_idx in TEST_OUTPUT_LAYER:
            assert torch.equal(truncated_hidden_states[layer_idx], full_hidden_states[layer_idx])
        for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
            assert torch.equal(actual_tensor, expected_tensor)

    def test_custom_output_layer_is_respected(self):
        custom = [0, 2, 4]
        # 3 layers × 8 hidden = 24 → use context_in_dim=24
        enc = _make_encoder(context_in_dim=24, output_layer=custom)
        assert enc.output_layer == custom
        layers = _layers(6)
        ctx, _ = enc._build_outputs(layers)
        expected = torch.cat([layers[k] for k in custom], dim=-1).to(torch.bfloat16)
        assert torch.equal(ctx, expected)


class _PackingProcessor:
    def apply_chat_template(self, messages, **kwargs) -> str:
        return str(messages[0]["content"])

    @property
    def tokenizer(self):
        return self

    def __call__(self, text, return_tensors=None, padding=None, truncation=None, max_length=None):
        real_len = len(str(text).split())
        if truncation and max_length is not None:
            real_len = min(real_len, max_length)
        length = max_length if padding == "max_length" and max_length is not None else real_len
        input_ids = torch.arange(length, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.zeros(1, length, dtype=torch.long)
        attention_mask[:, :real_len] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class TestTextPacking:
    def test_encode_pads_batch_to_multiple_of_configured_window(self):
        enc = Flux3TextEncoder(
            _FakeQwen3(),
            processor=_PackingProcessor(),
            context_in_dim=CONTEXT_IN_DIM,
            vec_in_dim=VEC_IN_DIM,
            max_sequence_length=25,
            pad_multiple=8,
            output_layer=TEST_OUTPUT_LAYER,
        )

        hidden_states, attention_mask = enc._encode_qwen3(
            ["one two three", "one two three four five six seven eight nine"],
            max_sequence_length=25,
        )

        assert attention_mask.shape == (2, 16)
        assert hidden_states[0].shape == (2, 16, QWEN_HIDDEN)

    def test_encode_caps_padded_length_at_max_sequence_length(self):
        enc = Flux3TextEncoder(
            _FakeQwen3(),
            processor=_PackingProcessor(),
            context_in_dim=CONTEXT_IN_DIM,
            vec_in_dim=VEC_IN_DIM,
            max_sequence_length=10,
            pad_multiple=8,
            output_layer=TEST_OUTPUT_LAYER,
        )

        _hidden_states, attention_mask = enc._encode_qwen3(
            ["one two three four five six seven eight nine"],
            max_sequence_length=10,
        )

        assert attention_mask.shape == (1, 10)


class TestContract:
    def test_encode_requires_prompt(self):
        enc = _make_encoder()
        with pytest.raises(TypeError):
            enc.encode()  # type: ignore[call-arg]

    def test_forward_is_encode_alias(self):
        assert Flux3TextEncoder.forward is Flux3TextEncoder.encode

    def test_stores_context_and_vec_dims(self):
        enc = _make_encoder(context_in_dim=64, vec_in_dim=8)
        assert enc.context_in_dim == 64
        assert enc.vec_in_dim == 8


@hardware_test(res={"cuda": ["L4", "H100", "B200"]}, num_cards=1)
def test_compiled_language_model_matches_eager() -> None:
    torch.manual_seed(0)
    enc = _make_encoder().eval().to(device="cuda", dtype=torch.bfloat16)
    enc.finalize_after_loading()
    inputs = _token_inputs("cuda")

    expected = enc._language_model_forward(*inputs)
    assert all(tensor.dtype == torch.bfloat16 for tensor in expected)
    enc.setup_compile(dynamic=False)
    assert enc._compiled_language_model_forward is not None
    actual = enc._compiled_language_model_forward(*inputs)

    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor, atol=5e-3, rtol=5e-3)
