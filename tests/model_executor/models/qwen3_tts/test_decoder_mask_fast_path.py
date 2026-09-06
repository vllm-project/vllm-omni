from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz import (
    modeling_qwen3_tts_tokenizer_v2 as decoder_module,
)
from vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2DecoderTransformerModel,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _RotaryStub(nn.Module):
    def forward(self, hidden_states, position_ids):
        return hidden_states, position_ids


def _make_decoder_transformer_stub():
    model = object.__new__(Qwen3TTSTokenizerV2DecoderTransformerModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        num_hidden_layers=0,
        max_position_embeddings=32,
        _attn_implementation="sdpa",
    )
    model.input_proj = nn.Identity()
    model.output_proj = nn.Identity()
    model.norm = nn.Identity()
    model.rotary_emb = _RotaryStub()
    model.layers = nn.ModuleList()
    model._sliding_attention_mask_cache = {}
    return model


def _install_mask_builder(monkeypatch: pytest.MonkeyPatch, calls: list[torch.Tensor]) -> None:
    def fake_create_sliding_window_causal_mask(
        *,
        config,
        attention_mask,
        past_key_values,
        position_ids,
        inputs_embeds,
    ):
        del config, attention_mask, past_key_values, position_ids
        calls.append(inputs_embeds)
        return torch.ones(
            1,
            1,
            inputs_embeds.shape[1],
            inputs_embeds.shape[1],
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )

    monkeypatch.setattr(
        decoder_module,
        "create_sliding_window_causal_mask",
        fake_create_sliding_window_causal_mask,
    )


def test_auto_position_ids_reuse_sliding_mask_cache(monkeypatch: pytest.MonkeyPatch):
    model = _make_decoder_transformer_stub()
    calls: list[torch.Tensor] = []
    _install_mask_builder(monkeypatch, calls)

    model(inputs_embeds=torch.randn(2, 7, 4))
    model(inputs_embeds=torch.randn(3, 7, 4))

    assert len(calls) == 1
    assert calls[0].shape == (1, 7, 4)
    assert len(model._sliding_attention_mask_cache) == 1


def test_explicit_non_contiguous_position_ids_fail_closed():
    model = _make_decoder_transformer_stub()
    position_ids = torch.tensor([[0, 1, 0, 1, 2, 3, 4]])

    with pytest.raises(ValueError, match="contiguous zero-based position_ids"):
        model(inputs_embeds=torch.randn(1, 7, 4), position_ids=position_ids)


def test_sliding_mask_cache_key_tracks_shape_dtype_and_attention_implementation(
    monkeypatch: pytest.MonkeyPatch,
):
    model = _make_decoder_transformer_stub()
    calls: list[torch.Tensor] = []
    _install_mask_builder(monkeypatch, calls)

    model._get_sliding_attention_mask(torch.randn(1, 4, 8, dtype=torch.float32))
    model._get_sliding_attention_mask(torch.randn(1, 5, 8, dtype=torch.float32))
    model._get_sliding_attention_mask(torch.randn(1, 5, 8, dtype=torch.float64))
    model.config._attn_implementation = "flash_attention_2"
    model._get_sliding_attention_mask(torch.randn(1, 5, 8, dtype=torch.float64))

    assert len(calls) == 4
    assert len(model._sliding_attention_mask_cache) == 4


def test_apply_clears_sliding_mask_cache():
    model = _make_decoder_transformer_stub()
    model._sliding_attention_mask_cache[(4, torch.float32, torch.device("cpu"), "sdpa")] = torch.ones(1)

    model._apply(lambda tensor: tensor)

    assert model._sliding_attention_mask_cache == {}
