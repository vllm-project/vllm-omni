# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect

import pytest
import torch
from transformers.masking_utils import create_sliding_window_causal_mask

from vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2DecoderConfig,
)
from vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2DecoderTransformerModel,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_config(attention_implementation: str) -> Qwen3TTSTokenizerV2DecoderConfig:
    config = Qwen3TTSTokenizerV2DecoderConfig(
        hidden_size=32,
        latent_dim=16,
        max_position_embeddings=512,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=64,
        num_hidden_layers=1,
        num_quantizers=2,
        decoder_dim=32,
        upsample_rates=(2,),
        upsampling_ratios=(2,),
    )
    config._attn_implementation = attention_implementation
    return config


def _make_model(
    attention_implementation: str,
    dtype: torch.dtype = torch.float32,
) -> Qwen3TTSTokenizerV2DecoderTransformerModel:
    model = Qwen3TTSTokenizerV2DecoderTransformerModel._from_config(_make_config(attention_implementation))
    return model.to(dtype=dtype).eval()


def _create_reference_mask(
    model: Qwen3TTSTokenizerV2DecoderTransformerModel,
    inputs_embeds: torch.Tensor,
) -> torch.Tensor | None:
    sequence_length = inputs_embeds.shape[1]
    mask_kwargs = {
        "config": model.config,
        "attention_mask": None,
        "past_key_values": None,
        "position_ids": None,
    }
    sig = inspect.signature(create_sliding_window_causal_mask)
    if "input_embeds" in sig.parameters:
        mask_kwargs["input_embeds"] = inputs_embeds[:1]
        mask_kwargs["cache_position"] = torch.arange(sequence_length)
    else:
        mask_kwargs["inputs_embeds"] = inputs_embeds[:1]
    return create_sliding_window_causal_mask(**mask_kwargs)


@pytest.mark.parametrize("sequence_length", [17, 72, 73, 169, 325])
@pytest.mark.parametrize(
    ("attention_implementation", "dtype"),
    [("sdpa", torch.float32), ("eager", torch.float32), ("eager", torch.float16)],
)
def test_cached_sliding_mask_matches_transformers(
    sequence_length: int,
    attention_implementation: str,
    dtype: torch.dtype,
):
    model = _make_model(attention_implementation, dtype)
    inputs_embeds = torch.empty(2, sequence_length, model.config.hidden_size, dtype=dtype)

    actual = model._get_sliding_attention_mask(inputs_embeds)
    expected = _create_reference_mask(model, inputs_embeds)

    if expected is None:
        assert actual is None
        assert model._get_sliding_attention_mask(inputs_embeds) is None
        return

    assert actual is not None
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert actual.shape == (1, 1, sequence_length, sequence_length)
    assert actual.is_contiguous()
    assert model._get_sliding_attention_mask(inputs_embeds) is actual


def test_cached_eager_mask_tracks_compute_dtype():
    model = _make_model("eager")

    fp32_inputs = torch.empty(1, 73, model.config.hidden_size)
    fp32_mask = model._get_sliding_attention_mask(fp32_inputs)
    assert fp32_mask is not None
    assert fp32_mask.dtype == torch.float32

    model.half()
    fp16_inputs = fp32_inputs.half()
    fp16_mask = model._get_sliding_attention_mask(fp16_inputs)
    assert fp16_mask is not None
    assert fp16_mask.dtype == torch.float16
    assert torch.isfinite(fp16_mask).all()
    assert fp16_mask.min() == torch.finfo(torch.float16).min


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"use_cache": True}, "use_cache=True"),
        ({"past_key_values": object()}, "past_key_values"),
        ({"cache_position": torch.arange(1, 18)}, "zero-based cache_position"),
        ({"position_ids": torch.arange(1, 18).unsqueeze(0)}, "zero-based position_ids"),
    ],
)
def test_cached_mask_rejects_unsupported_cache_and_positions(kwargs: dict, message: str):
    model = _make_model("sdpa")
    inputs_embeds = torch.randn(1, 17, model.config.latent_dim)

    with pytest.raises(ValueError, match=message):
        model(inputs_embeds=inputs_embeds, **kwargs)
