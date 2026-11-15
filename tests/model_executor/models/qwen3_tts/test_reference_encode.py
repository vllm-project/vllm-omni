# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Mel constants keep FP32 state independent of the caller's torch defaults."""

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_mel_caches_stay_fp32_when_first_built_under_bf16_default():
    from vllm_omni.model_executor.models.qwen3_tts import prompt_embeds_builder as builder

    builder._cached_mel_filter_bank.cache_clear()
    builder._cached_hann_window.cache_clear()
    default_dtype = torch.get_default_dtype()
    try:
        # Model loading runs under the model dtype; a first call there must not
        # cache BF16 constants that later FP32 requests multiply against.
        torch.set_default_dtype(torch.bfloat16)
        assert builder._cached_mel_filter_bank(24000, 1024, 128, 0, 12000).dtype == torch.float32
        assert builder._cached_hann_window(1024).dtype == torch.float32
    finally:
        torch.set_default_dtype(default_dtype)
    wav = torch.randn(1, 24000)
    mels = builder.mel_spectrogram(
        wav, n_fft=1024, num_mels=128, sampling_rate=24000, hop_size=256, win_size=1024, fmin=0, fmax=12000
    )
    assert mels.dtype == torch.float32 and mels.shape[1] == 128


def test_reference_encoder_loads_flat_mimi_checkpoint_and_encodes():
    from transformers import MimiConfig
    from vllm.model_executor.models.utils import AutoWeightsLoader

    from vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
        Qwen3TTSTokenizerV2Encoder,
    )

    config = MimiConfig(
        hidden_size=16,
        num_filters=4,
        upsampling_ratios=[2, 2],
        upsample_groups=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=32,
        codebook_size=8,
        codebook_dim=8,
        vector_quantization_hidden_dimension=8,
        num_quantizers=4,
        sampling_rate=160,
        frame_rate=20,
    )
    config._attn_implementation = "eager"
    encoder = Qwen3TTSTokenizerV2Encoder._from_config(config).eval()
    assert config.decoder_config is not None
    assert encoder.decoder is None
    holder = torch.nn.Module()
    holder.encoder = encoder
    # These are the original speech_tokenizer/ checkpoint's namespaces,
    # including a persistent VQ buffer rather than only model parameters.
    weights = [
        (
            "encoder.encoder.layers.0.conv.weight",
            torch.full_like(encoder.encoder.encoder_seanet.layers[0].conv.weight, 0.125),
        ),
        ("encoder.encoder_transformer.layers.0.input_layernorm.weight", torch.full((16,), 0.75)),
        ("encoder.downsample.conv.weight", torch.full_like(encoder.encoder.downsample.conv.weight, 0.25)),
        ("encoder.quantizer.semantic_residual_vector_quantizer.layers.0.codebook.embed_sum", torch.ones(8, 8)),
    ]
    loaded = AutoWeightsLoader(holder).load_weights(weights)
    assert len(loaded) == len(weights)
    assert torch.all(encoder.encoder.encoder_seanet.layers[0].conv.weight == 0.125)
    assert torch.all(encoder.encoder.encoder_transformer.layers[0].input_layernorm.weight == 0.75)
    assert torch.all(encoder.encoder.downsample.conv.weight == 0.25)
    assert torch.all(encoder.quantizer.semantic_residual_vector_quantizer.layers[0].codebook.embed_sum == 1)
    with torch.inference_mode():
        codes = encoder.encode(torch.zeros(1, 1, 160), num_quantizers=2, return_dict=True).audio_codes
    assert codes.shape == (1, 2, 20)
    assert torch.all((codes >= 0) & (codes < 8))
