# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2DecoderConfig,
)
from vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2Decoder,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_decoder() -> Qwen3TTSTokenizerV2Decoder:
    config = Qwen3TTSTokenizerV2DecoderConfig(
        codebook_size=32,
        hidden_size=16,
        latent_dim=16,
        codebook_dim=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=32,
        num_hidden_layers=1,
        num_quantizers=2,
        decoder_dim=32,
        upsample_rates=(8, 5, 4, 3),
        upsampling_ratios=(2, 2),
        sliding_window=72,
    )
    torch.manual_seed(0)
    decoder = Qwen3TTSTokenizerV2Decoder(config).eval()
    with torch.no_grad():
        for name, param in decoder.named_parameters():
            # Non-trivial SnakeBeta and ConvNeXt scales so every term matters.
            if name.endswith(("alpha", "beta")):
                param.uniform_(-0.5, 0.5)
            elif name.endswith("gamma"):
                param.uniform_(0.5, 1.0)
    decoder.precompute_snake_caches()
    return decoder


@torch.inference_mode()
def test_time_major_conv_decode_matches_nct_stack() -> None:
    decoder = _make_decoder()
    hidden = torch.randn(2, 5, decoder.config.latent_dim)
    expected = decoder._conv_decode(hidden)

    decoder.enable_time_major_conv()
    actual = decoder._conv_decode(hidden)

    assert actual.shape == expected.shape == (2, 1, 5 * int(decoder.total_upsample))
    scale = expected.abs().max()
    torch.testing.assert_close(actual / scale, expected / scale, rtol=0, atol=1e-5)
    # The stack holds weight copies outside the module tree.
    assert not any("time_major" in key for key in decoder.state_dict())


@torch.inference_mode()
def test_time_major_conv_decode_takes_sliced_hidden() -> None:
    decoder = _make_decoder()
    hidden = torch.randn(1, 9, decoder.config.latent_dim)
    expected = decoder._conv_decode(hidden[:, -4:, :])
    decoder.enable_time_major_conv()
    actual = decoder._conv_decode(hidden[:, -4:, :])  # non-contiguous, like the suffix decode's slice
    scale = expected.abs().max()
    torch.testing.assert_close(actual / scale, expected / scale, rtol=0, atol=1e-5)


@torch.inference_mode()
def test_first_frame_single_key_attention_matches_codec_without_graphs():
    from vllm_omni.model_executor.models.qwen3_tts.first_frame_decoder import Qwen3TTSFirstFrameDecoder

    decoder = _make_decoder()
    codes = torch.randint(0, 32, (3, 2, 1))
    expected = decoder._decode_xvec_first_chunk(codes, {})[:, 0, :]
    first = Qwen3TTSFirstFrameDecoder.__new__(Qwen3TTSFirstFrameDecoder)
    torch.nn.Module.__init__(first)
    first.decoder = decoder
    first._graphs = {}
    actual = first.decode(codes[:, :, 0])
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
