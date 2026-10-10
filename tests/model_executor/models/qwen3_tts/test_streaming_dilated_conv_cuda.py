# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Implicit-GEMM dilated unit convs and the tap-GEMM output conv match the direct paths across calls and slots."""

import pytest
import torch

from tests.helpers.mark import hardware_test

pytestmark = [pytest.mark.core_model, pytest.mark.tts]


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@torch.inference_mode()
def test_dilated_unit_conv_matches_patch_matrix_path():
    from vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import (
        Qwen3TTSTokenizerV2DecoderConfig,
    )
    from vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
        Qwen3TTSTokenizerV2Decoder,
    )
    from vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.streaming_decoder import (
        _DILATED_IGEMM,
        StreamingCodecDecoder,
    )

    # decoder_dim 768: unit widths 384, 192, 96 and 48, so every implicit-GEMM tile config runs.
    config = Qwen3TTSTokenizerV2DecoderConfig(
        codebook_size=32, hidden_size=16, latent_dim=16, codebook_dim=16, num_attention_heads=2,
        num_key_value_heads=2, intermediate_size=32, num_hidden_layers=1, num_quantizers=2, decoder_dim=768,
        upsample_rates=(8, 5, 4, 3), upsampling_ratios=(2, 2), sliding_window=72,
    )  # fmt: skip
    torch.manual_seed(0)
    decoder = Qwen3TTSTokenizerV2Decoder(config).eval()
    for name, param in decoder.named_parameters():
        if name.endswith(("alpha", "beta")):
            param.uniform_(-0.5, 0.5)
        elif name.endswith("gamma"):
            param.uniform_(0.5, 1.0)
        elif name.endswith("embedding_sum"):
            param.normal_()
    decoder.precompute_snake_caches()
    decoder = decoder.to(device="cuda", dtype=torch.bfloat16)
    decoder.config.head_dim = decoder.config.hidden_size // decoder.config.num_attention_heads
    stream = StreamingCodecDecoder(decoder, num_slots=17, dtype=torch.bfloat16)
    assert stream.dilated_igemm and stream.conv_out_gemm
    assert {blk["c_out"] for blk in stream.blocks} >= set(_DILATED_IGEMM)

    batch, frames = 16, 5
    codes = torch.randint(0, 32, (batch, frames + 3, 2), device="cuda", dtype=torch.int32)
    # Reversed, non-contiguous slots: rows of one tile belong to different requests.
    slots = torch.arange(batch, 0, -1, device="cuda", dtype=torch.int32)

    def decode(use_igemm: bool, use_out_gemm: bool) -> torch.Tensor:
        stream.dilated_igemm = use_igemm
        stream.conv_out_gemm = use_out_gemm
        out = []
        for t in range(frames):
            pos = torch.full((batch,), t, device="cuda", dtype=torch.int32)
            out.append(stream(codes[:, t : t + 1].contiguous(), slots, pos).clone())
        # A multi-frame call continues the same streams (history across calls).
        pos = torch.full((batch,), frames, device="cuda", dtype=torch.int32)
        out.append(stream(codes[:, frames:].contiguous(), slots, pos).clone())
        return torch.cat(out, dim=1)

    expected = decode(False, False)
    for flags in ((True, False), (False, True), (True, True)):
        actual = decode(*flags)
        assert actual.abs().max() > 0
        torch.testing.assert_close(actual, expected, rtol=0, atol=4e-3, msg=lambda m, flags=flags: f"{flags}: {m}")
