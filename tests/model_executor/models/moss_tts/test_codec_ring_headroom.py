# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Chunked streaming decode must match whole-sequence decode.

``RingKVCache.complete`` writes a whole chunk before attending. With a ring of
exactly ``context`` entries, a chunk of T tokens evicts up to T keys that its
own queries should still see; for T > context it evicts part of the chunk
itself. ``initialize_decoder_state_pool(chunk_frames=...)`` adds one chunk of
headroom per layer so streaming reproduces the non-streaming decoder.
"""

import pytest
import torch

from vllm_omni.model_executor.models.moss_tts.audio_tokenizer_v2 import MossAudioTokenizerModel
from vllm_omni.model_executor.models.moss_tts.configuration_moss_audio_tokenizer_v2 import (
    MossAudioTokenizerConfig,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

TRANSFORMER = {
    "module_type": "Transformer",
    "d_model": 16,
    "num_heads": 2,
    "num_layers": 1,
    "dim_feedforward": 32,
    "causal": True,
    "norm": "layer_norm",
    "positional_embedding": "rope",
    "max_period": 10000,
    "gating": "none",
    "layer_scale": 1.0,
    "conv_layout": True,
}


def tiny_codec() -> MossAudioTokenizerModel:
    # 64 Hz mono input, 4-sample code frames (16 code frames per second).
    # Decoder: transformer @16 Hz (context 0.5 s = 8 tokens) -> x2 -> transformer
    # @32 Hz (context 0.25 s = 8 tokens) -> x2 -> 64 Hz. A 6-frame chunk is 12
    # tokens at the second transformer, more than its 8-token ring.
    config = MossAudioTokenizerConfig(
        sampling_rate=64,
        downsample_rate=4,
        number_channels=1,
        enable_channel_interleave=False,
        encoder_kwargs=[
            {"module_type": "PatchedPretransform", "patch_size": 4},
            {**TRANSFORMER, "input_dimension": 4, "output_dimension": 8, "context_duration": 0.5},
        ],
        decoder_kwargs=[
            {**TRANSFORMER, "input_dimension": 8, "output_dimension": 16, "context_duration": 0.5},
            {"module_type": "PatchedPretransform", "patch_size": 2},
            {**TRANSFORMER, "input_dimension": 8, "output_dimension": 2, "context_duration": 0.25},
            {"module_type": "PatchedPretransform", "patch_size": 2},
        ],
        quantizer_kwargs={
            "input_dim": 8,
            "rvq_dim": 8,
            "output_dim": 8,
            "num_quantizers": 2,
            "codebook_size": 16,
            "codebook_dim": 4,
            "quantizer_type": "rlfq",
        },
    )
    torch.manual_seed(0)
    model = MossAudioTokenizerModel(config).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.5)
    model.quantizer.build_decode_lut(2, dtype=torch.float32)
    return model


def stream_decode(model, codes, chunk_frames, *, headroom):
    n_q, _, total = codes.shape
    model.initialize_decoder_state_pool(1, 0, chunk_frames=chunk_frames if headroom else 0)
    try:
        slot = torch.zeros(1, dtype=torch.long)
        valid = torch.ones(1, dtype=torch.bool)
        parts = []
        for start in range(0, total, chunk_frames):
            chunk = codes[:, :, start : start + chunk_frames]
            lengths = torch.full((1,), chunk.shape[-1], dtype=torch.long)
            parts.append(model.decode_streaming_batch(chunk, lengths, slot, valid).audio)
        return torch.cat(parts, dim=-1)
    finally:
        model.close_decoder_state_pool()


@torch.no_grad()
def test_ring_headroom_sizes_each_layer_for_one_chunk():
    model = tiny_codec()
    transformers = [m for m in model.decoder if hasattr(m, "tokens_per_input_frame")]
    assert [m.tokens_per_input_frame for m in transformers] == [1, 2]
    assert [m.transformer.layers[0].self_attn.context for m in transformers] == [8, 8]
    model.initialize_decoder_state_pool(1, 0, chunk_frames=6)
    try:
        rings = [m.transformer.layers[0].self_attn._streaming_state.kv_cache.capacity for m in transformers]
        assert rings == [8 + 6, 8 + 12]
    finally:
        model.close_decoder_state_pool()
    model.initialize_decoder_state_pool(1, 0)
    try:
        rings = [m.transformer.layers[0].self_attn._streaming_state.kv_cache.capacity for m in transformers]
        assert rings == [8, 8]
    finally:
        model.close_decoder_state_pool()


@torch.no_grad()
def test_chunked_streaming_matches_whole_sequence_only_with_headroom():
    model = tiny_codec()
    torch.manual_seed(1)
    codes = torch.randint(0, 16, (2, 1, 18))
    reference = model._decode_frame(codes).audio
    with_headroom = stream_decode(model, codes, 6, headroom=True)
    legacy = stream_decode(model, codes, 6, headroom=False)
    assert with_headroom.shape == reference.shape == legacy.shape
    torch.testing.assert_close(with_headroom, reference, rtol=1e-5, atol=1e-5)
    # The legacy context-sized ring lets the 12-token chunk evict keys inside
    # its own window, so streaming diverges from whole-sequence decoding.
    assert not torch.allclose(legacy, reference, rtol=1e-3, atol=1e-3)


@torch.no_grad()
def test_padded_terminal_tail_matches_exact_tail_with_headroom():
    model = tiny_codec()
    torch.manual_seed(2)
    codes = torch.randint(0, 16, (2, 1, 14))
    history, tail = codes[:, :, :12], codes[:, :, 12:]
    padded_tail = torch.nn.functional.pad(tail, (0, 4))  # 2 real frames padded to a 6-frame step
    outputs = {}
    for name, last in [("exact", tail), ("padded", padded_tail)]:
        model.initialize_decoder_state_pool(1, 0, chunk_frames=6)
        try:
            slot, valid = torch.zeros(1, dtype=torch.long), torch.ones(1, dtype=torch.bool)
            for chunk in (history[:, :, :6], history[:, :, 6:]):
                model.decode_streaming_batch(chunk, torch.tensor([6]), slot, valid)
            audio = model.decode_streaming_batch(last, torch.tensor([last.shape[-1]]), slot, valid).audio
            outputs[name] = audio[..., : 2 * model.downsample_rate]
        finally:
            model.close_decoder_state_pool()
    torch.testing.assert_close(outputs["padded"], outputs["exact"], rtol=1e-5, atol=1e-5)
