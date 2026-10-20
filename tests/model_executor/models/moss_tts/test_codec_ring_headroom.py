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

from tests.helpers.mark import hardware_test
from vllm_omni.model_executor.models.moss_tts.audio_tokenizer_v2 import MossAudioTokenizerModel
from vllm_omni.model_executor.models.moss_tts.configuration_moss_audio_tokenizer_v2 import (
    MossAudioTokenizerConfig,
)

pytestmark = [pytest.mark.core_model]

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


@pytest.mark.cpu
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


@pytest.mark.cpu
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


@pytest.mark.cpu
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


@pytest.mark.cpu
@torch.no_grad()
def test_packed_pool_preserves_distinct_scratch_slots_and_decode_state():
    """Packed padding owns separate slots; it must not use main's single null slot."""
    shared, packed = tiny_codec(), tiny_codec()
    shared.initialize_decoder_state_pool(2, 3)
    packed.initialize_decoder_state_pool(2, 3, packed=True)
    try:
        assert shared._decoder_state_capacity == 3
        assert packed._decoder_state_capacity == 5
        assert packed._decoder_null_slot is None
        slots = torch.tensor([1, 0])
        valid = torch.tensor([True, True])
        for _ in range(3):
            codes = torch.randint(0, 16, (2, 2, 2))
            lengths = torch.tensor([2, 2])
            expected = shared.decode_streaming_batch(codes, lengths, slots, valid).audio
            actual = packed.decode_streaming_batch(codes, lengths, slots, valid).audio
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        packed._decoder_slot_offsets[:, 2:].fill_(9)
        packed.reset_decoder_state_slots(torch.tensor([2, 3, 4]))
        assert packed._decoder_slot_offsets[:, 2:].count_nonzero() == 0
        assert packed._decoder_slot_offsets[:, :2].count_nonzero() > 0
    finally:
        shared.close_decoder_state_pool()
        packed.close_decoder_state_pool()


@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards=1)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@torch.no_grad()
def test_packed_session_matches_mixed_length_streams_and_slot_reuse():
    import copy

    from vllm.config import CompilationConfig, VllmConfig

    from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_codec import _MossCodecStreamSession
    from vllm_omni.model_executor.models.moss_tts.slot_attention import (
        slot_ring_flash_attention,
        slot_ring_flash_attention_rows,
    )

    config = tiny_codec().config
    for entry in config.decoder_kwargs:
        if entry.get("module_type") == "Transformer":
            entry.update(d_model=64, num_heads=1, dim_feedforward=128)
    reference = MossAudioTokenizerModel(config).eval().cuda()
    reference.decoder.to(torch.bfloat16)
    reference.quantizer.build_decode_lut(2, dtype=torch.bfloat16)
    candidate = copy.deepcopy(reference)
    for codec in (reference, candidate):
        for module in codec.decoder.modules():
            if hasattr(module, "_slot_attention"):
                module._slot_attention = slot_ring_flash_attention
                module._slot_attention_rows = slot_ring_flash_attention_rows
    vllm_config = VllmConfig(compilation_config=CompilationConfig(mode=0))
    kwargs = dict(
        state_capacity=2,
        n_vq=2,
        gpu_output=True,
        chunk_frames=3,
        graph_batch_sizes=[1, 2],
        graph_frame_sizes=[1, 2, 3],
        vllm_config=vllm_config,
    )
    plain = _MossCodecStreamSession(reference, **kwargs)
    packed = _MossCodecStreamSession(candidate, **kwargs, packed=True)
    slots = [packed.acquire(), packed.acquire()]
    assert slots == [plain.acquire(), plain.acquire()]
    saved = []
    for lengths, terminal in [([1, 3], False), ([2, 1], False), ([3, 2], True)]:
        codes = [torch.randint(0, 16, (2, n), device="cuda") for n in lengths]
        actual = packed.step_packed([(slot, c, terminal) for slot, c in zip(slots, codes)])
        for slot, c in zip(slots, codes):
            expected = plain.step({slot: c}, terminal_slots={slot} if terminal else set())[slot]
            torch.testing.assert_close(actual[slot], expected, rtol=0.03, atol=0.01)
            saved.append((actual[slot], actual[slot].clone()))
    for session in (plain, packed):
        for slot in slots:
            session.release(slot)
    slot = packed.acquire()
    assert slot == plain.acquire()
    codes = torch.randint(0, 16, (2, 2), device="cuda")
    actual = packed.step_packed([(slot, codes, True)])[slot]
    expected = plain.step({slot: codes}, terminal_slots={slot})[slot]
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.01)
    for audio, snapshot in saved:
        torch.testing.assert_close(audio, snapshot, rtol=0, atol=0)


@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards=1)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@torch.no_grad()
def test_packed_decoder_splits_long_chunks_into_ordered_rounds():
    import copy

    from vllm.config import CompilationConfig, VllmConfig

    from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_codec import (
        MossTTSCodecDecoder,
        _MossCodecStreamSession,
    )
    from vllm_omni.model_executor.models.moss_tts.slot_attention import (
        slot_ring_flash_attention,
        slot_ring_flash_attention_rows,
    )

    config = tiny_codec().config
    for entry in config.decoder_kwargs:
        if entry.get("module_type") == "Transformer":
            entry.update(d_model=64, num_heads=1, dim_feedforward=128)
    reference = MossAudioTokenizerModel(config).eval().cuda()
    reference.decoder.to(torch.bfloat16)
    reference.quantizer.build_decode_lut(2, dtype=torch.bfloat16)
    candidate = copy.deepcopy(reference)
    for codec in (reference, candidate):
        for module in codec.decoder.modules():
            if hasattr(module, "_slot_attention"):
                module._slot_attention = slot_ring_flash_attention
                module._slot_attention_rows = slot_ring_flash_attention_rows
    kwargs = dict(
        state_capacity=2,
        n_vq=2,
        gpu_output=True,
        chunk_frames=3,
        graph_batch_sizes=[1, 2],
        graph_frame_sizes=[1, 2, 3],
        vllm_config=VllmConfig(compilation_config=CompilationConfig(mode=0)),
    )
    plain = _MossCodecStreamSession(reference, **kwargs)
    packed = _MossCodecStreamSession(candidate, **kwargs, packed=True)
    assert packed._cudagraph_wrapper is None
    decoder = object.__new__(MossTTSCodecDecoder)
    torch.nn.Module.__init__(decoder)
    decoder._codec = candidate
    decoder._stream_session = packed
    decoder._stream_req_slots = {}
    decoder._stream_state_capacity = 2
    decoder._stream_max_step_frames = 3
    long, tail = torch.randint(0, 16, (2, 7), device="cuda"), torch.randint(0, 16, (2, 2), device="cuda")
    actual = decoder._decode_streaming_batch([(0, "long", long, False), (1, "tail", tail, True)])
    # The 7-frame chunk decodes as 3 + 3 + 1 frames, in order, next to the terminal tail.
    a, b = plain.acquire(), plain.acquire()
    expected_long = torch.cat([plain.step({a: long[:, s : s + 3]})[a] for s in (0, 3, 6)], dim=-1)
    expected_tail = plain.step({b: tail}, terminal_slots={b})[b]
    torch.testing.assert_close(actual[0], expected_long, rtol=0.03, atol=0.01)
    torch.testing.assert_close(actual[1], expected_tail, rtol=0.03, atol=0.01)
    assert set(decoder._stream_req_slots) == {"long"}


@hardware_test(res={"cuda": "L4", "rocm": "MI325"}, num_cards=1)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@torch.no_grad()
def test_one_frame_one_batch_retains_ordinary_decode():
    from vllm.config import CompilationConfig, VllmConfig

    from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_codec import (
        MossTTSCodecDecoder,
        _MossCodecStreamSession,
    )

    codec = tiny_codec().cuda()
    codec.decoder.to(torch.bfloat16)
    codec.quantizer.build_decode_lut(2, dtype=torch.bfloat16)
    session = _MossCodecStreamSession(
        codec,
        state_capacity=1,
        n_vq=2,
        chunk_frames=1,
        packed=True,
        graph_batch_sizes=[1],
        graph_frame_sizes=[1],
        gpu_output=True,
        vllm_config=VllmConfig(compilation_config=CompilationConfig(mode=0)),
    )
    decoder = object.__new__(MossTTSCodecDecoder)
    torch.nn.Module.__init__(decoder)
    decoder._codec = codec
    decoder._stream_session = session
    decoder._stream_req_slots = {}
    decoder._stream_state_capacity = 1
    decoder._stream_max_step_frames = 1
    codes = torch.randint(0, 16, (2, 1), device="cuda")
    output = decoder._decode_streaming_batch([(0, "request", codes, False)])[0]
    assert output.shape == (1, 4) and torch.isfinite(output).all()
