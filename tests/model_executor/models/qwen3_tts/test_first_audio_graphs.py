# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Prefix graphs must follow per-request first-audio delivery, not a capture hint."""

import math
from types import SimpleNamespace

import pytest
import torch

from tests.helpers.mark import hardware_test
from tests.model_executor.models.qwen3_tts.test_qwen3_tts_incremental_decode import _decoder_stub
from vllm_omni.model_executor.models.qwen3_tts.segmented_graph_wrapper import CUDAGraphDecoderWrapper
from vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2Decoder,
)

pytestmark = [pytest.mark.core_model]
DEVICE = torch.device("cuda:0")


@pytest.mark.cpu
@pytest.mark.parametrize("capacity,model_type", [(1, "base"), (128, "base"), (128, "custom_voice")])
def test_stream_priming_graph_workspace_respects_token_budget(monkeypatch, capacity, model_type):
    from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker import Qwen3TTSTalkerForConditionalGeneration
    from vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz import streaming_decoder

    captures = []

    def capture(stream, sizes, frames=1, pool=None):
        captures.append((sizes, frames))
        return SimpleNamespace(sizes=sizes, frames=frames, pool=pool)

    monkeypatch.setattr(streaming_decoder, "StreamingDecodeGraphs", capture)
    model = SimpleNamespace(
        stream_decoder=SimpleNamespace(max_frames=33),
        stream_graphs=None,
        stream_prime_graphs=None,
        stream_prime_pieces={},
        stream_ref_context_frames=25,
        stream_chunk_frames=25,
        config=SimpleNamespace(tts_model_type=model_type),
        vllm_config=SimpleNamespace(scheduler_config=SimpleNamespace(max_num_batched_tokens=512)),
    )
    sizes = [size for size in [1, 2, 4, 8, 16, 32, 64, 128] if size <= capacity]
    Qwen3TTSTalkerForConditionalGeneration.capture_stream_decode_graphs(model, sizes)
    expected = [(sizes, 1)]
    if model_type == "base":
        # The priming chunk plus the power-of-two pieces of a shorter tail.
        for frames in (25, 16, 8, 4, 2):
            expected.append((sorted({1, *(size for size in sizes if size * frames <= 512)}), frames))
    assert captures == expected
    Qwen3TTSTalkerForConditionalGeneration.capture_stream_decode_graphs(model, sizes)
    assert captures == expected


@pytest.mark.cpu
@pytest.mark.parametrize("skip_flags", [(False, False), (True, True), (False, True), (True, False)])
@pytest.mark.parametrize("initial_frames", [1, 3])
@pytest.mark.parametrize("state_only_available", [False, True])
def test_xvec_first_chunk_replays_prefix_graph(monkeypatch, skip_flags, initial_frames, state_only_available):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    wrapper = CUDAGraphDecoderWrapper.__new__(CUDAGraphDecoderWrapper)
    wrapper.prefix_length = 72
    wrapper.initial_chunk_frames = initial_frames
    wrapper.codec_chunk_frames = 25
    wrapper.capture_batch_sizes = [1, 2]
    wrapper._icl_previous_frames_by_target = {26: 1}
    wrapper._xvec_previous_frames_by_target = {initial_frames + 25: initial_frames}
    replayed = []
    wrapper.xvec_prefix_states = {
        2: {
            "graph": SimpleNamespace(replay=lambda: replayed.append("full")),
            "input": {"codes": torch.zeros(2, 2, initial_frames)},
            "output": torch.arange(4 * initial_frames, dtype=torch.float32).view(2, 1, 2 * initial_frames),
            "cache": {
                "ref_hidden": torch.zeros(2, 2, 0),
                "ref_conv": torch.zeros(2, 0, 3),
                "prefix_hidden": torch.zeros(2, 0, 3),
                "ref_upsample": torch.zeros(2, 3, 0),
                "ref_wav": torch.zeros(2, 1, 0),
                "suffix_quantized": torch.ones(2, 2, initial_frames),
                "suffix_conv": torch.ones(2, initial_frames, 3),
                "past_key_values": SimpleNamespace(layers=[]),
            },
        }
    }
    wrapper.xvec_prefix_state_only_states = {}
    if state_only_available and initial_frames == 1:
        wrapper.xvec_prefix_state_only_states[2] = {
            **wrapper.xvec_prefix_states[2],
            "graph": SimpleNamespace(replay=lambda: replayed.append("state")),
            "state_only": True,
        }
    wrapper._record_graph_hit = lambda *_args: None
    wrapper._record_graph_fallback = lambda *_args: None
    wrapper._ensure_suffix_buffers = lambda cache: None
    wrapper.decoder = _decoder_stub(
        capture_first_audio_state_only=True,
        total_upsample=2,
        _is_suffix_cache_rolling=lambda previous, cached: False,
        _decode_xvec_first_chunk=lambda *_args: pytest.fail("unexpected eager fallback"),
        _slice_dynamic_cache=Qwen3TTSTokenizerV2Decoder._slice_dynamic_cache,
    )

    caches = [{"prefix_frames": 0, "skip_first_audio": skip} for skip in skip_flags]
    outputs = wrapper._batched_request_decode(
        [torch.full((1, 2, initial_frames), 3), torch.full((1, 2, initial_frames), 4)],
        caches,
    )

    assert replayed == ["state" if all(skip_flags) and state_only_available and initial_frames == 1 else "full"]
    torch.testing.assert_close(wrapper.xvec_prefix_states[2]["input"]["codes"][0], torch.full((2, initial_frames), 3.0))
    torch.testing.assert_close(wrapper.xvec_prefix_states[2]["input"]["codes"][1], torch.full((2, initial_frames), 4.0))
    assert [cache["decoder_prefix_frames"] for cache in caches] == [0, 0]
    assert [cache["suffix_frames"] for cache in caches] == [initial_frames, initial_frames]
    assert len(outputs) == 2
    for row, skip in enumerate(skip_flags):
        expected = wrapper.xvec_prefix_states[2]["output"][row : row + 1, :, 2 if skip else 0 :]
        torch.testing.assert_close(outputs[row], expected)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("initial_frames", [1, 3])
@pytest.mark.parametrize("time_major", [False, True])
@torch.inference_mode()
def test_xvec_graphs_follow_request_first_audio_flags(initial_frames, time_major):
    from tests.model_executor.models.qwen3_tts.test_time_major_decoder import _make_decoder
    from vllm_omni.model_executor.models.qwen3_tts.segmented_graph_wrapper import (
        CUDAGraphDecoderWrapper as SegmentedWrapper,
    )

    decoder = _make_decoder().to(DEVICE)
    decoder.precompute_snake_caches()
    if time_major:
        decoder.enable_time_major_conv()
    decoder.capture_first_audio_state_only = True
    wrapper = SegmentedWrapper(
        decoder,
        capture_modes=("xvec",),
        capture_batch_sizes=[1, 2],
        num_quantizers=2,
        initial_chunk_frames=initial_frames,
    )
    # This regression covers prefix graphs. Avoid compiling unrelated suffix
    # shapes, especially the time-major kernels for long continuation chunks.
    wrapper._xvec_previous_frames_by_target = {}
    wrapper.warmup(DEVICE)
    assert set(wrapper.xvec_prefix_states) == {1, 2}
    assert set(wrapper.xvec_prefix_state_only_states) == ({1, 2} if initial_frames == 1 else set())

    for skip_flags in ((False, False), (True, True), (False, True), (True, False)):
        codes = [torch.randint(0, 32, (1, 2, initial_frames), device=DEVICE) for _ in skip_flags]
        caches = [{"prefix_frames": 0, "skip_first_audio": skip} for skip in skip_flags]
        expected_caches = [{"prefix_frames": 0, "skip_first_audio": skip} for skip in skip_flags]
        expected = [decoder._decode_stream_first_chunk(code, cache) for code, cache in zip(codes, expected_caches)]
        actual = wrapper._decode_xvec_prefix_batch(codes, caches)
        assert actual is not None  # Eligible batches must not silently become per-request eager calls.
        for got, want, cache, expected_cache in zip(actual, expected, caches, expected_caches):
            torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)
            for key in ("ref_hidden", "ref_conv", "prefix_hidden", "suffix_quantized", "suffix_conv"):
                torch.testing.assert_close(cache[key], expected_cache[key], atol=1e-5, rtol=1e-5)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@torch.inference_mode()
def test_stream_decoder_preemption_restores_all_state_into_a_different_slot():
    from tests.model_executor.models.qwen3_tts.test_time_major_decoder import _make_decoder
    from vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.streaming_decoder import StreamingCodecDecoder

    decoder = _make_decoder().to(device=DEVICE, dtype=torch.bfloat16)
    for name, param in decoder.named_parameters():
        if name.endswith("embedding_sum"):
            param.normal_()
    decoder.config.head_dim = decoder.config.hidden_size // decoder.config.num_attention_heads
    stream = StreamingCodecDecoder(decoder, num_slots=2, dtype=torch.bfloat16)
    codes = torch.randint(0, 32, (1, 8, 2), device=DEVICE)
    slot = torch.tensor([0], device=DEVICE, dtype=torch.int32)
    for pos in range(7):
        stream(codes[:, pos : pos + 1], slot, torch.tensor([pos], device=DEVICE, dtype=torch.int32))
    saved = stream.save_slot(0)
    expected = stream(codes[:, 7:], slot, torch.tensor([7], device=DEVICE, dtype=torch.int32)).clone()
    # Both the freed slot and the destination can hold an unrelated request.
    other_codes = (codes[:, :3] + 1) % 32
    for pos in range(3):
        stream(
            other_codes[:, pos : pos + 1].expand(2, -1, -1).contiguous(),
            torch.tensor([0, 1], device=DEVICE, dtype=torch.int32),
            torch.full((2,), pos, device=DEVICE, dtype=torch.int32),
        )
    without_restore = stream(
        codes[:, 7:],
        torch.tensor([1], device=DEVICE, dtype=torch.int32),
        torch.tensor([7], device=DEVICE, dtype=torch.int32),
    ).clone()
    assert not torch.equal(without_restore, expected)
    stream.restore_slot(1, saved)
    actual = stream(
        codes[:, 7:],
        torch.tensor([1], device=DEVICE, dtype=torch.int32),
        torch.tensor([7], device=DEVICE, dtype=torch.int32),
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("use_graph", [False, True, "pieces"])
@torch.inference_mode()
def test_batched_reference_priming_matches_serial_decoder_continuation(use_graph):
    from tests.model_executor.models.qwen3_tts.test_time_major_decoder import _make_decoder
    from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker import Qwen3TTSTalkerForConditionalGeneration
    from vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.streaming_decoder import (
        StreamingCodecDecoder,
        StreamingDecodeGraphs,
    )

    decoder = _make_decoder().to(device=DEVICE, dtype=torch.bfloat16)
    for name, param in decoder.named_parameters():
        if name.endswith("embedding_sum"):
            param.normal_()
    decoder.config.head_dim = decoder.config.hidden_size // decoder.config.num_attention_heads
    stream = StreamingCodecDecoder(decoder, num_slots=5, dtype=torch.bfloat16)
    graphs = StreamingDecodeGraphs(stream, [2], frames=25) if use_graph else None
    model = Qwen3TTSTalkerForConditionalGeneration.__new__(Qwen3TTSTalkerForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.stream_decoder = stream
    model.stream_prime_graphs = graphs
    model.stream_chunk_frames = 25
    model.stream_prime_pieces = {}
    if use_graph == "pieces":
        # Tails of 3 and 51 frames replay 2 + 1 and 25 + 25 + 1 frame graphs.
        model.stream_prime_pieces = {
            f: StreamingDecodeGraphs(stream, [2], frames=f, pool=graphs.pool) for f in (16, 8, 4, 2)
        }
        model.stream_graphs = StreamingDecodeGraphs(stream, [2])
    refs = [torch.randint(0, 32, (length, 2), device=DEVICE) for length in [25, 3, 51, 25]]
    slots = [2, 0, 3, 4]
    frame = torch.randint(0, 32, (4, 1, 2), device=DEVICE)
    slot_tensor = torch.tensor(slots, device=DEVICE, dtype=torch.int32)
    positions = torch.tensor([len(ref) for ref in refs], device=DEVICE, dtype=torch.int32)
    for slot, ref in zip(slots, refs, strict=True):
        for t0 in range(0, len(ref), 25):
            stream(
                ref[None, t0 : t0 + 25].to(torch.int32).contiguous(),
                torch.tensor([slot], device=DEVICE, dtype=torch.int32),
                torch.tensor([t0], device=DEVICE, dtype=torch.int32),
            )
    expected = stream(frame, slot_tensor, positions).clone()
    untouched = stream.save_slot(1)
    # Reuse dirty slots; pos=0 must reset every causal layer for each group.
    model.prime_stream_decoder(list(zip(slots, refs, strict=True)))
    actual = stream(frame, slot_tensor, positions)
    torch.testing.assert_close(actual, expected, rtol=0, atol=3e-5)
    for got, want in zip(stream.save_slot(1), untouched, strict=True):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("use_graph", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_stream_frames_match_fp32_exact_decoder(use_graph, dtype):
    from tests.model_executor.models.qwen3_tts.test_time_major_decoder import _make_decoder
    from vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.streaming_decoder import (
        StreamingCodecDecoder,
        StreamingDecodeGraphs,
    )

    decoder = _make_decoder().to(device=DEVICE, dtype=dtype)
    # The tiny HF default initialization shrinks the deep conv stack to
    # near-zero PCM. Keep the oracle signal large enough to catch lost state.
    for module in decoder.modules():
        if isinstance(module, (torch.nn.Conv1d, torch.nn.ConvTranspose1d, torch.nn.Linear)):
            torch.nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            if module.bias is not None:
                module.bias.uniform_(-0.01, 0.01)
    for name, param in decoder.named_parameters():
        if name.endswith("embedding_sum"):
            param.normal_()
    decoder.config.head_dim = decoder.config.hidden_size // decoder.config.num_attention_heads
    reference = _make_decoder().to(device=DEVICE, dtype=torch.float32)
    reference.load_state_dict(decoder.state_dict())
    reference.precompute_snake_caches()
    # Cross both the attention window and KV ring wrap, checking every frame.
    codes = torch.randint(0, 32, (2, 136, 2), device=DEVICE)
    expected = reference._forward_exact(codes.transpose(1, 2))[:, 0]
    assert expected.square().sum() > 1e-3
    stream = StreamingCodecDecoder(decoder, num_slots=3, dtype=dtype)
    decode = StreamingDecodeGraphs(stream, [2]) if use_graph else stream
    slots = torch.tensor([2, 0], device=DEVICE, dtype=torch.int32)
    spf = int(stream.spf)
    for frame in range(codes.shape[1]):
        positions = torch.full((2,), frame, device=DEVICE, dtype=torch.int32)
        actual = decode(codes[:, frame : frame + 1].contiguous().to(torch.int32), slots, positions).reshape(2, -1)
        exact_frame = expected[:, frame * spf : (frame + 1) * spf]
        relative_rms = (actual - exact_frame).square().mean(1).sqrt() / exact_frame.square().mean(1).sqrt()
        # BF16 kernels are approximate; compare every frame to the independent
        # FP32 full decoder, not only one streaming implementation to itself.
        tolerance = 0.05 if dtype == torch.bfloat16 else 5e-4
        assert torch.all(relative_rms < tolerance), (frame, relative_rms)
