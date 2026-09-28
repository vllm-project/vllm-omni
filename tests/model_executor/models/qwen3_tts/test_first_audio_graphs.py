# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Prefix graphs must follow per-request first-audio delivery, not a capture hint."""

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
