# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import math

import pytest
import torch

from vllm_omni.model_executor.models.audio8_tts import codec_utils
from vllm_omni.model_executor.models.audio8_tts.cudagraph_wrapper import (
    Audio8CodecCUDAGraphWrapper,
    decoder_capture_sizes,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _CausalCodec(torch.nn.Module):
    sample_rate = 8
    frame_length = 2

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))

    def _encode_eager(self, audio, lengths):
        frames = audio[:, :, :: self.frame_length].cumsum(-1)
        return frames.expand(-1, 10, -1).long(), torch.ceil(lengths.float() / self.frame_length).long()

    def _decode_eager(self, codes):
        return codes[:, :1].float().cumsum(-1).repeat_interleave(self.frame_length, -1)


class _Replay:
    def __init__(self, wrapper, key, calls):
        self.wrapper = wrapper
        self.key = key
        self.calls = calls

    def replay(self):
        kind, batch_size, size = self.key
        self.calls.append((kind, batch_size, size))
        inputs = self.wrapper._inputs[self.key]
        if kind == "encode":
            self.wrapper._outputs[self.key] = self.wrapper.codec._encode_eager(
                inputs, torch.full((inputs.shape[0],), size)
            )[0]
        else:
            self.wrapper._outputs[self.key] = self.wrapper.codec._decode_eager(inputs)


class _SimulatedGraphWrapper(Audio8CodecCUDAGraphWrapper):
    def __init__(self, codec, **kwargs):
        super().__init__(codec, **kwargs)
        self.calls = []

    def _capture(self, kind, batch_size, size):
        key = (kind, batch_size, size)
        if key in self._graphs:
            return
        self._inputs[key] = torch.zeros((batch_size, 1 if kind == "encode" else 10, size))
        self._outputs[key] = torch.empty(0)
        self._graphs[key] = _Replay(self, key, self.calls)


def test_decoder_capture_sizes_include_initial_and_transition_windows():
    assert decoder_capture_sizes(25, 25, 4) == (4, 8, 12, 16, 20, 24, 25, 49, 50)


def test_encode_groups_nearest_half_second_frame_buckets_and_preserves_order():
    codec = _CausalCodec()
    wrapper = _SimulatedGraphWrapper(codec, batch_size=5)
    requests = [torch.arange(n).float() for n in (9, 2, 16, 7, 10)]
    outputs = wrapper.encode(requests)
    assert wrapper.calls == [("encode", 1, 4), ("encode", 1, 8), ("encode", 2, 12), ("encode", 1, 16)]
    for source, output in zip(requests, outputs, strict=True):
        eager = codec._encode_eager(source.reshape(1, 1, -1), torch.tensor([len(source)]))[0][0]
        assert output.shape == (10, math.ceil(len(source) / codec.frame_length))
        torch.testing.assert_close(output, eager)


def test_encode_bucket_is_aligned_to_codec_frames():
    codec = _CausalCodec()
    codec.sample_rate = 44100
    codec.frame_length = 2048
    wrapper = _SimulatedGraphWrapper(codec, batch_size=1)
    assert wrapper._bucket("encode", 1) == 22528
    assert wrapper._bucket("encode", 22529) == 45056


def test_reference_encode_uses_waveform_length_without_reading_cuda_lengths(monkeypatch):
    class _FakeReferenceCodec:
        def encode(self, audio):
            assert isinstance(audio, list) and len(audio) == 1
            assert tuple(audio[0].shape) == (2049,)
            return [torch.arange(30).reshape(10, 3)]

    monkeypatch.setattr(codec_utils, "load_arktts_codec", lambda *args, **kwargs: _FakeReferenceCodec())
    monkeypatch.setattr(codec_utils, "prepare_reference_waveform", lambda *args, **kwargs: torch.zeros(2049))
    codes = codec_utils.encode_reference_audio_codes("unused", torch.zeros(1), 44100, device="cpu")
    assert tuple(codes.shape) == (2, 10)
    torch.testing.assert_close(codes, torch.arange(30).reshape(10, 3)[:, :2].T.contiguous())


def test_decode_groups_tail_chunks_and_padding_matches_eager():
    codec = _CausalCodec()
    wrapper = _SimulatedGraphWrapper(codec, batch_size=5, decoder_sizes=(4, 8, 12))
    wrapper.capture_decoder()
    requests = [torch.arange(n).expand(10, -1).long() for n in (9, 3, 8, 6, 2)]
    outputs = wrapper.decode(requests)
    assert wrapper.calls == [("decode", 2, 4), ("decode", 2, 8), ("decode", 1, 12)]
    for source, output in zip(requests, outputs, strict=True):
        eager = codec._decode_eager(source.unsqueeze(0))[0]
        assert output.shape == (1, len(source[0]) * codec.frame_length)
        torch.testing.assert_close(output, eager, atol=1e-5, rtol=1e-5)


def test_decode_reuses_larger_batch_and_length_graph_without_new_capture():
    codec = _CausalCodec()
    wrapper = _SimulatedGraphWrapper(codec, batch_size=4)
    long_audio = torch.arange(8).float()
    wrapper.encode([long_audio, long_audio.clone()])
    assert set(wrapper._graphs) == {("encode", 2, 8)}

    short_audio = torch.arange(3).float()
    output = wrapper.encode([short_audio])[0]
    assert set(wrapper._graphs) == {("encode", 2, 8)}
    assert wrapper.calls[-1] == ("encode", 2, 8)
    torch.testing.assert_close(output, codec._encode_eager(short_audio.reshape(1, 1, -1), torch.tensor([3]))[0][0])


def test_different_buckets_sharing_graph_replay_together():
    codec = _CausalCodec()
    wrapper = _SimulatedGraphWrapper(codec, batch_size=2)
    short_audio = torch.arange(3).float()
    long_audio = torch.arange(8).float()
    wrapper.encode([long_audio, long_audio.clone()])
    assert set(wrapper._graphs) == {("encode", 2, 8)}

    before = len(wrapper.calls)
    outputs = wrapper.encode([short_audio, long_audio])
    assert wrapper.calls[before:] == [("encode", 2, 8)]
    for source, output in zip((short_audio, long_audio), outputs, strict=True):
        eager = codec._encode_eager(source.reshape(1, 1, -1), torch.tensor([source.numel()]))[0][0]
        torch.testing.assert_close(output, eager)


def test_shared_graph_group_captures_combined_batch_instead_of_replaying_twice():
    codec = _CausalCodec()
    wrapper = _SimulatedGraphWrapper(codec, batch_size=2)
    short_audio = torch.arange(3).float()
    long_audio = torch.arange(8).float()
    wrapper.encode([long_audio])

    before = len(wrapper.calls)
    outputs = wrapper.encode([short_audio, long_audio])
    assert wrapper.calls[before:] == [("encode", 2, 8)]
    assert ("encode", 2, 8) in wrapper._graphs
    assert outputs[0].shape[-1] == 2
    assert outputs[1].shape[-1] == 4


def test_combined_encoder_batch_captures_current_length_instead_of_reusing_longer_graph():
    codec = _CausalCodec()
    wrapper = _SimulatedGraphWrapper(codec, batch_size=2)
    short_audio = torch.arange(3).float()
    long_audio = torch.arange(8).float()
    wrapper.encode([long_audio])
    wrapper.encode([torch.arange(16).float(), torch.arange(16).float()])

    before = len(wrapper.calls)
    wrapper.encode([short_audio, long_audio])
    assert wrapper.calls[before:] == [("encode", 2, 8)]
    assert ("encode", 2, 8) in wrapper._graphs


def test_encoder_lazy_capture_uses_actual_group_batch_size():
    codec = _CausalCodec()
    wrapper = _SimulatedGraphWrapper(codec, batch_size=4)
    audio = torch.arange(7).float()
    outputs = wrapper.encode([audio, audio.clone(), audio.clone()])
    assert set(wrapper._graphs) == {("encode", 3, 8)}
    assert wrapper.calls == [("encode", 3, 8)]
    assert len(outputs) == 3
