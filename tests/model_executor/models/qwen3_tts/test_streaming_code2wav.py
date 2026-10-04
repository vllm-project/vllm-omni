# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.model_executor.models.qwen3_tts import streaming_code2wav as module

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Codec:
    max_frames = 3
    spf = 2

    def __init__(self, _decoder, num_slots):
        self.state = [999] * num_slots
        self.calls = []

    def __call__(self, codes, slots, positions):
        self.calls.append((codes.clone(), slots.tolist(), positions.tolist()))
        out = torch.empty(codes.shape[0], codes.shape[1] * self.spf)
        for row, (slot, pos) in enumerate(zip(slots.tolist(), positions.tolist(), strict=True)):
            previous = 0 if pos == 0 else self.state[slot]
            values = codes[row, :, 0].cumsum(0) + previous
            self.state[slot] = int(values[-1])
            out[row] = values.repeat_interleave(self.spf)
        return out


class _Graphs:
    def __init__(self, sd, batch_sizes, frames):
        self.sd = sd
        self.frames = frames
        self.storage = torch.empty(max(batch_sizes), frames * sd.spf)

    def __call__(self, codes, slots, positions):
        out = self.sd(codes, slots, positions)
        self.storage[: out.shape[0]].copy_(out)
        return self.storage[: out.shape[0]]


@pytest.fixture
def backend(monkeypatch):
    monkeypatch.setattr(module, "StreamingCodecDecoder", _Codec)
    monkeypatch.setattr(module, "StreamingDecodeGraphs", _Graphs)
    return module.StreamingCode2Wav(
        None, num_slots=2, max_batch_size=1, batch_sizes=[1], frame_sizes=[1, 3], capture=True
    )


def _decode(backend, values, request_ids, caches, *, terminal=None, legacy=None):
    lengths = [len(row) for row in values]
    codes = torch.zeros(len(values), 1, max(lengths), dtype=torch.long)
    for index, row in enumerate(values):
        codes[index, 0, : len(row)] = torch.tensor(row)
    return backend.decode(
        codes,
        lengths,
        request_ids=request_ids,
        caches=caches,
        terminal=terminal or [False] * len(values),
        legacy_decoder=legacy,
        chunk_size=300,
        left_context_size=25,
    )


def test_reordered_chunks_keep_request_state_and_skip_direct_audio_once(backend):
    a = {"skip_first_audio": True}
    b: dict[str, object] = {}
    first = _decode(backend, [[2], [5]], ["a", "b"], [a, b])
    assert first[0].numel() == 0
    torch.testing.assert_close(first[1], torch.tensor([[5.0, 5.0]]))
    follow = _decode(backend, [[7, 11], [3, 4]], ["b", "a"], [b, a])
    torch.testing.assert_close(follow[0], torch.tensor([[12.0, 12.0, 23.0, 23.0]]))
    torch.testing.assert_close(follow[1], torch.tensor([[5.0, 5.0, 9.0, 9.0]]))
    assert backend.streams["a"].position == backend.streams["b"].position == 3
    assert [call[0].shape[1] for call in backend.decoder.calls[-2:]] == [2, 2]


def test_terminal_padding_does_not_leak_pcm_and_graph_replay_keeps_prior_outputs(backend):
    out = _decode(backend, [[2, 3], [7, 11]], ["a", "b"], [{}, {}], terminal=[True, True])
    assert [call[0].shape[1] for call in backend.decoder.calls] == [3, 3]
    torch.testing.assert_close(out[0], torch.tensor([[2.0, 2.0, 5.0, 5.0]]))
    torch.testing.assert_close(out[1], torch.tensor([[7.0, 7.0, 18.0, 18.0]]))
    backend.release(["a", "b"])
    _decode(backend, [[99, 99, 99]], ["new"], [{}])
    torch.testing.assert_close(out[0], torch.tensor([[2.0, 2.0, 5.0, 5.0]]))


def test_long_chunk_splits_at_ring_limit_and_dirty_slot_starts_at_zero(backend):
    out = _decode(backend, [[1, 2, 3, 4, 5, 6, 7]], ["a"], [{}])[0]
    expected = torch.tensor([[1.0, 3.0, 6.0, 10.0, 15.0, 21.0, 28.0]]).repeat_interleave(2, dim=1)
    torch.testing.assert_close(out, expected)
    assert [call[2] for call in backend.decoder.calls] == [[0], [3], [6]]
    backend.release(["a"])
    backend.release(["a"])
    assert len(backend.free_slots) == 2
    _decode(backend, [[10]], ["other"], [{}])
    fresh = _decode(backend, [[5]], ["reused"], [{}])[0]
    torch.testing.assert_close(fresh, torch.tensor([[5.0, 5.0]]))


def test_icl_and_overflow_keep_legacy_decoder_for_entire_request(backend, mocker):
    legacy = mocker.Mock(spec=module.Qwen3TTSTokenizerV2Decoder)
    legacy.batched_chunked_decode.side_effect = lambda codes, lengths, **kw: [
        torch.full((1, length * 2), 42.0) for length in lengths
    ]
    cache_a: dict[str, object] = {}
    cache_b: dict[str, object] = {}
    overflow: dict[str, object] = {}
    _decode(backend, [[1], [2]], ["a", "b"], [cache_a, cache_b])
    out = _decode(backend, [[3], [4]], ["icl", "overflow"], [{"prefix_frames": 2}, overflow], legacy=legacy)
    assert set(backend.streams) == {"a", "b"}
    assert backend.fallback_ids == {"overflow"}
    for wav in out:
        assert torch.all(wav == 42)
    backend.release(["a"])
    _decode(backend, [[5]], ["overflow"], [overflow], legacy=legacy)
    assert legacy.batched_chunked_decode.call_count == 2
    assert "overflow" not in backend.streams
    backend.release(["overflow"])
    assert not backend.fallback_ids


def test_cache_replacement_resets_stream_position(backend):
    _decode(backend, [[5]], ["a"], [{}])
    out = _decode(backend, [[7]], ["a"], [{}])[0]
    torch.testing.assert_close(out, torch.tensor([[7.0, 7.0]]))
    assert backend.streams["a"].position == 1


def test_dummy_run_uses_native_capture_without_allocating_stream_state(backend, mocker):
    legacy = mocker.Mock(spec=module.Qwen3TTSTokenizerV2Decoder)
    legacy.batched_chunked_decode.return_value = [torch.ones(1, 7)]
    actual = _decode(backend, [[1]], ["dummy"], [{"_is_dummy_run": True}], legacy=legacy)[0]
    torch.testing.assert_close(actual, torch.ones(1, 7))
    assert not backend.streams and len(backend.free_slots) == 2
    assert not backend.decoder.calls


def test_invalid_slot_capacity_is_rejected():
    with pytest.raises(ValueError, match="num_slots"):
        module.StreamingCode2Wav(None, num_slots=0, max_batch_size=1, batch_sizes=[1], frame_sizes=[1], capture=False)
