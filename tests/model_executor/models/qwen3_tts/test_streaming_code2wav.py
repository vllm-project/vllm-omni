# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.model_executor.models.qwen3_tts import streaming_code2wav as module

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Codec:
    max_frames = 3
    window = 3
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
    assert backend.request_states["a"].position == backend.request_states["b"].position == 3
    assert [call[0].shape[1] for call in backend.decoder.calls[-1:]] == [2]


def test_terminal_padding_does_not_leak_pcm_and_graph_replay_keeps_prior_outputs(backend):
    # Keep only the buckets needed here so two real frames exercise padding to three.
    backend.graphs = {frames: graph for frames, graph in backend.graphs.items() if frames in (1, 3)}
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
    assert set(backend.request_states) == {"a", "b"}
    assert backend.fallback_ids == {"icl", "overflow"}
    for wav in out:
        assert torch.all(wav == 42)
    backend.release(["a"])
    _decode(backend, [[5]], ["overflow"], [overflow], legacy=legacy)
    assert legacy.batched_chunked_decode.call_count == 2
    assert "overflow" not in backend.request_states
    backend.release(["icl", "overflow"])
    assert not backend.fallback_ids


def test_cache_replacement_resets_stream_position(backend):
    _decode(backend, [[5]], ["a"], [{}])
    out = _decode(backend, [[7]], ["a"], [{}])[0]
    torch.testing.assert_close(out, torch.tensor([[7.0, 7.0]]))
    assert backend.request_states["a"].position == 1


def test_dummy_run_uses_native_capture_without_allocating_stream_state(backend, mocker):
    legacy = mocker.Mock(spec=module.Qwen3TTSTokenizerV2Decoder)
    legacy.batched_chunked_decode.return_value = [torch.ones(1, 7)]
    actual = _decode(backend, [[1]], ["dummy"], [{"_is_dummy_run": True}], legacy=legacy)[0]
    torch.testing.assert_close(actual, torch.ones(1, 7))
    assert not backend.request_states and len(backend.free_slots) == 2
    assert not backend.decoder.calls


def test_invalid_slot_capacity_is_rejected():
    with pytest.raises(ValueError, match="num_slots"):
        module.StreamingCode2Wav(None, num_slots=0, max_batch_size=1, batch_sizes=[1], frame_sizes=[1], capture=False)


def test_icl_primes_once_and_continues_past_window(monkeypatch):
    monkeypatch.setattr(module, "StreamingCodecDecoder", _Codec)
    monkeypatch.setattr(module, "StreamingDecodeGraphs", _Graphs)
    backend = module.StreamingCode2Wav(
        None, num_slots=2, max_batch_size=1, batch_sizes=[1], frame_sizes=[1, 3], capture=True
    )
    cache = {"prefix_frames": 2}
    first = _decode(backend, [[2, 3, 4]], ["icl"], [cache])[0]
    torch.testing.assert_close(first, torch.tensor([[9.0, 9.0]]))
    assert backend.decoder.calls[0][0].shape[1] == 2  # exact prefix length, no padded codec frames
    growing = _decode(backend, [[1, 2]], ["icl"], [cache])[0]
    torch.testing.assert_close(growing, torch.tensor([[10.0, 10.0, 12.0, 12.0]]))
    rolling = _decode(backend, [[7]], ["icl"], [cache])[0]
    torch.testing.assert_close(rolling, torch.tensor([[19.0, 19.0]]))
    assert backend.request_states["icl"].position == 4
    assert backend.decoder.calls[-1][2] == [5]
    backend.release(["icl"])
    assert len(backend.free_slots) == 2


def test_icl_reordered_requests_keep_prefix_and_suffix_isolated(monkeypatch):
    monkeypatch.setattr(module, "StreamingCodecDecoder", _Codec)
    monkeypatch.setattr(module, "StreamingDecodeGraphs", _Graphs)
    backend = module.StreamingCode2Wav(
        None, num_slots=2, max_batch_size=1, batch_sizes=[1], frame_sizes=[1, 3], capture=True
    )
    a = {"prefix_frames": 1}
    b = {"prefix_frames": 2}
    first = _decode(backend, [[2, 4], [5, 1, 3]], ["a", "b"], [a, b])
    torch.testing.assert_close(first[0], torch.tensor([[6.0, 6.0]]))
    torch.testing.assert_close(first[1], torch.tensor([[9.0, 9.0]]))
    reordered = _decode(backend, [[7], [1, 2]], ["b", "a"], [b, a])
    torch.testing.assert_close(reordered[0], torch.tensor([[16.0, 16.0]]))
    torch.testing.assert_close(reordered[1], torch.tensor([[7.0, 7.0, 9.0, 9.0]]))
    rolling = _decode(backend, [[3]], ["a"], [a])[0]
    torch.testing.assert_close(rolling, torch.tensor([[12.0, 12.0]]))
    assert backend.request_states["b"].position == 2


def test_icl_equal_shapes_share_priming_and_incremental_batches(monkeypatch):
    monkeypatch.setattr(module, "StreamingCodecDecoder", _Codec)
    monkeypatch.setattr(module, "StreamingDecodeGraphs", _Graphs)
    backend = module.StreamingCode2Wav(
        None, num_slots=2, max_batch_size=2, batch_sizes=[1, 2], frame_sizes=[1, 3], capture=True
    )
    a = {"prefix_frames": 2}
    b = {"prefix_frames": 2}
    first = _decode(backend, [[2, 3, 4], [5, 1, 3]], ["a", "b"], [a, b])
    torch.testing.assert_close(first[0], torch.tensor([[9.0, 9.0]]))
    torch.testing.assert_close(first[1], torch.tensor([[9.0, 9.0]]))
    assert [call[0].shape[0] for call in backend.decoder.calls] == [2, 2]
    assert backend.decoder.calls[0][0].shape[1] == 2

    growing = _decode(backend, [[1, 2], [3, 4]], ["a", "b"], [a, b])
    torch.testing.assert_close(growing[0], torch.tensor([[10.0, 10.0, 12.0, 12.0]]))
    torch.testing.assert_close(growing[1], torch.tensor([[12.0, 12.0, 16.0, 16.0]]))
    assert backend.decoder.calls[-1][0].shape[:2] == (2, 2)

    rolling = _decode(backend, [[7], [8]], ["b", "a"], [b, a])
    torch.testing.assert_close(rolling[0], torch.tensor([[23.0, 23.0]]))
    torch.testing.assert_close(rolling[1], torch.tensor([[20.0, 20.0]]))
    assert [call[0].shape[0] for call in backend.decoder.calls[-1:]] == [2]
