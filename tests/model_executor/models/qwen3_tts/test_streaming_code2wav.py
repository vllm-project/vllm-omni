# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import threading
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_omni.distributed.omni_connectors.transfer_adapter.chunk_transfer_adapter import OmniChunkTransferAdapter
from vllm_omni.model_executor.models.qwen3_tts import streaming_code2wav as module
from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_code2wav import Qwen3TTSCode2Wav
from vllm_omni.model_executor.stage_input_processors.qwen3_tts import talker2code2wav_async_chunk

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Codec:
    device = torch.device("cpu")
    RING = 5
    max_frames = 3
    window = 3
    spf = 2

    def __init__(self, _decoder, num_slots):
        self.state = [999] * num_slots
        self.calls = []

    def __call__(self, codes, slots, positions, *, initial=False):
        self.calls.append((codes.clone(), slots.tolist(), positions.tolist()))
        out = torch.empty(codes.shape[0], codes.shape[1] * self.spf)
        for row, (slot, pos) in enumerate(zip(slots.tolist(), positions.tolist(), strict=True)):
            previous = 0 if pos == 0 else self.state[slot]
            values = codes[row, :, 0].cumsum(0) + previous
            self.state[slot] = int(values[-1])
            out[row] = values.repeat_interleave(self.spf)
        return out


class _Graphs:
    def __init__(self, sd, batch_sizes, frames, pool=None, *, initial=False):
        self.pool = object() if pool is None else pool
        self.sd = sd
        self.frames = frames
        self.storage = torch.empty(max(batch_sizes), frames * sd.spf)

    def __call__(self, codes, slots, positions):
        out = self.sd(codes, slots, positions)
        self.storage[: out.shape[0]].copy_(out)
        return self.storage[: out.shape[0]]


@pytest.fixture(autouse=True)
def mock_graph_pool_handle(monkeypatch):
    monkeypatch.setattr(torch.cuda, "graph_pool_handle", object)
    monkeypatch.setattr(torch.accelerator, "memory_allocated", lambda _device: 17, raising=False)
    monkeypatch.setattr(torch.accelerator, "memory_reserved", lambda _device: 23, raising=False)


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


def test_long_chunk_splits_at_decoder_limit(backend):
    out = _decode(backend, [[1, 2, 3, 4, 5, 6, 7]], ["a"], [{}])[0]
    expected = torch.tensor([[1.0, 3.0, 6.0, 10.0, 15.0, 21.0, 28.0]]).repeat_interleave(2, dim=1)
    torch.testing.assert_close(out, expected)
    assert [call[2] for call in backend.decoder.calls] == [[0], [3], [6]]


def test_dummy_run_uses_legacy_decoder_without_allocating_request_state(backend, mocker):
    legacy = mocker.Mock(spec=module.Qwen3TTSTokenizerV2Decoder)
    legacy.batched_chunked_decode.return_value = [torch.ones(1, 7)]
    actual = _decode(backend, [[1]], ["dummy"], [{"_is_dummy_run": True}], legacy=legacy)[0]
    torch.testing.assert_close(actual, torch.ones(1, 7))
    assert not backend.request_states and len(backend.free_slots) == 2
    assert not backend.decoder.calls


def test_icl_primes_once_and_continues_past_window(monkeypatch):
    monkeypatch.setattr(module, "StreamingCodecDecoder", _Codec)
    monkeypatch.setattr(module, "StreamingDecodeGraphs", _Graphs)
    backend = module.StreamingCode2Wav(
        None, num_slots=2, max_batch_size=1, batch_sizes=[1], frame_sizes=[1, 3], capture=True
    )
    cache = {"prefix_frames": 4}
    assert cache["prefix_frames"] > backend.decoder.max_frames
    first = _decode(backend, [[1, 2, 3, 4, 5]], ["icl"], [cache])[0]
    assert [call[0].shape[1] for call in backend.decoder.calls] == [4, 1]
    assert [call[2] for call in backend.decoder.calls] == [[0], [4]]
    torch.testing.assert_close(first, torch.tensor([[15.0, 15.0]]))
    follow = _decode(backend, [[1, 2]], ["icl"], [cache])[0]
    torch.testing.assert_close(follow, torch.tensor([[16.0, 16.0, 18.0, 18.0]]))
    past_window = _decode(backend, [[7]], ["icl"], [cache])[0]
    torch.testing.assert_close(past_window, torch.tensor([[25.0, 25.0]]))
    assert backend.request_states["icl"].position == 8
    assert [call[2] for call in backend.decoder.calls] == [[0], [4], [5], [7]]
    backend.release(["icl"])
    assert len(backend.free_slots) == 2


def test_icl_and_xvec_share_suffix_batches(monkeypatch):
    monkeypatch.setattr(module, "StreamingCodecDecoder", _Codec)
    monkeypatch.setattr(module, "StreamingDecodeGraphs", _Graphs)
    backend = module.StreamingCode2Wav(
        None, num_slots=2, max_batch_size=2, batch_sizes=[1, 2], frame_sizes=[1, 3], capture=True
    )
    caches = [{"prefix_frames": 2, "skip_first_audio": True}, {}]
    first = _decode(backend, [[1, 2, 3], [4]], ["icl", "xvec"], caches)
    assert first[0].numel() == 0
    torch.testing.assert_close(first[1], torch.tensor([[4.0, 4.0]]))
    assert [call[0].shape[:2] for call in backend.decoder.calls] == [(1, 2), (2, 1)]
    assert backend.decoder.calls[-1][2] == [2, 0]
    follow = _decode(backend, [[5], [6]], ["xvec", "icl"], caches[::-1])
    torch.testing.assert_close(follow[0], torch.tensor([[9.0, 9.0]]))
    torch.testing.assert_close(follow[1], torch.tensor([[12.0, 12.0]]))
    assert backend.decoder.calls[-1][0].shape[:2] == (2, 1)
    assert backend.decoder.calls[-1][2] == [1, 3]


def test_streaming_memory_snapshot_is_retained(backend, monkeypatch):
    monkeypatch.setattr(torch.accelerator, "memory_allocated", lambda _device: 99, raising=False)
    monkeypatch.setattr(torch.accelerator, "memory_reserved", lambda _device: 101, raising=False)
    model = SimpleNamespace(_streaming_codec=backend)
    stats = Qwen3TTSCode2Wav.get_stage_post_warmup_memory_stats(model)
    assert (stats.allocated_bytes, stats.reserved_bytes) == (17, 23)


def test_icl_segment_boundary_resends_reference_and_reprimes(backend):
    manager = object.__new__(OmniChunkTransferAdapter)
    manager.code_prompt_token_ids = defaultdict(list)
    manager.put_req_chunk = defaultdict(int)
    manager.ramp_chunk_count = defaultdict(int)
    manager._adaptive_states = {}
    manager.request_payload = {}
    manager.connector = SimpleNamespace(
        config={"extra": {"codec_chunk_frames": 25, "codec_left_context_frames": 72, "initial_codec_chunk_frames": 1}}
    )
    manager._sender_state_lock = threading.Lock()
    manager.connector.stage_id = 0
    manager.connector.put = Mock(return_value=(True, 0, None))
    manager.request_payload["icl"] = torch.full((2, 4), 2, dtype=torch.long)
    request = SimpleNamespace(external_req_id="icl", request_id="icl", is_finished=lambda: False)
    for _ in range(2):
        manager.code_prompt_token_ids["icl"] = [[1, 2, 3, 4]]
        payload = talker2code2wav_async_chunk(manager, {"codes": {"audio": torch.empty(0)}}, request, is_finished=True)
        assert payload.meta.ref_context_included
        assert payload.meta.ref_context_size == 2
        assert payload.codes.audio.numel() == 4 * 3
        codes = payload.codes.audio.reshape(1, 4, 3)
        result = backend.decode(
            codes,
            [3],
            request_ids=["icl"],
            caches=[{"prefix_frames": 2}],
            terminal=[True],
            legacy_decoder=None,
            chunk_size=300,
            left_context_size=25,
        )[0]
        torch.testing.assert_close(result, torch.tensor([[5.0, 5.0]]))
        assert backend.decoder.calls[-2][2] == [0]
        assert backend.decoder.calls[-1][2] == [2]
        backend.release(["icl"])
        manager._send_single_request_for_generation(
            {
                "multimodal_output": None,
                "request": request,
                "is_finished": False,
                "is_segment_finished": True,
                "payload": payload,
            }
        )
        assert "icl" not in manager._qwen3_tts_emitted_frames
    assert manager.put_req_chunk["icl"] == 2
