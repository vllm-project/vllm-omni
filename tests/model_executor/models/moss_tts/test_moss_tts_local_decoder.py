# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_omni.model_executor.models.moss_tts.moss_tts_local_decoder import (
    MossTTSDecoderModel,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_decode_one_request_decodes_all_zero_rvq_codes():
    model = object.__new__(MossTTSDecoderModel)
    model.n_vq = 3
    model.device = torch.device("cpu")
    model._streaming_states = {}
    model._batched_streaming_stack = None
    model._batched_streaming_request_ids = None
    model._first_chunk_dir = None
    model._first_chunk_seen = set()

    wav = torch.ones(7, dtype=torch.float32)
    model._codec = SimpleNamespace(decode=Mock(return_value=wav))

    out = MossTTSDecoderModel._decode_one_request(
        model,
        torch.zeros(6, dtype=torch.long),
    )

    model._codec.decode.assert_called_once()
    assert torch.equal(out, wav)


def test_decode_one_request_with_id_emits_stateless_audio_delta():
    model = object.__new__(MossTTSDecoderModel)
    model.n_vq = 3
    model.device = torch.device("cpu")
    model._streaming_states = {}
    model._batched_streaming_stack = None
    model._batched_streaming_request_ids = None
    model._request_code_buffers = {}
    model._request_audio_offsets = {}
    model._first_chunk_dir = None
    model._first_chunk_seen = set()

    def decode(codes):
        return torch.arange(codes.shape[-1] * 2, dtype=torch.float32)

    model._codec = SimpleNamespace(decode=Mock(side_effect=decode))

    first = MossTTSDecoderModel._decode_one_request(
        model,
        torch.arange(6, dtype=torch.long),
        request_id="req-0",
    )
    second = MossTTSDecoderModel._decode_one_request(
        model,
        torch.arange(6, 12, dtype=torch.long),
        request_id="req-0",
        is_finished=True,
    )

    assert torch.equal(first, torch.arange(4, dtype=torch.float32))
    assert torch.equal(second, torch.arange(4, 8, dtype=torch.float32))
    assert model._codec.decode.call_count == 2
    assert "req-0" not in model._request_code_buffers
    assert "req-0" not in model._request_audio_offsets


def test_single_request_with_id_uses_stateless_delta_path(monkeypatch):
    model = object.__new__(MossTTSDecoderModel)
    model._batched_streaming_stack = None
    model._streaming_states = {}

    per_request_calls = []
    batch_calls = []

    def fake_decode_one_request(req_codes, request_id=None, is_finished=False):
        per_request_calls.append((req_codes, request_id, is_finished))
        return torch.ones(3)

    def fake_batch_decode_streaming(*args, **kwargs):
        batch_calls.append((args, kwargs))
        return [torch.zeros(3)]

    monkeypatch.setattr(model, "_decode_one_request", fake_decode_one_request)
    monkeypatch.setattr(model, "_batch_decode_streaming", fake_batch_decode_streaming)

    out = MossTTSDecoderModel._batch_decode(
        model,
        [torch.arange(6)],
        request_ids=["req-0"],
        finished_flags=[False],
    )

    assert len(per_request_calls) == 1
    assert per_request_calls[0][1:] == ("req-0", False)
    assert batch_calls == []
    assert torch.equal(out[0], torch.ones(3))


def test_multi_request_with_ids_uses_batched_streaming(monkeypatch):
    model = object.__new__(MossTTSDecoderModel)
    model._batched_streaming_stack = None
    model._streaming_states = {}

    per_request_calls = []
    batch_calls = []

    def fake_decode_one_request(*args, **kwargs):
        per_request_calls.append((args, kwargs))
        return torch.ones(3)

    def fake_batch_decode_streaming(request_codes_list, request_ids, finished_flags):
        batch_calls.append((request_codes_list, request_ids, finished_flags))
        return [torch.full((2,), 1.0), torch.full((2,), 2.0)]

    monkeypatch.setattr(model, "_decode_one_request", fake_decode_one_request)
    monkeypatch.setattr(model, "_batch_decode_streaming", fake_batch_decode_streaming)

    out = MossTTSDecoderModel._batch_decode(
        model,
        [torch.arange(6), torch.arange(6, 12)],
        request_ids=["req-0", "req-1"],
        finished_flags=[False, True],
    )

    assert per_request_calls == []
    assert len(batch_calls) == 1
    assert batch_calls[0][1:] == (["req-0", "req-1"], [False, True])
    assert torch.equal(out[0], torch.full((2,), 1.0))
    assert torch.equal(out[1], torch.full((2,), 2.0))


def test_batch_decode_streaming_uses_codec_worker_batch_decode(monkeypatch):
    model = object.__new__(MossTTSDecoderModel)
    model.n_vq = 3
    model.device = torch.device("cpu")
    model._first_chunk_dir = None
    model._first_chunk_seen = set()

    monkeypatch.setattr(model, "_enter_batched_streaming", lambda request_ids: None)
    monkeypatch.setattr(model, "_exit_batched_streaming", lambda: None)
    model._codec = SimpleNamespace(
        decode_batch=Mock(
            return_value=[
                torch.full((2,), 1.0),
                torch.full((4,), 2.0),
            ]
        )
    )

    out = MossTTSDecoderModel._batch_decode_streaming(
        model,
        [torch.arange(6), torch.arange(6, 15)],
        request_ids=["req-0", "req-1"],
        finished_flags=[False, True],
    )

    model._codec.decode_batch.assert_called_once()
    padded, lengths = model._codec.decode_batch.call_args.args
    assert padded.shape == (3, 2, 3)
    assert torch.equal(lengths, torch.tensor([2, 3]))
    assert torch.equal(out[0], torch.full((2,), 1.0))
    assert torch.equal(out[1], torch.full((4,), 2.0))
