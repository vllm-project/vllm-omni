# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Single-stage audio consumes only valid samples at the existing eager hook."""

import queue
import threading
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_omni.model_executor.models.cosyvoice3.async_codec import AsyncCodec
from vllm_omni.model_executor.models.cosyvoice3.model_state import CosyVoice3ModelState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Decoder:
    def __init__(self):
        self.calls = []

    def forward_streaming(self, **kwargs):
        self.calls.append(kwargs)
        return torch.ones(1, 1, 4), None


@pytest.fixture(autouse=True)
def cpu_streams(monkeypatch):
    class Event:
        def record(self, *args):
            pass

        def query(self):
            return True

    from contextlib import nullcontext

    empty = torch.empty

    def host_empty(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", host_empty)
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda *args: None)


def run_step(state, batch, hidden, tokens, outputs):
    codec = state.codec
    codec.decoder = state.model.code2wav
    state.post_sample(batch, hidden, tokens, outputs)
    assert outputs == {}  # Producer never attaches pending audio to AR output.
    # Coalesced progress is consumed independently of producer notifications.
    codec.start_reads()
    codec.poll_reads()
    while codec.decode_ready():
        codec.publish()
    codec.publish()
    audio = [torch.empty(0) for _ in batch.req_ids]
    while not codec.outputs.empty():
        result = codec.outputs.get_nowait()
        if result.pcm is not None:
            audio[batch.req_ids.index(result.request_id)] = result.pcm
    outputs["audio"] = audio


def _fixture():
    state = object.__new__(CosyVoice3ModelState)
    state.vocab_size = 6561
    state.codec_max_model_len = 32
    codec = object.__new__(AsyncCodec)
    codec.states = {}
    codec.chunk_size, codec.lookahead = 1, 0
    state.audio_routes = {"prefill": 0, "decode": 0}
    codec.buffers, codec.mailbox, codec.retirements = {}, {}, {}
    codec.reads, codec.progress = {}, {}
    codec.released = set()
    codec.lock = threading.Lock()
    codec.tick = threading.Event()
    codec.outputs = queue.SimpleQueue()
    codec.max_model_len = 32
    codec.stream = SimpleNamespace(wait_event=lambda event: None)
    codec.copy_stream = SimpleNamespace(wait_event=lambda event: None)
    codec.control_stream = SimpleNamespace(wait_event=lambda event: None)
    codec.cancelled, codec.ended = set(), set()
    codec.pending = deque()
    codec.closed, codec.error = False, None
    state.codec = codec
    state.req_states = SimpleNamespace(max_seq_len=np.array([32, 32]), prompt_len=SimpleNamespace(np=np.array([4, 6])))
    decoder = _Decoder()
    state.model = SimpleNamespace(
        code2wav=decoder, config=SimpleNamespace(sample_rate=24000), _maybe_enable_code2wav_trt=lambda: None
    )
    payloads = [{"sampling_params": SimpleNamespace(max_tokens=2)} for _ in range(2)]
    state.intermediate_buffer = SimpleNamespace(gather=lambda batch: payloads)
    batch = SimpleNamespace(
        num_reqs=2,
        idx_mapping_np=np.array([0, 1]),
        req_ids=["prefill", "decode"],
        num_computed_tokens_np=np.array([0, 5]),
        num_computed_prefill_tokens_np=np.array([0, 5]),
        prefill_len_np=np.array([4, 5]),
        is_prefilling_np=np.array([True, False]),
        num_scheduled_tokens=np.array([2, 1], dtype=np.int32),
    )
    conditioning = {
        "speech_token": [torch.ones(1, 1, dtype=torch.long), torch.ones(1, 1, dtype=torch.long)],
        "speech_feat": [torch.ones(1, 2, 80), torch.ones(1, 2, 80)],
        "embedding": [torch.ones(1, 192), torch.ones(1, 192)],
    }
    return state, batch, conditioning, decoder


def test_final_prefill_creates_state_but_intermediate_prefill_does_not():
    state, batch, conditioning, decoder = _fixture()
    outputs = {"embed": conditioning}
    run_step(state, batch, torch.zeros(3, 8), torch.tensor([[0], [7]]), outputs)
    assert "prefill" not in state.codec.states
    batch.num_computed_prefill_tokens_np[0] = 2
    batch.num_computed_tokens_np[0] = 2
    batch.num_computed_tokens_np[1] += 1
    run_step(state, batch, torch.zeros(3, 8), torch.tensor([[12], [8]]), {"audio": []})
    assert state.codec.states["prefill"].token[:, :1].tolist() == [[12]]


@pytest.mark.parametrize("sample_before_retire", [False, True])
def test_retired_route_skips_in_flight_sample_without_recreating_buffer(sample_before_retire):
    state, batch, conditioning, _ = _fixture()
    tokens = torch.tensor([[0], [12]])
    if sample_before_retire:
        state.post_sample(batch, torch.zeros(3, 8), tokens, {"embed": conditioning})
        assert "decode" in state.codec.buffers
    # Same route removal/release order as scheduler.retire_request(), including
    # cancellation before this request ever produced its first sample.
    state.audio_routes.pop("decode")
    state.codec.release("decode")
    state.codec.consume_retirements()
    state.post_sample(batch, torch.zeros(3, 8), tokens, {"embed": conditioning})
    assert "decode" not in state.codec.buffers
    assert "decode" not in state.codec.mailbox


def test_scheduler_length_limit_flushes_last_token():
    state, batch, conditioning, decoder = _fixture()
    state.req_states.max_seq_len[1] = 7
    run_step(state, batch, torch.zeros(3, 8), torch.tensor([[0], [12]]), {"embed": conditioning})
    assert decoder.calls[-1]["finalize"] is True
    request = state.codec.states["decode"]
    assert request.token[:, :1].tolist() == [[12]]


def test_resumed_prefill_uses_original_prompt_length_for_append_position():
    state, batch, conditioning, _ = _fixture()
    run_step(state, batch, torch.zeros(3, 8), torch.tensor([[0], [12]]), {"embed": conditioning})
    batch.num_computed_tokens_np[1] = 6
    run_step(state, batch, torch.zeros(3, 8), torch.tensor([[0], [13]]), {"audio": []})
    # Replay prefill includes the two accepted speech tokens, but original
    # prompt_len remains 6. The new sample belongs at token-buffer offset 2.
    batch.is_prefilling_np[1] = True
    batch.prefill_len_np[1] = 8
    batch.num_computed_prefill_tokens_np[1] = 7
    batch.num_computed_tokens_np[1] = 7
    run_step(state, batch, torch.zeros(3, 8), torch.tensor([[0], [14]]), {"audio": []})
    assert state.codec.states["decode"].token[:, :3].tolist() == [[12, 13, 14]]


class _TaggedDecoder(_Decoder):
    def forward_streaming(self, **kwargs):
        self.calls.append(kwargs)
        final = kwargs["finalize"]
        size = kwargs["token"].shape[1] + (10 if final else 0)
        return torch.full((1, 1, size), 100.0 if final else 10.0), None


def test_eos_between_chunk_boundaries_flushes_without_intermediate_decodes():
    state, batch, conditioning, _ = _fixture()
    state.codec.chunk_size, state.codec.lookahead = 25, 3
    state.model.code2wav = _TaggedDecoder()
    for step, token in enumerate([12, 13, 6562]):
        batch.num_computed_tokens_np[1] = 5 + step
        outputs = {"embed": conditioning} if step == 0 else {"audio": []}
        run_step(state, batch, torch.zeros(3, 8), torch.tensor([[0], [token]]), outputs)
        if step < 2:
            assert outputs["audio"][1].numel() == 0
            assert state.model.code2wav.calls == []
        else:
            assert outputs["audio"][1].tolist() == [100.0] * 12
        request = state.codec.states["decode"]
        assert request.emitted == 0
        assert request.cache is None


def test_mixed_speech_and_empty_eos_keep_batch_order():
    state, batch, conditioning, _ = _fixture()
    batch.num_computed_tokens_np[0] = 2
    batch.num_computed_prefill_tokens_np[0] = 2
    outputs = {"embed": conditioning}
    run_step(state, batch, torch.zeros(3, 8), torch.tensor([[7], [6562]]), outputs)
    assert outputs["audio"][0].tolist() == [1.0] * 4
    assert outputs["audio"][1].numel() == 0


def test_coalesced_length_stop_preserves_final_chunk_boundary():
    from vllm_omni.model_executor.models.cosyvoice3.async_codec import SampleEntry

    state, _, _, decoder = _fixture()
    codec = state.codec
    codec.decoder = state.model.code2wav
    codec.chunk_size, codec.lookahead = 2, 0
    conditioning = {
        "speech_token": torch.empty(1, 0, dtype=torch.long),
        "speech_feat": torch.empty(1, 0, 80),
        "embedding": torch.ones(1, 192),
    }
    for position in range(2):
        codec.submit(
            [SampleEntry("decode", position, position == 1, conditioning)],
            torch.tensor([position + 1]),
            torch.tensor([True]),
        )
    codec.start_reads()
    codec.poll_reads()
    assert codec.decode_ready()
    assert not codec.decode_ready()
    assert [call["finalize"] for call in decoder.calls] == [True]
    assert decoder.calls[0]["token"].tolist() == [[1, 2]]


def test_cancel_keeps_pending_copy_alive_and_does_not_affect_another_request():
    from vllm_omni.model_executor.models.cosyvoice3.async_codec import SampleEntry

    state, _, _, _ = _fixture()
    codec = state.codec
    codec.decoder = state.model.code2wav
    conditioning = {
        "speech_token": torch.empty(1, 0, dtype=torch.long),
        "speech_feat": torch.empty(1, 0, 80),
        "embedding": torch.ones(1, 192),
    }
    codec.submit([SampleEntry("decode", 0, False, conditioning)], torch.tensor([12]), torch.tensor([True]))
    codec.start_reads()
    old = codec.reads["decode"]
    done = SimpleNamespace(value=False)
    old.copy_done = SimpleNamespace(query=lambda: done.value)
    codec.cancel("decode")
    codec.release("decode")
    codec.consume_retirements()
    codec.poll_reads()
    assert "decode" in codec.reads  # Keep pinned destination alive while its DMA is pending.
    codec.submit([SampleEntry("next", 0, True, conditioning)], torch.tensor([23]), torch.tensor([True]))
    codec.start_reads()
    codec.poll_reads()
    done.value = True
    codec.poll_reads()
    assert "decode" not in codec.progress
    codec.decode_ready()
    codec.publish()
    assert codec.outputs.get_nowait().request_id == "next"
    assert codec.outputs.empty()


def test_submit_copies_noncontiguous_source_rows_before_input_reuse():
    from vllm_omni.model_executor.models.cosyvoice3.async_codec import SampleEntry

    state, _, _, _ = _fixture()
    codec = state.codec
    tokens = torch.tensor([11, 22, 33, 44])
    token_mask = torch.tensor([True, False, True, True])
    entries = [SampleEntry("prefill", 0, False, {}), SampleEntry("decode", 0, False, {})]
    codec.submit(entries, tokens, token_mask, rows=[3, 1])
    tokens.fill_(999)
    token_mask.fill_(True)
    assert codec.buffers["prefill"].tokens[:, :1].tolist() == [[44]]
    assert codec.buffers["decode"].tokens[:, :1].tolist() == [[22]]
    assert codec.buffers["prefill"].mask[:1].tolist() == [True]
    assert codec.buffers["decode"].mask[:1].tolist() == [False]


def test_sample_snapshot_preserves_row_mapping_without_cpu_readback(monkeypatch):
    state = object.__new__(CosyVoice3ModelState)
    state.vocab_size = 100
    state.codec_max_model_len = 32
    state.req_states = SimpleNamespace(max_seq_len=np.full(5, 32), prompt_len=SimpleNamespace(np=np.full(5, 4)))
    state.intermediate_buffer = SimpleNamespace(gather=lambda batch: [{} for _ in range(5)])
    submitted = []
    state.codec = SimpleNamespace(submit=lambda *args: submitted.append(args))
    state.audio_routes = {"b": 0, "a": 0}
    batch = SimpleNamespace(
        num_reqs=5,
        req_ids=["prefill", "b", "_warmup_0", "a", "retired"],
        idx_mapping_np=np.arange(5),
        num_scheduled_tokens=np.ones(5, dtype=np.int32),
        num_computed_prefill_tokens_np=np.array([1, 4, 4, 4, 4]),
        prefill_len_np=np.full(5, 4),
        is_prefilling_np=np.array([True, False, False, False, False]),
        num_computed_tokens_np=np.array([1, 3, 3, 3, 3]),
    )
    original = torch.Tensor.__getitem__

    def basic_index_only(tensor, index):
        parts = index if isinstance(index, tuple) else (index,)
        if any(isinstance(part, (list, np.ndarray, torch.Tensor)) for part in parts):
            pytest.fail("CPU row metadata must not be used for CUDA advanced indexing")
        return original(tensor, index)

    monkeypatch.setattr(torch.Tensor, "__getitem__", basic_index_only)

    def no_cpu_readback(*args, **kwargs):
        pytest.fail("Sampling control tensors must remain on the GPU")

    monkeypatch.setattr(torch.Tensor, "cpu", no_cpu_readback)
    sampled = torch.tensor([[99], [12], [77], [100], [55]])
    outputs = {"audio": []}
    state.post_sample(batch, torch.zeros(5, 8), sampled, outputs)
    batch.req_ids[1] = "reused"
    batch.num_computed_tokens_np[1] = 99
    entries, tokens, token_mask, rows = submitted[0]
    assert rows == [1, 3]
    assert [entry.position for entry in entries] == [0, 0]
    assert [entry.request_id for entry in entries] == ["b", "a"]
    assert tokens.untyped_storage().data_ptr() == sampled.untyped_storage().data_ptr()
    assert tokens.tolist() == [99, 12, 77, 100, 55]
    assert token_mask.tolist() == [True, True, True, False, True]
    assert outputs == {}


def test_per_request_d2h_owns_pcm_and_publishes_between_submissions(monkeypatch):
    """Cover D2H readiness, graph-output reuse and early publication together."""
    from vllm_omni.model_executor.models.cosyvoice3.async_codec import SampleEntry

    class PendingEvent:
        def __init__(self):
            self.done = False

        def record(self, *args):
            pass

        def query(self):
            return self.done

    monkeypatch.setattr(torch.cuda, "Event", PendingEvent)
    state, _, _, decoder = _fixture()
    codec = state.codec
    codec.decoder = state.model.code2wav
    conditioning = {
        "speech_token": torch.empty(1, 0, dtype=torch.long),
        "speech_feat": torch.empty(1, 0, 80),
        "embedding": torch.ones(1, 192),
    }
    codec.submit(
        [SampleEntry("prefill", 0, False, conditioning), SampleEntry("decode", 0, True, conditioning)],
        torch.tensor([7, 8]),
        torch.tensor([True, True]),
    )
    codec.start_reads()
    for read in codec.reads.values():
        read.copy_done.done = True
    codec.poll_reads()
    shared = torch.ones(1, 1, 4)

    def decode(**kwargs):
        if decoder.calls:
            # A's copy was already queued before B's HiFT, without a GPU wait.
            assert len(codec.pending) == 1
            first = codec.pending[0]
            assert codec.outputs.empty() and not first.copy_done.done
            shared.fill_(2)  # Simulate a replay reusing its output allocation.
            assert first.source.tolist() == [1.0] * 4
            first.copy_done.done = True
        decoder.calls.append(kwargs)
        return shared, None

    decoder.forward_streaming = decode
    assert codec.decode_ready()
    # A is published although B's D2H is not done; the final flag stays with B.
    first = codec.outputs.get_nowait()
    assert first.request_id == "prefill" and not first.final
    assert first.pcm.tolist() == [1.0] * 4
    assert len(codec.pending) == 1
    final = codec.pending[0]
    assert not final.copy_done.done
    assert final.source.tolist() == [2.0] * 4
    final.copy_done.done = True
    codec.publish()
    last = codec.outputs.get_nowait()
    assert last.request_id == "decode" and last.final
    assert last.pcm.tolist() == [2.0] * 4


def _chunk_conditioning(prompt_len=3):
    return {
        "speech_token": torch.ones((1, prompt_len), dtype=torch.long),
        "speech_feat": torch.ones((1, prompt_len * 2, 80)),
        "embedding": torch.ones((1, 192)),
    }


@pytest.mark.parametrize("stop_known_early", [False, True])
def test_chunk_windows_match_connector_hop_padding_and_lookahead(stop_known_early):
    state, _, _, _ = _fixture()
    codec = state.codec
    codec.chunk_size, codec.lookahead = 4, 3
    key = "a"
    codec.initialize(key, _chunk_conditioning(), torch.arange(40).reshape(1, -1))
    # A coalesced EOS prefix must take exactly the same windows as incremental
    # token arrivals; it must not collapse the chunk ramp into one large tail.
    updates = [(40, 40)] * 4 if stop_known_early else [(n, None) for n in range(1, 41)] + [(40, 40)]
    items = []
    for length, stop_position in updates:
        item = codec.prepare(key, length, stop_position)
        if item is not None:
            items.append((item["token"].tolist(), item["token_offset_tokens"], item["finalize"]))
    assert [(len(tokens[0]), offset, end) for tokens, offset, end in items] == [
        (8, 0, False),
        (16, 5, False),
        (32, 13, False),
        (40, 29, True),
    ]
    assert items[-1][0] == [list(range(40))]
    assert codec.states[key].finished


def test_prompt_padding_is_removed_before_window_alignment():
    state, _, _, _ = _fixture()
    codec = state.codec
    codec.chunk_size, codec.lookahead = 4, 0
    conditioning = _chunk_conditioning(8)
    conditioning["speech_token_len"] = torch.tensor([[3]])
    key = "a"
    codec.initialize(key, conditioning, torch.arange(5).reshape(1, -1))
    item = codec.prepare(key, 5)
    assert item is not None
    assert item["prompt_token"].shape == (1, 3)
    assert item["prompt_feat"].shape == (1, 6, 80)
    assert item["token"].shape == (1, 5)
