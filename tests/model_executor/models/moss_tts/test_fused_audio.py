# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Chunking, delivery and stop handling of the single-stage MOSS codec path."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.helpers.mark import hardware_marks
from vllm_omni.data_entry_keys import FIRST_AUDIO_REQUIRED_KEY
from vllm_omni.model_executor.models.moss_tts.fused_audio import MossFusedAudio

pytestmark = [
    pytest.mark.core_model,
    *hardware_marks(res={"cuda": "L4"}, num_cards=1),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA streams"),
]

N_VQ = 3
SPF = 4
PAD = 99


class _FakeCodec:
    """Decodes each frame to SPF samples per channel holding its first code.

    Each decode first sleeps on the codec stream, so audio is never complete
    on the step that cut it.
    """

    def __init__(self, initial=1, chunk=3, packed=False, **options):
        # Decode as soon as a chunk is cut unless a test asks for batching.
        self.options = {"fused_decode_min_batch": 1, "fused_decode_max_wait_steps": 0, **options}
        self._stream_chunk_frames = chunk
        self._initial_stream_chunk_frames = initial
        self._stream_session = SimpleNamespace(_samples_per_frame=SPF)
        self._sr_tensor = torch.tensor(48000, dtype=torch.int32)
        self._codec_stream = None
        self.packed_decode_ready = packed
        self.calls = []
        self.packed_calls = []
        self.finished = []

    def _connector_int(self, name, default=0):
        return int(self.options.get(name, default))

    def _connector_extra(self):
        return self.options

    def _decode_streaming_packed(self, items):
        self.packed_calls.append(len(self.calls))
        return self._decode_streaming_batch(items)

    def _decode_streaming_batch(self, items):
        assert all(not codes.eq(PAD).any() for _, _, codes, _ in items), "padding reached the codec"
        assert len({req_id for _, req_id, _, _ in items}) == len(items), "two chunks of one stream in one call"
        self.calls.append([(req_id, codes[0].tolist(), finished) for _, req_id, codes, finished in items])
        torch.cuda._sleep(20_000_000)
        out = {}
        for index, _req_id, codes, _finished in items:
            first = codes[0].to(torch.float32)  # [T]
            out[index] = first.repeat_interleave(SPF).unsqueeze(0).repeat(2, 1)  # [C=2, T*SPF]
        return out

    def on_requests_finished(self, req_ids):
        self.finished.extend(req_ids)


class _FakeSender:
    """Records tail deliveries; the scheduler deferred these requests' finish."""

    def __init__(self):
        self.deferred = set()
        self.sent = []  # (request, codes, final)
        self.after_first = []  # after_first flag per submit

    def is_deferred(self, req_id):
        return req_id in self.deferred

    def submit(self, req_id, pieces, sample_rate, final, *, after_first=False):
        torch.accelerator.synchronize()
        codes = []
        for pcm, _event in pieces:
            codes.extend(pcm[0, ::SPF].to(torch.long).tolist())
        self.sent.append((req_id, codes, final))
        self.after_first.append(after_first)


class _Batch:
    def __init__(self, slots, req_ids):
        self.num_reqs = len(slots)
        self.idx_mapping_np = np.asarray(slots, dtype=np.int64)
        self.req_ids = list(req_ids)


def _fused(initial, chunk, **options):
    codec = _FakeCodec(initial=initial, chunk=chunk, **options)
    fused = MossFusedAudio(codec, capacity=8, n_vq=N_VQ, pad_id=PAD, device=torch.device("cuda"))
    sender = _FakeSender()
    fused.tail_sender = sender
    return codec, fused, sender


def _run(fused, slots, frames, markers=None):
    """One runner step: drain, then a decode step; ``frames`` maps slot -> first code (None = padding).

    Returns {request: (continue, delivered codes or None)}. ``markers`` collects
    the requests whose delivered audio must follow an early first frame.
    """
    fused.drain_detached_audio()
    req_ids = [fused.streams[slot].req_id for slot in slots]
    batch = _Batch(slots, req_ids)
    codes = torch.full((len(slots), N_VQ), PAD, dtype=torch.long, device="cuda")
    for row, slot in enumerate(slots):
        if frames[slot] is not None:
            codes[row] = frames[slot]
    keep = codes.ne(PAD).any(dim=-1)
    fused.step(batch, list(range(len(slots))), codes, keep, lambda data: torch.from_numpy(data).cuda())
    step = fused.take_step(batch.num_reqs)
    host = step.to_cpu(torch.cuda.current_stream(), lambda tensor: tensor.cpu())
    torch.accelerator.synchronize()
    output = host.get_output()
    if markers is not None:
        markers.update(
            req_id
            for req_id, payload in zip(req_ids, output, strict=True)
            if payload is not None and bool(payload.get(FIRST_AUDIO_REQUIRED_KEY, False))
        )
    return {
        req_id: (bool(go), None if payload is None else payload["model_outputs"][0, ::SPF].to(torch.long).tolist())
        for req_id, go, payload in zip(req_ids, keep.cpu().tolist(), output, strict=True)
    }


def _stop(fused, sender, req_id, steps=6):
    """The scheduler finished ``req_id`` at its stop and deferred its terminal output."""
    sender.deferred.add(req_id)
    fused.finish([req_id])
    for _ in range(steps):
        fused.drain_detached_audio()
        torch.accelerator.synchronize()


def _stream_audio(delivered, sender, req_id):
    audio = [codes for codes in delivered]
    audio += [codes for req, codes, _final in sender.sent if req == req_id and codes]
    finals = [final for req, _codes, final in sender.sent if req == req_id]
    return audio, finals


def test_live_chunks_then_tail_after_the_stop():
    codec, fused, sender = _fused(initial=1, chunk=3)
    fused.add_request(2, "a")
    delivered = []
    for code in [10, 11, 12, 13, 14, None]:
        go, audio = _run(fused, [2], {2: code})["a"]
        if audio is not None:
            delivered.append(audio)
    # The stop token is on the padding frame's own step.
    assert go is False
    _stop(fused, sender, "a")
    audio, finals = _stream_audio(delivered, sender, "a")
    assert audio == [[10], [11, 12, 13], [14]]
    assert finals[-1] is True and finals.count(True) == 1
    assert [[finished for _, _, finished in call] for call in codec.calls][-1] == [True]
    assert "a" not in fused.detached


def test_abort_releases_without_delivery():
    codec, fused, sender = _fused(initial=1, chunk=3)
    fused.add_request(0, "x")
    _run(fused, [0], {0: 5})
    _run(fused, [0], {0: 6})
    fused.finish(["x"])  # not deferred: aborted
    assert codec.finished == ["x"] and sender.sent == [] and not fused.detached


def test_abort_of_detached_request_releases_pending_codec_work():
    codec, fused, sender = _fused(initial=1, chunk=15)
    fused.add_request(0, "x")
    _run(fused, [0], {0: 5})
    _run(fused, [0], {0: 6})
    sender.deferred.add("x")
    fused.finish(["x"])
    assert "x" in fused.detached
    sender.deferred.remove("x")
    fused.finish(["x"])
    assert "x" not in fused.detached and codec.finished == ["x"]
    assert all(chunk.req_id != "x" for chunk in fused._ready)
    assert all(audio.req_id != "x" for audio in fused._decoded)


def test_undelivered_audio_goes_ahead_of_the_tail():
    _, fused, sender = _fused(initial=1, chunk=2)
    fused.add_request(1, "u")
    delivered = []
    for code in [7, 8, 9]:
        _, audio = _run(fused, [1], {1: code})["u"]
        if audio is not None:
            delivered.append(audio)
    # [8, 9] was cut on the last step and not yet delivered when the request stops.
    _run(fused, [1], {1: None})
    _stop(fused, sender, "u")
    audio, finals = _stream_audio(delivered, sender, "u")
    assert sum(audio, []) == [7, 8, 9]
    assert finals[-1] is True


def test_length_limit_delivers_every_frame():
    _, fused, sender = _fused(initial=1, chunk=4)
    fused.add_request(3, "l")
    delivered = []
    for code in [1, 2, 3]:
        _, audio = _run(fused, [3], {3: code})["l"]
        if audio is not None:
            delivered.append(audio)
    # Finished at its length limit: no padding frame, all frames are audio.
    _stop(fused, sender, "l")
    audio, finals = _stream_audio(delivered, sender, "l")
    assert sum(audio, []) == [1, 2, 3] and finals[-1] is True


def test_empty_tail_finishes_without_audio():
    codec, fused, sender = _fused(initial=1, chunk=2)
    fused.add_request(0, "e")
    delivered = []
    # 40: a frame of a step that ran before the scheduler saw the stop.
    for code in [7, 8, 9, None, 40]:
        _, audio = _run(fused, [0], {0: code})["e"]
        if audio is not None:
            delivered.append(audio)
    _stop(fused, sender, "e")
    audio, finals = _stream_audio(delivered, sender, "e")
    assert sum(audio, []) == [7, 8, 9]
    assert sender.sent[-1] == ("e", [], True)
    assert codec.finished == ["e"]


@pytest.mark.parametrize("ramp", [[], [2, 3]])
def test_detached_tail_survives_slot_reuse(ramp):
    _, fused, sender = _fused(initial=1, chunk=5, codec_chunk_ramp_frames=ramp)
    fused.add_request(4, "old")
    delivered = []
    for code in [1, 2, 3, None]:
        _, audio = _run(fused, [4], {4: code})["old"]
        if audio is not None:
            delivered.append(audio)
    sender.deferred.add("old")
    fused.finish(["old"])
    fused.add_request(4, "new")
    new_audio = []
    for code in [50, 51, 52, 53, 54, 55]:
        _, audio = _run(fused, [4], {4: code})["new"]
        if audio is not None:
            new_audio.append(audio)
    _stop(fused, sender, "x-none", steps=3)
    audio, finals = _stream_audio(delivered, sender, "old")
    assert sum(audio, []) == [1, 2, 3] and finals[-1] is True
    assert new_audio[0] == [50]


@pytest.mark.parametrize("early", [False, True])
@pytest.mark.parametrize("padding_stop", [False, True])
@pytest.mark.parametrize(
    "frames,chunk_sizes",
    [
        (1, [1]),
        (3, [1, 2]),
        (4, [1, 3]),
        (10, [1, 3, 6]),
        (11, [1, 3, 7]),
        (26, [1, 3, 7, 15]),
        (27, [1, 3, 7, 15, 1]),
        (42, [1, 3, 7, 15, 15, 1]),
    ],
)
def test_chunk_ramp_preserves_frames_at_stop_boundaries(early, padding_stop, frames, chunk_sizes):
    codec, fused, sender = _fused(initial=1, chunk=15, packed=True, codec_chunk_ramp_frames=[3, 7])
    first = _FakeFirstSender()
    fused.first_sender = first
    # Reusing the slot must restart the ramp, including an early first frame.
    for req_id in ["old", "new"]:
        fused.add_request(0, req_id)
        if early:
            assert _first_frames(fused, {req_id: 1}) == [req_id]
        delivered = []
        for code in range(1, frames + 1):
            _, audio = _run(fused, [0], {0: code})[req_id]
            if audio is not None:
                delivered.append(audio)
        if padding_stop:
            _, audio = _run(fused, [0], {0: None})[req_id]
            if audio is not None:
                delivered.append(audio)
        _stop(fused, sender, req_id)
        audio, finals = _stream_audio(delivered, sender, req_id)
        assert ([1] if early else []) + sum(audio, []) == list(range(1, frames + 1))
        chunks = [codes for call in codec.calls for rid, codes, _ in call if rid == req_id]
        assert [len(codes) for codes in chunks] == chunk_sizes
        assert sum(chunks, []) == list(range(1, frames + 1))
        assert finals.count(True) == 1 and finals[-1]
    expected_first = [([req_id], [1], [True]) for req_id in ["old", "new"]] if early else []
    assert first.sent == expected_first


@pytest.mark.parametrize("ramp", [[0], [-1], [True], [3, 3], [7, 3], [3, 15], "3,7"])
def test_chunk_ramp_rejects_invalid_sizes(ramp):
    with pytest.raises(ValueError, match="codec_chunk_ramp_frames"):
        _fused(initial=1, chunk=15, codec_chunk_ramp_frames=ramp)


def test_ramp_chunks_use_the_first_chunk_wait_budget():
    codec, fused, _ = _fused(
        initial=1,
        chunk=15,
        packed=True,
        codec_chunk_ramp_frames=[3, 7],
        fused_decode_min_batch=99,
        fused_decode_max_wait_steps=20,
        fused_decode_first_wait_steps=1,
    )
    fused.add_request(0, "ramp")
    for code in range(1, 7):
        _run(fused, [0], {0: code})
    assert codec.calls == [[("ramp", [1], False)], [("ramp", [2, 3, 4], False)]]


def test_decodes_wait_for_a_batch_or_the_wait_limit():
    codec, fused, _ = _fused(
        initial=1,
        chunk=2,
        fused_decode_min_batch=3,
        fused_decode_max_wait_steps=2,
        fused_decode_first_wait_steps=5,
    )
    for slot in range(3):
        fused.add_request(slot, f"s{slot}")
    _run(fused, [0, 1, 2], {0: 10, 1: 20, 2: 30})
    _run(fused, [0, 1, 2], {0: 11, 1: 21, 2: 31})
    assert [sorted(req for req, _, _ in call) for call in codec.calls] == [["s0", "s1", "s2"]]
    calls = len(codec.calls)
    for code in [12, 13, 14, 15]:
        _run(fused, [0], {0: code})
    assert len(codec.calls) > calls
    assert codec.calls[calls] == [("s0", [11, 12], False)]


def test_queued_chunks_of_one_stream_decode_in_order():
    codec, fused, _ = _fused(initial=1, chunk=2, fused_decode_min_batch=99, fused_decode_max_wait_steps=4)
    fused.add_request(0, "o")
    audio = []
    for code in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
        _, codes = _run(fused, [0], {0: code})["o"]
        if codes is not None:
            audio.extend(codes)
    order = [codes for call in codec.calls for _, codes, _ in call]
    assert order == sorted(order, key=lambda c: c[0])
    assert audio == sorted(audio) and audio[:3] == [1, 2, 3]


def test_a_due_first_chunk_does_not_take_steady_chunks_along():
    codec, fused, _ = _fused(
        initial=1,
        chunk=2,
        fused_decode_min_batch=99,
        fused_decode_max_wait_steps=6,
        fused_decode_first_wait_steps=1,
    )
    fused.add_request(0, "a")
    for code in [1, 2, 3]:
        _run(fused, [0], {0: code})
    fused.add_request(1, "b")
    for step in range(3):
        _run(fused, [0, 1], {0: 4 + step, 1: 10 + step})
    calls = [[(req, codes) for req, codes, _ in call] for call in codec.calls]
    # b's first chunk decodes alone; a's steady chunk [2, 3] keeps waiting for its batch.
    assert [("b", [10])] in calls
    assert all(("a", [2, 3]) not in call for call in calls[: calls.index([("b", [10])]) + 1])
    for step in range(6):
        _run(fused, [0, 1], {0: 7 + step, 1: 13 + step})
    order = [codes for call in codec.calls for req, codes, _ in call if req == "a"]
    assert order[:2] == [[1], [2, 3]]


def test_packed_decode_mixes_chunk_lengths_and_keeps_the_interval():
    codec, fused, _ = _fused(
        initial=1,
        chunk=2,
        packed=True,
        fused_decode_min_batch=99,
        fused_decode_max_wait_steps=4,
        fused_decode_first_wait_steps=1,
        fused_decode_packed_interval=2,
    )
    steps = []
    decode = codec._decode_streaming_packed

    def recording_decode(items):
        steps.append(fused._steps)
        return decode(items)

    codec._decode_streaming_packed = recording_decode
    fused.add_request(0, "a")
    for code in [1, 2, 3]:
        _run(fused, [0], {0: code})
    fused.add_request(1, "b")
    for step in range(8):
        _run(fused, [0, 1], {0: 4 + step, 1: 10 + step})
    assert codec.calls and len(codec.packed_calls) == len(codec.calls)
    # A launch may hold several waves (calls); launches are at least the interval apart.
    launches = sorted(set(steps))
    assert all(later - earlier >= 2 for earlier, later in zip(launches, launches[1:]))
    # One packed call carries chunks of different lengths: b's first chunk with a's steady one.
    assert any(len({len(codes) for _, codes, _ in call}) > 1 for call in codec.calls)
    order = [codes for call in codec.calls for req, codes, _ in call if req == "a"]
    assert order[:2] == [[1], [2, 3]]


def test_packed_last_chunk_launches_after_last_wait_steps():
    def steps_until_tail(last_wait):
        codec, fused, sender = _fused(
            initial=1,
            chunk=3,
            packed=True,
            fused_decode_min_batch=99,
            fused_decode_max_wait_steps=6,
            fused_decode_first_wait_steps=99,
            fused_decode_last_wait_steps=last_wait,
        )
        fused.add_request(0, "a")
        fused.add_request(1, "b")
        # a: chunks [10], [11, 12, 13], then a two-frame last chunk [14, 15].
        for step in range(6):
            _run(fused, [0, 1], {0: 10 + step, 1: 30 + step})
        _run(fused, [0, 1], {0: None, 1: 36})  # a stops on its padding frame
        sender.deferred.add("a")
        fused.finish(["a"])
        start = len(codec.calls)
        for step in range(10):
            _run(fused, [1], {1: 37 + step})
            if any(req == "a" and last for call in codec.calls[start:] for req, _, last in call):
                return step
        return None

    eager, lazy = steps_until_tail(0), steps_until_tail(6)
    assert eager is not None and lazy is not None and eager < lazy


class _FakeFirstSender:
    """Prepares routes for live requests and records first frames once decoded."""

    def __init__(self):
        self.sent = []  # (request ids, first codes, valid)

    def prepare(self, req_ids):
        return SimpleNamespace(sink="sink", routes={req_id: 0 for req_id in req_ids})

    def submit_prepared(self, delivery, req_ids, pcm, sample_rate, valid):
        torch.accelerator.synchronize()
        self.sent.append((list(req_ids), pcm[:, 0, ::SPF].to(torch.long).reshape(-1).tolist(), valid.tolist()))


def _first_frames(fused, frames):
    """Frames just sampled after the final prefill: {request: first code (None = padding)}."""
    ids = list(frames)
    codes = torch.full((len(ids), N_VQ), PAD, dtype=torch.long, device="cuda")
    for row, req_id in enumerate(ids):
        if frames[req_id] is not None:
            codes[row] = frames[req_id]
    valid = codes.ne(PAD).any(dim=-1)
    return fused.add_first_frames(ids, codes, valid)


def _early_fused(**options):
    codec, fused, sender = _fused(initial=1, chunk=3, packed=True, **options)
    first = _FakeFirstSender()
    fused.first_sender = first
    return codec, fused, sender, first


def test_early_first_frame_decodes_once_and_goes_to_the_sender():
    codec, fused, sender, first = _early_fused()
    fused.add_request(0, "a")
    fused.add_request(1, "b")
    assert _first_frames(fused, {"a": 10}) == ["a"]
    delivered: dict[str, list[list[int]]] = {"a": [], "b": []}
    markers: set[str] = set()
    for step, code in enumerate([10, 11, 12, 13, 14, None]):
        result = _run(fused, [0, 1], {0: code, 1: 30 + step}, markers)
        for req_id, (_go, audio) in result.items():
            if audio is not None:
                delivered[req_id].append(audio)
    _stop(fused, sender, "a")
    # The first frame went to the sender as soon as its decode completed, decoded once.
    assert first.sent == [(["a"], [10], [True])]
    order = [codes for call in codec.calls for req, codes, _ in call if req == "a"]
    assert order[0] == [10] and sum(order, []) == [10, 11, 12, 13, 14]
    audio, finals = _stream_audio(delivered["a"], sender, "a")
    assert sum(audio, []) == [11, 12, 13, 14] and finals[-1] is True
    assert "a" in markers and "b" not in markers and delivered["b"][0] == [30]
    assert not fused.after_first and not fused.early


def test_early_padding_first_frame_finishes_without_audio():
    codec, fused, sender, first = _early_fused()
    fused.add_request(0, "p")
    _first_frames(fused, {"p": None})
    _run(fused, [0], {0: None})
    _stop(fused, sender, "p")
    # Decoded as code 0 and dropped by validity; the terminal waits for nothing.
    assert first.sent == [(["p"], [0], [False])]
    assert sender.sent == [("p", [], True)] and sender.after_first == [False]
    assert codec.finished == ["p"] and not fused.early and not fused.after_first


def test_early_first_frame_of_a_length_stopped_request():
    _, fused, sender, first = _early_fused()
    fused.add_request(0, "l")
    _first_frames(fused, {"l": 7})
    _run(fused, [0], {0: 7})
    _stop(fused, sender, "l")
    # Its only audio went ahead; the terminal output must follow it.
    assert first.sent == [(["l"], [7], [True])]
    assert sender.sent == [("l", [], True)] and sender.after_first == [True]


def test_early_first_frame_launches_at_once_and_skips_the_mask_wait():
    codec, fused, _sender, first = _early_fused(fused_decode_min_batch=99, fused_decode_max_wait_steps=9)
    fused.add_request(0, "a")
    _first_frames(fused, {"a": 10})
    # The first step after the final prefill decodes it: no continue-mask wait.
    _run(fused, [0], {0: 10})
    assert codec.calls == [[("a", [10], False)]] and first.sent == [(["a"], [10], [True])]


def test_early_first_frame_without_a_route_keeps_the_regular_first_chunk():
    _, fused, _sender, first = _early_fused()
    first.prepare = lambda req_ids: SimpleNamespace(sink="sink", routes={})
    fused.add_request(0, "r")
    assert _first_frames(fused, {"r": 4}) == []
    delivered = []
    for code in [4, 5, 6, 7, 8]:
        _, audio = _run(fused, [0], {0: code})["r"]
        if audio is not None:
            delivered.append(audio)
    assert delivered[0] == [4] and first.sent == []


def test_early_first_frame_released_on_abort():
    codec, fused, sender, first = _early_fused()
    fused.add_request(0, "x")
    _first_frames(fused, {"x": 5})
    _run(fused, [0], {0: 5})
    _run(fused, [0], {0: 6})
    fused.finish(["x"])  # not deferred: aborted
    assert codec.finished == ["x"] and sender.sent == []
    assert not fused.early and not fused.after_first
