# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project


import pytest
import torch

from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_codec import (
    MossTTSCodecDecoder,
    _MossCodecStreamSession,
)

pytestmark = [pytest.mark.core_model, pytest.mark.tts, pytest.mark.cpu]


class FakeGraphs:
    batch_sizes = [1, 2, 4, 8]

    def __init__(self):
        self.graphs = {(b, t): object() for b in self.batch_sizes for t in [1, 15]}

    def _select_frame_size(self, frames, allow_padding):
        return next((t for t in [1, 15] if t >= frames), None)

    def decode(self, codes, slots, **kwargs):
        # One sample per input token, preserving each row's zero padding.
        audio = codes.sum(0).unsqueeze(1).float()
        return audio, None, codes.shape[1]


def make_session():
    session = object.__new__(_MossCodecStreamSession)
    session._cudagraph_wrapper = FakeGraphs()
    session._device = torch.device("cpu")
    session._samples_per_frame = 1
    session._state_capacity = 8
    session._leased_slots = {0, 1, 2, 3}
    session._reset_slot_ids = lambda slots: None
    return session


def test_terminal_graph_buckets_require_complete_capture():
    session = make_session()
    assert session.terminal_graph_frame_size(1) == 1
    assert session.terminal_graph_frame_size(7) == 15
    assert session.terminal_graph_frame_size(16) == 16
    del session._cudagraph_wrapper.graphs[(4, 15)]
    assert session.terminal_graph_frame_size(7) == 7
    session._cudagraph_wrapper = None
    assert session.terminal_graph_frame_size(7) == 7


def test_mixed_terminal_tails_trim_to_individual_lengths():
    session = make_session()
    plan = {0: torch.ones(12, 2), 1: torch.full((12, 7), 2.0), 2: torch.full((12, 15), 3.0)}
    out = session.step(plan, terminal_slots={0, 1}, pad_to_frames=15)
    for slot, codes in plan.items():
        torch.testing.assert_close(out[slot], codes.sum(0).unsqueeze(0))


def test_never_pad_nonterminal_rows_or_silently_fall_back():
    session = make_session()
    plan = {0: torch.ones(12, 2), 1: torch.ones(12, 15)}
    with pytest.raises(ValueError, match="Only terminal"):
        session.step(plan, pad_to_frames=15)
    result = session.step(plan, terminal_slots={0})
    assert result[0].shape[-1] == 2
    session._cudagraph_wrapper = None
    with pytest.raises(RuntimeError, match="padded CUDA graph"):
        session.step(plan, terminal_slots={0}, pad_to_frames=15)


@pytest.mark.parametrize("enabled,expected_calls", [(False, 4), (True, 2)])
def test_dispatch_merges_only_tails_with_same_original_graph(enabled, expected_calls):
    calls = []
    session = make_session()
    free = iter(range(4))
    session.acquire = lambda: next(free)

    def step(plan, **kwargs):
        calls.append((plan, kwargs))
        return {slot: torch.zeros(2, codes.shape[1]) for slot, codes in plan.items()}

    session.step = step
    owner = object.__new__(MossTTSCodecDecoder)
    torch.nn.Module.__init__(owner)
    owner._codec_batch_io = enabled
    owner._stream_max_step_frames = 15
    owner._stream_req_slots = {}
    owner._ensure_stream_session = lambda: session
    owner._finish_stream_request = lambda *args, **kwargs: None
    # First-packet graph stays separate; T=2 and T=7 tails merge with T=15.
    items = [
        (i, f"r{i}", torch.zeros(12, t), finished)
        for i, (t, finished) in enumerate([(1, False), (2, True), (7, True), (15, False)])
    ]
    outputs = MossTTSCodecDecoder._decode_streaming_batch(owner, items)
    assert len(calls) == expected_calls
    assert [outputs[i].shape[-1] for i in range(4)] == [1, 2, 7, 15]
    if enabled:
        assert calls[1][1] == dict(terminal_slots={1, 2}, pad_to_frames=15)
