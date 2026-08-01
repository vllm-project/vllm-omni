# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU coverage for Gepard's prompt layout and streaming-decode window.

The e2e test needs a GPU, NeMo and the checkpoint, so it only runs nightly.
These pin the parts that are pure logic and break silently: which frames each
decode covers, which samples survive the lookback trim, and the prompt layout.

The codec is replaced by a shift-invariant stand-in whose output depends only
on the codes, which makes "chunked emissions concatenate to one whole-history
decode" an exact assertion — the contract lookback exists to provide.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from vllm_omni.model_executor.models.gepard.configuration_gepard import GepardConfig
from vllm_omni.model_executor.models.gepard.gepard_talker import (
    CHUNK_FRAMES,
    FIRST_CHUNK_FRAMES,
    GepardTalkerForConditionalGeneration,
    _GepardState,
)
from vllm_omni.model_executor.models.gepard.nanocodec import (
    LOOKBACK_FRAMES,
    NanoCodec,
    apply_end_of_speech_tail,
)
from vllm_omni.model_executor.models.gepard.prompt import build_gepard_prompt_ids

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

SAMPLES_PER_FRAME = 1024
NUM_HEADS = 32
SAMPLE_RATE = 22050


class _ShiftInvariantCodec:
    """Frame tagged ``v`` -> samples ``v*SPF .. v*SPF+SPF-1``, wherever it sits."""

    def __init__(self) -> None:
        self.windows: list[int] = []  # frame count of every decode call

    def decode_from_codes(self, codes: torch.Tensor, codes_len: torch.Tensor):
        assert codes.shape[0] == 1, "streaming decode is per-request"
        assert codes.shape[1] == NUM_HEADS, f"expected {NUM_HEADS} code channels"
        tags = codes[0, 0, :].to(torch.int64)  # (T,)
        self.windows.append(int(tags.shape[0]))
        offsets = torch.arange(SAMPLES_PER_FRAME, dtype=torch.int64)
        audio = (tags.unsqueeze(1) * SAMPLES_PER_FRAME + offsets).reshape(1, -1)
        return audio.to(torch.float32), codes_len * SAMPLES_PER_FRAME


def _make_codec() -> tuple[NanoCodec, _ShiftInvariantCodec]:
    codec = NanoCodec(codec_id="test/fake-codec", sample_rate=SAMPLE_RATE)
    fake = _ShiftInvariantCodec()
    codec._codec = fake  # bypasses load(); NeMo is never imported
    assert codec.is_loaded
    return codec, fake


def _make_talker(codec: NanoCodec) -> GepardTalkerForConditionalGeneration:
    """A real talker with only the audio-path attributes populated.

    ``__init__`` needs a VllmConfig and builds the backbone; the methods under
    test touch only these four attributes.
    """
    talker = GepardTalkerForConditionalGeneration.__new__(GepardTalkerForConditionalGeneration)
    nn.Module.__init__(talker)  # sets up _parameters/_buffers/_modules only
    talker._codec = codec
    talker._audio_queue = []
    talker._active_states = {}
    talker._deferred_cleanup_ids = set()
    return talker


def _frame(tag: int) -> torch.Tensor:
    """One committed frame: 32 per-dimension codes, all carrying the tag."""
    return torch.full((NUM_HEADS,), tag, dtype=torch.long)


def _expected_samples(num_frames: int) -> torch.Tensor:
    offsets = torch.arange(SAMPLES_PER_FRAME, dtype=torch.int64)
    tags = torch.arange(num_frames, dtype=torch.int64).unsqueeze(1)
    return (tags * SAMPLES_PER_FRAME + offsets).reshape(-1).to(torch.float32)


def _drain(talker) -> torch.Tensor:
    """Concatenate and clear the queued deltas, as make_omni_output does."""
    if not talker._audio_queue:
        return torch.empty(0)
    out = torch.cat([a.reshape(-1) for _, a in talker._audio_queue])
    talker._audio_queue.clear()
    return out


def _generate(talker, state, num_frames: int, *, stop_at_end: bool) -> None:
    """Drive the commit loop the way forward() does, one frame per step."""
    for tag in range(num_frames):
        state.frames.append(_frame(tag))
        state.frame_count += 1
        state.past_first_step = True
        talker._emit_audio(state, is_final=stop_at_end and tag == num_frames - 1)


# Counts straddling both cadence thresholds, where an off-by-one hides.
@pytest.mark.parametrize(
    "num_frames",
    [
        1,
        2,
        FIRST_CHUNK_FRAMES - 1,
        FIRST_CHUNK_FRAMES,
        FIRST_CHUNK_FRAMES + 1,
        CHUNK_FRAMES - 1,
        CHUNK_FRAMES,
        CHUNK_FRAMES + 1,
        FIRST_CHUNK_FRAMES + CHUNK_FRAMES,
        79,
        200,
    ],
)
def test_chunked_emissions_equal_a_whole_history_decode(num_frames: int) -> None:
    codec, _fake = _make_codec()
    talker = _make_talker(codec)
    state = _GepardState(request_id="req-0")

    _generate(talker, state, num_frames, stop_at_end=True)

    torch.testing.assert_close(_drain(talker), _expected_samples(num_frames))
    assert state.emitted_frames == num_frames


def test_every_frame_is_emitted_exactly_once() -> None:
    """The trim must not drop or duplicate samples at a chunk boundary."""
    codec, _fake = _make_codec()
    talker = _make_talker(codec)
    state = _GepardState(request_id="req-0")
    num_frames = FIRST_CHUNK_FRAMES + 2 * CHUNK_FRAMES + 3

    _generate(talker, state, num_frames, stop_at_end=True)
    audio = _drain(talker)

    assert audio.numel() == num_frames * SAMPLES_PER_FRAME
    # Tags are globally unique, so a duplicated or dropped frame shows up here.
    assert len(torch.unique(audio)) == audio.numel()


def test_decode_windows_carry_lookback_context() -> None:
    """Every decode after the first must re-decode LOOKBACK_FRAMES of context."""
    codec, fake = _make_codec()
    talker = _make_talker(codec)
    state = _GepardState(request_id="req-0")
    num_frames = FIRST_CHUNK_FRAMES + 2 * CHUNK_FRAMES

    _generate(talker, state, num_frames, stop_at_end=True)

    assert len(fake.windows) >= 3, "expected a first chunk plus steady-state chunks"
    assert fake.windows[0] == FIRST_CHUNK_FRAMES, "the first chunk has no left context"
    for window in fake.windows[1:]:
        assert window == CHUNK_FRAMES + LOOKBACK_FRAMES


def test_codec_runs_once_per_chunk_not_once_per_frame() -> None:
    codec, fake = _make_codec()
    talker = _make_talker(codec)
    state = _GepardState(request_id="req-0")
    num_frames = FIRST_CHUNK_FRAMES + 3 * CHUNK_FRAMES

    _generate(talker, state, num_frames, stop_at_end=True)

    assert len(fake.windows) == 4
    assert sum(fake.windows) < num_frames * 2  # far below one decode per frame


def test_tail_is_flushed_when_a_request_ends_without_stop() -> None:
    """max_tokens / abort finish the request with no STOP frame; the frames
    committed since the last chunk must still reach the queue."""
    codec, _fake = _make_codec()
    talker = _make_talker(codec)
    state = _GepardState(request_id="req-0")
    num_frames = FIRST_CHUNK_FRAMES + CHUNK_FRAMES + 5
    talker._active_states["req-0"] = state

    _generate(talker, state, num_frames, stop_at_end=False)
    assert state.emitted_frames < num_frames, "a partial chunk should still be pending"

    talker._deferred_cleanup_ids.add("req-0")  # on_requests_finished's effect
    talker._flush_deferred_cleanup()

    torch.testing.assert_close(_drain(talker), _expected_samples(num_frames))
    assert "req-0" not in talker._active_states


def test_emitting_with_nothing_pending_is_a_no_op() -> None:
    codec, fake = _make_codec()
    talker = _make_talker(codec)
    state = _GepardState(request_id="req-0")

    _generate(talker, state, 4, stop_at_end=True)
    calls_after_flush = len(fake.windows)
    talker._audio_queue.clear()

    talker._emit_audio(state, is_final=True)

    assert len(fake.windows) == calls_after_flush, "codec ran with no new frames"
    assert talker._audio_queue == []


def test_concurrent_requests_keep_separate_windows() -> None:
    codec, _fake = _make_codec()
    talker = _make_talker(codec)
    states = [_GepardState(request_id=f"req-{i}") for i in range(2)]

    for state in states:
        _generate(talker, state, FIRST_CHUNK_FRAMES, stop_at_end=True)

    by_req: dict[str, list[torch.Tensor]] = {}
    for req_id, audio in talker._audio_queue:
        by_req.setdefault(req_id, []).append(audio.reshape(-1))
    assert set(by_req) == {"req-0", "req-1"}
    for chunks in by_req.values():
        torch.testing.assert_close(torch.cat(chunks), _expected_samples(FIRST_CHUNK_FRAMES))


def test_end_of_speech_tail_fades_and_pads() -> None:
    audio = torch.ones(SAMPLE_RATE)  # 1 s of full-scale signal
    out = apply_end_of_speech_tail(audio, SAMPLE_RATE, fade_ms=10.0, silence_ms=20.0)

    fade = int(SAMPLE_RATE * 10.0 / 1000.0)
    pad = int(SAMPLE_RATE * 20.0 / 1000.0)
    assert out.numel() == SAMPLE_RATE + pad
    assert torch.all(out[-pad:] == 0.0), "trailing pad must be silent"
    assert out[SAMPLE_RATE - fade - 1] == 1.0, "the fade must not reach back past fade_ms"
    # Monotonically ramped to zero — the pad alone would leave the click.
    ramp = out[SAMPLE_RATE - fade : SAMPLE_RATE]
    assert torch.all(ramp[:-1] >= ramp[1:])
    assert ramp[-1] == 0.0


def test_end_of_speech_tail_is_a_no_op_by_default() -> None:
    audio = torch.ones(128)
    out = apply_end_of_speech_tail(audio, SAMPLE_RATE, fade_ms=0.0, silence_ms=0.0)
    assert out is audio


def test_gepard_prompt_layout() -> None:
    """The prompt carries the speaker slots and exactly one SOS, at the end."""
    cfg = GepardConfig()
    ids = build_gepard_prompt_ids(
        [1, 2, 3],
        start_of_text=cfg.start_of_text,
        end_of_text=cfg.end_of_text,
        start_of_speech=cfg.start_of_speech,
        speaker_token_base=cfg.speaker_token_base,
        num_speaker_prefix=cfg.num_speaker_prefix,
    )

    slots = [cfg.speaker_token_base + i for i in range(cfg.num_speaker_prefix)]
    assert ids[: cfg.num_speaker_prefix] == slots
    assert ids.count(cfg.start_of_speech) == 1, "only the canonical copy may carry SOS"
    assert ids[-1] == cfg.start_of_speech, "the first frame is sampled from the SOS position"
