# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""MiniCPM-o 4.5 support for input-clocked sessions: where a speaking unit's audio ends, and audio-only input."""

from __future__ import annotations

import base64
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from tests.engine.duplex.test_session_runner import (
    _fake_encode_audio,
    append_audio,
    close_harness,
    open_harness,
    tts_output,
    types,
)
from vllm_omni.engine.duplex.config import apply_input_clock_capabilities
from vllm_omni.engine.duplex.plugin import DuplexUnitOutputs
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.capabilities import minicpmo45_native_capabilities
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.input import MiniCPMO45PcmAppendBuffer
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.plugin import MiniCPMO45DuplexPlugin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

SPEAK, CHUNK_EOS = 151706, 151718
INPUT_CLOCK = {"clock": "input"}


def _plugin() -> MiniCPMO45DuplexPlugin:
    return MiniCPMO45DuplexPlugin(_fake_encode_audio)


def _unit(*, upstream: bool, final_segment: bool = False, index: int = 0) -> DuplexUnitOutputs:
    return DuplexUnitOutputs(index=index, upstream_segment_finished=upstream, final_segment_finished=final_segment)


def _code2wav(*, last_chunk: bool, nested: bool = False) -> SimpleNamespace:
    meta = {"tts_is_last_chunk": np.array([last_chunk])}
    mm = (
        {"audio": np.zeros(240, np.float32), "meta": meta}
        if nested
        else {"meta.tts_is_last_chunk": meta["tts_is_last_chunk"]}
    )
    return SimpleNamespace(outputs=[SimpleNamespace(multimodal_output=mm)], multimodal_output=None)


def _complete(
    plugin: MiniCPMO45DuplexPlugin, unit: DuplexUnitOutputs | None, state: dict, output: object = None
) -> bool:
    return plugin.unit_output_complete(
        unit=unit, final_stage_id=2, new_output=output, new_context=None, state=state, runtime_config={}
    )


def test_minicpmo_opts_into_the_input_clock_for_audio() -> None:
    assert MiniCPMO45DuplexPlugin.supports_input_clock is True
    assert MiniCPMO45DuplexPlugin.input_clock_input_modalities == {"audio"}


# --------------------------------------------------------------------------- #
# Where a speaking unit's audio ends                                           #
# --------------------------------------------------------------------------- #


def test_a_speaking_unit_needs_its_talker_segment_end_and_a_flagged_code2wav_output() -> None:
    plugin = _plugin()
    flagged = _code2wav(last_chunk=True, nested=True)

    state: dict = {}  # Code2Wav's flushed tail first, then the Talker segment end
    assert not _complete(plugin, _unit(upstream=False), state, flagged)
    assert _complete(plugin, _unit(upstream=True), state)

    state = {}  # the Talker segment end first
    assert not _complete(plugin, _unit(upstream=True), state)
    assert not _complete(plugin, _unit(upstream=True), state, _code2wav(last_chunk=False))
    assert _complete(plugin, _unit(upstream=True), state, _code2wav(last_chunk=True))

    state = {}  # a final-stage segment end without the flag does not complete it
    assert not _complete(plugin, _unit(upstream=True, final_segment=True), state, object())

    state = {}  # the flag of unit 0 does not complete unit 1
    assert _complete(plugin, _unit(upstream=True), state, flagged)
    assert not _complete(plugin, _unit(upstream=True, index=1), state)
    assert not _complete(plugin, None, state, flagged), "no speaking unit: nothing to complete"


def test_a_trailing_code2wav_segment_end_of_a_completed_unit_does_not_complete_the_next_unit() -> None:
    """A repeated control-only turn-end snapshot of A arrives while B is pending: the engine holds it for B."""
    from tests.engine.duplex.test_input_clock import _append, _clock, _progress

    clock, sent = _clock(_plugin())
    (a,) = _append(clock)
    _progress(clock, stage_id=0, final_stage_id=2)  # A speaks
    _progress(clock, stage_id=1, final_stage_id=2)  # A's Talker segment end
    _progress(clock, stage_id=2, final_stage_id=2, output=_code2wav(last_chunk=True, nested=True))
    assert a.complete
    (b,) = _append(clock)  # B pending
    _progress(clock, stage_id=2, final_stage_id=2)  # A's trailing segment end, without the flag
    _progress(clock, stage_id=0, final_stage_id=2)  # B speaks

    _progress(clock, stage_id=1, final_stage_id=2)  # B's Talker segment end
    assert not b.complete, "B completed by A's trailing segment end before B's flagged output"
    _progress(clock, stage_id=2, final_stage_id=2, segment_finished=False, output=_code2wav(last_chunk=True))
    assert b.complete


def _speak_segment_end(request_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=request_id,
        finished=False,
        outputs=[SimpleNamespace(text="hi", token_ids=[SPEAK, 9, CHUNK_EOS], multimodal_output={})],
        multimodal_output={},
    )


@pytest.mark.parametrize("talker_first", [True, False], ids=["talker-then-code2wav", "code2wav-then-talker"])
async def test_a_speaking_unit_is_acknowledged_after_code2wav_flushes_it(talker_first: bool) -> None:
    h = await open_harness(extra_body=INPUT_CLOCK, stage_count=3)
    try:
        await h.run(append_audio())
        request_id = h.stage0_request_id()
        h.deliver(_speak_segment_end(request_id), stage_id=0, segment_finished=True)
        talker = SimpleNamespace(request_id=request_id, finished=False, outputs=[], multimodal_output={})
        flush = tts_output(request_id, samples=24000, text="hi", tts_is_last_chunk=True)
        if talker_first:
            h.deliver(talker, stage_id=1, segment_finished=True)
            events = await h.deliver_and_settle(flush, stage_id=2)
        else:
            events = await h.deliver_and_settle(flush, stage_id=2)
            assert "input_audio_buffer.processed" not in types(events)
            events = await h.deliver_and_settle(talker, stage_id=1, segment_finished=True)
        assert types(events)[-1] == "input_audio_buffer.processed"
        assert events[-1].units == ({"end_ms": 1000, "decision": "speak"},)
    finally:
        await close_harness(h)


async def test_a_talker_end_and_flush_that_overtake_their_stage0_segment_end_still_complete_the_unit() -> None:
    """The engine holds both until the unit speaks (no plugin bookkeeping needed)."""
    h = await open_harness(extra_body=INPUT_CLOCK, stage_count=3)
    try:
        await h.run(append_audio())
        request_id = h.stage0_request_id()
        talker = SimpleNamespace(request_id=request_id, finished=False, outputs=[], multimodal_output={})
        h.deliver(talker, stage_id=1, segment_finished=True)
        h.deliver(tts_output(request_id, samples=24000, text="hi", tts_is_last_chunk=True), stage_id=2)
        assert [e for e in await h.settle() if e.type == "input_audio_buffer.processed"] == []

        events = await h.deliver_and_settle(_speak_segment_end(request_id), stage_id=0, segment_finished=True)

        assert events[-1].type == "input_audio_buffer.processed"
        assert events[-1].units == ({"end_ms": 1000, "decision": "speak"},)
    finally:
        await close_harness(h)


# --------------------------------------------------------------------------- #
# Backlog, and audio-only input                                                #
# --------------------------------------------------------------------------- #


def test_the_input_buffer_hands_out_every_whole_chunk_an_append_completed() -> None:
    buffer = MiniCPMO45PcmAppendBuffer()
    payload = {
        "type": "audio",
        "audio": base64.b64encode(bytes(4 * 16000 * 3 + 4 * 3200)).decode(),
        "format": "pcm_f32le",
        "sample_rate_hz": 16000,
        "is_speech": True,
    }
    first = buffer.prepare_append(payload, operation_id="a", chunk_period_ms=1000)
    assert first is not None and first.byte_count == 4 * 16000
    backlog = [buffer.prepare_backlog(operation_id=f"b{i}", chunk_period_ms=1000) for i in range(3)]
    assert [r.byte_count if r is not None else None for r in backlog] == [4 * 16000, 4 * 16000, None]
    assert buffer.pending_byte_count == 4 * 3200


def test_an_input_clocked_session_keeps_only_the_models_clocked_input_modalities() -> None:
    capabilities = minicpmo45_native_capabilities()
    assert capabilities.accepts_input_modality("video")

    clocked = apply_input_clock_capabilities(capabilities, INPUT_CLOCK, input_modalities=frozenset({"audio"}))
    assert (clocked.required_input_modalities, clocked.optional_input_modalities) == (frozenset(), {"audio"})
    # Without the clock, or for a model that names no clocked modalities, nothing changes.
    assert apply_input_clock_capabilities(capabilities, {}, input_modalities=frozenset({"audio"})) is capabilities
    assert apply_input_clock_capabilities(capabilities, INPUT_CLOCK).optional_input_modalities == {"audio", "video"}


async def test_an_input_clocked_session_refuses_video_frames_and_acknowledges_the_append() -> None:
    """A committed turn with video frames would split into Stage-0 submissions the clock cannot account for."""
    h = await open_harness(extra_body=INPUT_CLOCK)
    try:
        assert h.session.capabilities.optional_input_modalities == {"audio"}
        events = await h.run(replace(append_audio(event_id="evt-video"), video_frames=("ZnJhbWU=",)))

        assert types(events) == ["error", "input_audio_buffer.processed"]
        assert (events[0].code, events[0].related_event_id) == ("invalid_input_modality", "evt-video")
        assert (events[1].input_index, events[1].units) == (1, ())
        assert (events[1].decision, events[1].reason) == ("rejected", "invalid_input_modality")

        events = await h.run(append_audio())  # audio is unaffected
        assert "error" not in types(events)
    finally:
        await close_harness(h)


async def test_without_the_clock_video_frames_are_accepted_as_before() -> None:
    h = await open_harness()
    try:
        assert h.session.capabilities.accepts_input_modality("video")
        events = await h.run(replace(append_audio(event_id="evt-video"), video_frames=("ZnJhbWU=",)))
        assert "error" not in types(events)
    finally:
        await close_harness(h)
