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
    LISTEN_TOKEN_ID,
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
from vllm_omni.protocol.duplex.events import InputProcessed

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

SPEAK, CHUNK_EOS = 151706, 151718
INPUT_CLOCK: dict[str, object] = {"clock": "input"}


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


def test_the_flushes_of_queued_speaking_units_are_consumed_in_unit_order() -> None:
    """The engine hands every final-stage output to the oldest speaking unit, whichever unit it belongs to."""
    plugin = _plugin()
    flagged = _code2wav(last_chunk=True, nested=True)

    state: dict = {}  # both flushes before either Talker segment end
    assert not _complete(plugin, _unit(upstream=False), state, flagged)
    assert not _complete(plugin, _unit(upstream=False), state, flagged)
    assert _complete(plugin, _unit(upstream=True), state)
    assert _complete(plugin, _unit(upstream=True, index=1), state)
    assert not _complete(plugin, _unit(upstream=True, index=2), state), "a third unit has no flush yet"

    state = {}  # both Talker segment ends before either flush
    assert not _complete(plugin, _unit(upstream=True), state)
    assert _complete(plugin, _unit(upstream=True), state, flagged)
    assert not _complete(plugin, _unit(upstream=True, index=1), state)
    assert _complete(plugin, _unit(upstream=True, index=1), state, flagged)


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


@pytest.mark.parametrize("flush_ends_segment", [False, True], ids=["mid-turn flush", "turn-end flush"])
@pytest.mark.parametrize("order", ["flushes-first", "talker-ends-first", "interleaved"])
async def test_two_queued_speaking_units_are_each_acknowledged_after_their_own_audio(
    order: str, flush_ends_segment: bool
) -> None:
    """The orchestrator reads the Talker and Code2Wav streams independently, so both flushes can precede the first Talker end.

    A flush that ends the turn is also a Code2Wav segment end (``flush_ends_segment``), which the engine credits
    separately; the hook must not depend on it.
    """
    steps = {
        "flushes-first": ["flush", "flush", "talker", "talker"],
        "talker-ends-first": ["talker", "talker", "flush", "flush"],
        "interleaved": ["talker", "flush", "flush", "talker"],
    }[order]
    h = await open_harness(extra_body=INPUT_CLOCK, stage_count=3)
    try:
        await h.run(append_audio())
        await h.run(append_audio())
        request_id = h.stage0_request_id()
        h.deliver(_speak_segment_end(request_id), stage_id=0, segment_finished=True)
        h.deliver(_speak_segment_end(request_id), stage_id=0, segment_finished=True)
        assert "input_audio_buffer.processed" not in types(await h.settle())

        acks: list[InputProcessed] = []
        talker_ends = flushes = 0
        for step in steps:
            if step == "talker":
                talker_ends += 1
                talker = SimpleNamespace(request_id=request_id, finished=False, outputs=[], multimodal_output={})
                events = await h.deliver_and_settle(talker, stage_id=1, segment_finished=True)
            else:
                flushes += 1
                flush = tts_output(request_id, samples=24000, text=f"unit {flushes}", tts_is_last_chunk=True)
                events = await h.deliver_and_settle(flush, stage_id=2, segment_finished=flush_ends_segment)
            acks.extend(e for e in events if e.type == "input_audio_buffer.processed")
            assert len(acks) == min(talker_ends, flushes), f"after {step} ({talker_ends=}, {flushes=}): {acks}"
        assert [(ack.input_index, ack.units) for ack in acks] == [
            (1, ({"end_ms": 1000, "decision": "speak"},)),
            (2, ({"end_ms": 2000, "decision": "speak"},)),
        ]
    finally:
        await close_harness(h)


async def test_flushes_of_two_queued_units_that_overtake_both_stage0_segment_ends_complete_them_in_order() -> None:
    """Both Code2Wav flushes are held by the engine until the units speak, then replayed to the oldest one."""
    h = await open_harness(extra_body=INPUT_CLOCK, stage_count=3)
    try:
        await h.run(append_audio())
        await h.run(append_audio())
        request_id = h.stage0_request_id()
        h.deliver(tts_output(request_id, samples=24000, text="unit 1", tts_is_last_chunk=True), stage_id=2)
        h.deliver(tts_output(request_id, samples=24000, text="unit 2", tts_is_last_chunk=True), stage_id=2)
        h.deliver(_speak_segment_end(request_id), stage_id=0, segment_finished=True)
        h.deliver(_speak_segment_end(request_id), stage_id=0, segment_finished=True)
        assert "input_audio_buffer.processed" not in types(await h.settle()), "no Talker end yet"

        talker = SimpleNamespace(request_id=request_id, finished=False, outputs=[], multimodal_output={})
        events = await h.deliver_and_settle(talker, stage_id=1, segment_finished=True)
        acks = [e for e in events if e.type == "input_audio_buffer.processed"]
        assert [(a.input_index, a.units) for a in acks] == [(1, ({"end_ms": 1000, "decision": "speak"},))]
        talker = SimpleNamespace(request_id=request_id, finished=False, outputs=[], multimodal_output={})
        events = await h.deliver_and_settle(talker, stage_id=1, segment_finished=True)
        acks = [e for e in events if e.type == "input_audio_buffer.processed"]
        assert [(a.input_index, a.units) for a in acks] == [(2, ({"end_ms": 2000, "decision": "speak"},))]
    finally:
        await close_harness(h)


async def test_a_unit_that_opens_with_the_chunk_terminator_is_acknowledged_before_the_next_speaking_unit() -> None:
    """Stage 0 can draw the chunk terminator as a unit's first decision; ``llm2tts`` hands the Talker an empty condition
    for it, so the unit has a Talker segment end and an empty flagged Code2Wav marker of its own. The speaking unit
    after it is acknowledged after its own audio instead of standing in for the empty unit."""
    h = await open_harness(extra_body=INPUT_CLOCK, stage_count=3)
    try:
        await h.run(append_audio())
        await h.run(append_audio())
        request_id = h.stage0_request_id()
        terminator_only = SimpleNamespace(
            request_id=request_id,
            finished=False,
            outputs=[SimpleNamespace(text="", token_ids=[CHUNK_EOS], multimodal_output={})],
            multimodal_output={
                "special_token_ids": {"listen_token_id": LISTEN_TOKEN_ID, "chunk_eos_token_id": CHUNK_EOS}
            },
        )
        h.deliver(terminator_only, stage_id=0, segment_finished=True)
        h.deliver(_speak_segment_end(request_id), stage_id=0, segment_finished=True)
        assert "input_audio_buffer.processed" not in types(await h.settle())

        talker = SimpleNamespace(request_id=request_id, finished=False, outputs=[], multimodal_output={})
        events = await h.deliver_and_settle(talker, stage_id=1, segment_finished=True)
        assert "input_audio_buffer.processed" not in types(events), "the empty marker has not arrived"
        marker = tts_output(request_id, samples=0, text="", tts_is_last_chunk=True)
        events = await h.deliver_and_settle(marker, stage_id=2)
        acks = [e for e in events if e.type == "input_audio_buffer.processed"]
        assert [(a.input_index, a.units) for a in acks] == [(1, ({"end_ms": 1000, "decision": "speak"},))]

        flush = tts_output(request_id, samples=24000, text="unit 2", tts_is_last_chunk=True)
        events = await h.deliver_and_settle(flush, stage_id=2)
        assert "input_audio_buffer.processed" not in types(events), "unit 2 still waits for its Talker end"
        talker = SimpleNamespace(request_id=request_id, finished=False, outputs=[], multimodal_output={})
        events = await h.deliver_and_settle(talker, stage_id=1, segment_finished=True)
        acks = [e for e in events if e.type == "input_audio_buffer.processed"]
        assert [(a.input_index, a.units) for a in acks] == [(2, ({"end_ms": 2000, "decision": "speak"},))]
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
    assert apply_input_clock_capabilities(capabilities, INPUT_CLOCK).optional_input_modalities == {
        "audio",
        "video",
        "text",
    }


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
