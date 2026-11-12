# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# SPDX-FileCopyrightText: Copyright contributors to the VLLM-Omni project

from dataclasses import dataclass, field

import pytest

from vllm_omni.engine.duplex.plugin import PartialStageForward
from vllm_omni.model_executor.models.joyai_vl_interaction.duplex.sentence_tts import (
    JOYAI_TTS_CLOSE_ONLY_KEY,
    JOYAI_TTS_FINISH_ON_NEXT_STOP_KEY,
    JOYAI_TTS_PARTIAL_KEY,
    incomplete_delegation_hold,
    incomplete_marker_prefix,
    next_spoken_sentence_chunk,
    plan_partial_stage_output,
    spoken_payload,
    spoken_text_ready,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("raw", ["", "</res", "</respon", "</sile", "</sil"])
def test_marker_prefix_is_held(raw):
    """A marker still arriving must not be rendered as speech."""
    assert incomplete_marker_prefix(raw)
    assert not spoken_text_ready(raw, finished=False)


@pytest.mark.parametrize("raw", ["The pot is smoking.", "先帝创业未半"])
def test_plain_text_without_marker_streams(raw):
    """Markers are optional: JoyAI may answer with bare text."""
    assert not incomplete_marker_prefix(raw)
    assert spoken_text_ready(raw, finished=False)


def test_response_marker_opens_the_answer():
    assert spoken_text_ready("</response> The pot is smoking.", finished=False)
    assert spoken_text_ready("</response>", finished=False)
    assert spoken_text_ready("The pot is smoking.", finished=True)


def test_silence_action_never_streams_text():
    assert not spoken_text_ready("</silence>", finished=True)
    assert not spoken_text_ready("</silence>", finished=False)


def test_delegation_prefix_holds_then_releases():
    assert incomplete_delegation_hold("Checking.</delega", finished=False)
    assert not incomplete_delegation_hold("Checking. </delegation> Why?", finished=False)
    assert not incomplete_delegation_hold("Checking.</delega", finished=True)
    assert not spoken_text_ready("</response> Checking.</deleg", finished=False)
    assert spoken_text_ready("</response> Checking. </delegation> Why?", finished=False)


def test_spoken_payload_drops_markers_and_silence():
    assert spoken_payload("</response> The pot is smoking.", finished=False) == "The pot is smoking."
    assert spoken_payload("</silence>", finished=True) is None
    assert spoken_payload("</response> Checking.</deleg", finished=False) is None


def test_next_sentence_emits_before_the_action_finishes():
    state: dict[str, object] = {}
    text = "水开了。关火再离开。"
    emitted = None
    for index in range(1, len(text) + 1):
        chunk = next_spoken_sentence_chunk(state, text[:index], finished=False)
        if chunk:
            emitted = index
            break
    assert emitted is not None
    assert emitted < len(text)
    assert "水开了" in chunk
    assert "关火" not in chunk


def test_next_sentence_emits_the_second_sentence():
    state: dict[str, object] = {}
    # The splitter holds the terminator until it sees the following character:
    # "。" alone may still grow into "。。" or "..." so it is not a unit yet.
    first = next_spoken_sentence_chunk(state, "水开了。关", finished=False)
    assert first is not None
    assert "水开了" in first
    second = next_spoken_sentence_chunk(state, "水开了。关火。", finished=True)
    assert second is not None
    assert "关火" in second


def test_next_sentence_flushes_a_terminator_free_tail_on_finish():
    state: dict[str, object] = {}
    assert next_spoken_sentence_chunk(state, "还在烧", finished=False) is None
    assert next_spoken_sentence_chunk(state, "还在烧", finished=True) == "还在烧"


def joyai_action_to_tts() -> None:
    return None


@dataclass
class _FakeStageClient:
    custom_process_input_func: object = joyai_action_to_tts


@dataclass
class _FakePool:
    stage_client: _FakeStageClient = field(default_factory=_FakeStageClient)


@dataclass
class _FakeOrchestrator:
    # Match production: Orchestrator.stage_pools is a list indexed by stage id.
    stage_pools: list[_FakePool] | dict[int, _FakePool] = field(default_factory=lambda: [_FakePool(), _FakePool()])
    chunk_fed: bool = False
    pending_talker_followups: list[tuple[object, bool]] = field(default_factory=list)

    def _stage_receives_async_chunks(self, stage_id: int) -> bool:
        del stage_id
        return self.chunk_fed


@dataclass
class _FakeSegment:
    finished: bool = False


@dataclass
class _FakeStreaming:
    bridge_states: dict[str, object] = field(default_factory=dict)
    enabled: bool = False
    segments: dict[int, _FakeSegment] = field(default_factory=dict)

    def segment(self, stage_id: int | None) -> _FakeSegment:
        if stage_id is None:
            return _FakeSegment()
        return self.segments.get(stage_id, _FakeSegment())


@dataclass
class _FakeReqState:
    session_owned: bool = True
    final_stage_id: int = 2
    request_id: str = "req-1"
    prompt: dict[str, object] = field(default_factory=lambda: {"additional_information": {}})
    streaming: _FakeStreaming = field(default_factory=_FakeStreaming)


@dataclass
class _FakeCompletion:
    text: str
    cumulative_text: str | None = None
    token_ids: list[int] = field(default_factory=list)


@dataclass
class _FakeOutput:
    request_id: str
    outputs: list[_FakeCompletion]
    finished: bool = False


def _stage0_output(text: str, *, finished: bool = False) -> _FakeOutput:
    return _FakeOutput(
        finished=finished,
        request_id="req-1",
        outputs=[_FakeCompletion(text=text, cumulative_text=text if finished else None)],
    )


def test_plan_partial_hands_the_first_sentence_resumable():
    orchestrator = _FakeOrchestrator()
    req_state = _FakeReqState()
    opened = plan_partial_stage_output(
        orchestrator, 0, 0, _stage0_output("</response> 水开了。关"), req_state, joyai_stage_id=0
    )
    assert opened is not None
    assert opened.is_final_update is False
    assert opened.close_only is False
    assert opened.queue_close_after is False
    assert "水开了" in opened.output.text
    assert req_state.streaming.enabled is True
    info = req_state.prompt["additional_information"]
    assert info[JOYAI_TTS_PARTIAL_KEY] is True
    assert info[JOYAI_TTS_CLOSE_ONLY_KEY] is False


def test_plan_partial_holds_later_sentences_while_talker_is_busy():
    orchestrator = _FakeOrchestrator()
    req_state = _FakeReqState()
    first = plan_partial_stage_output(
        orchestrator, 0, 0, _stage0_output("</response> 水开了。关"), req_state, joyai_stage_id=0
    )
    assert first is not None
    assert "水开了" in first.output.text
    later = plan_partial_stage_output(
        orchestrator,
        0,
        0,
        _stage0_output("</response> 水开了。关火。", finished=True),
        req_state,
        joyai_stage_id=0,
    )
    assert later is not None
    assert later.output.text == ""
    assert later.is_final_update is False
    assert later.close_only is False
    assert later.queue_close_after is False
    bridge = req_state.streaming.bridge_states["joyai_sentence_tts"]
    queued = bridge["queued"]
    assert isinstance(queued, list)
    assert any(isinstance(item, str) and "关火" in item for item in queued)
    info = req_state.prompt["additional_information"]
    assert info[JOYAI_TTS_CLOSE_ONLY_KEY] is False


def test_plan_partial_does_not_close_while_talker_is_in_flight():
    orchestrator = _FakeOrchestrator()
    req_state = _FakeReqState()
    opened = plan_partial_stage_output(
        orchestrator, 0, 0, _stage0_output("</response> 水开了。关"), req_state, joyai_stage_id=0
    )
    assert opened is not None
    done = plan_partial_stage_output(
        orchestrator,
        0,
        0,
        _stage0_output("</response> 水开了。关", finished=True),
        req_state,
        joyai_stage_id=0,
    )
    assert done is not None
    assert done.output.text == ""
    assert done.close_only is False
    assert done.is_final_update is False
    assert done.queue_close_after is False
    info = req_state.prompt["additional_information"]
    assert info[JOYAI_TTS_CLOSE_ONLY_KEY] is False


def test_plan_partial_does_not_submit_a_second_sentence_mid_generation():
    orchestrator = _FakeOrchestrator()
    req_state = _FakeReqState()
    first = plan_partial_stage_output(
        orchestrator, 0, 0, _stage0_output("</response> 水开了。关"), req_state, joyai_stage_id=0
    )
    assert first is not None
    held = plan_partial_stage_output(
        orchestrator, 0, 0, _stage0_output("</response> 水开了。关火。还"), req_state, joyai_stage_id=0
    )
    assert held is None
    bridge = req_state.streaming.bridge_states["joyai_sentence_tts"]
    queued = bridge["queued"]
    assert isinstance(queued, list)
    assert any(isinstance(item, str) and "关火" in item for item in queued)


def test_talker_park_releases_the_next_queued_sentence():
    orchestrator = _FakeOrchestrator()
    req_state = _FakeReqState()
    opened = plan_partial_stage_output(
        orchestrator, 0, 0, _stage0_output("</response> 水开了。关"), req_state, joyai_stage_id=0
    )
    assert opened is not None
    plan_partial_stage_output(
        orchestrator,
        0,
        0,
        _stage0_output("</response> 水开了。关火。", finished=True),
        req_state,
        joyai_stage_id=0,
    )
    req_state.streaming.segments[1] = _FakeSegment(finished=True)
    parked = _stage0_output("codec")
    parked.outputs[0].token_ids = [100, 101, 2150]
    park = plan_partial_stage_output(orchestrator, 1, 0, parked, req_state, joyai_stage_id=0)
    assert park is None
    # Last queued sentence stays resumable; its 2150 is the request terminal.
    assert len(orchestrator.pending_talker_followups) == 1
    view, is_final_update = orchestrator.pending_talker_followups[0]
    assert is_final_update is False
    assert "关火" in view.text
    assert not view.text.startswith("。")
    leftover = _stage0_output("codec-again")
    leftover.outputs[0].token_ids = [100, 101, 2150]
    assert plan_partial_stage_output(orchestrator, 1, 0, leftover, req_state, joyai_stage_id=0) is None
    assert len(orchestrator.pending_talker_followups) == 1
    info = req_state.prompt["additional_information"]
    assert info[JOYAI_TTS_PARTIAL_KEY] is True
    assert info[JOYAI_TTS_CLOSE_ONLY_KEY] is False
    assert info[JOYAI_TTS_FINISH_ON_NEXT_STOP_KEY] is True
    bridge = req_state.streaming.bridge_states["joyai_sentence_tts"]
    assert bridge.get("close_sent") is True

    req_state.streaming.segments[1] = _FakeSegment(finished=False)
    generating = _stage0_output("last-sentence")
    generating.outputs[0].token_ids = [7]
    assert plan_partial_stage_output(orchestrator, 1, 0, generating, req_state, joyai_stage_id=0) is None
    req_state.streaming.segments[1] = _FakeSegment(finished=True)
    assert (
        plan_partial_stage_output(orchestrator, 1, 0, _stage0_output("park-close"), req_state, joyai_stage_id=0) is None
    )
    assert len(orchestrator.pending_talker_followups) == 1
    assert info[JOYAI_TTS_CLOSE_ONLY_KEY] is False


def test_talker_park_closes_after_the_queue_drains():
    """A talker that is still busy at Stage0 finish is closed by the park poll."""
    orchestrator = _FakeOrchestrator()
    req_state = _FakeReqState()
    opened = plan_partial_stage_output(
        orchestrator, 0, 0, _stage0_output("</response> 水开了。关"), req_state, joyai_stage_id=0
    )
    assert opened is not None
    # The action ends without another spoken unit, while sentence 1 is in flight:
    # the finish is a skip-legacy marker, not the close.
    plan_partial_stage_output(
        orchestrator,
        0,
        0,
        _stage0_output("</silence>", finished=True),
        req_state,
        joyai_stage_id=0,
    )
    assert orchestrator.pending_talker_followups == []
    req_state.streaming.segments[1] = _FakeSegment(finished=True)
    park = plan_partial_stage_output(orchestrator, 1, 0, _stage0_output("codec"), req_state, joyai_stage_id=0)
    assert park is None
    assert len(orchestrator.pending_talker_followups) == 1
    view, is_final_update = orchestrator.pending_talker_followups[0]
    assert is_final_update is True
    assert view.text == ""
    info = req_state.prompt["additional_information"]
    assert info[JOYAI_TTS_CLOSE_ONLY_KEY] is True


def test_talker_empty_unpark_does_not_dump_the_queued_sentence():
    orchestrator = _FakeOrchestrator()
    req_state = _FakeReqState()
    opened = plan_partial_stage_output(
        orchestrator, 0, 0, _stage0_output("</response> 水开了。关"), req_state, joyai_stage_id=0
    )
    assert opened is not None
    # Seed the Talker queue directly: incremental SpeechTextSplitter tails are
    # not under test here, only that an empty unpark cannot drain it.
    bridge = req_state.streaming.bridge_states["joyai_sentence_tts"]
    bridge["queued"] = ["关火。", "还在烧。"]
    bridge["talker_busy"] = True
    req_state.streaming.segments[1] = _FakeSegment(finished=True)
    park_tokens = [100, 101, 2150]
    parked = _stage0_output("park-1")
    parked.outputs[0].token_ids = list(park_tokens)
    assert plan_partial_stage_output(orchestrator, 1, 0, parked, req_state, joyai_stage_id=0) is None
    assert len(orchestrator.pending_talker_followups) == 1
    req_state.streaming.segments[1] = _FakeSegment(finished=False)
    leftover = _stage0_output("leftover")
    leftover.outputs[0].token_ids = list(park_tokens)
    assert plan_partial_stage_output(orchestrator, 1, 0, leftover, req_state, joyai_stage_id=0) is None
    req_state.streaming.segments[1] = _FakeSegment(finished=True)
    park_again = _stage0_output("park-2")
    park_again.outputs[0].token_ids = list(park_tokens)
    assert plan_partial_stage_output(orchestrator, 1, 0, park_again, req_state, joyai_stage_id=0) is None
    assert len(orchestrator.pending_talker_followups) == 1
    req_state.streaming.segments[1] = _FakeSegment(finished=False)
    generating = _stage0_output("codec")
    generating.outputs[0].token_ids = [7]
    assert plan_partial_stage_output(orchestrator, 1, 0, generating, req_state, joyai_stage_id=0) is None
    req_state.streaming.segments[1] = _FakeSegment(finished=True)
    assert plan_partial_stage_output(orchestrator, 1, 0, _stage0_output("park-3"), req_state, joyai_stage_id=0) is None
    assert len(orchestrator.pending_talker_followups) == 2
    assert "还在烧" in orchestrator.pending_talker_followups[1][0].text


def test_talker_park_is_ignored_until_the_segment_stops():
    orchestrator = _FakeOrchestrator()
    req_state = _FakeReqState()
    opened = plan_partial_stage_output(
        orchestrator, 0, 0, _stage0_output("</response> 水开了。 "), req_state, joyai_stage_id=0
    )
    assert opened is None  # a bare terminator is not a unit yet
    opened = plan_partial_stage_output(
        orchestrator, 0, 0, _stage0_output("</response> 水开了。关"), req_state, joyai_stage_id=0
    )
    assert opened is not None
    req_state.streaming.segments[1] = _FakeSegment(finished=False)
    assert plan_partial_stage_output(orchestrator, 1, 0, _stage0_output("codec"), req_state, joyai_stage_id=0) is None
    assert orchestrator.pending_talker_followups == []


def test_plan_partial_still_indexes_a_dict_of_pools():
    orchestrator = _FakeOrchestrator(stage_pools={1: _FakePool()})
    req_state = _FakeReqState()
    opened = plan_partial_stage_output(
        orchestrator, 0, 0, _stage0_output("</response> 水开了。关"), req_state, joyai_stage_id=0
    )
    assert opened is not None
    assert "水开了" in opened.output.text


def test_plan_partial_skips_when_talker_is_chunk_fed():
    orchestrator = _FakeOrchestrator(chunk_fed=True)
    req_state = _FakeReqState()
    assert (
        plan_partial_stage_output(
            orchestrator, 0, 0, _stage0_output("</response> 水开了。关"), req_state, joyai_stage_id=0
        )
        is None
    )
    assert req_state.streaming.enabled is False


def test_single_sentence_finish_is_a_final_talker_submit():
    orchestrator = _FakeOrchestrator()
    req_state = _FakeReqState()
    plan = plan_partial_stage_output(
        orchestrator,
        0,
        0,
        _stage0_output("</response> 还在烧", finished=True),
        req_state,
        joyai_stage_id=0,
    )
    assert plan is not None
    assert plan.output.text == "还在烧"
    assert plan.is_final_update is True
    assert plan.close_only is False
    assert plan.queue_close_after is False
    assert isinstance(plan, PartialStageForward)


def test_last_sentence_after_a_parked_talker_stays_resumable():
    """The second sentence arrives after 2150 left the Talker idle.

    Submitting it non-resumable would make it an end sentinel: the scheduler
    aborts the parked session and the sentence is never spoken. It must be a
    resumable update; 2150 of this sentence is the request terminal.
    """
    orchestrator = _FakeOrchestrator()
    req_state = _FakeReqState()
    first = plan_partial_stage_output(
        orchestrator, 0, 0, _stage0_output("</response> 水开了。关"), req_state, joyai_stage_id=0
    )
    assert first is not None
    assert "水开了" in first.output.text
    # Talker parked on 2150 with nothing left to release: this is what
    # _plan_after_talker_park leaves behind when the queue is empty.
    req_state.streaming.bridge_states["joyai_sentence_tts"]["talker_busy"] = False

    last = plan_partial_stage_output(
        orchestrator,
        0,
        0,
        _stage0_output("</response> 水开了。关火再离开。", finished=True),
        req_state,
        joyai_stage_id=0,
    )
    assert last is not None
    assert "关火" in last.output.text
    assert last.is_final_update is False
    assert last.close_only is False
    assert last.queue_close_after is True
    info = req_state.prompt["additional_information"]
    assert info[JOYAI_TTS_CLOSE_ONLY_KEY] is False
    assert info[JOYAI_TTS_FINISH_ON_NEXT_STOP_KEY] is True
    assert req_state.streaming.bridge_states["joyai_sentence_tts"].get("close_sent") is True


def test_close_sentinel_is_sent_at_most_once():
    """The idle-at-finish close and the 2150 park poll must not both close.

    Two terminals for one turn would finish a session that is already closing,
    so the second path has to see the flag the first one set.
    """
    orchestrator = _FakeOrchestrator()
    req_state = _FakeReqState()
    opened = plan_partial_stage_output(
        orchestrator, 0, 0, _stage0_output("</response> 水开了。关"), req_state, joyai_stage_id=0
    )
    assert opened is not None
    # Talker went idle again, and the action finishes with nothing new to speak.
    bridge = req_state.streaming.bridge_states["joyai_sentence_tts"]
    bridge["talker_busy"] = False
    close = plan_partial_stage_output(
        orchestrator,
        0,
        0,
        _stage0_output("</silence>", finished=True),
        req_state,
        joyai_stage_id=0,
    )
    assert close is not None
    assert close.close_only is True
    assert close.is_final_update is True
    assert bridge["close_sent"] is True

    # The park poll that follows must not schedule a second terminal.
    req_state.streaming.segments[1] = _FakeSegment(finished=True)
    assert plan_partial_stage_output(orchestrator, 1, 0, _stage0_output("codec"), req_state, joyai_stage_id=0) is None
    assert orchestrator.pending_talker_followups == []
