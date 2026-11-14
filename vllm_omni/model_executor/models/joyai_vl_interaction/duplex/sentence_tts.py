# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""JoyAI streaming-text gates and sentence-level Talker handoff.

The duplex profile runs the JoyAI stage with ``RequestOutputKind.DELTA`` so the
session can project the spoken answer while the action is still being generated.
These helpers decide when a partial action is safe to show.

JoyAI may answer with ``</response> <text>`` or with bare ``<text>``, so a marker
is optional. What must not happen is showing a marker that is still arriving: a
partial ``</sil`` rendered as speech is wrong, and a ``</delegation>`` note that
is still open may not be the note after all.

Sentence TTS follows AURA's orchestrator contract: each confirmed sentence is a
resumable Talker update on the same request id. Only one sentence is in flight
at a time: later sentences wait in ``queued`` until Talker parks on 2150, then
the plugin submits the next one from the Talker-stage poll. MRv2 async_chunk
Talker only emits those 2150 parks (no generating deltas), so a new park with
different token ids must drain even while ``releasing`` is still set. The last
sentence is a resumable unpark stamped ``joyai_finish_on_next_stop`` so that
utterance's 2150 is Code2Wav's request terminal. A later parked close_only is
too late: the vocoder already treated 2150 as a segment stop. Do not prefix
the next sentence with ``。``: Talker vocalizes that as a filler. The pause is
the terminator already in the sentence text. ``joyai_action_to_tts`` reads the
flags written here; do not duplicate the key strings elsewhere.
"""

from __future__ import annotations

import asyncio
import time

import regex as re
from vllm.logger import init_logger

from vllm_omni.engine.duplex.plugin import PartialStageForward
from vllm_omni.entrypoints.openai.speech_text_splitter import SpeechTextSplitter
from vllm_omni.experimental.fullduplex.joyvl.decision.output_parser import (
    DELEGATION_TAG,
    RESPONSE_TOKEN,
    SILENCE_TOKEN,
    parse_action,
)
from vllm_omni.experimental.fullduplex.joyvl.decision.prompts import USER_QUERY_HEADER

logger = init_logger(__name__)

_DELEGATION_MARKERS = (DELEGATION_TAG, "<delegation>")
_LEADING_MARKERS = (SILENCE_TOKEN, RESPONSE_TOKEN)
_BRIDGE_STATE_KEY = "joyai_sentence_tts"
JOYAI_TTS_PARTIAL_KEY = "joyai_tts_partial"
JOYAI_TTS_CLOSE_ONLY_KEY = "joyai_tts_close_only"
# Stamped on the last queued sentence so Talker's 2150 is a request terminal
# (Code2Wav ``finished``) instead of a parked close that never reaches the vocoder.
JOYAI_TTS_FINISH_ON_NEXT_STOP_KEY = "joyai_finish_on_next_stop"
# CustomVoice speaker rows are deterministic; codec sampling is not. Pin Talker
# (and its MTP) so each turn starts from the same voice instead of a new RNG.
JOYAI_TTS_SEED = 0
# Official Talker sampling is 0.9, which wanders in rate/pitch across tokens
# and sentences. 0.4 is the Qwen3-Omni Talker recommendation: still sampled,
# but stable enough that recitation does not change speaker mid-paragraph.
JOYAI_TTS_TEMPERATURE = 0.4
# V1 can rebroadcast the same 2150 in the following polls. MRv2 DELTA parks
# often look like ``(2150,)`` every sentence, so same ids after this window
# are the next utterance, not a leftover.
_PARK_REBROADCAST_WINDOW_S = 0.2
# JoyAI wraps the user turn with USER_QUERY_HEADER and may echo chunk-duration
# tags plus a trailing </silence>. None of that is speech.
_DURATION_TAG = re.compile(r"<\s*[\d.]+\s*seconds\s*>", re.IGNORECASE)


def incomplete_marker_prefix(raw: str) -> bool:
    """True while ``raw`` is still a prefix of an action marker.

    ``</res`` may yet become ``</response>`` or ``</silence>``; deciding now
    would show marker characters as speech.
    """
    stripped = (raw or "").strip()
    if not stripped:
        return True
    return any(marker.startswith(stripped) for marker in _LEADING_MARKERS)


def incomplete_delegation_hold(payload: str, *, finished: bool) -> bool:
    """True while streamed text may still grow into a delegation tag."""
    if finished:
        return False
    for tag in _DELEGATION_MARKERS:
        if tag in payload:
            return False
    stripped = payload.rstrip()
    for tag in _DELEGATION_MARKERS:
        for n in range(1, len(tag)):
            if stripped.endswith(tag[:n]):
                return True
    return False


def spoken_text_ready(raw: str, *, finished: bool) -> bool:
    """Whether the action in ``raw`` may be shown to the client now."""
    text = raw or ""
    silence_at = text.find(SILENCE_TOKEN)
    response_at = text.find(RESPONSE_TOKEN)
    if response_at == -1 and silence_at == -1:
        # No marker yet: either plain text (markers are optional) or a marker
        # still arriving. Only the latter has to wait.
        return finished or not incomplete_marker_prefix(text)
    if response_at == -1 or (silence_at != -1 and silence_at < response_at):
        # Silence is a decision, not speech.
        return False
    return not incomplete_delegation_hold(text[response_at + len(RESPONSE_TOKEN) :], finished=finished)


def sanitize_spoken_text(text: str) -> str:
    """Drop prompt scaffolding that must not be handed to Talker.

    Recitation and normal turns both see ``USER_QUERY_HEADER`` in the prompt.
    Copying it verbatim, then a ``<0.0 seconds></silence>`` tail, is what the
    Seed-TTS WER bad cases spoke as "User query, important..." / "sillings".
    """
    payload = (text or "").lstrip()
    if payload and USER_QUERY_HEADER.startswith(payload) and payload != USER_QUERY_HEADER:
        # Header still arriving; do not start TTS on ``[User Query (IMP``.
        return ""
    header_at = payload.find(USER_QUERY_HEADER)
    if header_at != -1:
        payload = payload[header_at + len(USER_QUERY_HEADER) :]
    payload = payload.replace(SILENCE_TOKEN, " ")
    payload = _DURATION_TAG.sub(" ", payload)
    return " ".join(payload.split())


def spoken_payload(raw: str, *, finished: bool) -> str | None:
    """The text JoyAI chose to speak, or None when nothing may be handed over yet."""
    if not spoken_text_ready(raw, finished=finished):
        return None
    parsed = parse_action(raw)
    if not parsed.spoke:
        return None
    cleaned = sanitize_spoken_text(parsed.text)
    return cleaned or None


def _pending_units(state: dict[str, object]) -> list[str]:
    units = state.get("pending_units")
    if isinstance(units, list):
        return units
    pending: list[str] = []
    state["pending_units"] = pending
    return pending


def next_spoken_sentence_chunk(state: dict[str, object], spoken: str, *, finished: bool) -> str | None:
    """Return the next Talker sentence, or None when this output must not start TTS.

    ``state`` is per JoyAI request and is mutated. Mid-generation pops one
    confirmed sentence so Talker can start before the action finishes. Stage0
    finish flushes leftover units as one remainder so a single orchestrator
    tick does not drop the tail.
    """
    if state.get("closed"):
        return None
    text = spoken or ""
    splitter = state.get("splitter")
    if not isinstance(splitter, SpeechTextSplitter):
        splitter = SpeechTextSplitter("sentence")
        state["splitter"] = splitter
        state["fed_text"] = ""
        state["pending_units"] = []
    pending_units = _pending_units(state)
    fed = str(state.get("fed_text", ""))
    if text.startswith(fed):
        tail = text[len(fed) :]
    else:
        splitter = SpeechTextSplitter("sentence")
        state["splitter"] = splitter
        pending_units.clear()
        tail = text
    state["fed_text"] = text
    if tail:
        pending_units.extend(unit.strip() for unit in splitter.feed(tail) if unit.strip())
    if finished:
        pending_units.extend(unit.strip() for unit in splitter.flush() if unit.strip())
    if not pending_units:
        return None
    if finished:
        chunk = "".join(pending_units)
        pending_units.clear()
    else:
        chunk = pending_units.pop(0)
    if not chunk:
        return None
    state["emits"] = int(state.get("emits", 0)) + 1
    return chunk


def mark_talker_prompt_flags(
    req_state: object,
    *,
    close_only: bool,
    finish_on_next_stop: bool = False,
) -> None:
    """Stamp the Talker prompt so ``joyai_action_to_tts`` sees a partial or close."""
    prompt = getattr(req_state, "prompt", None)
    if not isinstance(prompt, dict):
        return
    raw_info = prompt.get("additional_information")
    if not isinstance(raw_info, dict):
        raw_info = {}
        prompt["additional_information"] = raw_info
    raw_info[JOYAI_TTS_PARTIAL_KEY] = True
    raw_info[JOYAI_TTS_CLOSE_ONLY_KEY] = close_only
    raw_info[JOYAI_TTS_FINISH_ON_NEXT_STOP_KEY] = finish_on_next_stop


def enable_talker_streaming(req_state: object) -> None:
    """Keep Talker resumable across sentence stops so Code2Wav is not a turn end."""
    streaming = getattr(req_state, "streaming", None)
    if streaming is not None:
        streaming.enabled = True


class SentenceTtsOutput:
    """JoyAI view that exposes only the sentence Talker should speak."""

    def __init__(self, request_id: str, text: str) -> None:
        self.request_id = request_id
        self.finished = False
        self.text = text
        self.cumulative_text = text
        self.outputs = [self]
        self.token_ids: list[int] = []


def _stage_pool(orchestrator: object, stage_id: int) -> object | None:
    """The orchestrator stores pools in a list; tests may use a dict."""
    pools = getattr(orchestrator, "stage_pools", None)
    try:
        return pools[stage_id]
    except (TypeError, KeyError, IndexError):
        return None


def _stage1_processor_name(orchestrator: object, stage_id: int) -> str:
    pool = _stage_pool(orchestrator, stage_id + 1)
    client = getattr(pool, "stage_client", None)
    processor = getattr(client, "custom_process_input_func", None)
    name = getattr(processor, "__name__", "")
    return name if isinstance(name, str) else ""


def _raw_action_text(output: object, observation: object | None) -> tuple[str, bool]:
    finished = bool(getattr(output, "finished", False))
    if observation is not None:
        absorb = getattr(observation, "absorb", None)
        if callable(absorb):
            raw, finished = absorb(output)
            return raw if isinstance(raw, str) else "", bool(finished)
    completion = next(iter(getattr(output, "outputs", ()) or ()), None)
    raw = getattr(completion, "cumulative_text", None) or getattr(completion, "text", "") or ""
    return raw if isinstance(raw, str) else "", finished


def _sentence_tts_bridge(req_state: object) -> dict[str, object] | None:
    streaming = getattr(req_state, "streaming", None)
    bridge_states = getattr(streaming, "bridge_states", None)
    if not isinstance(bridge_states, dict):
        return None
    bridge = bridge_states.setdefault(_BRIDGE_STATE_KEY, {})
    if isinstance(bridge, dict):
        return bridge
    bridge = {}
    bridge_states[_BRIDGE_STATE_KEY] = bridge
    return bridge


def _queued_sentences(bridge: dict[str, object]) -> list[str]:
    queued = bridge.get("queued")
    if isinstance(queued, list):
        return queued
    pending: list[str] = []
    bridge["queued"] = pending
    return pending


def mark_close_sent(req_state: object) -> None:
    """Record that this turn's end sentinel was already submitted.

    Three paths can reach the close: the idle Talker at finish, the follow-up
    after a resumable last sentence, and the park poll that drains an empty
    queue. Only the first one to fire may send it; a second terminal would
    finish a session that is already closing.
    """
    bridge = _sentence_tts_bridge(req_state)
    if bridge is not None:
        bridge["close_sent"] = True


def _pop_queued_sentence(bridge: dict[str, object]) -> str:
    queued = _queued_sentences(bridge)
    while queued:
        item = queued.pop(0)
        if isinstance(item, str) and item:
            return item
    return ""


def _skip_legacy_plan(request_id: str) -> PartialStageForward:
    """Keep Stage0's full action off Talker while a sentence is already speaking."""
    return PartialStageForward(
        output=SentenceTtsOutput(request_id, ""),
        is_final_update=False,
        close_only=False,
        queue_close_after=False,
    )


def _schedule_talker_followup_sequence(
    orchestrator: object,
    joyai_stage_id: int,
    req_state: object,
    steps: list[tuple[SentenceTtsOutput, bool, bool, bool]],
) -> None:
    """Submit Talker updates from JoyAI's stage, not from Talker→Code2Wav.

    ``DuplexOrchestrator._route_output`` always forwards ``stage_id + 1``. A
    Talker-stage plan would therefore hit Code2Wav. Scheduling from
    ``joyai_stage_id`` keeps ``joyai_action_to_tts`` on the Talker path.
    """
    if not steps:
        return
    pending = getattr(orchestrator, "pending_talker_followups", None)
    if isinstance(pending, list):
        pending.extend((view, is_final_update) for view, is_final_update, _close_only, _finish in steps)
    last = steps[-1]
    mark_talker_prompt_flags(req_state, close_only=last[2], finish_on_next_stop=last[3])
    forward = getattr(orchestrator, "_forward_to_next_stage", None)
    if not callable(forward):
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return

    async def _run() -> None:
        for view, is_final_update, close_only, finish_on_next_stop in steps:
            mark_talker_prompt_flags(req_state, close_only=close_only, finish_on_next_stop=finish_on_next_stop)
            await forward(
                str(view.request_id),
                joyai_stage_id,
                view,
                req_state,
                is_streaming_session=True,
                is_final_update=is_final_update,
            )

    loop.create_task(_run())


def _schedule_talker_followup(
    orchestrator: object,
    joyai_stage_id: int,
    view: SentenceTtsOutput,
    req_state: object,
    *,
    is_final_update: bool,
    close_only: bool = False,
) -> None:
    _schedule_talker_followup_sequence(
        orchestrator,
        joyai_stage_id,
        req_state,
        [(view, is_final_update, close_only, False)],
    )


def _talker_segment_finished(req_state: object, talker_stage_id: int, output: object | None = None) -> bool:
    """Whether this Talker poll is a 2150 park.

    Prefer ``output.is_segment_finished`` from this poll. MRv2 async_chunk Talker
    only emits EngineCoreOutputs at 2150, so ``streaming.segments`` can keep a
    leftover True from the previous sentence and hide the next park.
    """
    poll_flag = getattr(output, "is_segment_finished", None) if output is not None else None
    if poll_flag is not None:
        return bool(poll_flag)
    streaming = getattr(req_state, "streaming", None)
    if streaming is None or not getattr(streaming, "enabled", False):
        return False
    segment = getattr(streaming, "segment", None)
    if not callable(segment):
        return False
    return bool(getattr(segment(talker_stage_id), "finished", False))


def _coerce_token_ids(ids: object) -> tuple[int, ...]:
    if not isinstance(ids, list) or not ids:
        return ()
    try:
        return tuple(int(token_id) for token_id in ids)
    except (TypeError, ValueError):
        return ()


def _talker_token_ids(output: object) -> tuple[int, ...]:
    ids = _coerce_token_ids(getattr(output, "token_ids", None))
    if ids:
        return ids
    outputs = getattr(output, "outputs", None)
    if not isinstance(outputs, (list, tuple)) or not outputs:
        return ()
    return _coerce_token_ids(getattr(outputs[0], "token_ids", None))


def _talker_park_identity(output: object) -> tuple[int, ...]:
    """Identity of this 2150. Prefer cumulative ids; DELTA parks are often (2150,)."""
    outputs = getattr(output, "outputs", None)
    completion = outputs[0] if isinstance(outputs, (list, tuple)) and outputs else output
    cumulative = _coerce_token_ids(getattr(completion, "cumulative_token_ids", None))
    if cumulative:
        return cumulative
    return _talker_token_ids(output)


def _talker_output_has_progress(output: object, park_ids: object) -> bool:
    """True when this unparked delta is a new sentence, not a leftover rebroadcast.

    Production Talker leftovers still carry the previous sentence's cumulative
    ``token_ids``. Treating any non-empty list as progress used to clear
    ``releasing`` and dump the queue in one second.
    """
    ids = _talker_token_ids(output)
    if not ids:
        return False
    if not isinstance(park_ids, tuple) or not park_ids:
        return True
    if len(ids) < len(park_ids):
        return True
    return ids[: len(park_ids)] != park_ids


def _is_same_park_rebroadcast(output: object, park_ids: object, *, released_at: object) -> bool:
    """True when this 2150 is the previous park replayed, not the next sentence.

    V1 leftovers still carry the parked sentence's ids and arrive in the next
    polls. MRv2 DELTA often reports ``(2150,)`` for every sentence; those later
    parks must drain after ``_PARK_REBROADCAST_WINDOW_S``.
    """
    if not isinstance(park_ids, tuple) or not park_ids:
        return False
    ids = _talker_park_identity(output)
    if ids != park_ids:
        return False
    try:
        age = time.monotonic() - float(released_at or 0)
    except (TypeError, ValueError):
        return True
    return age < _PARK_REBROADCAST_WINDOW_S


def _plan_after_talker_park(
    orchestrator: object,
    output: object,
    req_state: object,
    *,
    joyai_stage_id: int,
) -> PartialStageForward | None:
    """Release the next queued sentence, or close, once Talker parks on 2150."""
    bridge = _sentence_tts_bridge(req_state)
    if bridge is None:
        return None
    parked = _talker_segment_finished(req_state, joyai_stage_id + 1, output)
    if not parked:
        # Only a real decode delta unlocks the next park. Empty leftovers
        # after 2150 used to clear ``releasing`` and dump the queue.
        if _talker_output_has_progress(output, bridge.get("park_token_ids")):
            bridge["releasing"] = False
        return None
    # MRv2 never sends the generating heartbeat that clears ``releasing``.
    # A new 2150 (ids that are not the previous park replayed) must drain.
    if bridge.get("releasing") and _is_same_park_rebroadcast(
        output, bridge.get("park_token_ids"), released_at=bridge.get("park_released_at")
    ):
        return None
    request_id = str(getattr(output, "request_id", getattr(req_state, "request_id", "")))
    chunk = _pop_queued_sentence(bridge)
    if chunk:
        bridge["releasing"] = True
        bridge["park_token_ids"] = _talker_park_identity(output)
        bridge["park_released_at"] = time.monotonic()
        bridge["talker_busy"] = True
        enable_talker_streaming(req_state)
        queued = _queued_sentences(bridge)
        close_after = bool(bridge.get("closed")) and not queued
        logger.info(
            "[JoyAI] sentence TTS park-follow req=%s text_len=%d queued=%d close_after=%s",
            request_id,
            len(chunk),
            len(queued),
            close_after,
        )
        # Last sentence stays resumable so this 2150 unpark is spoken. Stamp
        # finish-on-next-stop so that utterance's 2150 is Code2Wav's request
        # terminal. A later parked close_only is too late: the vocoder already
        # treated 2150 as a segment stop and never sees ``finished``.
        if close_after:
            mark_close_sent(req_state)
        _schedule_talker_followup_sequence(
            orchestrator,
            joyai_stage_id,
            req_state,
            [(SentenceTtsOutput(request_id, chunk), False, False, close_after)],
        )
        return None
    bridge["talker_busy"] = False
    if not bridge.get("closed") or bridge.get("close_sent"):
        # ``close_sent``: the idle-at-finish path already handed the sentinel to
        # the orchestrator right after the last sentence. Sending a second one
        # would abort/finish a session that is already closing.
        return None
    bridge["releasing"] = True
    mark_close_sent(req_state)
    enable_talker_streaming(req_state)
    logger.info("[JoyAI] sentence TTS park-close req=%s", request_id)
    _schedule_talker_followup(
        orchestrator,
        joyai_stage_id,
        SentenceTtsOutput(request_id, ""),
        req_state,
        is_final_update=True,
        close_only=True,
    )
    return None


def plan_partial_stage_output(
    orchestrator: object,
    stage_id: int,
    replica_id: int,
    output: object,
    req_state: object,
    *,
    joyai_stage_id: int,
    observation: object | None = None,
) -> PartialStageForward | None:
    """Hand a finished sentence to Talker before JoyAI finishes.

    Same request id as the turn (Code2Wav was prewarmed on it). Text goes
    through ``joyai_action_to_tts``. Only one Talker sentence is submitted at
    a time; later sentences wait until Talker parks.
    """
    del replica_id
    if not getattr(req_state, "session_owned", False):
        return None
    if stage_id == joyai_stage_id + 1:
        return _plan_after_talker_park(orchestrator, output, req_state, joyai_stage_id=joyai_stage_id)
    if stage_id != joyai_stage_id:
        return None
    final_stage_id = getattr(req_state, "final_stage_id", -1)
    if not isinstance(final_stage_id, int) or stage_id + 1 > final_stage_id:
        return None
    if _stage1_processor_name(orchestrator, stage_id) != "joyai_action_to_tts":
        return None
    receives_async_chunks = getattr(orchestrator, "_stage_receives_async_chunks", None)
    if callable(receives_async_chunks) and receives_async_chunks(stage_id + 1):
        return None

    bridge = _sentence_tts_bridge(req_state)
    if bridge is None or bridge.get("closed"):
        return None

    raw, finished = _raw_action_text(output, observation)
    spoken = spoken_payload(raw, finished=finished)
    request_id = str(getattr(output, "request_id", getattr(req_state, "request_id", "")))
    prior_emits = int(bridge.get("emits", 0))
    chunk = ""
    if spoken is not None:
        popped = next_spoken_sentence_chunk(bridge, spoken, finished=finished)
        if popped is not None:
            chunk = popped
        elif not (finished and prior_emits):
            return None
    elif not (finished and prior_emits):
        return None

    queued = _queued_sentences(bridge)
    talker_busy = bool(bridge.get("talker_busy"))
    if chunk and talker_busy:
        queued.append(chunk)
        logger.info(
            "[JoyAI] sentence TTS hold req=%s text_len=%d queued=%d — Talker busy",
            request_id,
            len(chunk),
            len(queued),
        )
        chunk = ""
    if finished:
        bridge["closed"] = True

    # A later StreamingUpdate during Talker prefill replaces additional_information
    # and aborts the in-flight sentence. Keep one sentence on Talker; skip-legacy
    # so Stage0 finish does not one-shot the full recitation.
    if talker_busy:
        if not finished:
            return None
        logger.info(
            "[JoyAI] sentence TTS skip-legacy req=%s queued=%d — Talker still in-flight",
            request_id,
            len(queued),
        )
        mark_talker_prompt_flags(req_state, close_only=False)
        enable_talker_streaming(req_state)
        return _skip_legacy_plan(request_id)

    if chunk:
        bridge["talker_busy"] = True
        bridge["releasing"] = False
        enable_talker_streaming(req_state)
        # A final sentence that follows already-spoken ones is a new utterance on
        # a Talker that may be parked on 2150, not an end sentinel:
        # ``is_final_update=True`` submits non-resumable
        # (``StreamingUpdate.from_request`` returns None), which aborts the parked
        # session and drops this sentence. Submit it resumable; 2150 of this
        # utterance is the Code2Wav request terminal (finish_on_next_stop).
        queue_close_after = bool(finished and prior_emits and not queued)
        mark_talker_prompt_flags(req_state, close_only=False, finish_on_next_stop=queue_close_after)
        logger.info(
            "[JoyAI] sentence TTS req=%s finished=%s close_only=%s text_len=%d queued=%d close_after=%s",
            request_id,
            finished,
            False,
            len(chunk),
            len(queued),
            queue_close_after,
        )
        is_final_update = finished and not queued and not queue_close_after
        if is_final_update or queue_close_after:
            # ``is_final_update=True`` submits non-resumable, so this chunk *is*
            # the end sentinel. ``queue_close_after`` leaves it resumable and
            # finishes on the coming 2150; either way a later park-close must
            # not send a second terminal.
            mark_close_sent(req_state)
        return PartialStageForward(
            output=SentenceTtsOutput(request_id, chunk),
            is_final_update=is_final_update,
            close_only=False,
            queue_close_after=queue_close_after,
        )

    if finished and prior_emits:
        mark_talker_prompt_flags(req_state, close_only=True)
        enable_talker_streaming(req_state)
        mark_close_sent(req_state)
        logger.info("[JoyAI] sentence TTS close_only req=%s — Talker idle", request_id)
        return PartialStageForward(
            output=SentenceTtsOutput(request_id, ""),
            is_final_update=True,
            close_only=True,
            queue_close_after=False,
        )
    return None


__all__ = [
    "JOYAI_TTS_CLOSE_ONLY_KEY",
    "JOYAI_TTS_FINISH_ON_NEXT_STOP_KEY",
    "JOYAI_TTS_PARTIAL_KEY",
    "JOYAI_TTS_SEED",
    "JOYAI_TTS_TEMPERATURE",
    "SentenceTtsOutput",
    "enable_talker_streaming",
    "incomplete_delegation_hold",
    "incomplete_marker_prefix",
    "mark_close_sent",
    "mark_talker_prompt_flags",
    "next_spoken_sentence_chunk",
    "plan_partial_stage_output",
    "sanitize_spoken_text",
    "spoken_payload",
    "spoken_text_ready",
]
