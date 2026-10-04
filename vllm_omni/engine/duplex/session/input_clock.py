# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Input-clocked sessions: model time advances only with client input.

A session opts in with ``extra_body.clock == "input"`` at creation (on a model
whose plugin sets ``DuplexModelPlugin.supports_input_clock``). The server then
never invents input of its own (no silence continuation; the idle window
defaults to 10 minutes) and acknowledges every client input that can advance
the model -- each ``input_audio_buffer.append``, ``input_audio_buffer.commit``
and ``response.create`` -- with exactly one ``input_audio_buffer.processed``
event, in input order, sent only once every output that input caused has been
sent. A client can therefore step the model as fast as acknowledgements come
back, or pause between inputs while it thinks, and always knows which outputs
belong to which input position.

How "every output that input caused" is known, without model specifics:

* A *unit* is one Stage-0 submission: whatever the model plugin's PCM buffer
  cut on an append (a 1 s chunk, an 80 ms frame) or a committed turn. The
  runner creates the unit when it starts the append that submits it; when
  Stage 0 accepts the submission the unit gets its *ordinal*, its position
  among the Stage-0 submissions of its epoch.
* Every Stage-0 submission ends in exactly one Stage-0 segment end, in
  submission order, so the n-th Stage-0 segment end of an epoch *decides* the
  unit with ordinal n. A plugin decision that ends the unit's pipeline (a
  listen, a silent turn) completes it (``DuplexModelPlugin.unit_decision``).
  Otherwise the unit is *speaking*.
* A speaking unit completes only on the session's final stage: when the plugin
  says its final-stage output is complete
  (``DuplexModelPlugin.unit_output_complete``; default: the final stage marks
  one segment end per unit). Speaking units take final-stage output in the
  order they were decided. A non-final stage (an intermediate text stage)
  finishing never completes a unit; a non-final stage ends a unit only through
  a decision with ``ends_unit=True`` (e.g. a silent turn that never reaches
  TTS). Later-stage progress that overtakes the Stage-0 segment end of its
  unit is held and replayed once that unit is speaking.
* Units are acknowledged strictly in order. An acknowledgement waits for every
  unit created up to the end of its input (its *watermark*) and for the
  acknowledgements before it.

Completion is kept apart from receiving a stage output: the runner turns each
stage output into a ``StageProgress`` item on its mailbox queued *after* that
output (and after any projected intermediate output or metrics it produced),
so an acknowledgement can never overtake the events it covers.

Units that will never produce model output are *settled* explicitly, each with
a defined ``units[].decision``: an append that never reached Stage 0
(``dropped``); open units of a cancelled epoch, an append the cancel called
off, and a deferred committed turn that was not submitted because a cancel
ended its response or its audio was dropped (``cancelled``); session teardown
(``aborted``); and timeouts (``timed_out``): the liveness valve when the
pipeline produced nothing for
``extra_body.input_clock_unit_timeout_s`` (default 15 s) while a unit was
open, and a maximum unit age, ``extra_body.input_clock_unit_max_s`` (default
60 s), for a unit that keeps producing but never completes.

Settling only concerns the acknowledgement. A unit settled before its output
arrived keeps its place in the pipeline: its Stage-0 segment end still decides
it (and only it), and if it speaks it still takes its own final-stage output
until that output is complete, so late output is never credited to the next
unit. Output of a cancelled epoch is dropped by the session (the
cancelled-output boundary), so its units leave the pipeline with the cancel.

The plugin hooks run on the session's output path: a hook that raises is
logged (once per session and hook) and treated as "no decision" / "not
complete", so the unit is left to the timeouts instead of failing every later
stage output.
"""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import TypeGuard

from vllm.logger import init_logger

from vllm_omni.engine.duplex.config import INPUT_CLOCK_IDLE_TIMEOUT_S, INPUT_CLOCK_KEY, input_clocked
from vllm_omni.engine.duplex.contracts import DuplexOutputContext, DuplexOutputDecision
from vllm_omni.engine.duplex.plugin import (
    DuplexModelPlugin,
    DuplexRuntimeConfigError,
    DuplexUnitDecision,
    DuplexUnitOutputs,
)
from vllm_omni.protocol.duplex.events import DuplexEvent, InputProcessed

logger = init_logger(__name__)

UNIT_TIMEOUT_KEY = "input_clock_unit_timeout_s"
DEFAULT_UNIT_TIMEOUT_S = 15.0
UNIT_MAX_AGE_KEY = "input_clock_unit_max_s"
DEFAULT_UNIT_MAX_AGE_S = 60.0

#: Client events an input-clocked session acknowledges (``wire_type`` of the command).
ACKNOWLEDGED_INPUTS = frozenset({"input_audio_buffer.append", "input_audio_buffer.commit", "response.create"})

#: ``units[].decision`` of a unit that was settled instead of completed by the model.
DECISION_DROPPED = "dropped"
DECISION_CANCELLED = "cancelled"
DECISION_ABORTED = "aborted"
DECISION_TIMED_OUT = "timed_out"
#: ``units[].decision`` of a unit whose pipeline continued past Stage 0, unless
#: the plugin labelled the decision.
DECISION_SPEAK = "speak"

#: ``units[].reason`` values the clock sets itself (cancel and close reasons come from the session).
REASON_NO_PROGRESS = "no_progress"
REASON_MAX_AGE = "max_age"

#: Stage progress held for a unit that is not speaking yet (``InputClock._hold``), at most.
_MAX_HELD_PROGRESS = 256
#: At most one warning per session this often; the rest of a burst is logged at debug level.
_WARNING_INTERVAL_S = 30.0


def _is_positive_seconds(raw: object) -> TypeGuard[int | float]:
    return isinstance(raw, int | float) and not isinstance(raw, bool) and math.isfinite(raw) and raw > 0


def _positive_seconds(extra_body: object, key: str, default: float) -> float:
    raw = extra_body.get(key) if isinstance(extra_body, Mapping) else None
    if _is_positive_seconds(raw):
        return float(raw)
    return default


def unit_timeout_s(extra_body: object) -> float:
    return _positive_seconds(extra_body, UNIT_TIMEOUT_KEY, DEFAULT_UNIT_TIMEOUT_S)


def unit_max_age_s(extra_body: object) -> float:
    return _positive_seconds(extra_body, UNIT_MAX_AGE_KEY, DEFAULT_UNIT_MAX_AGE_S)


def check_input_clock_supported(plugin: DuplexModelPlugin, extra_body: object) -> None:
    """Refuse ``clock: "input"`` on a model whose plugin has not opted in, or with an invalid timeout.

    The timeouts are only checked for an input-clocked session (a session
    without the clock ignores them); ``null`` means the default, and any other
    value that is not a positive number of seconds is refused
    (``invalid_duplex_runtime_config``) rather than replaced by the default.
    """
    if not input_clocked(extra_body):
        return
    if not plugin.supports_input_clock:
        raise DuplexRuntimeConfigError(
            f'extra_body.clock "input" is not supported by {plugin.plugin_id or "this model"}',
            code="input_clock_unsupported",
        )
    assert isinstance(extra_body, Mapping)
    for key in (UNIT_TIMEOUT_KEY, UNIT_MAX_AGE_KEY):
        value = extra_body.get(key)
        if value is not None and not _is_positive_seconds(value):
            raise DuplexRuntimeConfigError(
                f"extra_body.{key} must be a positive number of seconds, got {extra_body[key]!r}",
                code="invalid_duplex_runtime_config",
            )


def check_input_clock_unchanged(current: Mapping[str, object], candidate: Mapping[str, object]) -> None:
    """Refuse a ``session.update`` that changes the clock or its timeouts: they are fixed at creation.

    The timeouts are only compared for an input-clocked session; a session
    without the clock is unaffected by them.
    """
    settings: list[tuple[str, Callable[[object], object]]] = [(INPUT_CLOCK_KEY, input_clocked)]
    if input_clocked(current):
        settings += [(UNIT_TIMEOUT_KEY, unit_timeout_s), (UNIT_MAX_AGE_KEY, unit_max_age_s)]
    for key, effective in settings:
        if effective(current) != effective(candidate):
            raise DuplexRuntimeConfigError(
                f"session.update cannot change extra_body.{key}: the input clock is fixed at session creation",
                code="input_clock_update_unsupported",
            )


def input_clock_lease_idle_s(session_idle_s: float, deploy_idle_ttl_s: float | None) -> float | None:
    """Engine lease idle window of an input-clocked session.

    The session's own window (``idle_timeout_s``, default 10 minutes), capped
    at the deploy's ``idle_ttl_s`` or 10 minutes, whichever is longer, so a
    client cannot hold its admission slot and stage state indefinitely. A
    deploy without an idle TTL (``None``) expires no session for idleness.
    """
    if deploy_idle_ttl_s is None:
        return None
    return min(session_idle_s, max(deploy_idle_ttl_s, INPUT_CLOCK_IDLE_TIMEOUT_S))


@dataclass(frozen=True, slots=True)
class StageProgress:
    """Mailbox item queued behind one stage output's events: what it means for the input clock."""

    stage_id: int
    final_stage_id: int
    #: Session epoch of the output's request.
    epoch: int
    segment_finished: bool
    #: The plugin's classification of a decision taken on this output, if any.
    decision: DuplexUnitDecision | None = None
    #: The final-stage output and its context (``None`` for other stages).
    output: object | None = None
    context: DuplexOutputContext | None = None


@dataclass(slots=True, eq=False)
class InputClockUnit:
    """One Stage-0 submission, tracked until all of its output has been emitted."""

    seq: int
    #: Client input this unit consumed, in microseconds (exact: samples / rate).
    input_us: int | Fraction
    #: Session epoch the unit was submitted in (a cancel advances it).
    epoch: int = 0
    #: Clock time the unit was submitted (claimed, for a reserved turn).
    started_at: float = 0.0
    #: Reserved for a deferred committed turn that has not been submitted yet,
    #: keyed by that turn (its committed-audio operation id).
    placeholder: bool = False
    turn: str | None = None
    #: 1-based position among the Stage-0 submissions of its epoch, once Stage 0 accepted it.
    ordinal: int | None = None
    #: The acknowledgement no longer waits for this unit (completed by the model, or settled).
    complete: bool = False
    decision: str | None = None
    #: Why a settled unit ended (cancel / close / timeout reason), reported with the decision.
    reason: str | None = None
    # Pipeline view, kept after the unit was settled so its late output stays its own.
    upstream_segment_finished: bool = False
    final_segment_finished: bool = False

    @property
    def index(self) -> int:
        return self.seq - 1


@dataclass(slots=True)
class PendingAck:
    index: int
    trigger: str
    audio_end_us: int
    #: Units created before this input was handled; ``None`` until then.
    watermark: int | None = None


class InputClock:
    """Unit tracking and input acknowledgements for one session."""

    def __init__(
        self,
        *,
        emit: Callable[[list[DuplexEvent]], None],
        plugin: DuplexModelPlugin,
        runtime_config: Callable[[], Mapping[str, object]],
        unit_timeout_s: float = DEFAULT_UNIT_TIMEOUT_S,
        unit_max_age_s: float = DEFAULT_UNIT_MAX_AGE_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._emit = emit
        self._plugin = plugin
        self._runtime_config = runtime_config
        self._clock = clock
        self.unit_timeout_s = unit_timeout_s
        self.unit_max_age_s = unit_max_age_s
        #: Units the acknowledgements still wait for, in creation order (the completed prefix is dropped).
        self._units: deque[InputClockUnit] = deque()
        self._acks: deque[PendingAck] = deque()
        self._units_created = 0
        self._units_completed = 0
        self._input_count = 0
        # Exact (samples / rate), so per-append rounding never accumulates.
        self._audio_end = Fraction(0)
        self._unit_end = Fraction(0)
        #: Settled units not yet reported, oldest first: (seq, ``units[]`` entry, unit end after it).
        self._completed_since_ack: deque[tuple[int, dict[str, object], Fraction]] = deque()
        #: ``unit_end_ms`` of the last acknowledgement (the units it reported end there).
        self._reported_unit_end = Fraction(0)
        #: Plugin-owned state for ``unit_output_complete``, new in every epoch.
        self._plugin_state: dict[str, object] = {}
        #: Stage outputs of an epoch below this were cancelled (their events are dropped).
        self._live_epoch = 0
        #: Per epoch: Stage-0 submissions accepted, and Stage-0 segment ends seen.
        self._accepted: dict[int, int] = {}
        self._segment_ends: dict[int, int] = {}
        #: Accepted units waiting for their Stage-0 segment end, by (epoch, ordinal).
        self._undecided: dict[tuple[int, int], InputClockUnit] = {}
        #: Speaking units waiting for their final-stage output, in decision order.
        self._speaking: deque[InputClockUnit] = deque()
        #: Reserved slots settled before their deferred turn was submitted, by turn: a submission
        #: of that turn that no client input made (the end of the response) takes the settled unit.
        self._settled_turns: dict[str, InputClockUnit] = {}
        #: Later-stage progress that arrived before any unit of its epoch was speaking (``_hold``).
        self._held: deque[StageProgress] = deque()
        #: Epochs whose hold overflowed: their later-stage progress is no longer held (``_hold``).
        self._hold_overflowed: set[int] = set()
        self._closed = False
        #: Last time the pipeline produced something (or became busy): the liveness valve's reference.
        self.last_progress = clock()
        self._failed_hooks: set[str] = set()
        self._last_warning_at: float | None = None
        self._suppressed_warnings = 0

    @property
    def audio_end_us(self) -> int:
        return math.floor(self._audio_end)

    @property
    def unit_end_us(self) -> int:
        return math.floor(self._unit_end)

    @property
    def input_count(self) -> int:
        """Client inputs admitted so far (the ``input_index`` of the latest one)."""
        return self._input_count

    # ------------------------------------------------------------------ #
    # Input side                                                         #
    # ------------------------------------------------------------------ #

    def begin_input(self, trigger: str) -> PendingAck:
        """A client input left the mailbox; its acknowledgement is owed from now on."""
        self._input_count += 1
        ack = PendingAck(index=self._input_count, trigger=trigger, audio_end_us=self.audio_end_us)
        if not self._closed:
            self._acks.append(ack)
        return ack

    def note_input_audio(self, ack: PendingAck, *, samples: int, sample_rate_hz: int) -> None:
        """The input being handled carries ``samples`` of audio: it counts in ``audio_end_ms`` from now on.

        Audio the session later discards without submitting it (an overlap
        ``drop``, a silent chunk skipped in turn mode, a cleared buffer) stays
        counted: ``audio_end_ms`` is the client's input position.
        """
        if samples <= 0 or sample_rate_hz <= 0:
            return
        self._audio_end += Fraction(samples * 1_000_000, int(sample_rate_hz))
        ack.audio_end_us = self.audio_end_us

    def end_input(self, ack: PendingAck) -> None:
        """The input is handled (its units, if any, exist): fix its watermark.

        An input that only buffered audio created no unit, so its watermark is
        already reached: it is acknowledged as soon as the acknowledgements
        before it are out. An input that submitted several units is
        acknowledged once, after all of them, and lists each in ``units``.
        """
        if ack.watermark is None:
            ack.watermark = self._units_created
        self.flush()

    def _note_busy(self, now: float) -> None:
        # The valve measures how long the pipeline has been silent while it
        # had work: client input that arrives while units are open does not
        # hold it off, but a pipeline that was idle starts a fresh window.
        if not self._units:
            self.last_progress = now

    def new_unit(
        self, *, input_us: int | Fraction, epoch: int, turn: str | None = None, from_input: bool = True
    ) -> InputClockUnit:
        """A Stage-0 submission is starting; the submission of a deferred turn claims its reserved slot.

        A deferred turn whose slot was already settled (timed out, or released
        by a cancel that kept its audio) takes that settled unit's place when
        the end of the response submits it (``from_input=False``): its output
        stays its own and no acknowledgement waits for it. Submitted by a
        client input (a ``response.create``, a commit it was merged into), it
        is that input's new unit.
        """
        now = self._clock()
        input_us = max(0, input_us)
        if turn is not None:
            for unit in self._units:
                if unit.placeholder and unit.turn == turn:
                    unit.placeholder = False
                    unit.input_us = input_us
                    unit.epoch = epoch
                    unit.started_at = now
                    return unit
            settled = self._settled_turns.pop(turn, None)
            if settled is not None and not from_input and not self._closed:
                # Its acknowledgement entry is out; it only keeps its place in the
                # pipeline (a command waiting for this append treats it as healthy).
                settled.decision = settled.reason = None
                settled.epoch = epoch
                settled.started_at = now
                return settled
        self._note_busy(now)
        self._units_created += 1
        unit = InputClockUnit(seq=self._units_created, input_us=input_us, epoch=epoch, started_at=now)
        if self._closed:
            unit.complete = True  # nothing is acknowledged after teardown
        else:
            self._units.append(unit)
        return unit

    def reserve_unit(self, turn: str | None) -> None:
        """A committed turn was deferred behind the active response: count it now, submit it later.

        A second commit deferred before the first was submitted is merged into
        the same turn (one submission), so it takes over the open slot instead
        of reserving another.
        """
        if self._closed:
            return
        # The retained turn is now this one: an earlier settled slot's turn will not be submitted as such.
        self._settled_turns.clear()
        for unit in self._units:
            if unit.placeholder:
                unit.turn = turn
                return
        self._note_busy(self._clock())
        self._units_created += 1
        self._units.append(InputClockUnit(seq=self._units_created, input_us=0, placeholder=True, turn=turn))

    def rekey_placeholder(self, old_turn: str | None, new_turn: str | None) -> None:
        """The deferred turn is now carried by another commit (merged into it)."""
        if old_turn is not None:
            self._settled_turns.pop(old_turn, None)
        for unit in self._units:
            if unit.placeholder and unit.turn == old_turn:
                unit.turn = new_turn
                return

    def placeholder_turns(self) -> list[str | None]:
        return [unit.turn for unit in self._units if unit.placeholder]

    def release_placeholder(self, turn: str | None, reason: str) -> None:
        """The deferred turn was not submitted: its audio was dropped, or the response it waited for was cancelled."""
        for unit in list(self._units):
            if unit.placeholder and unit.turn == turn:
                self.settle_unit(unit, DECISION_CANCELLED, reason=reason)
                return

    def unit_submitted(self, unit: InputClockUnit) -> None:
        """Stage 0 accepted the unit's submission: it takes the next ordinal of its epoch."""
        if unit.ordinal is not None or self._closed or unit.epoch < self._live_epoch:
            # Already claimed, or its epoch was cancelled (its output is dropped).
            return
        ordinal = self._accepted.get(unit.epoch, 0) + 1
        self._accepted[unit.epoch] = ordinal
        unit.ordinal = ordinal
        self._undecided[(unit.epoch, ordinal)] = unit

    def append_finished(self, unit: InputClockUnit) -> None:
        """The append task that carried ``unit`` ended (not cancelled).

        A unit that never reached Stage 0 will never finish (``dropped``). An
        append that fails after Stage 0 accepted it closes the session, whose
        teardown settles the unit (``aborted``).
        """
        if unit.ordinal is None:
            self.settle_unit(unit, DECISION_DROPPED)
            self._drop_stray_held()

    # ------------------------------------------------------------------ #
    # Settling (units that will produce no further model output)         #
    # ------------------------------------------------------------------ #

    def _settle(self, unit: InputClockUnit, decision: str, reason: str | None) -> None:
        if unit.complete:
            return
        if unit.placeholder and unit.turn is not None and not self._closed:
            # Its turn may still be submitted when the response ends: that submission stays this unit's.
            self._settled_turns[unit.turn] = unit
        unit.placeholder = False
        unit.decision = decision
        unit.reason = reason
        self._complete(unit)

    def settle_unit(self, unit: InputClockUnit, decision: str, *, reason: str | None = None) -> None:
        """Complete ``unit`` without its model output, reporting ``decision`` (and ``reason``)."""
        if self._closed:
            return
        self._settle(unit, decision, reason)
        self.flush()

    def cancel_open(self, reason: str, *, before_epoch: int) -> None:
        """A cancel advanced the session epoch: settle the units submitted before it.

        Stage outputs of those epochs are dropped by the session (the
        cancelled-output boundary), so their units would never complete and
        leave the pipeline here. A deferred turn that is still waiting keeps
        its slot; it is released with ``release_placeholder`` when the cancel
        also dropped its audio.
        """
        if self._closed:
            return
        if before_epoch > self._live_epoch:
            # The plugin's per-epoch view (e.g. a count of the epoch's output
            # frames) starts over: the cancelled epochs' output is dropped.
            self._plugin_state = {}
        self._live_epoch = max(self._live_epoch, before_epoch)
        self._held = deque(progress for progress in self._held if progress.epoch >= before_epoch)
        self._hold_overflowed = {epoch for epoch in self._hold_overflowed if epoch >= before_epoch}
        for unit in list(self._units):
            if not unit.placeholder and unit.epoch < before_epoch:
                self._settle(unit, DECISION_CANCELLED, reason)
        self._undecided = {key: unit for key, unit in self._undecided.items() if key[0] >= before_epoch}
        self._speaking = deque(unit for unit in self._speaking if unit.epoch >= before_epoch)
        for counts in (self._accepted, self._segment_ends):
            for epoch in [epoch for epoch in counts if epoch < before_epoch]:
                del counts[epoch]
        self.flush()

    def close(self, reason: str) -> None:
        """Session teardown: settle what is open, acknowledge every owed input at once, then stop.

        Open units and reserved turns are settled as ``aborted``. All inputs
        whose acknowledgement is still owed -- including one still being
        handled -- are acknowledged by a single event (``first_input_index``
        .. ``input_index``, or a plain acknowledgement when only one is owed),
        emitted before the session's terminal event, so a teardown adds at
        most one event to the session's output. Afterwards the clock emits
        nothing.
        """
        if self._closed:
            return
        for unit in list(self._units):
            self._settle(unit, DECISION_ABORTED, reason)
        self._closed = True
        self._settled_turns.clear()
        self._held.clear()
        self._undecided.clear()
        self._speaking.clear()
        if not self._acks:
            return
        first, last = self._acks[0], self._acks[-1]
        self._acks.clear()
        self._emit([self._ack_event(last, first_index=first.index if first is not last else None, teardown=True)])

    # ------------------------------------------------------------------ #
    # Model side (called from the mailbox, after the referenced output)  #
    # ------------------------------------------------------------------ #

    def note_progress(self) -> None:
        """The session's pipeline produced something (resets the liveness valve)."""
        self.last_progress = self._clock()

    def on_stage_progress(self, progress: StageProgress) -> None:
        """Account for one stage output whose events have just been emitted."""
        if self._closed or progress.epoch < self._live_epoch:
            # Teardown, or a cancelled epoch: its events were dropped and its units settled.
            return
        if progress.stage_id == 0:
            if progress.segment_finished:
                self._on_stage0_segment_end(progress)
                self._drop_stray_held()
        else:
            self._on_later_stage_progress(progress)
        self.flush()

    def _on_later_stage_progress(self, progress: StageProgress) -> None:
        if progress.stage_id < progress.final_stage_id:
            self._on_upstream_progress(progress)
            return
        if not self._speaking:
            # Ahead of the Stage-0 segment end that makes its unit speaking.
            self._hold(progress)
            return
        if progress.segment_finished:
            self._speaking[0].final_segment_finished = True
        self._advance_speaking(progress.final_stage_id, progress.output, progress.context)

    def _hold(self, progress: StageProgress) -> None:
        """Keep a later stage's progress that no speaking unit can take yet, for the next unit that speaks.

        The orchestrator delivers different stages' outputs to the session in
        either order, so a later stage's output can overtake the Stage-0
        segment end that makes its unit speaking. That unit's Stage-0
        submission is then still undecided: progress is held only while its
        epoch has one, and replayed, in arrival order, when the next unit of
        the epoch starts speaking. Progress with no such submission is stray
        and dropped, so it never ends a unit submitted after it arrived.
        """
        epoch = progress.epoch
        if epoch in self._hold_overflowed or not self._stage0_pending(epoch):
            self._warn("input clock: stage %s progress of epoch %s matches no unit; dropped", progress.stage_id, epoch)
            return
        if len(self._held) >= _MAX_HELD_PROGRESS:
            # Which held item belongs to which unit is lost: credit none of them. The epoch's later
            # units are left to the timeouts; a cancel (a new epoch) recovers.
            logger.error(
                "input clock: more than %d stage progress events overtook their Stage-0 segment ends;"
                " dropped all of them and stop holding progress for epoch %s",
                _MAX_HELD_PROGRESS,
                epoch,
            )
            self._held.clear()
            self._hold_overflowed.add(epoch)
            return
        self._held.append(progress)

    def _stage0_pending(self, epoch: int) -> bool:
        """A Stage-0 submission of ``epoch`` has no segment end yet (accepted or not)."""
        return any(key[0] == epoch for key in self._undecided) or any(
            unit.epoch == epoch and unit.ordinal is None and not unit.placeholder and not unit.complete
            for unit in self._units
        )

    def _drop_stray_held(self) -> None:
        """Held progress whose epoch has no undecided Stage-0 submission left will never find its unit."""
        if not self._held:
            return
        kept = deque(progress for progress in self._held if self._stage0_pending(progress.epoch))
        if len(kept) != len(self._held):
            self._warn(
                "input clock: dropped %d held stage progress events that match no unit", len(self._held) - len(kept)
            )
            self._held = kept

    def _replay_held(self) -> None:
        held, self._held = self._held, deque()
        for progress in held:
            if progress.epoch >= self._live_epoch:
                self._on_later_stage_progress(progress)

    def _on_stage0_segment_end(self, progress: StageProgress) -> None:
        epoch = progress.epoch
        ordinal = self._segment_ends.get(epoch, 0) + 1
        self._segment_ends[epoch] = ordinal
        unit = self._undecided.pop((epoch, ordinal), None)
        if unit is None:
            unit = self._claim_unaccepted(epoch, ordinal)
        if unit is None:
            self._warn("input clock: Stage-0 segment end %s of epoch %s matches no submission", ordinal, epoch)
            return
        decision = progress.decision
        ends_unit = (decision is not None and decision.ends_unit) or progress.final_stage_id == 0
        if not unit.complete:
            # A unit settled before its segment end arrived keeps its settled decision.
            unit.decision = decision.label if decision is not None and decision.label else DECISION_SPEAK
        if ends_unit:
            if not unit.complete:
                self._complete(unit)
            return
        unit.upstream_segment_finished = progress.final_stage_id == 1
        self._speaking.append(unit)
        if self._held:
            self._replay_held()
        self._advance_speaking(progress.final_stage_id, None, None)

    def _claim_unaccepted(self, epoch: int, ordinal: int) -> InputClockUnit | None:
        """A segment end that overtook its submission's acceptance callback: give that unit the ordinal now.

        Appends run in wire order, so only the oldest open unit of the epoch
        that Stage 0 has not acknowledged yet can be the one in submission.
        """
        for unit in self._units:
            if unit.epoch == epoch and unit.ordinal is None and not unit.placeholder and not unit.complete:
                unit.ordinal = ordinal
                self._accepted[epoch] = max(self._accepted.get(epoch, 0), ordinal)
                return unit
        return None

    def _on_upstream_progress(self, progress: StageProgress) -> None:
        """A non-final stage after Stage 0: it ends a unit only through a decision that ends the pipeline."""
        upstream = progress.stage_id == progress.final_stage_id - 1
        target = next(
            (unit for unit in self._speaking if not (upstream and unit.upstream_segment_finished)),
            None,
        )
        if target is None:
            if progress.segment_finished or (progress.decision is not None and progress.decision.ends_unit):
                # Ahead of the Stage-0 segment end that makes its unit speaking.
                self._hold(progress)
            return
        decision = progress.decision
        if decision is not None and decision.ends_unit:
            self._speaking.remove(target)
            if not target.complete:
                if decision.label:
                    target.decision = decision.label
                self._complete(target)
        elif upstream and progress.segment_finished:
            target.upstream_segment_finished = True
        else:
            return
        self._advance_speaking(progress.final_stage_id, None, None)

    def unit_decision(
        self,
        *,
        stage_id: int,
        decision: DuplexOutputDecision | None,
        output: object = None,
        context: DuplexOutputContext | None = None,
    ) -> DuplexUnitDecision | None:
        """The plugin's classification of a segment end for the clock (``None``: none, or the hook raised)."""
        try:
            return self._plugin.unit_decision(
                stage_id=stage_id,
                decision=decision,
                output=output,
                context=context,
                runtime_config=self._runtime_config(),
            )
        except Exception:
            self._hook_failed("unit_decision", "no decision")
            return None

    def _advance_speaking(
        self,
        final_stage_id: int,
        new_output: object | None,
        new_context: DuplexOutputContext | None,
    ) -> None:
        """Ask the plugin about the oldest speaking unit; cascade while it keeps saying yes."""
        runtime_config = self._runtime_config()
        while self._speaking:
            unit = self._speaking[0]
            try:
                done = self._plugin.unit_output_complete(
                    unit=DuplexUnitOutputs(
                        index=unit.index,
                        epoch=unit.epoch,
                        # Set for every speaking unit (its Stage-0 segment end gave it one).
                        ordinal=unit.ordinal or 0,
                        upstream_segment_finished=unit.upstream_segment_finished,
                        final_segment_finished=unit.final_segment_finished,
                    ),
                    final_stage_id=final_stage_id,
                    new_output=new_output,
                    new_context=new_context,
                    state=self._plugin_state,
                    runtime_config=runtime_config,
                )
            except Exception:
                self._hook_failed("unit_output_complete", "not complete")
                return
            if not done:
                return
            self._speaking.popleft()
            if not unit.complete:
                self._complete(unit)
            # The output was already counted; the next unit is asked with no new output.
            new_output = new_context = None

    def _hook_failed(self, hook: str, treated_as: str) -> None:
        if hook in self._failed_hooks:
            logger.debug("input clock: plugin hook %s raised again", hook, exc_info=True)
            return
        self._failed_hooks.add(hook)
        logger.exception(
            "input clock: plugin hook %s raised; treated as %s, the unit is left to the timeouts"
            " (further failures of this hook in the session are logged at debug level)",
            hook,
            treated_as,
        )

    def _warn(self, message: str, *args: object) -> None:
        """A warning at most every ``_WARNING_INTERVAL_S`` per session; the rest of a burst goes to debug."""
        now = self._clock()
        if self._last_warning_at is not None and now - self._last_warning_at < _WARNING_INTERVAL_S:
            self._suppressed_warnings += 1
            logger.debug(message, *args)
            return
        self._last_warning_at = now
        if self._suppressed_warnings:
            message += " (%d similar messages since the previous warning were logged at debug level)"
            args = (*args, self._suppressed_warnings)
            self._suppressed_warnings = 0
        logger.warning(message, *args)

    def _complete(self, unit: InputClockUnit) -> None:
        unit.complete = True
        # Units are acknowledged in order: advance over the completed prefix.
        while self._units and self._units[0].complete:
            head = self._units.popleft()
            self._units_completed += 1
            self._unit_end += head.input_us
            entry: dict[str, object] = {"end_ms": self.unit_end_us // 1000, "decision": head.decision}
            if head.reason is not None:
                entry["reason"] = head.reason
            self._completed_since_ack.append((head.seq, entry, self._unit_end))

    # ------------------------------------------------------------------ #
    # Acknowledgements                                                   #
    # ------------------------------------------------------------------ #

    def _ack_event(self, ack: PendingAck, *, first_index: int | None = None, teardown: bool = False) -> InputProcessed:
        """The acknowledgement of ``ack``: it reports the settled units created up to its input, no later ones.

        Units of later inputs that settled already (inputs sent ahead, or
        several units settled by one timeout check) wait for the
        acknowledgement of the input that created them, so ``unit_end_ms``
        never runs ahead of the acknowledged input. A teardown acknowledgement
        reports every settled unit.
        """
        units: list[dict[str, object]] = []
        pending = self._completed_since_ack
        while pending and (teardown or ack.watermark is None or pending[0][0] <= ack.watermark):
            _, entry, self._reported_unit_end = pending.popleft()
            units.append(entry)
        return InputProcessed(
            audio_end_ms=ack.audio_end_us // 1000,
            unit_end_ms=math.floor(self._reported_unit_end) // 1000,
            input_index=ack.index,
            first_input_index=first_index,
            trigger=ack.trigger,
            units=tuple(units),
        )

    def flush(self) -> None:
        if self._closed:
            return
        events: list[DuplexEvent] = []
        while self._acks:
            ack = self._acks[0]
            if ack.watermark is None or self._units_completed < ack.watermark:
                break
            self._acks.popleft()
            events.append(self._ack_event(ack))
        if events:
            self._emit(events)

    def has_open_units(self) -> bool:
        return bool(self._units)

    def expire(self) -> bool:
        """Timeouts: settle units the pipeline will evidently not complete (``timed_out``).

        * Liveness valve: the session's pipeline produced nothing for
          ``unit_timeout_s`` while units were open -- the oldest open unit is
          settled (``no_progress``) and the window restarts.
        * Maximum unit age: a submitted unit still open ``unit_max_age_s``
          after its submission -- e.g. a model that keeps streaming but never
          marks the unit complete -- is settled (``max_age``).

        A model that breaks the unit contract degrades into slow
        acknowledgements instead of a client that waits forever.
        """
        if self._closed or not self._units:
            return False
        now = self._clock()
        expired: list[tuple[InputClockUnit, str]] = []
        if now - self.last_progress >= self.unit_timeout_s:
            expired.append((self._units[0], REASON_NO_PROGRESS))
        for unit in self._units:
            if not unit.placeholder and not unit.complete and now - unit.started_at >= self.unit_max_age_s:
                if all(unit is not seen for seen, _ in expired):
                    expired.append((unit, REASON_MAX_AGE))
        if not expired:
            return False
        for unit, reason in expired:
            if unit in self._speaking:
                self._warn(
                    "input clock: speaking unit %s settled after a timeout (%s). It keeps its place for its"
                    " final-stage output: if that never comes, the speaking units after it in this epoch are"
                    " acknowledged through their own timeouts (output stays correctly attributed)",
                    unit.seq,
                    reason,
                )
            else:
                self._warn(
                    "input clock: settling unit %s (decision=%s) after a timeout (%s)", unit.seq, unit.decision, reason
                )
            self._settle(unit, DECISION_TIMED_OUT, reason)
        self.last_progress = now
        self.flush()
        return True


__all__ = [
    "ACKNOWLEDGED_INPUTS",
    "DECISION_ABORTED",
    "DECISION_CANCELLED",
    "DECISION_DROPPED",
    "DECISION_SPEAK",
    "DECISION_TIMED_OUT",
    "DEFAULT_UNIT_MAX_AGE_S",
    "DEFAULT_UNIT_TIMEOUT_S",
    "InputClock",
    "InputClockUnit",
    "PendingAck",
    "StageProgress",
    "check_input_clock_supported",
    "check_input_clock_unchanged",
    "input_clock_lease_idle_s",
    "input_clocked",
    "unit_max_age_s",
    "unit_timeout_s",
]
