# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Per-session output pacing for full-duplex auto responses (``duplex_session.pacing``).

A duplex reply is paced by its input: one model unit per chunk period, each
continuation unit of synthesized silence submitted on the session's nominal
cadence (``compute_silence_continuation_deadline``). Played back with a jitter
buffer of ``B`` (``DUPLEX_CLIENT_PREBUFFER_S``, counted once per session), the
second unit's audio must arrive within ``B + d1`` of the first chunk's, where
``d1`` is that first chunk's length. With a short first chunk (0.12 s is common
at 16 concurrent sessions) the gap at unit 2 is about one chunk period plus the
latency difference of the two units -- a structural stall no Stage-2 batching
removes.

``SessionPacing`` lets a client that went quiet after its commit ("client 1")
submit its speech continuations up to ``onset_lead_max_s`` (< one unit) ahead of
the nominal cadence, so the model timeline runs that much ahead of the wall
clock. Quiet means no input append since the commit for at least
``max(QUIET_GUARD_MIN_S, 1.5 x the longest append gap of the committed
utterance)``: a client that keeps its microphone open and streams in long
chunks is never mistaken for a quiet one. Any input append after the commit
makes the rest of the reply wall-clock paced again (lead 0); unsubmitted
silence is then dropped by the runner's existing validity checks. The nominal
deadline chain is never moved: the lead only shortens each sleep, so it does
not accumulate.

The per-reply state (first chunk, playback mirror) follows the session's
active response lazily: whichever path ends a reply, the next decision sees a
different response id and resets, and a quiet client's pacing ends with the
reply its commit started.

``PlaybackMirror`` is the API's model of that client's playback (the same rule
as ``duplex_rt``): ``B`` before the first chunk of the session, then play on
arrival, re-anchoring after a late chunk, truncated by a cut.

``barge_cut_on_model_yield`` reuses the mirror for the playback cut: a new
user utterance over audible assistant audio (``barge_arm_min_silence_s`` of
non-speech input, then ``barge_arm_min_speech_s`` of speech, where pauses of
up to ``BARGE_SPEECH_HANGOVER_S`` do not end the utterance, starting at least
``barge_arm_min_playback_s`` after the reply's first audio) arms the session;
when the model then yields -- listens, or ends its turn -- while the client
still holds ``barge_cut_min_unplayed_s`` of unplayed audio, the reply is
cancelled like an explicit barge-in instead of draining its tail. Stage 0
drops a session's context with its aborted request, so the real input units of
the interrupting utterance (at most ``BARGE_REPLAY_MAX_UNITS``, the latest) are
replayed as the new epoch's first appends. Until the client appends again or a
reply opens, a model listen there gets a silence unit, as the cut reply's
continuations would have given a client that went quiet.

``fire_grid_ms`` (PersonaPlex-style lockstep) snaps the paced speech
continuations of quiet clients onto one grid of the shared monotonic clock, so
units of different sessions reach Stage 0 together: only earlier, within the
lead budget (the phase is paid by the lead). ``idle_grid`` rounds the
continuations after a non-terminal listen (no audio deadline) up to the grid.
Eager units -- the commit, the onset, CRITICAL, any real input -- never snap.

Everything here is pure Python on the orchestrator loop; with both switches
off no ``SessionPacing`` exists and the runner keeps its historical path.
"""

from __future__ import annotations

import math
from collections import deque
from enum import Enum
from typing import TYPE_CHECKING

from vllm.logger import init_logger

from vllm_omni.engine.duplex.config import DUPLEX_CLIENT_PREBUFFER_S

if TYPE_CHECKING:
    from vllm_omni.config.stage_config import DuplexPacingConfig, DuplexSessionRuntimeConfig
    from vllm_omni.engine.duplex.session.engine_session import DuplexEngineSession

logger = init_logger(__name__)

#: How often a paced continuation re-reads its lead while it waits.
PACE_RECHECK_S = 0.1
#: Shortest quiet after a commit before the onset lead applies.
QUIET_GUARD_MIN_S = 0.25
#: Non-speech input inside a user utterance that does not end it (playback-cut arming).
BARGE_SPEECH_HANGOVER_S = 0.3
#: Input units of the interrupting utterance replayed into the new epoch after a cut.
BARGE_REPLAY_MAX_UNITS = 6


class PlaybackMirror:
    """The client's playback timeline as the API expects it (``duplex_rt`` rule), in API monotonic time."""

    def __init__(self, *, prebuffer_s: float = DUPLEX_CLIENT_PREBUFFER_S) -> None:
        self.prebuffer_s = prebuffer_s
        #: Time the last scheduled audio ends (None before any audio).
        self.play_end: float | None = None
        #: Time of the current response's first audio (None until it emits).
        self.response_first_audio_t: float | None = None

    def new_response(self) -> None:
        self.response_first_audio_t = None

    def on_emit(self, now: float, seconds: float, *, current: bool = True) -> float:
        """Account one emitted audio delta; return the stall it caused (0 for a response's first chunk).

        ``current=False``: the tail of an older, draining reply. It still plays
        before anything newer, but is no part of the current response.
        """
        stall = 0.0
        if self.play_end is None:
            self.play_end = now + self.prebuffer_s + seconds
        else:
            if now > self.play_end:
                stall = now - self.play_end
                self.play_end = now
            self.play_end += seconds
        if not current:
            return 0.0
        if self.response_first_audio_t is None:
            # A gap before a response's first chunk is the model listening, not a stall.
            self.response_first_audio_t = now
            stall = 0.0
        return stall

    def unplayed(self, now: float) -> float:
        """Seconds of scheduled audio the client has not played yet."""
        if self.play_end is None:
            return 0.0
        return max(0.0, self.play_end - now)

    def cut(self, now: float) -> None:
        """The client stopped playing at ``now`` (a cancel or barge-in)."""
        if self.play_end is not None:
            self.play_end = min(self.play_end, now)
        self.new_response()


class PaceState(str, Enum):
    LISTEN = "listen"
    #: Committed; the reply's first audio has not been emitted yet.
    ONSET = "onset"
    PACED = "paced"
    #: The mirror saw a stall above ``critical_stall_s``: full lead until it recovers.
    CRITICAL = "critical"


class SessionPacing:
    """Client class, pacing state and lead of one duplex session (see module docstring)."""

    def __init__(self, runtime_config: DuplexSessionRuntimeConfig, session: DuplexEngineSession) -> None:
        self.runtime_config = runtime_config
        self.config: DuplexPacingConfig = runtime_config.pacing
        self.session = session
        self.mirror = PlaybackMirror()
        self.state = PaceState.LISTEN
        #: A commit happened and no input append arrived since (client 1).
        self.quiet_since_commit = False
        #: Length (s) of the current response's first audio chunk.
        self.d1: float | None = None
        self._clean_deltas = 0
        #: The response the per-reply state belongs to.
        self._response_id: str | None = None
        self._commit_t: float | None = None
        self._quiet_guard_s = QUIET_GUARD_MIN_S
        self._last_append_t: float | None = None
        self._utterance_gap_s = 0.0
        # Playback cut (``barge_cut_on_model_yield``).
        self.barge_enabled = runtime_config.barge_cut_on_model_yield is True
        self._quiet_input_s = 0.0
        self._speech_run_s = 0.0
        self._speech_run_over_playback = False
        #: When the current reply was armed for a cut (None: not armed).
        self.armed_at: float | None = None
        #: Real input units Stage 0 took during the current user utterance (replayed after a cut).
        self._utterance_units: deque[dict[str, object]] = deque(maxlen=BARGE_REPLAY_MAX_UNITS)
        #: The epoch a cut's replay opened, until the client appends again or a reply opens there.
        self._replay_epoch: int | None = None

    @property
    def paces_continuations(self) -> bool:
        return self.config.enabled is True

    def _sync(self) -> None:
        """Reset the per-reply state once the session's active response changed (any end path)."""
        response_id = self.session.active_response_id
        if response_id == self._response_id:
            return
        if self._response_id is not None:
            # The reply this state followed is gone: wall clock until the next commit.
            self.quiet_since_commit = False
            self._set_state(PaceState.LISTEN)
        self._response_id = response_id
        self._new_response()

    def _new_response(self) -> None:
        self.mirror.new_response()
        self.d1 = None
        self._clean_deltas = 0
        self._disarm()
        self._replay_epoch = None

    def quiet(self, now: float) -> bool:
        """Quiet since the commit, for at least the guard (see module docstring)."""
        return self.quiet_since_commit and self._commit_t is not None and now - self._commit_t >= self._quiet_guard_s

    # ---- inputs -------------------------------------------------------- #

    def on_commit(self, now: float) -> None:
        self._sync()
        self.quiet_since_commit = True
        self._commit_t = now
        # A client streaming in long chunks would look quiet between two of them.
        self._quiet_guard_s = min(1.0, max(QUIET_GUARD_MIN_S, 1.5 * self._utterance_gap_s))
        self._utterance_gap_s = 0.0
        # A reply that already spoke before the commit (Stage-0 handoff) is paced from its known d1.
        self._set_state(PaceState.ONSET if self.d1 is None else PaceState.PACED)

    def on_client_append(self, now: float, *, is_speech: bool, duration_s: float) -> None:
        """Any input append after the commit hands the rest of the reply back to the wall clock."""
        self._sync()
        self.quiet_since_commit = False
        last = self._last_append_t
        if last is not None and (self._commit_t is None or last > self._commit_t):
            # Gaps inside one utterance only: the first append after a commit opens a new one.
            self._utterance_gap_s = max(self._utterance_gap_s, now - last)
        self._last_append_t = now
        if self.barge_enabled:
            self._replay_epoch = None
            self._track_utterance(now, is_speech=is_speech, duration_s=duration_s)

    def _track_utterance(self, now: float, *, is_speech: bool, duration_s: float) -> None:
        """Arm the playback cut once a new user utterance runs over audible reply audio."""
        if not is_speech:
            self._quiet_input_s += duration_s
            if self._quiet_input_s > BARGE_SPEECH_HANGOVER_S:
                # A pause, not a dip inside the utterance: the speech run ends.
                self._speech_run_s = 0.0
                self._speech_run_over_playback = False
                if self.armed_at is None:
                    self._utterance_units.clear()
            return
        if self._speech_run_s == 0.0:
            # A speech run starts: it is a new utterance over the reply only
            # after enough quiet input and once the reply is audibly playing.
            self._speech_run_over_playback = (
                self._quiet_input_s >= self.runtime_config.barge_arm_min_silence_s and self._playing(now)
            )
        self._quiet_input_s = 0.0
        self._speech_run_s += duration_s
        if (
            self.armed_at is None
            and self._speech_run_over_playback
            and self._speech_run_s >= self.runtime_config.barge_arm_min_speech_s
        ):
            self.armed_at = now

    def _playing(self, now: float) -> bool:
        first_audio_t = self.mirror.response_first_audio_t
        return (
            first_audio_t is not None
            and now - first_audio_t >= self.runtime_config.barge_arm_min_playback_s
            and self.mirror.unplayed(now) > 0.0
        )

    @property
    def armed(self) -> bool:
        return self.armed_at is not None

    def cut_ready(self, now: float) -> bool:
        """Whether a model yield now should cut the reply the client is still playing."""
        self._sync()
        return (
            self.barge_enabled
            and self.armed_at is not None
            and self._response_id is not None
            and self.mirror.unplayed(now) >= self.runtime_config.barge_cut_min_unplayed_s
        )

    def note_cut(self, now: float, reason: str) -> None:
        logger.info(
            "Duplex barge cut (%s) response=%s armed %.2f s ago, %.2f s unplayed",
            reason,
            self._response_id,
            now - (self.armed_at if self.armed_at is not None else now),
            self.mirror.unplayed(now),
        )

    def on_input_unit(self, payload: dict[str, object]) -> None:
        """A real input unit was planned for Stage 0 while a user utterance runs: kept for the cut replay.

        Recorded when the append is planned, not when Stage 0 accepts it: an
        append still queued when the cut moves the epoch is dropped by the
        stale-epoch check, so the replay is its only delivery.
        """
        if not self.barge_enabled or (self._speech_run_s == 0.0 and self.armed_at is None):
            return
        if (
            self.armed_at is not None
            and payload.get("is_speech") is not True
            and len(self._utterance_units) == BARGE_REPLAY_MAX_UNITS
        ):
            # Non-speech input after arming is what the model hears before it
            # yields: kept while there is room, but it never pushes the start
            # of the utterance out of the replay.
            return
        self._utterance_units.append(dict(payload))

    def take_replay(self) -> list[dict[str, object]]:
        """The units of the utterance that armed the cut (empty when not armed); call before the cancel.

        The arming of a reply that already ended (the route-M listen abort does
        not ask ``cut_ready`` first) is reset here, not replayed.
        """
        if not self.barge_enabled:
            return []
        self._sync()
        units = list(self._utterance_units) if self.armed_at is not None else []
        self._utterance_units.clear()
        return units

    def on_replay(self, epoch: int) -> None:
        """A cut replayed the interrupting utterance as the first appends of ``epoch``."""
        self._replay_epoch = epoch

    def drives_replay(self, epoch: int) -> bool:
        """Whether a model listen in ``epoch`` should get a silence unit.

        Only in the epoch a cut's replay opened, while no reply opened there and
        the client appended nothing since the cut: the cut reply's
        continuations would have kept Stage 0 of a quiet client moving (it
        answers from the silence after the utterance), the new epoch has none.
        """
        if self._replay_epoch != epoch:
            return False
        self._sync()  # a reply that opened meanwhile ends it
        return self._replay_epoch == epoch

    def _disarm(self) -> None:
        self.armed_at = None
        self._speech_run_over_playback = False
        self._utterance_units.clear()

    def on_epoch_advance(self, now: float) -> None:
        """A barge-in or cancel: the client stops the old reply now."""
        self.mirror.cut(now)
        self.quiet_since_commit = False
        self._response_id = None
        self._new_response()
        self._set_state(PaceState.LISTEN)

    # ---- outputs ------------------------------------------------------- #

    def on_audio_emit(self, now: float, seconds: float, *, draining: bool = False) -> None:
        if seconds <= 0:
            return
        self._sync()
        if draining:
            # An older reply's tail: playback only, never the current reply's d1.
            self.mirror.on_emit(now, seconds, current=False)
            return
        first = self.mirror.response_first_audio_t is None
        stall = self.mirror.on_emit(now, seconds)
        if first:
            self.d1 = seconds
            if self.state == PaceState.ONSET:
                self._set_state(PaceState.PACED)
            return
        if stall > self.config.critical_stall_s:
            self._clean_deltas = 0
            if self.state == PaceState.PACED:
                self._set_state(PaceState.CRITICAL)
        elif self.state == PaceState.CRITICAL:
            self._clean_deltas += 1
            if self._clean_deltas >= 2:
                self._set_state(PaceState.PACED)

    # ---- decisions ----------------------------------------------------- #

    def lead_s(self, now: float) -> float:
        """How far ahead of its nominal deadline the next speech continuation may submit."""
        self._sync()
        lead_max = float(self.config.onset_lead_max_s)
        if not self.config.enabled or lead_max <= 0:
            return 0.0
        if not self.quiet(now) or self.state == PaceState.LISTEN:
            return 0.0
        if self.state == PaceState.CRITICAL or self.d1 is None:
            return lead_max
        return min(lead_max, max(0.0, float(self.config.onset_lead_target_s) - self.d1))

    def recheck_in(self, now: float) -> float:
        """How long a waiting continuation may sleep before its lead can change."""
        if self.quiet_since_commit and self._commit_t is not None:
            guard_left = self._commit_t + self._quiet_guard_s - now
            if guard_left > 0:
                return min(PACE_RECHECK_S, guard_left)
        return PACE_RECHECK_S

    def skip_response_wait(self, now: float) -> bool:
        """Whether the stop-token trigger may continue unit 2 before its response opens."""
        self._sync()
        return (
            self.config.enabled is True
            and self.config.onset_skip_response_wait is True
            and self.quiet(now)
            and self.state == PaceState.ONSET
        )

    def speech_fire_at(self, nominal: float, now: float) -> float:
        """Submission time of a speech continuation whose nominal deadline is ``nominal``."""
        return nominal - self.lead_s(now)

    def idle_fire_at(self, nominal: float, now: float) -> float:
        """Submission time of a continuation after a non-terminal listen (no audio deadline)."""
        self._sync()
        grid_s = self.config.fire_grid_ms / 1000.0
        if grid_s <= 0 or self.config.idle_grid is not True or not self.quiet(now):
            return nominal
        return grid_s * math.ceil(nominal / grid_s - 1e-9)

    def snap(self, fire_at: float, nominal: float, now: float) -> float:
        """Move a paced speech continuation earlier onto the fire grid, within the lead budget.

        Identity without ``fire_grid_ms``, outside PACED, or when the earlier
        grid point is already past or would lead more than ``onset_lead_max_s``.
        """
        if not self.snap_holds(now):
            return fire_at
        grid_s = self.config.fire_grid_ms / 1000.0
        lower = grid_s * math.floor(fire_at / grid_s + 1e-9)
        if lower >= now and nominal - lower <= self.config.onset_lead_max_s + 1e-9:
            return lower
        return fire_at

    def snap_holds(self, now: float) -> bool:
        """Whether a grid point applies (a grid, a paced reply and a quiet client)."""
        self._sync()
        return self.config.fire_grid_ms > 0 and self.state == PaceState.PACED and self.quiet(now)

    def _set_state(self, state: PaceState) -> None:
        self.state = state


def _unit_continuation(manager: object) -> object:
    return getattr(getattr(manager, "runtime_config", None), "unit_continuation", None)


def unit_continuation_depth(manager: object) -> int:
    """``duplex_session.unit_continuation.depth``: continuations of one session in flight (default 1)."""
    depth = getattr(_unit_continuation(manager), "depth", 1)
    return depth if isinstance(depth, int) and not isinstance(depth, bool) and depth >= 1 else 1


def continue_on_stop_token(manager: object) -> bool:
    """``unit_continuation.continue_on: stop_token``: the runner plans the next unit itself."""
    return getattr(_unit_continuation(manager), "continue_on", None) == "stop_token"


def same_turn_response_keeps_continuation(pace: object) -> bool:
    """Whether a model-turn continuation stays valid once its own turn's response opens.

    Only with ``onset_skip_response_wait``: the stop-token trigger then submits
    unit 2 under the model-turn owner before the response exists, and the
    response opening while it waits must not drop it.
    """
    if not isinstance(pace, SessionPacing):
        return False
    return pace.config.enabled is True and pace.config.onset_skip_response_wait is True
