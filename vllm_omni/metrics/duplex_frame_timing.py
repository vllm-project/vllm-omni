# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in per-frame timing instrumentation for the duplex async-chunk path.

Enabled with ``VLLM_OMNI_DUPLEX_FRAME_TIMING=1`` (off by default); per-event
lines can be throttled with ``VLLM_OMNI_DUPLEX_FRAME_TIMING_LOG_EVERY``
(default ``1`` = every event). Every line shares the ``DUPLEX_FRAME_TIMING``
prefix and a ``t_ns`` stamp (``time.monotonic_ns()``), the same clock family
the client-side duplex timeline uses, so events from the API server and the
stage worker processes can be joined on one host and, eventually, aligned
with the client/benchmark metrics proposed in #7242 / #7025.

Measurement points — one line per frame/chunk at each site, emitted through
a dedicated ``log_*_event`` helper. The event schema (fields, gating, cadence
state) lives in this module only: call sites are single calls, so a hook can
follow its site across layer moves unchanged (the append and audio_emit hooks
moved from the deleted serving-layer session runner / runtime bridge to the
engine-resident session owner with the #7413 rework).

- ``append`` (duplex session runner, engine): a tick-sized user frame was
  framed and reserved for the engine. Reports inter-append ``jitter_ms`` and
  input ``drift_ms`` against the tick budget.
- ``connector_put`` (chunk transfer adapter, stage worker): chunk written
  into the inter-stage connector; ``wrap_ms`` is the ``connector.put`` wall
  time.
- ``connector_get`` (chunk transfer adapter, next stage worker): chunk read
  out of the connector; ``wrap_ms`` is the ``connector.get`` wall time and
  ``chunk_age_ms`` the put→get age when both sides run in one process. Across
  processes, join ``key`` + ``t_ns`` from the two lines offline.
- ``stage1_decode`` (PersonaPlex Code2Wav, stage-1 worker): streaming Mimi
  decode time for the new frames of a request.
- ``audio_emit`` (model channel, engine): an audio delta was projected for
  the client. Reports emit ``jitter_ms`` and output ``drift_ms`` against
  the tick budget — the server-side counterpart of the client receive
  timeline.

Name and clock alignment with #7242 / #7025:

- Clock: every ``t_ns`` is ``time.monotonic_ns()`` — the same CLOCK_MONOTONIC
  origin as the ``time.monotonic()`` instants #7242 publishes as
  ``stream_start`` / ``*_at_s``, so client and server lines join on one host
  by ``t_ns / 1e9``.
- Units: every duration field ends in ``_ms``, matching #7242 / #7025.
- ``audio_duration_ms`` is the per-emit counterpart of #7242's session-level
  ``stream_audio_duration_ms``; the ``audio_emit`` cadence lines are the
  server-side counterpart of the client TTFP receive timeline (#7242
  ``ttfp``).
- ``decode_ms`` is the per-chunk counterpart of the stage-1 "stage generation
  time" #7025 reports per stage.
- ``tick_period_ms`` mirrors the capabilities ``chunk_period_ms`` (RFC #7389
  ``duplex_tick_period_ms``); ``chunk_age_ms`` is that RFC's "chunk age".

All hooks are no-ops when the flag is unset; none of them changes control
flow, payload contents, or ordering.
"""

from __future__ import annotations

import itertools
import os
import threading
import time
from collections import OrderedDict
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

_TRUTHY_FLAGS = ("1", "true", "yes", "on")

# Tick budget when capabilities do not advertise one. Every native duplex
# serving adapter ships 80 ms ticks (one duplex frame of audio).
DEFAULT_TICK_PERIOD_MS = 80.0

# Streams (session-scoped pacers) and connector handoff stamps come and go
# without a close hook at these sites, so both registries are LRU-capped
# instead of relying on explicit lifecycle plumbing from the callers.
_MAX_TRACKED_STREAMS = 1024
_MAX_TRACKED_HANDOFFS = 4096

_log_sequence = itertools.count(start=1)
_handoff_lock = threading.Lock()


def duplex_frame_timing_enabled() -> bool:
    """Whether per-frame duplex timing instrumentation is switched on."""
    return os.environ.get("VLLM_OMNI_DUPLEX_FRAME_TIMING", "").lower() in _TRUTHY_FLAGS


def _log_every() -> int:
    try:
        return max(1, int(os.environ.get("VLLM_OMNI_DUPLEX_FRAME_TIMING_LOG_EVERY", "1") or 1))
    except ValueError:
        return 1


def _format_field(value: Any) -> str:
    if value is None:
        return "na"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def log_frame_timing(event: str, /, **fields: Any) -> None:
    """Emit one greppable structured line for a frame-timing event.

    A no-op unless ``VLLM_OMNI_DUPLEX_FRAME_TIMING`` is set; subject to the
    ``..._LOG_EVERY`` throttle otherwise.
    """
    if not duplex_frame_timing_enabled():
        return
    if (next(_log_sequence) - 1) % _log_every() != 0:
        return
    body = " ".join(f"{key}={_format_field(value)}" for key, value in fields.items())
    logger.info("DUPLEX_FRAME_TIMING event=%s t_ns=%d %s", event, time.monotonic_ns(), body)


def frame_timing_clock() -> float:
    """Start stamp for a wrap-time interval; ``0.0`` (never logged) when the
    flag is unset, so disabled paths stay free of ``perf_counter`` calls."""
    return time.perf_counter() if duplex_frame_timing_enabled() else 0.0


class DuplexTickPacer:
    """Inter-arrival jitter and drift of a periodic stream vs its budget.

    ``jitter_s`` is how much each observed interval deviates from the expected
    ``period_s * ticks``; ``drift_s`` is how far the stream has slipped from
    its anchor (``now - anchor - ticks_since_anchor * period_s`` — the anchor
    arrival itself is not counted, so a stream that keeps delivering one
    period of content per period of wall time reads zero drift). Duplex
    streams pause and resume (the user stops talking, drain tasks restart),
    so the anchor re-bases whenever the drift exceeds ``reanchor_s`` — after
    a gap the accumulated number is a pause artifact, not cadence
    information.
    """

    def __init__(self, period_s: float, *, reanchor_s: float = 2.0) -> None:
        self.period_s = float(period_s)
        self.reanchor_s = float(reanchor_s)
        self._last_ns: int | None = None
        self._anchor_ns: int | None = None
        self._ticks = 0

    def observe(self, now_ns: int | None = None, *, ticks: int = 1) -> tuple[float | None, float]:
        """Record one arrival worth ``ticks`` periods; return (jitter_s, drift_s)."""
        now = time.monotonic_ns() if now_ns is None else now_ns
        ticks = max(1, ticks)
        jitter_s: float | None = None
        if self._last_ns is not None:
            jitter_s = (now - self._last_ns) / 1e9 - self.period_s * ticks
        if self._anchor_ns is None:
            self._anchor_ns = now
            self._ticks = 0
            drift_s = 0.0
        else:
            drift_s = (now - self._anchor_ns) / 1e9 - (self._ticks + ticks) * self.period_s
            if abs(drift_s) > self.reanchor_s:
                self._anchor_ns = now
                self._ticks = 0
                drift_s = 0.0
            else:
                self._ticks += ticks
        self._last_ns = now
        return jitter_s, drift_s


_tick_pacers: OrderedDict[tuple[str, str], DuplexTickPacer] = OrderedDict()


def get_tick_pacer(scope: str, stream_id: str, period_s: float) -> DuplexTickPacer:
    """Return the (LRU-capped) pacer for one measured stream, e.g. an append
    or emit cadence of a duplex session."""
    key = (scope, str(stream_id))
    pacer = _tick_pacers.get(key)
    if pacer is None or pacer.period_s != float(period_s):
        pacer = DuplexTickPacer(period_s)
        _tick_pacers[key] = pacer
    else:
        _tick_pacers.move_to_end(key)
    while len(_tick_pacers) > _MAX_TRACKED_STREAMS:
        _tick_pacers.popitem(last=False)
    return pacer


_put_stamps_ns: OrderedDict[str, int] = OrderedDict()


def record_chunk_put(key: str, t_ns: int | None = None) -> None:
    """Stamp a connector chunk at put time so the receiving side can report
    its handoff age when both run in the same process."""
    if not duplex_frame_timing_enabled():
        return
    stamp = time.monotonic_ns() if t_ns is None else t_ns
    with _handoff_lock:
        _put_stamps_ns[key] = stamp
        while len(_put_stamps_ns) > _MAX_TRACKED_HANDOFFS:
            _put_stamps_ns.popitem(last=False)


def pop_chunk_put_age_ms(key: str, now_ns: int | None = None) -> float | None:
    """Consume the put stamp for ``key`` and return its age in ms, or None
    when the matching put happened in another process (join offline via the
    ``connector_put`` / ``connector_get`` log lines instead)."""
    with _handoff_lock:
        stamp = _put_stamps_ns.pop(key, None)
    if stamp is None:
        return None
    now = time.monotonic_ns() if now_ns is None else now_ns
    return (now - stamp) / 1e6


def _tick_period_ms(tick_period_ms: float | None) -> float:
    return float(tick_period_ms) if tick_period_ms else DEFAULT_TICK_PERIOD_MS


def log_append_event(
    session_id: str,
    epoch: int | None,
    byte_count: int,
    tick_period_ms: float | None,
) -> None:
    """Site hook (duplex session runner): a tick-sized user frame was framed
    and reserved for the engine."""
    if not duplex_frame_timing_enabled() or byte_count <= 0:
        return
    period_ms = _tick_period_ms(tick_period_ms)
    jitter_s, drift_s = get_tick_pacer("append", session_id, period_ms / 1e3).observe()
    log_frame_timing(
        "append",
        session=session_id,
        epoch=epoch,
        bytes=byte_count,
        tick_period_ms=period_ms,
        jitter_ms=None if jitter_s is None else jitter_s * 1e3,
        drift_ms=drift_s * 1e3,
    )


def log_audio_emit_event(
    session_id: str,
    epoch: int | None,
    request_id: str | None,
    audio_result: Any,
    tick_period_ms: float | None,
) -> None:
    """Site hook (runtime bridge): an audio delta was projected for the
    client. ``audio_result`` is the projected native event; non-dict results
    and events without a positive ``audio_duration_ms`` are not cadence
    points."""
    if not duplex_frame_timing_enabled() or not isinstance(audio_result, dict):
        return
    duration_ms = audio_result.get("audio_duration_ms")
    if not isinstance(duration_ms, (int, float)) or duration_ms <= 0:
        return
    period_ms = _tick_period_ms(tick_period_ms)
    frames = max(1, round(duration_ms / period_ms))
    jitter_s, drift_s = get_tick_pacer("emit", session_id, period_ms / 1e3).observe(ticks=frames)
    log_frame_timing(
        "audio_emit",
        session=session_id,
        epoch=epoch,
        request_id=request_id,
        frames=frames,
        # Coerced so the field renders like every other duration regardless
        # of the int/float the projected event carried.
        audio_duration_ms=float(duration_ms),
        jitter_ms=None if jitter_s is None else jitter_s * 1e3,
        drift_ms=drift_s * 1e3,
    )


def log_connector_put_event(
    key: str,
    stage_id: int,
    ok: object,
    size: object,
    started_at: float,
) -> None:
    """Site hook (chunk transfer adapter): chunk written into the connector.
    ``started_at`` is the ``frame_timing_clock()`` stamp taken before the
    put."""
    if not duplex_frame_timing_enabled():
        return
    record_chunk_put(key)
    log_frame_timing(
        "connector_put",
        key=key,
        stage=stage_id,
        ok=bool(ok),
        bytes=int(size or 0),
        wrap_ms=(time.perf_counter() - started_at) * 1e3,
    )


def log_connector_get_event(
    key: str,
    stage_id: int,
    size: int,
    started_at: float,
) -> None:
    """Site hook (chunk transfer adapter): chunk read out of the connector.
    ``started_at`` is the ``frame_timing_clock()`` stamp taken before the
    get."""
    if not duplex_frame_timing_enabled():
        return
    log_frame_timing(
        "connector_get",
        key=key,
        stage=stage_id,
        bytes=int(size),
        wrap_ms=(time.perf_counter() - started_at) * 1e3,
        chunk_age_ms=pop_chunk_put_age_ms(key),
    )


def log_stage1_decode_event(
    request_id: str | None,
    frames: int,
    started_at: float,
) -> None:
    """Site hook (PersonaPlex Code2Wav): streaming Mimi decode of the new
    frames of a request. ``started_at`` is the ``frame_timing_clock()`` stamp
    taken before the decode."""
    if not duplex_frame_timing_enabled():
        return
    log_frame_timing(
        "stage1_decode",
        request_id=request_id if request_id is not None else "unknown",
        frames=int(frames),
        decode_ms=(time.perf_counter() - started_at) * 1e3,
    )
