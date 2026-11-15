# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Per-session record of stage work and emitted audio, by turn (RFC #8325).

The ledger keeps raw data only: each chunk's acceptance and handling times and
the stage processes' step spans, and each turn's emitted audio. Clients derive
busy RTF and the stall ratio from a ``session.stats`` snapshot
(``vllm_omni.benchmarks.duplex_session_metrics``).

A turn is **final** once its numbers can no longer change: the model turn
completed, every response bound to it ended (a draining TTS response included),
and every chunk was handled at least once. Closing a turn alone does not make it
final. A cancelled turn (barge-in, session close) never becomes final: the engine
reports no abort completion, so its in-flight work has no end time. Spans that
arrive for a final turn are counted in ``late_spans`` instead of changing it.

It lives on ``DuplexEngineSession`` and is mutated only on the orchestrator loop,
so it needs no locking. Times are ``time.monotonic_ns()`` from the API process
and the stage processes, so they are comparable only when every stage runs on
the same host; ``duplex_session.collect_stats`` is single-host only. Snapshots
report times relative to the session's open time.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from vllm_omni.core.sched.duplex_stage_work import PERF_CHUNK_KEY

SCHEMA_VERSION = 1
DEFAULT_MAX_TURNS = 256
DEFAULT_MAX_SPANS_PER_CHUNK = 4096

TURN_OPEN = "open"
TURN_DONE = "done"
TURN_CANCELLED = "cancelled"
TURN_LISTEN = "listen"


@dataclass
class AudioPiece:
    response_id: str | None
    emitted_ns: int
    duration_ms: float


@dataclass
class ChunkPerfRecord:
    seq: int
    turn_seq: int
    submitted_ns: int
    # stage_id -> [[start_ns, duration_ns], ...] sorted by start, absolute monotonic time
    stage_work: dict[int, list[list[int]]] = field(default_factory=dict)
    # Latest time the API process handled a stage output of this chunk. It only
    # gates finality: None while the chunk is still queued.
    handled_ns: int | None = None
    truncated: bool = False
    num_spans: int = 0

    def done_ns(self) -> int | None:
        """End of the in-engine interval: the last span end; None until handled.

        API-side handling is left out: a paced input's output is often handled
        only after the next input arrives, and that wait is not engine time.
        """
        if self.handled_ns is None:
            return None
        span_ends = (start + duration for spans in self.stage_work.values() for start, duration in spans)
        return max([self.submitted_ns, *span_ends])


@dataclass
class TurnPerfRecord:
    epoch: int
    turn_id: int
    closed: bool = False
    cancelled: bool = False
    response_ids: list[str] = field(default_factory=list)
    open_responses: set[str] = field(default_factory=set)
    chunks: dict[int, ChunkPerfRecord] = field(default_factory=dict)
    audio: list[AudioPiece] = field(default_factory=list)
    late_spans: int = 0

    @property
    def key(self) -> tuple[int, int]:
        return self.epoch, self.turn_id

    @property
    def status(self) -> str:
        if self.cancelled:
            return TURN_CANCELLED
        if not self.closed:
            return TURN_OPEN
        return TURN_DONE if self.audio else TURN_LISTEN

    @property
    def final(self) -> bool:
        return (
            self.closed
            and not self.cancelled
            and not self.open_responses
            and all(chunk.handled_ns is not None and not chunk.truncated for chunk in self.chunks.values())
        )


class SessionPerfLedger:
    """Bounded per-session store of turns and their chunks."""

    def __init__(
        self,
        *,
        stages: Sequence[tuple[int, str]],
        opened_at_ns: int | None = None,
        max_turns: int = DEFAULT_MAX_TURNS,
        max_spans_per_chunk: int = DEFAULT_MAX_SPANS_PER_CHUNK,
        clock: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self._clock = clock
        self.opened_at_ns = clock() if opened_at_ns is None else opened_at_ns
        self.stages = [(int(stage_id), str(name)) for stage_id, name in stages]
        self.max_turns = max(1, int(max_turns))
        self.max_spans_per_chunk = max(1, int(max_spans_per_chunk))
        self.dropped_turns = 0
        self.dropped_spans = 0
        # Insertion order is turn order; eviction takes the oldest final or cancelled turn.
        self._turns: dict[tuple[int, int], TurnPerfRecord] = {}
        # Chunk id -> turn. A chunk's id is its (epoch, seq) unless that seq was dropped before.
        self._chunk_turns: dict[tuple[int, int], tuple[int, int]] = {}
        # Dropped (epoch, seq) of the latest epoch; older epochs never reuse a seq.
        self._dropped_chunks: set[tuple[int, int]] = set()
        self._next_retry_id = -1
        self._response_turns: dict[str, tuple[int, int]] = {}

    def now_ns(self) -> int:
        return self._clock()

    # ------------------------------------------------------------------ #
    # Recording                                                          #
    # ------------------------------------------------------------------ #

    def open_chunk(self, *, epoch: int, seq: int, turn_id: int, turn_seq: int, submitted_ns: int) -> tuple[int, int]:
        """Register a chunk and return the id its append carries as ``perf_chunk``.

        The id is ``(epoch, seq)``. A seq reused after :meth:`drop_chunk` gets a
        fresh negative id instead, so late spans of the aborted submit, still
        stamped with the old id, cannot land on the retry.
        """
        chunk_id = (int(epoch), int(seq))
        self._prune_dropped(chunk_id[0])
        if chunk_id in self._dropped_chunks:
            chunk_id = (chunk_id[0], self._next_retry_id)
            self._next_retry_id -= 1
        turn = self._turn(epoch, turn_id)
        turn.chunks[chunk_id[1]] = ChunkPerfRecord(seq=seq, turn_seq=turn_seq, submitted_ns=submitted_ns)
        self._chunk_turns[chunk_id] = turn.key
        return chunk_id

    def drop_chunk(self, chunk_id: tuple[int, int]) -> None:
        """Forget a chunk whose submission was compensated; its spans are ignored from now on."""
        chunk_id = (int(chunk_id[0]), int(chunk_id[1]))
        if chunk_id[1] >= 0:  # a retry id is never opened again
            self._prune_dropped(chunk_id[0])
            self._dropped_chunks.add(chunk_id)
        turn_key = self._chunk_turns.pop(chunk_id, None)
        turn = self._turns.get(turn_key) if turn_key is not None else None
        if turn is None:
            return
        turn.chunks.pop(chunk_id[1], None)
        if not turn.chunks and not turn.response_ids:
            del self._turns[turn.key]

    def record_stage_work(self, rows: Iterable[Sequence[int]]) -> None:
        """Add ``(stage_id, epoch, seq, start_ns, duration_ns)`` spans; unknown chunks are ignored."""
        for stage_id, epoch, seq, start_ns, duration_ns in rows:
            found = self._chunk(epoch, seq)
            if found is None:
                continue
            turn, chunk = found
            if turn.final:
                # Reported numbers must not move; count the span instead.
                turn.late_spans += 1
                continue
            if chunk.num_spans >= self.max_spans_per_chunk:
                chunk.truncated = True
                self.dropped_spans += 1
                continue
            spans = chunk.stage_work.get(int(stage_id), [])
            before = len(spans)
            merged = _merge_span(spans, int(start_ns), int(start_ns) + int(duration_ns))
            chunk.stage_work[int(stage_id)] = merged
            chunk.num_spans += len(merged) - before

    def mark_handled(self, chunks: Iterable[tuple[int, int]]) -> None:
        now = self._clock()
        for epoch, seq in chunks:
            found = self._chunk(epoch, seq)
            if found is None or found[0].final:
                continue
            chunk = found[1]
            if chunk.handled_ns is None or now > chunk.handled_ns:
                chunk.handled_ns = now

    def record_audio(self, *, response_id: str, duration_ms: float) -> None:
        """Add one emitted audio delta to the turn its response belongs to."""
        turn_key = self._response_turns.get(response_id)
        turn = self._turns.get(turn_key) if turn_key is not None else None
        if turn is None or turn.final or duration_ms <= 0:
            return
        turn.audio.append(AudioPiece(response_id=response_id, emitted_ns=self._clock(), duration_ms=float(duration_ms)))

    def bind_response(self, *, epoch: int, turn_id: int, response_id: str) -> None:
        turn = self._turn(epoch, turn_id)
        if response_id not in turn.response_ids:
            turn.response_ids.append(response_id)
            turn.open_responses.add(response_id)
            self._response_turns[response_id] = turn.key

    def end_response(self, response_id: str) -> None:
        """The response finished (``response.done``); its turn may now become final."""
        turn_key = self._response_turns.get(response_id)
        turn = self._turns.get(turn_key) if turn_key is not None else None
        if turn is not None:
            turn.open_responses.discard(response_id)

    def close_turns(self, *, epoch: int, up_to_turn_id: int) -> None:
        """The model turn completed: close every open turn of ``epoch`` up to ``up_to_turn_id``."""
        for turn in self._turns.values():
            if turn.epoch == epoch and turn.turn_id <= up_to_turn_id:
                turn.closed = True

    def cancel_epoch(self, epoch: int) -> None:
        """Barge-in: every turn of the old epoch that is not final yet ends cancelled."""
        for turn in self._turns.values():
            if turn.epoch == epoch and not turn.final:
                turn.cancelled = True

    def close_all(self) -> None:
        """Session close: every turn that is not final yet ends cancelled."""
        for turn in self._turns.values():
            if not turn.final:
                turn.cancelled = True

    # ------------------------------------------------------------------ #
    # Query                                                              #
    # ------------------------------------------------------------------ #

    def snapshot(self) -> dict[str, object]:
        """The ``session.stats`` document (RFC #8325 §3.3), times relative to the session open."""
        origin = self.opened_at_ns
        return {
            "schema_version": SCHEMA_VERSION,
            "clock": {"source": "CLOCK_MONOTONIC", "unit": "ns", "origin": "session_opened"},
            "generated_at_ns": self._clock() - origin,
            "stages": [{"stage_id": stage_id, "name": name} for stage_id, name in self.stages],
            "turns": [_turn_document(turn, origin) for turn in self._turns.values()],
            "dropped_turns": self.dropped_turns,
            "dropped_spans": self.dropped_spans,
        }

    # ------------------------------------------------------------------ #
    # Internals                                                          #
    # ------------------------------------------------------------------ #

    def _turn(self, epoch: int, turn_id: int) -> TurnPerfRecord:
        key = (int(epoch), int(turn_id))
        turn = self._turns.get(key)
        if turn is None:
            if len(self._turns) >= self.max_turns:
                self._evict()
            turn = TurnPerfRecord(epoch=key[0], turn_id=key[1])
            self._turns[key] = turn
        return turn

    def _prune_dropped(self, epoch: int) -> None:
        if any(dropped_epoch < epoch for dropped_epoch, _ in self._dropped_chunks):
            self._dropped_chunks = {key for key in self._dropped_chunks if key[0] >= epoch}

    def _chunk(self, epoch: int, seq: int) -> tuple[TurnPerfRecord, ChunkPerfRecord] | None:
        turn_key = self._chunk_turns.get((int(epoch), int(seq)))
        turn = self._turns.get(turn_key) if turn_key is not None else None
        chunk = turn.chunks.get(int(seq)) if turn is not None else None
        return (turn, chunk) if turn is not None and chunk is not None else None

    def _evict(self) -> None:
        victim = next((turn for turn in self._turns.values() if turn.final or turn.cancelled), None)
        if victim is None:
            victim = next(iter(self._turns.values()))
        del self._turns[victim.key]
        for chunk_seq in victim.chunks:
            if self._chunk_turns.get((victim.epoch, chunk_seq)) == victim.key:
                del self._chunk_turns[(victim.epoch, chunk_seq)]
        for response_id in victim.response_ids:
            if self._response_turns.get(response_id) == victim.key:
                del self._response_turns[response_id]
        self.dropped_turns += 1


def _merge_span(spans: list[list[int]], start_ns: int, end_ns: int) -> list[list[int]]:
    """Add ``[start_ns, end_ns)`` to start-sorted ``spans``, joining only spans it overlaps.

    Spans arrive in flush order, which need not be start order.
    """
    if not spans or start_ns > spans[-1][0] + spans[-1][1]:
        spans.append([start_ns, end_ns - start_ns])
        return spans
    merged: list[list[int]] = []
    for span_start, span_end in sorted([(s, s + d) for s, d in spans] + [(start_ns, end_ns)]):
        if merged and span_start <= merged[-1][0] + merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], span_end - merged[-1][0])
        else:
            merged.append([span_start, span_end - span_start])
    return merged


def _turn_document(turn: TurnPerfRecord, origin: int) -> dict[str, object]:
    final = turn.final
    return {
        "epoch": turn.epoch,
        "turn_id": turn.turn_id,
        "response_ids": list(turn.response_ids),
        "status": turn.status,
        "final": final,
        "late_spans": turn.late_spans,
        "chunks": [_chunk_document(chunk, origin, final=final) for chunk in turn.chunks.values()],
        "audio": [
            {
                "response_id": piece.response_id,
                "emitted_ns": piece.emitted_ns - origin,
                "duration_ms": piece.duration_ms,
            }
            for piece in turn.audio
        ],
    }


def _chunk_document(chunk: ChunkPerfRecord, origin: int, *, final: bool) -> dict[str, object]:
    done_ns = chunk.done_ns() if final else None
    return {
        "seq": chunk.seq,
        "turn_seq": chunk.turn_seq,
        "submitted_ns": chunk.submitted_ns - origin,
        "done_ns": None if done_ns is None else done_ns - origin,
        "complete": final,
        "truncated": chunk.truncated,
        "stage_work": [
            {"stage_id": stage_id, "spans": [[start - origin, duration] for start, duration in spans]}
            for stage_id, spans in sorted(chunk.stage_work.items())
        ],
    }


def with_perf_chunk(prompt: dict[str, object], chunk: tuple[int, int]) -> dict[str, object]:
    """Return ``prompt`` with ``duplex.perf_chunk`` set, so stage schedulers time its steps.

    Prompts without a duplex envelope are returned unchanged; their chunks get no spans.
    """
    buffer = prompt.get("model_intermediate_buffer")
    duplex = buffer.get("duplex") if isinstance(buffer, Mapping) else None
    if not isinstance(buffer, Mapping) or not isinstance(duplex, Mapping):
        return prompt
    return {
        **prompt,
        "model_intermediate_buffer": {**buffer, "duplex": {**duplex, PERF_CHUNK_KEY: [int(chunk[0]), int(chunk[1])]}},
    }
