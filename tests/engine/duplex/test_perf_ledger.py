# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The per-session ledger behind ``session.stats`` (RFC #8325)."""

import pytest

from vllm_omni.benchmarks.duplex_session_metrics import session_stats_turn_metrics
from vllm_omni.engine.duplex.session.perf_ledger import SessionPerfLedger, with_perf_chunk

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

MS = 1_000_000


class _Clock:
    def __init__(self, now: int = 1_000 * MS) -> None:
        self.now = now

    def __call__(self) -> int:
        return self.now


def _ledger(**kwargs) -> tuple[SessionPerfLedger, _Clock]:
    clock = _Clock()
    ledger = SessionPerfLedger(stages=[(0, "thinker"), (1, "talker")], clock=clock, **kwargs)
    return ledger, clock


def _open(ledger: SessionPerfLedger, *, seq: int, turn_id: int = 0, epoch: int = 0, at: int = 0) -> None:
    ledger.open_chunk(epoch=epoch, seq=seq, turn_id=turn_id, turn_seq=seq, submitted_ns=ledger.opened_at_ns + at)


def _handle(ledger: SessionPerfLedger, clock: _Clock, chunk: tuple[int, int], *, at: int) -> None:
    clock.now = ledger.opened_at_ns + at
    ledger.mark_handled([chunk])


def _turn(ledger: SessionPerfLedger, turn_id: int = 0, epoch: int = 0) -> dict:
    return next(t for t in ledger.snapshot()["turns"] if (t["epoch"], t["turn_id"]) == (epoch, turn_id))


def test_snapshot_is_session_relative_and_final_after_turn_and_responses_end() -> None:
    ledger, clock = _ledger()
    origin = ledger.opened_at_ns
    _open(ledger, seq=1, at=5 * MS)
    ledger.record_stage_work([(0, 0, 1, origin + 6 * MS, 4 * MS), (1, 0, 1, origin + 8 * MS, 10 * MS)])
    ledger.bind_response(epoch=0, turn_id=0, response_id="resp-1")
    clock.now = origin + 20 * MS
    ledger.record_audio(response_id="resp-1", duration_ms=480.0)
    _handle(ledger, clock, (0, 1), at=21 * MS)

    stats = ledger.snapshot()
    assert stats["schema_version"] == 1
    assert stats["stages"] == [{"stage_id": 0, "name": "thinker"}, {"stage_id": 1, "name": "talker"}]
    (turn,) = stats["turns"]
    assert (turn["status"], turn["final"], turn["response_ids"]) == ("open", False, ["resp-1"])
    (chunk,) = turn["chunks"]
    assert chunk["submitted_ns"] == 5 * MS
    assert chunk["stage_work"] == [
        {"stage_id": 0, "spans": [[6 * MS, 4 * MS]]},
        {"stage_id": 1, "spans": [[8 * MS, 10 * MS]]},
    ]
    assert turn["audio"] == [{"response_id": "resp-1", "emitted_ns": 20 * MS, "duration_ms": 480.0}]
    assert (chunk["complete"], chunk["done_ns"]) == (False, None)

    ledger.close_turns(epoch=0, up_to_turn_id=0)
    assert _turn(ledger)["final"] is False  # the response is still open
    ledger.end_response("resp-1")
    turn = _turn(ledger)
    assert (turn["status"], turn["final"]) == ("done", True)
    # done_ns is the last span end; the API-side handling after it is not engine time.
    assert turn["chunks"][0]["done_ns"] == 18 * MS


def test_draining_tts_keeps_the_turn_open_and_late_spans_do_not_move_final_numbers() -> None:
    ledger, clock = _ledger()
    origin = ledger.opened_at_ns
    _open(ledger, seq=1)
    ledger.bind_response(epoch=0, turn_id=0, response_id="resp-1")
    ledger.record_stage_work([(0, 0, 1, origin + MS, 10 * MS)])
    _handle(ledger, clock, (0, 1), at=12 * MS)
    ledger.record_audio(response_id="resp-1", duration_ms=1000.0)
    # The next turn starts while this turn's TTS still drains.
    ledger.close_turns(epoch=0, up_to_turn_id=0)
    before = _turn(ledger)
    assert before["final"] is False
    assert session_stats_turn_metrics(before)["final"] is False

    # Delayed TTS work still lands on the turn before it is final.
    ledger.record_stage_work([(1, 0, 1, origin + 30 * MS, 20 * MS)])
    _handle(ledger, clock, (0, 1), at=55 * MS)
    ledger.end_response("resp-1")
    after = _turn(ledger)
    assert after["final"] is True
    metrics = session_stats_turn_metrics(after)
    assert metrics["generation_ms"] == 30.0
    assert after["chunks"][0]["done_ns"] == 50 * MS

    # Anything later is counted, not applied.
    ledger.record_stage_work([(1, 0, 1, origin + 80 * MS, 5 * MS)])
    _handle(ledger, clock, (0, 1), at=90 * MS)
    late = _turn(ledger)
    assert late["late_spans"] == 1
    assert session_stats_turn_metrics(late) == metrics


@pytest.mark.parametrize("cancel", ["barge_in", "session_close"])
def test_a_queued_chunk_of_a_cancelled_turn_never_reports_numbers(cancel: str) -> None:
    ledger, clock = _ledger()
    _open(ledger, seq=1)
    _handle(ledger, clock, (0, 1), at=10 * MS)
    _open(ledger, seq=2, at=1000 * MS)  # queued, no span and never handled
    if cancel == "barge_in":
        ledger.cancel_epoch(0)
    else:
        ledger.close_all()
    turn = _turn(ledger)
    assert (turn["status"], turn["final"]) == ("cancelled", False)
    assert [chunk["complete"] for chunk in turn["chunks"]] == [False, False]
    assert [chunk["done_ns"] for chunk in turn["chunks"]] == [None, None]
    assert session_stats_turn_metrics(turn)["final"] is False


def test_a_closed_turn_waits_for_its_queued_chunk_and_keeps_its_wait() -> None:
    ledger, clock = _ledger()
    origin = ledger.opened_at_ns
    _open(ledger, seq=1, at=0)
    ledger.close_turns(epoch=0, up_to_turn_id=0)
    assert _turn(ledger)["final"] is False
    ledger.record_stage_work([(0, 0, 1, origin + 290 * MS, 10 * MS)])
    _handle(ledger, clock, (0, 1), at=320 * MS)
    turn = _turn(ledger)
    assert (turn["status"], turn["final"]) == ("listen", True)
    # The queueing time stays inside the chunk's in-engine interval.
    assert session_stats_turn_metrics(turn)["turn_span_ms"] == 300.0


def test_chunks_done_long_before_the_turn_closes_keep_their_own_end() -> None:
    ledger, clock = _ledger()
    origin = ledger.opened_at_ns
    _open(ledger, seq=1, at=0)
    ledger.record_stage_work([(0, 0, 1, origin + MS, 5 * MS)])
    _handle(ledger, clock, (0, 1), at=10 * MS)
    _open(ledger, seq=2, at=1000 * MS)
    ledger.record_stage_work([(0, 0, 2, origin + 1001 * MS, 5 * MS)])
    _handle(ledger, clock, (0, 2), at=1010 * MS)
    clock.now = origin + 5000 * MS
    ledger.close_turns(epoch=0, up_to_turn_id=0)
    turn = _turn(ledger)
    assert [chunk["done_ns"] for chunk in turn["chunks"]] == [6 * MS, 1006 * MS]
    # Waiting for input between the chunks and before the close is not counted.
    assert session_stats_turn_metrics(turn)["turn_span_ms"] == 12.0


def test_a_paced_chunk_handled_after_the_next_input_ends_at_its_last_span() -> None:
    ledger, clock = _ledger()
    origin = ledger.opened_at_ns
    _open(ledger, seq=1, at=0)
    ledger.record_stage_work([(0, 0, 1, origin + MS, 50 * MS)])
    _open(ledger, seq=2, at=1000 * MS)
    # The listen output of chunk 1 is handled only after chunk 2 arrived.
    _handle(ledger, clock, (0, 1), at=1030 * MS)
    ledger.record_stage_work([(0, 0, 2, origin + 1001 * MS, 50 * MS)])
    _handle(ledger, clock, (0, 2), at=2030 * MS)
    ledger.close_turns(epoch=0, up_to_turn_id=0)
    turn = _turn(ledger)
    assert [chunk["done_ns"] for chunk in turn["chunks"]] == [51 * MS, 1051 * MS]
    # The wait for the next input is not engine time.
    assert session_stats_turn_metrics(turn)["turn_span_ms"] == 102.0


def test_a_final_chunk_without_spans_ends_at_its_submission() -> None:
    ledger, clock = _ledger()
    _open(ledger, seq=1, at=3 * MS)
    _handle(ledger, clock, (0, 1), at=10 * MS)
    ledger.close_turns(epoch=0, up_to_turn_id=0)
    turn = _turn(ledger)
    assert turn["final"] is True
    assert turn["chunks"][0]["done_ns"] == 3 * MS


def test_late_spans_of_an_old_epoch_land_on_that_epochs_chunk() -> None:
    ledger, _ = _ledger()
    origin = ledger.opened_at_ns
    _open(ledger, seq=1, epoch=0)
    ledger.cancel_epoch(0)
    _open(ledger, seq=1, epoch=1)
    ledger.record_stage_work([(1, 0, 1, origin + MS, MS), (0, 1, 1, origin + 3 * MS, MS)])
    by_epoch = {turn["epoch"]: turn["chunks"][0]["stage_work"] for turn in ledger.snapshot()["turns"]}
    assert by_epoch == {0: [{"stage_id": 1, "spans": [[MS, MS]]}], 1: [{"stage_id": 0, "spans": [[3 * MS, MS]]}]}


def test_spans_of_unknown_chunks_and_audio_of_unbound_responses_are_ignored() -> None:
    ledger, _ = _ledger()
    _open(ledger, seq=1)
    ledger.record_stage_work([(0, 0, 99, 0, MS)])
    ledger.record_audio(response_id="unbound", duration_ms=100.0)
    turn = _turn(ledger)
    assert turn["chunks"][0]["stage_work"] == []
    assert turn["audio"] == []


def test_touching_spans_coalesce_across_flushes() -> None:
    ledger, _ = _ledger()
    origin = ledger.opened_at_ns
    _open(ledger, seq=1)
    ledger.record_stage_work([(0, 0, 1, origin, 10)])
    ledger.record_stage_work([(0, 0, 1, origin + 10, 5)])
    ledger.record_stage_work([(0, 0, 1, origin + 30, 5)])
    assert _turn(ledger)["chunks"][0]["stage_work"][0]["spans"] == [[0, 15], [30, 5]]


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        ((5, 20), (0, 20), [[0, 25]]),  # later flush, earlier start, overlapping
        ((100, 5), (0, 10), [[0, 10], [100, 5]]),  # later flush, earlier start, disjoint
        ((0, 10), (2, 3), [[0, 10]]),  # contained
    ],
)
def test_spans_arriving_out_of_start_order_keep_their_union(
    first: tuple[int, int], second: tuple[int, int], expected: list[list[int]]
) -> None:
    ledger, _ = _ledger()
    origin = ledger.opened_at_ns
    _open(ledger, seq=1)
    ledger.record_stage_work([(0, 0, 1, origin + first[0], first[1])])
    ledger.record_stage_work([(0, 0, 1, origin + second[0], second[1])])
    assert _turn(ledger)["chunks"][0]["stage_work"][0]["spans"] == expected


def test_out_of_order_spans_bridge_the_spans_they_overlap() -> None:
    ledger, _ = _ledger()
    origin = ledger.opened_at_ns
    _open(ledger, seq=1)
    ledger.record_stage_work([(0, 0, 1, origin + start, 5) for start in (0, 20, 40)])
    ledger.record_stage_work([(0, 0, 1, origin + 3, 20)])
    assert _turn(ledger)["chunks"][0]["stage_work"][0]["spans"] == [[0, 25], [40, 5]]


def test_a_dropped_chunk_frees_its_empty_turn_and_its_retry_gets_a_fresh_id() -> None:
    ledger, _ = _ledger()
    origin = ledger.opened_at_ns
    _open(ledger, seq=1)
    ledger.drop_chunk((0, 1))
    assert ledger.snapshot()["turns"] == []
    retry = ledger.open_chunk(epoch=0, seq=1, turn_id=0, turn_seq=1, submitted_ns=origin + 2 * MS)
    assert retry != (0, 1)
    # The aborted submit's spans arrive after the retry opened and are ignored.
    ledger.record_stage_work([(0, 0, 1, origin + 3 * MS, MS)])
    assert _turn(ledger)["chunks"][0]["stage_work"] == []
    ledger.record_stage_work([(0, *retry, origin + 4 * MS, MS)])
    (chunk,) = _turn(ledger)["chunks"]
    assert (chunk["seq"], chunk["stage_work"]) == (1, [{"stage_id": 0, "spans": [[4 * MS, MS]]}])
    # A retry that fails again does not hand its id to the next one.
    ledger.drop_chunk(retry)
    assert ledger.open_chunk(epoch=0, seq=1, turn_id=0, turn_seq=1, submitted_ns=origin) not in {(0, 1), retry}


def test_dropped_ids_are_forgotten_once_a_newer_epoch_opens() -> None:
    ledger, _ = _ledger()
    _open(ledger, seq=1)
    ledger.drop_chunk((0, 1))
    _open(ledger, seq=1, epoch=1, turn_id=0)
    assert ledger._dropped_chunks == set()


def test_span_cap_marks_the_chunk_truncated_and_the_turn_never_final() -> None:
    ledger, clock = _ledger(max_spans_per_chunk=2)
    origin = ledger.opened_at_ns
    _open(ledger, seq=1)
    ledger.record_stage_work([(0, 0, 1, origin + start, 1) for start in (0, 10, 20)])
    _handle(ledger, clock, (0, 1), at=30)
    ledger.close_turns(epoch=0, up_to_turn_id=0)
    stats = ledger.snapshot()
    assert stats["dropped_spans"] == 1
    assert stats["turns"][0]["chunks"][0]["truncated"] is True
    assert stats["turns"][0]["final"] is False


def test_retention_evicts_the_oldest_finished_turn_first() -> None:
    ledger, clock = _ledger(max_turns=2)
    _open(ledger, seq=1, turn_id=0)
    _open(ledger, seq=2, turn_id=1)
    _handle(ledger, clock, (0, 1), at=MS)
    ledger.close_turns(epoch=0, up_to_turn_id=0)
    _open(ledger, seq=3, turn_id=2)
    stats = ledger.snapshot()
    assert [turn["turn_id"] for turn in stats["turns"]] == [1, 2]
    assert stats["dropped_turns"] == 1
    # The evicted turn's chunk is no longer addressable.
    ledger.record_stage_work([(0, 0, 1, 0, MS)])


def test_with_perf_chunk_copies_the_envelope_and_skips_prompts_without_one() -> None:
    duplex = {"epoch": 0, "seq": 3}
    prompt = {"prompt_token_ids": [1], "model_intermediate_buffer": {"duplex": duplex}}
    stamped = with_perf_chunk(prompt, (0, 3))
    assert stamped["model_intermediate_buffer"]["duplex"]["perf_chunk"] == [0, 3]
    assert "perf_chunk" not in duplex
    plain = {"prompt_token_ids": [1]}
    assert with_perf_chunk(plain, (0, 3)) is plain
