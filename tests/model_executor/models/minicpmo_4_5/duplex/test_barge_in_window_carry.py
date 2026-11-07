# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""A barge-in keeps the MiniCPM-o 4.5 Stage-0 window.

The barge-in aborts the epoch's Stage-0 request and the next append opens a new
one. The scheduler and the worker each keep the aborted request's completed
window units, and the new request's first window plan puts them back. These
tests check that both sides keep the same units and agree on the rebuilt
prompt length, and the greedy answer decision after the user's input.
"""

from __future__ import annotations

import base64
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.request import Request, RequestStatus, StreamingUpdate

from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.stage0 import (
    MiniCPMO45Stage0DuplexRuntime,
    _MiniCPMO45CarriedWindow,
    _MiniCPMO45Stage0SessionState,
    _MiniCPMO45WindowUnit,
)
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import MiniCPMO45OmniForConditionalGeneration

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

MARKER = [70, 71]
LISTEN, UNIT_END, TURN_EOS, CHUNK_EOS, CHUNK_TTS_EOS = 151705, 151684, 151717, 151718, 151721


def _stage0_id(session_id: str, epoch: int) -> str:
    encoded = base64.urlsafe_b64encode(session_id.encode()).decode().rstrip("=")
    return f"duplex-s.{encoded}.e.{epoch}.r.stage0"


def _scheduler() -> OmniARScheduler:
    sched = OmniARScheduler.__new__(OmniARScheduler)
    sched._new_prompt_len_snapshot = {}
    sched._hold_payloadless_outputs = False
    sched.vllm_config = SimpleNamespace(model_config=SimpleNamespace(stage_id=0, session_mode="duplex"))
    sched.num_waiting_for_streaming_input = 0
    sched.log_stats = False
    sched.hash_block_size = 16
    sched.chunk_transfer_adapter = None
    sched.kv_holding_waiting = set()
    sched.deferred_waiting = set()
    sched._free_request_blocks = MagicMock()
    sched.encoder_cache_manager = MagicMock()
    sched._inflight_prefills = set()
    sched._outputs_awaiting_stage_payload = {}
    sched.requests = {}
    return sched


def _new_request(request_id: str) -> Request:
    """A new epoch's request after its first append: 3 context tokens, 6 unit tokens, 1 sample."""
    request = Request(
        request_id=request_id,
        prompt_token_ids=[0] * 9,
        sampling_params=SamplingParams(max_tokens=8),
        pooling_params=None,
        arrival_time=100.0,
        block_hasher=None,
    )
    request.append_output_token_ids([40])
    request.num_computed_tokens = 10
    request.status = RequestStatus.WAITING_FOR_STREAMING_REQ
    return request


def _second_append(mode: str, *, high: int = 100, low: int = 80, context_max_units: int = 24) -> StreamingUpdate:
    """The new request's second append: closing terminator + </unit>, then 6 unit tokens."""
    update = StreamingUpdate(
        mm_features=None,
        prompt_token_ids=[0] * 8,
        max_tokens=32,
        arrival_time=101.0,
        sampling_params=SamplingParams(max_tokens=16),
    )
    update.model_intermediate_buffer = {
        "duplex": {
            "data_plane": True,
            "seq": 2,
            "runtime_config": {
                "duplex_scheduler_token_id": 0,
                "duplex_first_append_context_tokens": 3,
                "duplex_window_prefix_tokens": 2,
                "duplex_window_suffix_token_ids": [3],
                "duplex_window_previous_marker_token_ids": MARKER,
                "duplex_window_special_token_ids": [99],
                "duplex_window_config": {
                    "sliding_window_mode": mode,
                    "basic_window_high_tokens": high,
                    "basic_window_low_tokens": low,
                    "context_previous_max_tokens": 4,
                    "context_max_units": context_max_units,
                },
            },
        }
    }
    return update


CARRIED = [{"length": 5, "generated_token_ids": [41]}, {"length": 6, "generated_token_ids": []}]


def test_first_plan_puts_the_aborted_window_back_ahead_of_the_new_unit() -> None:
    sched = _scheduler()
    session = _new_request(_stage0_id("s", 1))
    session._minicpmo45_carried_window = {"units": [dict(unit) for unit in CARRIED], "previous": []}
    update = _second_append("basic")

    sched._update_request_as_session(session, update)

    plan = update.model_intermediate_buffer["duplex"]["stage0_window"]
    # Context 3 + carried 5 + 6 + the new unit (6 tokens, 1 sample, 2 closing) + the pending 6.
    assert plan["replace"] is True
    assert plan["carried_units"] == 2
    assert plan["drop_units"] == 0
    assert plan["replacement_prompt_len"] == 29
    assert session.prompt_token_ids == [0] * 29
    assert session.num_computed_tokens == 0
    assert [unit["length"] for unit in session._minicpmo45_window_units] == [5, 6, 9]
    assert session._minicpmo45_window_open_start == 3 + 5 + 6 + 9
    assert session._minicpmo45_carried_window is None


def test_carried_window_over_the_high_watermark_is_trimmed_from_the_oldest_unit() -> None:
    sched = _scheduler()
    session = _new_request(_stage0_id("s", 1))
    session._minicpmo45_carried_window = {"units": [dict(unit) for unit in CARRIED], "previous": []}
    update = _second_append("basic", high=25, low=20)

    sched._update_request_as_session(session, update)

    plan = update.model_intermediate_buffer["duplex"]["stage0_window"]
    assert plan["carried_units"] == 2
    assert plan["drop_units"] == 2
    assert plan["replacement_prompt_len"] == 29 - 11
    assert [unit["length"] for unit in session._minicpmo45_window_units] == [9]


def test_context_window_keeps_the_carried_previous_text() -> None:
    sched = _scheduler()
    session = _new_request(_stage0_id("s", 1))
    session._minicpmo45_carried_window = {"units": [dict(unit) for unit in CARRIED], "previous": [33]}
    update = _second_append("context", context_max_units=1)

    sched._update_request_as_session(session, update)

    plan = update.model_intermediate_buffer["duplex"]["stage0_window"]
    assert plan["mode"] == "context"
    assert plan["carried_units"] == 2
    assert plan["drop_units"] == 2
    assert plan["previous_token_ids"] == [33, 41]
    # 29 + marker 2 + previous 2 - the 11 carried tokens moved into the previous text.
    assert plan["replacement_prompt_len"] == 29 + 4 - 11
    assert session._minicpmo45_window_previous_len == 4


def test_window_mode_off_drops_the_carried_window() -> None:
    sched = _scheduler()
    session = _new_request(_stage0_id("s", 1))
    session._minicpmo45_carried_window = {"units": [dict(unit) for unit in CARRIED], "previous": []}
    update = _second_append("off")

    sched._update_request_as_session(session, update)

    assert "stage0_window" not in update.model_intermediate_buffer["duplex"]
    assert session._minicpmo45_carried_window is None


@pytest.mark.parametrize(("scheduled", "kept"), [(False, [5, 6]), (True, [9])])
def test_abort_keeps_the_window_the_worker_has(scheduled: bool, kept: list[int]) -> None:
    """The plan runs when the append arrives; the worker changes its units only when it runs that append.

    The plan below drops both old units and closes a new one. Aborted before the
    append is scheduled, the worker still has the two old units; aborted after,
    it has applied the plan.
    """
    sched = _scheduler()
    request_id = _stage0_id("s", 0)
    # Context 3, two closed units (5 + 6) and the open one (6 tokens, 1 sample).
    session = Request(
        request_id=request_id,
        prompt_token_ids=[0] * 20,
        sampling_params=SamplingParams(max_tokens=8),
        pooling_params=None,
        arrival_time=100.0,
        block_hasher=None,
    )
    session.append_output_token_ids([40])
    session.num_computed_tokens = 21
    session.status = RequestStatus.WAITING_FOR_STREAMING_REQ
    session._minicpmo45_window_units = [dict(unit) for unit in CARRIED]
    session._minicpmo45_window_open_start = 14
    sched.requests[request_id] = session
    update = _second_append("basic", high=25, low=20)
    sched._update_request_as_session(session, update)
    assert update.model_intermediate_buffer["duplex"]["stage0_window"]["drop_units"] == 2
    if scheduled:
        sched._commit_minicpmo45_window_snapshots(SimpleNamespace(num_scheduled_tokens={request_id: 8}))

    sched._stash_minicpmo45_windows([request_id])

    stashed = sched._minicpmo45_carried_windows["s"]
    assert [unit["length"] for unit in stashed["units"]] == kept
    assert request_id not in sched._minicpmo45_window_snapshot_ids


def test_a_second_append_before_booking_keeps_the_first_snapshot() -> None:
    sched = _scheduler()
    request_id = _stage0_id("s", 0)
    session = _new_request(request_id)
    session._minicpmo45_window_units = [dict(unit) for unit in CARRIED]
    sched.requests[request_id] = session

    sched._snapshot_minicpmo45_window(session)
    session._minicpmo45_window_units = [{"length": 9, "generated_token_ids": []}]
    sched._snapshot_minicpmo45_window(session)
    sched._stash_minicpmo45_windows([request_id])

    assert [unit["length"] for unit in sched._minicpmo45_carried_windows["s"]["units"]] == [5, 6]


def test_abort_before_the_first_plan_of_an_empty_window_keeps_nothing() -> None:
    """Epoch 0's second append records its first unit; aborted before it is booked, the worker has no closed unit."""
    sched = _scheduler()
    request_id = _stage0_id("s", 0)
    session = _new_request(request_id)
    sched.requests[request_id] = session
    sched._update_request_as_session(session, _second_append("basic"))
    assert session._minicpmo45_window_units

    sched._stash_minicpmo45_windows([request_id])

    assert "s" not in (getattr(sched, "_minicpmo45_carried_windows", None) or {})


def test_restash_keeps_the_time_of_the_first_abort(monkeypatch: pytest.MonkeyPatch) -> None:
    sched = _scheduler()
    old_id = _stage0_id("s", 0)
    old = _new_request(old_id)
    old._minicpmo45_window_units = [dict(unit) for unit in CARRIED]
    sched.requests[old_id] = old
    sched._stash_minicpmo45_windows([old_id])
    first_time = sched._minicpmo45_carried_windows["s"]["time"]

    new_id = _stage0_id("s", 1)
    new = _new_request(new_id)
    sched._attach_minicpmo45_carried_window(new)
    sched.requests[new_id] = new
    later = first_time + 5
    monkeypatch.setattr(time, "monotonic", lambda: later)
    sched._stash_minicpmo45_windows([new_id])

    assert sched._minicpmo45_carried_windows["s"]["time"] == first_time


def test_abort_before_the_first_plan_passes_the_carried_window_on() -> None:
    sched = _scheduler()
    request_id = _stage0_id("s", 1)
    request = _new_request(request_id)
    request._minicpmo45_carried_window = {"units": [dict(unit) for unit in CARRIED], "previous": [33]}
    sched.requests[request_id] = request

    sched._stash_minicpmo45_windows([request_id])

    stashed = sched._minicpmo45_carried_windows["s"]
    assert (stashed["units"], stashed["previous"]) == (CARRIED, [33])


def test_second_barge_in_before_the_carry_is_placed_keeps_the_carried_units() -> None:
    """The new request's first plan takes the carried units, then a second abort lands before the worker runs it."""
    sched = _scheduler()
    request_id = _stage0_id("s", 1)
    session = _new_request(request_id)
    session._minicpmo45_carried_window = {"units": [dict(unit) for unit in CARRIED], "previous": []}
    sched.requests[request_id] = session
    sched._update_request_as_session(session, _second_append("basic", high=25, low=20))
    assert session._minicpmo45_carried_window is None

    sched._stash_minicpmo45_windows([request_id])

    assert [unit["length"] for unit in sched._minicpmo45_carried_windows["s"]["units"]] == [5, 6]


def test_a_carry_older_than_its_ttl_is_not_attached(monkeypatch: pytest.MonkeyPatch) -> None:
    sched = _scheduler()
    old_id = _stage0_id("s", 0)
    old = _new_request(old_id)
    old._minicpmo45_window_units = [dict(unit) for unit in CARRIED]
    sched.requests[old_id] = old
    sched._stash_minicpmo45_windows([old_id])

    later = time.monotonic() + OmniARScheduler._MINICPMO45_CARRIED_WINDOW_TTL_S + 1
    monkeypatch.setattr(time, "monotonic", lambda: later)
    new = _new_request(_stage0_id("s", 1))
    sched._attach_minicpmo45_carried_window(new)

    assert getattr(new, "_minicpmo45_carried_window", None) is None
    assert "s" not in sched._minicpmo45_carried_windows
    # The worker keeps its copy longer, so it never misses units the scheduler places.
    assert MiniCPMO45Stage0DuplexRuntime._CARRIED_WINDOW_TTL_S > OmniARScheduler._MINICPMO45_CARRIED_WINDOW_TTL_S


def test_only_duplex_stage0_requests_are_kept_and_attached() -> None:
    sched = _scheduler()
    other = _new_request("chatcmpl-1")
    other._minicpmo45_window_units = [dict(unit) for unit in CARRIED]
    sched.requests["chatcmpl-1"] = other
    sched._stash_minicpmo45_windows(["chatcmpl-1"])
    assert not getattr(sched, "_minicpmo45_carried_windows", None)

    old_id = _stage0_id("s", 0)
    old = _new_request(old_id)
    old._minicpmo45_window_units = [dict(unit) for unit in CARRIED]
    old._minicpmo45_window_open_start = 30
    old.num_computed_tokens = 30
    sched.requests[old_id] = old
    sched._stash_minicpmo45_windows([old_id])

    new = _new_request(_stage0_id("s", 1))
    sched._attach_minicpmo45_carried_window(new)
    assert [unit["length"] for unit in new._minicpmo45_carried_window["units"]] == [5, 6]
    assert "s" not in sched._minicpmo45_carried_windows


def _worker() -> MiniCPMO45Stage0DuplexRuntime:
    helper = MiniCPMO45Stage0DuplexRuntime.__new__(MiniCPMO45Stage0DuplexRuntime)
    helper._carried_window_units = {}
    helper._embed_token = lambda token_id: torch.full((1, 4), float(token_id))
    return helper


def _unit(first: int, length: int) -> _MiniCPMO45WindowUnit:
    return _unit_of(list(range(first, first + length)))


def _unit_of(ids: list[int]) -> _MiniCPMO45WindowUnit:
    return _MiniCPMO45WindowUnit(embeds=[torch.full((1, 4), float(i)) for i in ids], token_ids=list(ids))


def _state(**fields) -> _MiniCPMO45Stage0SessionState:
    state = _MiniCPMO45Stage0SessionState(session_id="s")
    state.window_enabled = True
    state.context_token_ids = [1, 2, 3]
    state.context_embeds = [torch.zeros(1, 4) for _ in range(3)]
    for name, value in fields.items():
        setattr(state, name, value)
    return state


def test_worker_rebuild_matches_the_scheduler_plan() -> None:
    sched = _scheduler()
    session = _new_request(_stage0_id("s", 1))
    session._minicpmo45_carried_window = {"units": [dict(unit) for unit in CARRIED], "previous": []}
    update = _second_append("basic")
    sched._update_request_as_session(session, update)
    plan = update.model_intermediate_buffer["duplex"]["stage0_window"]

    helper = _worker()
    old = _state(window_units=[_unit(100, 5), _unit(200, 6)])
    helper.stash_window_units("s", old)
    new = _state()
    helper.restore_window_units(new)
    # The new request's first unit, closed while building its second append, and that append's pending unit.
    new.window_units = [_unit(300, 9)]
    new.pending_window_unit = _unit(400, 6)

    embeds, token_ids = helper._window_replacement_parts(new, plan)

    assert len(token_ids) == plan["replacement_prompt_len"] == 29
    assert token_ids[:3] == [1, 2, 3]
    assert token_ids[3:8] == list(range(100, 105))
    assert token_ids[8:14] == list(range(200, 206))
    assert token_ids[14:23] == list(range(300, 309))
    assert len(embeds) == len(token_ids)
    assert new.carried_window_units == []
    assert [len(unit.token_ids) for unit in new.window_units] == [5, 6, 9]


@pytest.mark.parametrize(
    ("body", "terminator", "closed_with"),
    [
        (202, CHUNK_EOS, LISTEN),
        (202, CHUNK_TTS_EOS, LISTEN),
        (202, LISTEN, LISTEN),
        (202, TURN_EOS, TURN_EOS),
        # The model had already ended its turn in this unit: its tokens are kept.
        (TURN_EOS, CHUNK_TTS_EOS, CHUNK_TTS_EOS),
    ],
)
def test_worker_ends_the_carried_speech_with_listen(body: int, terminator: int, closed_with: int) -> None:
    helper = _worker()
    helper.listen_token_id, helper.unit_end_token_id = LISTEN, UNIT_END
    helper.chunk_eos_token_id, helper.chunk_tts_eos_token_id = CHUNK_EOS, CHUNK_TTS_EOS
    helper.turn_eos_token_id = TURN_EOS
    spoke = _unit_of([100, 101, CHUNK_EOS, UNIT_END])
    last = _unit_of([200, 201, body, terminator, UNIT_END])
    state = _state(carried_window_units=[spoke, last], window_units=[_unit(300, 9)])
    plan = {"replace": True, "mode": "basic", "drop_units": 0, "carried_units": 2, "replacement_prompt_len": 21}

    embeds, token_ids = helper._window_replacement_parts(state, plan)

    # Only the last unit before the cut is closed; an earlier speech unit is left as it was.
    assert token_ids[3:7] == [100, 101, CHUNK_EOS, UNIT_END]
    assert token_ids[7:12] == [200, 201, body, closed_with, UNIT_END]
    assert len(embeds) == len(token_ids) == 21
    assert float(embeds[10][0, 0]) == float(closed_with)


def test_worker_drops_carried_units_no_plan_places() -> None:
    helper = _worker()
    state = _state(carried_window_units=[_unit(100, 5)], window_units=[_unit(300, 9)])

    assert helper._window_replacement_parts(state, {"completed_token_ids": [40]}) is None

    assert state.carried_window_units == []
    assert [len(unit.token_ids) for unit in state.window_units] == [9]


def test_worker_places_the_newest_units_when_it_kept_more() -> None:
    helper = _worker()
    state = _state(carried_window_units=[_unit(100, 4), _unit(200, 5), _unit(300, 6)], window_units=[_unit(400, 9)])
    plan = {"replace": True, "mode": "basic", "drop_units": 0, "carried_units": 2, "replacement_prompt_len": 23}

    _, token_ids = helper._window_replacement_parts(state, plan)

    assert token_ids[3:8] == list(range(200, 205))
    assert token_ids[8:14] == list(range(300, 306))


def test_worker_refuses_a_plan_that_carries_more_than_it_kept() -> None:
    helper = _worker()
    state = _state(carried_window_units=[_unit(100, 5)])
    plan = {"replace": True, "mode": "basic", "drop_units": 0, "carried_units": 2, "replacement_prompt_len": 8}

    with pytest.raises(RuntimeError, match="carried window history is shorter"):
        helper._window_replacement_parts(state, plan)


def test_worker_keeps_the_stashed_window_in_host_memory() -> None:
    helper = _worker()
    helper.stash_window_units("s", _state(window_units=[_unit(100, 3), _unit(200, 2)]))

    kept = helper._carried_window_units["s"]
    assert kept.embeds.device.type == "cpu"
    assert kept.embeds.shape == (5, 4)
    restored = _state()
    helper.restore_window_units(restored)
    assert [unit.token_ids for unit in restored.carried_window_units] == [[100, 101, 102], [200, 201]]
    assert [float(embed[0, 0]) for embed in restored.carried_window_units[1].embeds] == [200.0, 201.0]


def test_worker_finish_hook_stashes_and_a_stale_stash_is_not_restored(monkeypatch: pytest.MonkeyPatch) -> None:
    helper = _worker()
    helper.sessions = {"s": _state(window_units=[_unit(100, 5)])}
    model = SimpleNamespace(
        _minicpmo45_duplex_request_sessions={_stage0_id("s", 0): "s"},
        _minicpmo45_duplex_data_plane_helper=helper,
        model=SimpleNamespace(),
    )

    MiniCPMO45OmniForConditionalGeneration.on_requests_finished(model, {_stage0_id("s", 0)})

    assert helper.sessions == {}
    assert "s" in helper._carried_window_units
    later = time.monotonic() + MiniCPMO45Stage0DuplexRuntime._CARRIED_WINDOW_TTL_S + 1
    monkeypatch.setattr(time, "monotonic", lambda: later)
    state = _state()
    helper.restore_window_units(state)
    assert state.carried_window_units == []
    assert helper._carried_window_units == {}


def test_worker_stash_keeps_units_not_yet_placed() -> None:
    helper = _worker()
    state = _state(carried_window_units=[_unit(100, 5)], window_units=[_unit(300, 9)])

    helper.stash_window_units("s", state)
    restored = _state()
    helper.restore_window_units(restored)

    assert [unit.token_ids[0] for unit in restored.carried_window_units] == [100, 300]
    assert helper._carried_window_units == {}


def test_worker_without_a_window_keeps_nothing() -> None:
    helper = _worker()
    helper.stash_window_units("s", _state(window_enabled=False, window_units=[_unit(100, 5)]))
    assert helper._carried_window_units == {}

    helper._carried_window_units["s"] = _MiniCPMO45CarriedWindow.pack([_unit(100, 5)], time.monotonic())
    state = _state(window_enabled=False)
    helper.restore_window_units(state)
    assert state.carried_window_units == []


def test_answer_decision_keeps_listen_only_when_it_is_the_most_likely_token() -> None:
    listen_id = 3
    logits = torch.tensor(
        [
            [0.0, 1.0, 6.0, 5.0, 0.0],  # another token leads: listen is masked
            [0.0, 1.0, 6.0, 7.0, 0.0],  # listen leads: unchanged
        ]
    )

    MiniCPMO45OmniForConditionalGeneration._minicpmo45_answer_unless_listen_wins(logits, 0, listen_id)
    MiniCPMO45OmniForConditionalGeneration._minicpmo45_answer_unless_listen_wins(logits, 1, listen_id)

    assert logits[0, listen_id] == float("-inf")
    assert logits[0, 2] == 6.0
    assert logits[1].tolist() == [0.0, 1.0, 6.0, 7.0, 0.0]


def test_placing_carried_units_marks_the_state() -> None:
    helper = _worker()
    state = _state(carried_window_units=[_unit(100, 5)], window_units=[_unit(300, 9)])
    plan = {"replace": True, "mode": "basic", "drop_units": 0, "carried_units": 1, "replacement_prompt_len": 17}

    helper._window_replacement_parts(state, plan)

    assert state.carried_context is True
    fresh = _state(window_units=[_unit(300, 9)])
    helper._window_replacement_parts(
        fresh, {"replace": True, "mode": "basic", "drop_units": 0, "replacement_prompt_len": 12}
    )
    assert fresh.carried_context is False


def _upstream(cls: type, name: str) -> type:
    """The class whose ``name`` an ``OmniARScheduler`` method reaches through ``super()``."""
    return next(base for base in cls.__mro__[1:] if name in base.__dict__)


@pytest.mark.parametrize(
    ("status", "carried"), [(RequestStatus.FINISHED_ABORTED, 1), (RequestStatus.FINISHED_STOPPED, 0)]
)
def test_scheduler_round_trip_carries_only_an_aborted_window(
    monkeypatch: pytest.MonkeyPatch, status: RequestStatus, carried: int
) -> None:
    """append -> booked -> finish -> the session's next request -> its second append, through the public methods."""
    monkeypatch.setattr(
        _upstream(OmniARScheduler, "add_request"),
        "add_request",
        lambda self, request: self.requests.__setitem__(request.request_id, request),
    )
    monkeypatch.setattr(_upstream(OmniARScheduler, "finish_requests"), "finish_requests", lambda self, ids, st: [])
    monkeypatch.setattr(
        _upstream(OmniARScheduler, "_update_after_schedule"), "_update_after_schedule", lambda self, out: None
    )
    sched = _scheduler()
    sched._enforce_kstep_span_uniformity = lambda out: None
    sched._clear_kv_wait_starts = lambda ids: None

    old_id = _stage0_id("s", 0)
    old = Request(
        request_id=old_id,
        prompt_token_ids=[0] * 20,
        sampling_params=SamplingParams(max_tokens=8),
        pooling_params=None,
        arrival_time=100.0,
        block_hasher=None,
    )
    old.append_output_token_ids([40])
    old.num_computed_tokens = 21
    old.status = RequestStatus.WAITING_FOR_STREAMING_REQ
    old._minicpmo45_window_units = [dict(unit) for unit in CARRIED]
    old._minicpmo45_window_open_start = 14
    sched.add_request(old)
    sched._update_request_as_session(old, _second_append("basic", high=25, low=20))
    sched._update_after_schedule(SimpleNamespace(num_scheduled_tokens={old_id: 8}))
    sched.finish_requests([old_id], status)
    sched.requests.pop(old_id)

    new = _new_request(_stage0_id("s", 1))
    sched.add_request(new)
    update = _second_append("basic")
    sched._update_request_as_session(new, update)

    plan = update.model_intermediate_buffer["duplex"].get("stage0_window") or {}
    assert plan.get("carried_units", 0) == carried


def test_answer_rule_applies_only_to_a_request_that_carries_units() -> None:
    listen_id, turn_eos_id = 3, 9
    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    model.model_stage = "llm"
    model._minicpmo45_native_duplex_token_ids_cache = {"listen_token_id": listen_id, "turn_eos_token_id": turn_eos_id}

    def state(carried: bool) -> SimpleNamespace:
        return SimpleNamespace(current_turn_ended=True, pending_speech_context=True, carried_context=carried)

    model._minicpmo45_duplex_data_plane_helper = SimpleNamespace(sessions={"a": state(True), "b": state(False)})
    rows = tuple(
        SimpleNamespace(
            row_idx=i,
            request_id=f"req-{sid}",
            session_id=sid,
            seq=3,
            payload={"is_speech": False},
            temperature=None,
            top_k=None,
            top_p=None,
            max_tokens=None,
        )
        for i, sid in enumerate(("a", "b"))
    )
    # <|speak|> (id 2) leads <|listen|> on both rows.
    logits = torch.tensor([[0.0, 1.0, 6.0, 5.0, 0.0] + [0.0] * 5] * 2)

    model.prepare_duplex_sampling(logits, None, rows)

    assert logits[0, listen_id] == float("-inf")
    assert logits[1, listen_id] == 5.0


def test_an_abort_with_nothing_to_keep_drops_the_sessions_earlier_window() -> None:
    """The worker drops a session's kept units on every finish, so the scheduler does too."""
    sched = _scheduler()
    old_id = _stage0_id("s", 0)
    old = _new_request(old_id)
    old._minicpmo45_window_units = [dict(unit) for unit in CARRIED]
    sched.requests[old_id] = old
    sched._stash_minicpmo45_windows([old_id])
    assert "s" in sched._minicpmo45_carried_windows

    empty_id = _stage0_id("s", 1)
    sched.requests[empty_id] = _new_request(empty_id)
    sched._stash_minicpmo45_windows([empty_id])

    assert "s" not in sched._minicpmo45_carried_windows


def test_a_first_plan_that_places_nothing_drops_the_carry_on_both_sides() -> None:
    sched = _scheduler()
    session = _new_request(_stage0_id("s", 1))
    session._minicpmo45_carried_window = {"units": [dict(unit) for unit in CARRIED], "previous": []}
    # The open unit starts at the closing boundary, so this plan records no unit.
    session._minicpmo45_window_open_start = 12
    update = _second_append("basic")

    sched._update_request_as_session(session, update)

    plan = update.model_intermediate_buffer["duplex"]["stage0_window"]
    assert "replace" not in plan
    assert "carried_units" not in plan
    assert session._minicpmo45_carried_window is None

    helper = _worker()
    state = _state(carried_window_units=[_unit(100, 5), _unit(200, 6)])
    assert helper._window_replacement_parts(state, plan) is None
    assert state.carried_window_units == []


def test_two_sessions_keep_their_own_windows() -> None:
    sched = _scheduler()
    for name, units in (("a", CARRIED[:1]), ("b", CARRIED)):
        request_id = _stage0_id(name, 0)
        request = _new_request(request_id)
        request._minicpmo45_window_units = [dict(unit) for unit in units]
        sched.requests[request_id] = request
    sched._stash_minicpmo45_windows([_stage0_id("a", 0), _stage0_id("b", 0)])

    new_b = _new_request(_stage0_id("b", 1))
    sched._attach_minicpmo45_carried_window(new_b)
    assert [unit["length"] for unit in new_b._minicpmo45_carried_window["units"]] == [5, 6]
    assert list(sched._minicpmo45_carried_windows) == ["a"]

    helper = _worker()
    helper.stash_window_units("a", _state(session_id="a", window_units=[_unit(100, 5)]))
    helper.stash_window_units("b", _state(session_id="b", window_units=[_unit(200, 5), _unit(300, 6)]))
    restored = _state(session_id="b")
    helper.restore_window_units(restored)
    assert [unit.token_ids[0] for unit in restored.carried_window_units] == [200, 300]
    assert list(helper._carried_window_units) == ["a"]
