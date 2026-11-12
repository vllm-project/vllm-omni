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
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.window_kv import (
    MiniCPMO45CarriedWindows,
)
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.window_kv import (
    MiniCPMO45DuplexSchedulerHelper as Helper,
)
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.window_plan import (
    CARRIED_WINDOW_MAX_SESSIONS,
    CARRIED_WINDOW_OFFER_TTL_S,
)
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import MiniCPMO45OmniForConditionalGeneration

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

MARKER = [70, 71]
LISTEN, UNIT_END, TURN_EOS, CHUNK_EOS, CHUNK_TTS_EOS = 151705, 151684, 151717, 151718, 151721


def _stage0_id(session_id: str, epoch: int) -> str:
    encoded = base64.urlsafe_b64encode(session_id.encode()).decode().rstrip("=")
    return f"duplex-s.{encoded}.e.{epoch}.r.stage0"


def _carry(sched: OmniARScheduler) -> MiniCPMO45CarriedWindows:
    if getattr(sched, "_minicpmo45_carry", None) is None:
        sched._minicpmo45_carry = MiniCPMO45CarriedWindows()
    return sched._minicpmo45_carry


def _kept(sched: OmniARScheduler) -> dict:
    return _carry(sched).windows


def _finish(sched: OmniARScheduler, request_ids: list[str], *, aborted: bool = True) -> None:
    """Requests finish, then the next step hands their finish to the worker (``schedule`` -> ``on_schedule``)."""
    _carry(sched)
    Helper.release_windows(sched, request_ids, aborted=aborted)
    Helper.on_schedule(sched, SimpleNamespace(finished_req_ids=set(request_ids), scheduled_new_reqs=[]))


def _start(sched: OmniARScheduler, request: Request) -> None:
    """A session's new Stage-0 request gets its first scheduled step."""
    sched.requests[request.request_id] = request
    _carry(sched)
    Helper.on_schedule(
        sched, SimpleNamespace(finished_req_ids=set(), scheduled_new_reqs=[SimpleNamespace(req_id=request.request_id)])
    )


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
    request._minicpmo45_started = True  # it has had its first scheduled step
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

    # The segment boundary still goes to the worker without a window. The worker
    # drops its carried units at that plan, so the scheduler must count it as one.
    window = update.model_intermediate_buffer["duplex"]["stage0_window"]
    assert window == {"completed_token_ids": [40]}
    assert session._minicpmo45_carried_window is None
    assert session._minicpmo45_window_planned is True


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
    session._minicpmo45_started = True  # it has had its first scheduled step
    session._minicpmo45_window_units = [dict(unit) for unit in CARRIED]
    session._minicpmo45_window_open_start = 14
    sched.requests[request_id] = session
    update = _second_append("basic", high=25, low=20)
    sched._update_request_as_session(session, update)
    assert update.model_intermediate_buffer["duplex"]["stage0_window"]["drop_units"] == 2
    if scheduled:
        Helper.commit_window_snapshots(sched, SimpleNamespace(num_scheduled_tokens={request_id: 8}))

    _finish(sched, [request_id])

    stashed = _kept(sched)["s"]
    assert [unit["length"] for unit in stashed["units"]] == kept
    assert request_id not in _carry(sched).snapshot_ids


def test_a_second_append_before_booking_keeps_the_first_snapshot() -> None:
    sched = _scheduler()
    request_id = _stage0_id("s", 0)
    session = _new_request(request_id)
    session._minicpmo45_window_units = [dict(unit) for unit in CARRIED]
    sched.requests[request_id] = session

    Helper.snapshot_window(sched, session)
    session._minicpmo45_window_units = [{"length": 9, "generated_token_ids": []}]
    Helper.snapshot_window(sched, session)
    _finish(sched, [request_id])

    assert [unit["length"] for unit in _kept(sched)["s"]["units"]] == [5, 6]


def test_abort_before_the_first_plan_of_an_empty_window_keeps_nothing() -> None:
    """Epoch 0's second append records its first unit; aborted before it is booked, the worker has no closed unit."""
    sched = _scheduler()
    request_id = _stage0_id("s", 0)
    session = _new_request(request_id)
    sched.requests[request_id] = session
    sched._update_request_as_session(session, _second_append("basic"))
    assert session._minicpmo45_window_units

    _finish(sched, [request_id])

    assert "s" not in _kept(sched)


def test_restash_keeps_the_time_of_the_first_abort(monkeypatch: pytest.MonkeyPatch) -> None:
    sched = _scheduler()
    old_id = _stage0_id("s", 0)
    old = _new_request(old_id)
    old._minicpmo45_window_units = [dict(unit) for unit in CARRIED]
    sched.requests[old_id] = old
    _finish(sched, [old_id])
    first_time = _kept(sched)["s"]["time"]

    new_id = _stage0_id("s", 1)
    new = _new_request(new_id)
    _start(sched, new)
    sched.requests[new_id] = new
    later = first_time + 5
    monkeypatch.setattr(time, "monotonic", lambda: later)
    _finish(sched, [new_id])

    assert _kept(sched)["s"]["time"] == first_time


def test_abort_before_the_first_plan_passes_the_carried_window_on() -> None:
    sched = _scheduler()
    request_id = _stage0_id("s", 1)
    request = _new_request(request_id)
    request._minicpmo45_carried_window = {"units": [dict(unit) for unit in CARRIED], "previous": [33]}
    sched.requests[request_id] = request

    _finish(sched, [request_id])

    stashed = _kept(sched)["s"]
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

    _finish(sched, [request_id])

    assert [unit["length"] for unit in _kept(sched)["s"]["units"]] == [5, 6]


def test_a_carry_older_than_its_ttl_is_not_attached(monkeypatch: pytest.MonkeyPatch) -> None:
    sched = _scheduler()
    old_id = _stage0_id("s", 0)
    old = _new_request(old_id)
    old._minicpmo45_window_units = [dict(unit) for unit in CARRIED]
    sched.requests[old_id] = old
    _finish(sched, [old_id])

    later = time.monotonic() + CARRIED_WINDOW_OFFER_TTL_S + 1
    monkeypatch.setattr(time, "monotonic", lambda: later)
    new = _new_request(_stage0_id("s", 1))
    _start(sched, new)

    assert getattr(new, "_minicpmo45_carried_window", None) is None
    assert "s" not in _kept(sched)


def test_only_duplex_stage0_requests_are_kept_and_attached() -> None:
    sched = _scheduler()
    other = _new_request("chatcmpl-1")
    other._minicpmo45_window_units = [dict(unit) for unit in CARRIED]
    sched.requests["chatcmpl-1"] = other
    _finish(sched, ["chatcmpl-1"])
    assert not _kept(sched)

    old_id = _stage0_id("s", 0)
    old = _new_request(old_id)
    old._minicpmo45_window_units = [dict(unit) for unit in CARRIED]
    old._minicpmo45_window_open_start = 30
    old.num_computed_tokens = 30
    sched.requests[old_id] = old
    _finish(sched, [old_id])

    new = _new_request(_stage0_id("s", 1))
    _start(sched, new)
    assert [unit["length"] for unit in new._minicpmo45_carried_window["units"]] == [5, 6]
    assert "s" not in _kept(sched)


def _worker() -> MiniCPMO45Stage0DuplexRuntime:
    helper = MiniCPMO45Stage0DuplexRuntime.__new__(MiniCPMO45Stage0DuplexRuntime)
    helper._carried_window_units = {}
    helper._handed_window_units = {}
    helper._embed_token = lambda token_id: torch.full((1, 4), float(token_id))
    helper.listen_token_id = LISTEN
    return helper


def _keep(helper: MiniCPMO45Stage0DuplexRuntime, session_id: str, state, request_id: str = "old") -> None:
    """The scheduler's ``keep`` for a finished request whose state is ``state``."""
    helper.apply_carry_ops([("keep", session_id, request_id)], {request_id: state}.get, lambda sid: None)


def _hand(helper: MiniCPMO45Stage0DuplexRuntime, session_id: str, live=None) -> None:
    """The scheduler's ``hand`` to the session's new request (``live``: a state the session already has)."""
    helper.apply_carry_ops([("hand", session_id, "new")], lambda rid: None, lambda sid: live)


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
    _keep(helper, "s", old)
    _hand(helper, "s")
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


@pytest.mark.parametrize("kept", [[], [5], [4, 6]])
def test_worker_fills_carried_units_it_does_not_have_instead_of_failing(kept: list[int]) -> None:
    """Fewer or different units than the plan: <|listen|> fillers keep the planned length; the step goes on."""
    helper = _worker()
    state = _state(carried_window_units=[_unit(100 + 10 * k, n) for k, n in enumerate(kept)])
    plan = {
        "replace": True,
        "mode": "basic",
        "drop_units": 0,
        "carried_units": 2,
        "carried_unit_lengths": [5, 6],
        "replacement_prompt_len": 14,
    }

    _, token_ids = helper._window_replacement_parts(state, plan)

    assert len(token_ids) == 14
    assert token_ids[3:14] == [LISTEN] * 11
    assert state.carried_context is False


def test_worker_keeps_the_stashed_window_in_host_memory() -> None:
    helper = _worker()
    _keep(helper, "s", _state(window_units=[_unit(100, 3), _unit(200, 2)]))

    kept = helper._carried_window_units["s"]
    assert kept.embeds.device.type == "cpu"
    assert kept.embeds.shape == (5, 4)
    _hand(helper, "s")
    restored = _state()
    helper.restore_window_units(restored)
    assert [unit.token_ids for unit in restored.carried_window_units] == [[100, 101, 102], [200, 201]]
    assert [float(embed[0, 0]) for embed in restored.carried_window_units[1].embeds] == [200.0, 201.0]


def test_model_follows_the_schedulers_decisions_before_freeing_the_state() -> None:
    """``on_scheduler_output`` keeps the finished request's units before ``on_requests_finished`` frees its state."""
    helper = _worker()
    helper.sessions = {"s": _state(window_units=[_unit(100, 5)]), "t": _state(session_id="t")}
    old_id, other_id = _stage0_id("s", 0), _stage0_id("t", 0)
    model = SimpleNamespace(
        _minicpmo45_duplex_request_sessions={old_id: "s", other_id: "t"},
        _minicpmo45_duplex_data_plane_helper=helper,
        model=SimpleNamespace(),
    )
    finished = {old_id, other_id}
    output = SimpleNamespace(finished_req_ids=finished, duplex_carry_ops=[("keep", "s", old_id)])

    MiniCPMO45OmniForConditionalGeneration.on_scheduler_output(model, output)
    MiniCPMO45OmniForConditionalGeneration.on_requests_finished(model, finished)

    assert helper.sessions == {}
    assert list(helper._carried_window_units) == ["s"]  # only what the scheduler kept
    new_id = _stage0_id("s", 1)
    MiniCPMO45OmniForConditionalGeneration.on_scheduler_output(
        model, SimpleNamespace(finished_req_ids=set(), duplex_carry_ops=[("hand", "s", new_id)])
    )
    state = _state()
    helper.restore_window_units(state)
    assert [unit.token_ids[0] for unit in state.carried_window_units] == [100]
    assert helper._carried_window_units == {} and helper._handed_window_units == {}


def test_a_hand_to_a_session_that_still_has_a_live_state_goes_to_that_state() -> None:
    helper = _worker()
    _keep(helper, "s", _state(window_units=[_unit(100, 5)]))
    live = _state()

    _hand(helper, "s", live=live)

    assert [unit.token_ids[0] for unit in live.carried_window_units] == [100]
    assert helper._handed_window_units == {}


def test_worker_drops_what_the_scheduler_drops() -> None:
    helper = _worker()
    _keep(helper, "s", _state(window_units=[_unit(100, 5)]))

    helper.apply_carry_ops([("drop", "s", "")], lambda rid: None, lambda sid: None)

    assert helper._carried_window_units == {}


def test_worker_stash_keeps_units_not_yet_placed() -> None:
    helper = _worker()
    state = _state(carried_window_units=[_unit(100, 5)], window_units=[_unit(300, 9)])

    _keep(helper, "s", state)
    _hand(helper, "s")
    restored = _state()
    helper.restore_window_units(restored)

    assert [unit.token_ids[0] for unit in restored.carried_window_units] == [100, 300]
    assert helper._carried_window_units == {}


def test_worker_without_a_window_keeps_nothing() -> None:
    helper = _worker()
    _keep(helper, "s", _state(window_enabled=False, window_units=[_unit(100, 5)]))
    _hand(helper, "s")
    state = _state()
    helper.restore_window_units(state)
    assert state.carried_window_units == []

    helper._handed_window_units["s"] = _MiniCPMO45CarriedWindow.pack([_unit(100, 5)], time.monotonic())
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
    old._minicpmo45_started = True  # it has had its first scheduled step
    old._minicpmo45_window_units = [dict(unit) for unit in CARRIED]
    old._minicpmo45_window_open_start = 14
    sched.add_request(old)
    sched._update_request_as_session(old, _second_append("basic", high=25, low=20))
    sched._update_after_schedule(SimpleNamespace(num_scheduled_tokens={old_id: 8}))
    sched.finish_requests([old_id], status)
    sched.requests.pop(old_id)

    new = _new_request(_stage0_id("s", 1))
    sched.add_request(new)
    # The step that hands the finish to the worker and schedules the new request (OmniARScheduler.schedule).
    Helper.on_schedule(
        sched, SimpleNamespace(finished_req_ids={old_id}, scheduled_new_reqs=[SimpleNamespace(req_id=new.request_id)])
    )
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
    """A later finish with nothing to keep replaces the session's earlier window; the worker is told to drop it."""
    sched = _scheduler()
    old_id = _stage0_id("s", 0)
    old = _new_request(old_id)
    old._minicpmo45_window_units = [dict(unit) for unit in CARRIED]
    sched.requests[old_id] = old
    _finish(sched, [old_id])
    assert "s" in _kept(sched)

    empty_id = _stage0_id("s", 1)
    sched.requests[empty_id] = _new_request(empty_id)
    _finish(sched, [empty_id])

    assert "s" not in _kept(sched)


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
    _finish(sched, [_stage0_id("a", 0), _stage0_id("b", 0)])

    new_b = _new_request(_stage0_id("b", 1))
    _start(sched, new_b)
    assert [unit["length"] for unit in new_b._minicpmo45_carried_window["units"]] == [5, 6]
    assert list(_kept(sched)) == ["a"]

    helper = _worker()
    _keep(helper, "a", _state(session_id="a", window_units=[_unit(100, 5)]))
    _keep(helper, "b", _state(session_id="b", window_units=[_unit(200, 5), _unit(300, 6)]))
    _hand(helper, "b")
    restored = _state(session_id="b")
    helper.restore_window_units(restored)
    assert [unit.token_ids[0] for unit in restored.carried_window_units] == [200, 300]
    assert list(helper._carried_window_units) == ["a"]


def test_the_scheduler_keeps_at_most_the_session_cap_and_the_worker_follows() -> None:
    names = [f"s{k}" for k in range(CARRIED_WINDOW_MAX_SESSIONS + 1)]
    # s0 is kept again just before the last one, so it is the newest and s1 is the one dropped.
    order = [*names[:-1], names[0], names[-1]]

    sched = _scheduler()
    helper = _worker()
    for epoch, name in enumerate(order):
        request_id = _stage0_id(name, epoch)
        request = _new_request(request_id)
        request._minicpmo45_window_units = [dict(unit) for unit in CARRIED[:1]]
        sched.requests[request_id] = request
        _finish(sched, [request_id])
        worker_state = _state(session_id=name, window_units=[_unit(100, 5)])
        helper.apply_carry_ops(Helper.take_carry_ops(sched), {request_id: worker_state}.get, lambda sid: None)

    expected = [*names[2:-1], names[0], names[-1]]
    assert list(_kept(sched)) == expected
    assert list(helper._carried_window_units) == expected


def test_scheduler_hooks_stay_out_until_a_duplex_stage0_append_was_planned(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a duplex Stage-0 request the scheduler hooks never call into the MiniCPM-o helper."""

    def fail(*args, **kwargs):
        raise AssertionError("MiniCPM-o carry helper called")

    for name in ("on_schedule", "release_windows", "commit_window_snapshots"):
        monkeypatch.setattr(Helper, name, classmethod(fail))
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

    request = _new_request("chatcmpl-1")
    sched.add_request(request)
    sched._update_after_schedule(SimpleNamespace(num_scheduled_tokens={"chatcmpl-1": 1}))
    sched.finish_requests(["chatcmpl-1"], RequestStatus.FINISHED_ABORTED)


def test_an_abort_and_the_next_request_in_one_step_hand_the_window_over() -> None:
    """The worker adds finished requests' entries before it starts the step's new requests; so does the scheduler."""
    sched = _scheduler()
    old_id, new_id = _stage0_id("s", 0), _stage0_id("s", 1)
    old = _new_request(old_id)
    old._minicpmo45_window_units = [dict(unit) for unit in CARRIED]
    sched.requests[old_id] = old
    _carry(sched)
    Helper.release_windows(sched, [old_id], aborted=True)
    new = _new_request(new_id)
    sched.requests[new_id] = new

    Helper.on_schedule(
        sched, SimpleNamespace(finished_req_ids={old_id}, scheduled_new_reqs=[SimpleNamespace(req_id=new_id)])
    )

    assert [unit["length"] for unit in new._minicpmo45_carried_window["units"]] == [5, 6]
    assert "s" not in _kept(sched)


def test_a_request_whose_first_plan_already_went_out_gets_no_carry() -> None:
    """The worker drops its carried units at the first window plan; a carry handed over after it could not be placed."""
    sched = _scheduler()
    old_id = _stage0_id("s", 0)
    old = _new_request(old_id)
    old._minicpmo45_window_units = [dict(unit) for unit in CARRIED]
    sched.requests[old_id] = old
    _finish(sched, [old_id])
    new = _new_request(_stage0_id("s", 1))
    new._minicpmo45_window_planned = True

    _start(sched, new)

    assert getattr(new, "_minicpmo45_carried_window", None) is None
    assert "s" not in _kept(sched)


@pytest.mark.parametrize("seed", range(5))
def test_the_worker_restores_exactly_what_the_scheduler_hands_over(monkeypatch: pytest.MonkeyPatch, seed: int) -> None:
    """Both sides through the same random steps: aborts, normal finishes, requests aborted before they ever ran,
    new requests, a moving clock and a cap of 3. The worker applies the scheduler's ops; every request the
    scheduler hands a window gets exactly those units on the worker, and no other request gets any."""
    import random

    import vllm_omni.model_executor.models.minicpmo_4_5.duplex.window_kv as window_kv

    monkeypatch.setattr(window_kv, "CARRIED_WINDOW_MAX_SESSIONS", 3)
    rng = random.Random(seed)
    clock = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    sched = _scheduler()
    _carry(sched)
    helper = _worker()
    helper.sessions = {}
    sessions = [f"s{k}" for k in range(8)]
    epoch = dict.fromkeys(sessions, -1)
    running: dict[str, Request] = {}
    session_of: dict[str, str] = {}
    handed = 0
    for _ in range(400):
        clock[0] += rng.choice([0.0, 0.5, 2.0, 7.0])
        finishing = [s for s in sessions if s in running and rng.random() < 0.3]
        starting = [s for s in sessions if s not in running and s not in finishing and rng.random() < 0.3]
        never_ran = {s for s in starting if rng.random() < 0.2}
        aborted, normal = [], []
        for s in finishing:
            (aborted if rng.random() < 0.7 else normal).append(running[s].request_id)
        new_reqs = []
        for s in starting:
            epoch[s] += 1
            req = _new_request(_stage0_id(s, epoch[s]))
            sched.requests[req.request_id] = req
            session_of[req.request_id] = s
            if s in never_ran:
                req._minicpmo45_started = False
                aborted.append(req.request_id)  # added and aborted before it was ever scheduled
            else:
                new_reqs.append(req)
        finished_ids = set(aborted) | set(normal)
        Helper.release_windows(sched, aborted, aborted=True)
        Helper.release_windows(sched, normal, aborted=False)
        Helper.on_schedule(
            sched,
            SimpleNamespace(
                finished_req_ids=finished_ids,
                scheduled_new_reqs=[SimpleNamespace(req_id=r.request_id) for r in new_reqs],
            ),
        )
        ops = Helper.take_carry_ops(sched)
        # The worker's step: the scheduler's ops while the finished requests' state still exists, then the frees,
        # then the new requests' first prefill.
        states = {rid: helper.sessions.get(session_of[rid]) for rid in finished_ids}
        helper.apply_carry_ops(ops, states.get, lambda sid: None)
        for request_id in finished_ids:
            helper.sessions.pop(session_of[request_id], None)
            running.pop(session_of[request_id], None)
            sched.requests.pop(request_id, None)
        for req in new_reqs:
            s = session_of[req.request_id]
            state = _state(session_id=s)
            helper.restore_window_units(state)
            handed_window = getattr(req, "_minicpmo45_carried_window", None)
            if handed_window is not None:
                handed += 1
                assert [len(unit.token_ids) for unit in state.carried_window_units] == [
                    unit["length"] for unit in handed_window["units"]
                ]
            else:
                assert state.carried_window_units == []
            # This request closes 0-2 units of its own before it finishes; both sides see the same ones.
            n = rng.randint(0, 2)
            req._minicpmo45_window_units = [{"length": 5, "generated_token_ids": []} for _ in range(n)]
            state.window_units = [_unit(100 + 10 * k, 5) for k in range(n)]
            helper.sessions[s] = state
            running[s] = req
        assert set(helper._carried_window_units) == set(_kept(sched))
    assert handed > 0


def test_finishes_of_one_session_are_ordered_by_epoch_number() -> None:
    """Epochs 9 and 10 finish in one step: the later epoch's window is the one kept, as on the worker."""
    sched = _scheduler()
    for ep, units in ((9, CARRIED[:1]), (10, CARRIED)):
        request_id = _stage0_id("s", ep)
        request = _new_request(request_id)
        request._minicpmo45_window_units = [dict(unit) for unit in units]
        sched.requests[request_id] = request

    _finish(sched, [_stage0_id("s", 10), _stage0_id("s", 9)])

    assert [unit["length"] for unit in _kept(sched)["s"]["units"]] == [5, 6]
    assert [op[2] for op in Helper.take_carry_ops(sched) if op[0] == "keep"] == [
        _stage0_id("s", 9),
        _stage0_id("s", 10),
    ]


def test_an_abort_of_a_request_that_already_finished_keeps_nothing() -> None:
    """Upstream skips such a request, so its finish never reaches the worker again; nothing may wait for it."""
    sched = _scheduler()
    request_id = _stage0_id("s", 0)
    request = _new_request(request_id)
    request._minicpmo45_window_units = [dict(unit) for unit in CARRIED]
    request.status = RequestStatus.FINISHED_STOPPED
    sched.requests[request_id] = request

    Helper.release_windows(_carry(sched) and sched, [request_id], aborted=True)

    assert _carry(sched).pending == {}


def test_a_request_aborted_before_its_first_step_leaves_the_sessions_window_for_the_next_one() -> None:
    """A second barge-in while the replacement request still waits: the window goes on to the request after it."""
    sched = _scheduler()
    old_id = _stage0_id("s", 0)
    old = _new_request(old_id)
    old._minicpmo45_window_units = [dict(unit) for unit in CARRIED]
    sched.requests[old_id] = old
    _finish(sched, [old_id])
    queued = _new_request(_stage0_id("s", 1))
    queued._minicpmo45_started = False
    sched.requests[queued.request_id] = queued

    _finish(sched, [queued.request_id])
    newer = _new_request(_stage0_id("s", 2))
    _start(sched, newer)

    assert [unit["length"] for unit in newer._minicpmo45_carried_window["units"]] == [5, 6]


def test_units_handed_to_a_request_that_never_got_its_state_go_on_with_its_abort() -> None:
    helper = _worker()
    _keep(helper, "s", _state(window_units=[_unit(100, 5)]), request_id="w")
    helper.apply_carry_ops([("hand", "s", "x")], lambda rid: None, lambda sid: None)

    # x is aborted before its state was created; the scheduler keeps the units it carried.
    helper.apply_carry_ops([("keep", "s", "x")], lambda rid: None, lambda sid: None)
    helper.apply_carry_ops([("hand", "s", "z")], lambda rid: None, lambda sid: None)
    state = _state()
    helper.restore_window_units(state)

    assert [unit.token_ids[0] for unit in state.carried_window_units] == [100]


def test_units_handed_to_a_request_that_finished_without_a_state_are_freed() -> None:
    helper = _worker()
    _keep(helper, "s", _state(window_units=[_unit(100, 5)]))
    helper.apply_carry_ops([("hand", "s", "x")], lambda rid: None, lambda sid: None)

    helper.forget_handed_window_units("s", "other")
    assert "s" in helper._handed_window_units
    helper.forget_handed_window_units("s", "x")
    assert helper._handed_window_units == {}


def test_the_real_schedule_hands_the_window_and_sends_the_decision_with_the_step() -> None:
    """Through OmniARScheduler.schedule() with the real KV allocator: the hand happens in the request's first
    scheduled step and the op travels in that step's output."""
    from vllm.config import CacheConfig, SchedulerConfig, VllmConfig
    from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
    from vllm.v1.structured_output import StructuredOutputManager

    config = VllmConfig()
    config.model_config = SimpleNamespace(
        multimodal_config=None,
        uses_mrope=False,
        is_encoder_decoder=False,
        max_model_len=64,
        runner_type="generate",
        is_diffusion=False,
        supports_multimodal_inputs=False,
        return_sampling_mask=False,
        recompute_preemption="fail",
    )
    config.scheduler_config = SchedulerConfig(
        max_model_len=64, max_num_seqs=2, max_num_batched_tokens=64, is_encoder_decoder=False
    )
    config.cache_config = CacheConfig(block_size=16, enable_prefix_caching=False)
    config.cache_config.num_gpu_blocks = 4
    register_all_kvcache_specs(config)
    kv_config = KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"], FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=8, dtype=torch.float16)
            )
        ],
    )
    sched = OmniARScheduler(config, kv_config, MagicMock(spec=StructuredOutputManager), 16)
    _carry(sched).windows["s"] = {"units": [dict(unit) for unit in CARRIED], "previous": [], "time": time.monotonic()}
    request = Request(
        request_id=_stage0_id("s", 1),
        prompt_token_ids=[1] * 8,
        sampling_params=SamplingParams(max_tokens=4),
        pooling_params=None,
        arrival_time=0.0,
        block_hasher=None,
    )
    sched.add_request(request)
    assert getattr(request, "_minicpmo45_carried_window", None) is None  # not at add_request any more

    output = sched.schedule()

    assert output.duplex_carry_ops == [("hand", "s", request.request_id)]
    assert [unit["length"] for unit in request._minicpmo45_carried_window["units"]] == [5, 6]
    assert sched.schedule().duplex_carry_ops == []


def test_a_keep_this_side_cannot_fill_stores_nothing_and_says_so(caplog: pytest.LogCaptureFixture) -> None:
    """The scheduler keeps only windows with units; a keep the worker has nothing for is the disagreement itself."""
    helper = _worker()
    _keep(helper, "s", _state(window_units=[_unit(100, 5)]), request_id="w")

    with caplog.at_level("WARNING"):
        helper.apply_carry_ops([("keep", "s", "x")], lambda rid: None, lambda sid: None)

    assert helper._carried_window_units == {}
    assert "kept request x for session s" in caplog.text
