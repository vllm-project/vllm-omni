# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Scheduler side of the async-chunk prewarm hand-off.

``add_request`` pops the flat ``f"{ASYNC_CHUNK_PREWARM_NS}.<name>"`` keys off a
new request; ``_wrap_omni_scheduler_output`` hands each payload to the runner
once, only for ids that are still live. A malformed payload must never raise.
"""

from __future__ import annotations

from collections import defaultdict

import msgspec
import pytest
import torch
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import RequestStatus

import vllm_omni.core.sched.omni_generation_scheduler as gen_sched_mod
from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler
from vllm_omni.core.sched.omni_generation_scheduler import OmniGenerationScheduler
from vllm_omni.core.sched.output import OmniSchedulerOutput
from vllm_omni.data_entry_keys import ASYNC_CHUNK_PREWARM_NS
from vllm_omni.engine import AdditionalInformationEntry, AdditionalInformationPayload
from vllm_omni.engine.serialization import deserialize_additional_information, serialize_additional_information

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_REF_AUDIO_KEY = f"{ASYNC_CHUNK_PREWARM_NS}.ref_audio"
_REF_AUDIO_SR_KEY = f"{ASYNC_CHUNK_PREWARM_NS}.ref_audio_sr"

_SCHEDULERS = pytest.mark.parametrize(
    "scheduler_cls",
    [pytest.param(OmniGenerationScheduler, id="generation"), pytest.param(OmniARScheduler, id="ar")],
)


class _StubRequest:
    """The Request surface add_request, finish_requests and error outputs touch."""

    def __init__(self, request_id: str, additional_information=None, status=RequestStatus.WAITING) -> None:
        self.request_id = request_id
        self.additional_information = additional_information
        self.status = status
        self.resumable = False
        self.client_index = 0
        self.stop_reason = None
        self.trace_headers = None

    def is_finished(self) -> bool:
        return RequestStatus.is_finished(self.status)

    def get_finished_reason(self):
        return RequestStatus.get_finished_reason(self.status)

    def take_events(self):
        return None


@pytest.fixture
def base_scheduler(monkeypatch: pytest.MonkeyPatch) -> list[_StubRequest]:
    """Stub the upstream Scheduler add/finish; returns what upstream add_request received."""
    added: list[_StubRequest] = []

    def fake_add_request(self, request) -> None:
        added.append(request)
        self.requests[request.request_id] = request

    def fake_finish_requests(self, request_ids, finished_status):
        ids = [request_ids] if isinstance(request_ids, str) else list(request_ids)
        finished = [self.requests.pop(request_id) for request_id in ids if request_id in self.requests]
        for request in finished:
            request.status = finished_status
        return finished

    monkeypatch.setattr(gen_sched_mod.VLLMScheduler, "add_request", fake_add_request)
    monkeypatch.setattr(gen_sched_mod.VLLMScheduler, "finish_requests", fake_finish_requests)
    return added


def _make_scheduler(scheduler_cls):
    scheduler = scheduler_cls.__new__(scheduler_cls)
    scheduler.requests = {}
    scheduler.running = []
    scheduler.waiting = []
    scheduler.chunk_transfer_adapter = None
    scheduler.input_coordinator = None
    scheduler._pending_request_prewarms = {}
    return scheduler


def _ref_audio() -> torch.Tensor:
    return torch.linspace(-1.0, 1.0, 8, dtype=torch.float32)


def _placeholder_info() -> AdditionalInformationPayload:
    return serialize_additional_information({_REF_AUDIO_KEY: _ref_audio(), _REF_AUDIO_SR_KEY: 16000})


def _drain(scheduler) -> list:
    return scheduler._wrap_omni_scheduler_output(SchedulerOutput.make_empty()).pending_request_prewarms


@_SCHEDULERS
@pytest.mark.parametrize("as_dict", [False, True], ids=["wire-payload", "plain-dict"])
@pytest.mark.parametrize(
    "extra",
    [{"speaker": "default", f"{ASYNC_CHUNK_PREWARM_NS}_not_ns": 7}, {}],
    ids=["with-other-info", "prewarm-only"],
)
def test_add_request_pops_prewarm_and_delivers_it_once(scheduler_cls, base_scheduler, as_dict, extra) -> None:
    scheduler = _make_scheduler(scheduler_cls)
    info = {_REF_AUDIO_KEY: _ref_audio(), _REF_AUDIO_SR_KEY: 16000, **extra}
    if not as_dict:
        # What EngineCore decodes after the msgpack hop from the orchestrator.
        payload = serialize_additional_information(info)
        info = msgspec.msgpack.decode(msgspec.msgpack.encode(payload), type=AdditionalInformationPayload)
    request = _StubRequest("req-1", info)

    scheduler.add_request(request)

    assert base_scheduler == [request]
    remaining = request.additional_information
    if extra:
        # All but the dotted-prefix keys (the ``_not_ns`` look-alike stays) reach the runner.
        assert type(remaining) is type(info)
        assert deserialize_additional_information(remaining) == extra
    else:
        # Same shape as a placeholder without a prewarm (serialize_payload({}) is None).
        assert remaining is None
    drained = _drain(scheduler)
    assert [p.request_id for p in drained] == ["req-1"]
    prewarm = drained[0].payload
    assert set(prewarm) == {"ref_audio", "ref_audio_sr"}
    assert prewarm["ref_audio"].dtype == torch.float32
    assert torch.equal(prewarm["ref_audio"], _ref_audio())
    assert prewarm["ref_audio_sr"] == 16000 and isinstance(prewarm["ref_audio_sr"], int)
    assert _drain(scheduler) == []


@_SCHEDULERS
@pytest.mark.parametrize(
    ("status", "resumable", "owes_terminal"),
    [
        pytest.param(RequestStatus.RUNNING, False, False, id="running"),
        pytest.param(RequestStatus.WAITING_FOR_STREAMING_REQ, True, False, id="parked-resumable"),
        pytest.param(RequestStatus.WAITING_FOR_STREAMING_REQ, False, True, id="parked-terminal"),
    ],
)
def test_readd_of_live_id_takes_no_prewarm(scheduler_cls, base_scheduler, status, resumable, owes_terminal) -> None:
    """A re-add of a live id (streaming update) passes its info through and keeps
    the first payload; the prewarm branch must not shadow the #6670 path, where a
    final update on a parked async-chunk sender ends the session locally."""
    scheduler = _make_scheduler(scheduler_cls)
    first = _StubRequest("req-1", _placeholder_info())
    scheduler.add_request(first)
    first_prewarm = scheduler._pending_request_prewarms["req-1"]
    first.status = status
    adapter, finished_calls = object(), []
    scheduler._adapter_owing_terminal = lambda request: adapter if owes_terminal else None
    scheduler._finish_parked_streaming_session = lambda request, owner: finished_calls.append((request, owner))
    update_info = _placeholder_info()
    update = _StubRequest("req-1", update_info)
    update.resumable = resumable

    scheduler.add_request(update)

    assert base_scheduler == ([first] if owes_terminal else [first, update])
    assert finished_calls == ([(first, adapter)] if owes_terminal else [])
    assert update.additional_information is update_info
    assert set(scheduler._pending_request_prewarms) == {"req-1"}
    assert scheduler._pending_request_prewarms["req-1"] is first_prewarm


def _bad_ref_audio(**entry_fields) -> dict:
    sr_entry = AdditionalInformationEntry(scalar_data=16000)
    return {_REF_AUDIO_KEY: AdditionalInformationEntry(**entry_fields), _REF_AUDIO_SR_KEY: sr_entry}


@_SCHEDULERS
@pytest.mark.parametrize(
    "prewarm_entries",
    [
        pytest.param(_bad_ref_audio(tensor_data=b"\x00" * 4, tensor_shape=[1], tensor_dtype="bad"), id="unknown-dtype"),
        pytest.param(_bad_ref_audio(tensor_data=b"\x00" * 3, tensor_shape=[1], tensor_dtype="float32"), id="truncated"),
        pytest.param({_REF_AUDIO_KEY: AdditionalInformationEntry()}, id="empty-namespace"),
    ],
)
def test_malformed_prewarm_is_dropped_without_raising(scheduler_cls, base_scheduler, prewarm_entries) -> None:
    scheduler = _make_scheduler(scheduler_cls)
    speaker = AdditionalInformationEntry(scalar_data="default")
    request = _StubRequest("req-1", AdditionalInformationPayload(entries={**prewarm_entries, "speaker": speaker}))

    scheduler.add_request(request)

    # Still admitted, minus the prewarm keys; nothing reaches the runner.
    assert base_scheduler == [request]
    assert request.additional_information.entries == {"speaker": speaker}
    assert _drain(scheduler) == []


@_SCHEDULERS
@pytest.mark.parametrize("aborted", [False, True], ids=["stale-undrained", "aborted"])
@pytest.mark.parametrize("new_sr", [None, 24000], ids=["reuse-without-prewarm", "reuse-with-prewarm"])
def test_reused_id_gets_only_its_own_payload(scheduler_cls, base_scheduler, aborted, new_sr) -> None:
    scheduler = _make_scheduler(scheduler_cls)
    scheduler.add_request(_StubRequest("req-other", _placeholder_info()))
    if aborted:
        scheduler.add_request(_StubRequest("req-1", _placeholder_info()))
        scheduler.finish_requests(["req-1"], RequestStatus.FINISHED_ABORTED)
    else:  # left behind by a request that is gone without being drained or finished
        scheduler._pending_request_prewarms["req-1"] = {"ref_audio_sr": 8000}
    new_info = {"speaker": "default"} if new_sr is None else {_REF_AUDIO_KEY: torch.zeros(4), _REF_AUDIO_SR_KEY: new_sr}
    new_payload = serialize_additional_information(new_info)
    request = _StubRequest("req-1", new_payload)

    scheduler.add_request(request)

    drained = {p.request_id: p.payload for p in _drain(scheduler)}
    assert set(drained) == ({"req-other"} if new_sr is None else {"req-other", "req-1"})
    if new_sr is None:
        # Info without a prewarm reaches the runner untouched.
        assert request.additional_information is new_payload
    else:
        assert drained["req-1"]["ref_audio_sr"] == new_sr
        assert torch.equal(drained["req-1"]["ref_audio"], torch.zeros(4))


@_SCHEDULERS
def test_wrap_output_drains_live_ids_exactly_once(scheduler_cls) -> None:
    scheduler = _make_scheduler(scheduler_cls)
    live_payload = {"ref_audio_sr": 16000}
    scheduler.requests = {"req-live": _StubRequest("req-live")}
    scheduler._pending_request_prewarms = {"req-live": live_payload, "req-gone": {"ref_audio_sr": 16000}}

    first = _drain(scheduler)

    assert [p.request_id for p in first] == ["req-live"]
    assert first[0].payload is live_payload
    # The freed id is cleared too, so nothing is ever re-sent.
    assert scheduler._pending_request_prewarms == {}
    assert _drain(scheduler) == []


def test_scheduler_output_defaults_to_no_prewarms() -> None:
    base = SchedulerOutput.make_empty()
    output = OmniSchedulerOutput(**{name: getattr(base, name) for name in SchedulerOutput.__dataclass_fields__})
    assert output.pending_request_prewarms == []


def _finish_via_error_outputs(scheduler, request_id: str) -> None:
    scheduler.grammar_compile_error_reqs = {request_id}
    scheduler._finish_error_requests(defaultdict(list))


@_SCHEDULERS
@pytest.mark.parametrize(
    "finish",
    [
        pytest.param(lambda s, rid: s.finish_requests([rid], RequestStatus.FINISHED_ABORTED), id="abort-list"),
        pytest.param(lambda s, rid: s.finish_requests(rid, RequestStatus.FINISHED_ABORTED), id="abort-single-id"),
        pytest.param(lambda s, rid: s._finish_input_timeout_requests({rid}), id="input-timeout"),
        pytest.param(_finish_via_error_outputs, id="error-outputs"),
    ],
)
def test_finish_paths_discard_undrained_prewarm(scheduler_cls, base_scheduler, finish) -> None:
    """A finish between add_request and the next schedule() must not hand the
    runner a payload for a request it will never see."""
    scheduler = _make_scheduler(scheduler_cls)
    scheduler.add_request(_StubRequest("req-done", _placeholder_info()))
    scheduler.add_request(_StubRequest("req-kept", _placeholder_info()))

    finish(scheduler, "req-done")

    assert "req-done" not in scheduler.requests
    # Checked before the drain, which would also filter the freed id.
    assert set(scheduler._pending_request_prewarms) == {"req-kept"}
    assert [p.request_id for p in _drain(scheduler)] == ["req-kept"]
