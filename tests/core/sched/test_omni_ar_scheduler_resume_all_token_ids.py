from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_scheduler(requests: dict) -> OmniARScheduler:
    sched = OmniARScheduler.__new__(OmniARScheduler)
    sched.requests = requests
    return sched


def _make_output(req_ids, all_token_ids):
    cached = SimpleNamespace(req_ids=req_ids, all_token_ids=all_token_ids)
    return SimpleNamespace(scheduled_cached_reqs=cached)


def test_refill_adds_missing_all_token_ids() -> None:
    # The AR talker leaves the persistent batch each decode step, so the base
    # scheduler omits all_token_ids; the resume path needs it refilled.
    requests = {"a": SimpleNamespace(all_token_ids=[1, 2, 3])}
    out = _make_output(["a"], {})

    _make_scheduler(requests)._refill_cached_all_token_ids(out)

    assert out.scheduled_cached_reqs.all_token_ids == {"a": [1, 2, 3]}


def test_refill_copies_so_request_mutation_does_not_leak() -> None:
    src = [1, 2, 3]
    requests = {"a": SimpleNamespace(all_token_ids=src)}
    out = _make_output(["a"], {})

    _make_scheduler(requests)._refill_cached_all_token_ids(out)

    src.append(4)
    assert out.scheduled_cached_reqs.all_token_ids["a"] == [1, 2, 3]


def test_refill_does_not_overwrite_present_entry() -> None:
    requests = {"a": SimpleNamespace(all_token_ids=[1, 2, 3])}
    out = _make_output(["a"], {"a": [9, 9]})

    _make_scheduler(requests)._refill_cached_all_token_ids(out)

    # Present entries are left untouched (base scheduler already supplied them).
    assert out.scheduled_cached_reqs.all_token_ids["a"] == [9, 9]


def test_refill_skips_request_absent_from_tracking() -> None:
    # A cached id no longer in self.requests must not raise and must not be added.
    out = _make_output(["gone"], {})

    _make_scheduler({})._refill_cached_all_token_ids(out)

    assert out.scheduled_cached_reqs.all_token_ids == {}


def test_refill_noop_when_no_cached_reqs() -> None:
    for req_ids in (None, []):
        out = _make_output(req_ids, {})
        _make_scheduler({})._refill_cached_all_token_ids(out)
        assert out.scheduled_cached_reqs.all_token_ids == {}
