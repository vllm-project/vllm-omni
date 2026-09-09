# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import threading
from collections import defaultdict
from types import SimpleNamespace

import pytest

from vllm_omni.worker_v2.omni_data_plane import OmniRunnerDataPlane

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _plane():
    plane = OmniRunnerDataPlane.__new__(OmniRunnerDataPlane)
    plane._lock = threading.Lock()
    plane._stop_event = threading.Event()
    plane._stage_id = 1
    plane._pending_load_reqs = {"r": object()}
    plane._request_ids_mapping = {"r": "external"}
    plane._get_req_chunk = defaultdict(int)
    plane._local_stage_payload_cache = {}
    plane._local_request_metadata = {}
    plane._finished_load_reqs = set()
    plane._payload_is_consumable = bool
    return plane


def test_notification_selects_only_the_arriving_request_and_cancellation_drops_interest():
    plane = _plane()
    calls = []

    def poll(ids):
        calls.append(ids)
        if len(calls) == 2:
            plane._stop_event.set()

    def wait():
        # The old request was cancelled while waiting. A new request's data
        # was published before its registration reached this receiver.
        plane._pending_load_reqs = {"new": object()}
        return {"external_0_0", "new_0_0"}, False

    plane._poll_pending_requests_once = poll
    plane._recv_ready_loop(SimpleNamespace(wait=wait))
    assert calls == [["r"], ["new"]]


def test_staged_payload_is_not_polled_and_consumption_rearms_already_published_next_key():
    plane = _plane()
    plane._local_stage_payload_cache["r"] = {"codes": [1]}
    plane._get_req_chunk["r"] = 1
    calls = []

    def poll(ids):
        calls.append(ids)
        plane._stop_event.set()

    def wait():
        assert calls == []
        plane._local_stage_payload_cache.pop("r")
        # Local consumption wakes us; no further upstream notification is
        # needed for a key whose publication event happened while staged.
        return set(), False

    plane._poll_pending_requests_once = poll
    plane._recv_ready_loop(SimpleNamespace(wait=wait))
    assert calls == [["r"]]


def test_overflow_reconciles_all_interested_keys():
    plane = _plane()
    plane._pending_load_reqs["r2"] = object()
    calls = []

    def poll(ids):
        calls.append(ids)
        if len(calls) == 2:
            plane._stop_event.set()

    plane._poll_pending_requests_once = poll
    plane._recv_ready_loop(SimpleNamespace(wait=lambda: (set(), True)))
    assert calls == [["r", "r2"], ["r", "r2"]]
