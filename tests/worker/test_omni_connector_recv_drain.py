# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Receive-loop drain: do not rescan requests whose next chunk is absent."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest

from vllm_omni.worker.omni_connector_model_runner_mixin import OmniConnectorModelRunnerMixin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_N_REQS = 32
_QUEUED = 8
_DEPTH = 4


class _Host(OmniConnectorModelRunnerMixin):
    pass


class _Connector:
    def __init__(self) -> None:
        self.gets = 0
        self._store: dict[str, Any] = {}
        self._generation = 0
        self._signaled = False
        self._wake = threading.Condition()
        self._store_lock = threading.Lock()

    def put(self, from_stage, to_stage, put_key, data):
        with self._store_lock:
            self._store[f"{from_stage}_{to_stage}_{put_key}"] = data
        with self._wake:
            self._signaled = True
            self._wake.notify_all()
        return True, 1, None

    def get(self, from_stage, to_stage, get_key, metadata=None):
        self.gets += 1
        # Stand in for the shared-memory open and msgpack walk on every poll.
        total = 0
        for i in range(800):
            total += i
        with self._store_lock:
            data = self._store.pop(f"{from_stage}_{to_stage}_{get_key}", None)
        del total
        if data is None:
            return None
        return data, 1

    def close(self) -> None:
        return None

    def get_wakeup_generation(self) -> int:
        return self._generation

    def wait_for_change(self, generation: int, *, timeout: float) -> bool:
        with self._wake:
            if self._generation != generation:
                return True
            if not self._signaled:
                self._wake.wait(timeout=timeout)
            if self._generation != generation:
                return True
            if not self._signaled:
                return False
            self._signaled = False
            self._generation += 1
            return True


def _host(connector: _Connector) -> _Host:
    host = _Host()
    host.init_omni_connectors(
        SimpleNamespace(
            stage_connector_config=None,
            async_chunk=True,
            worker_type="ar",
            custom_process_next_stage_input_func=None,
            scheduling_metadata_adapter=None,
            stage_id=1,
        )
    )
    host._omni_connector = connector
    for i in range(_N_REQS):
        request = SimpleNamespace(
            request_id=f"r{i}",
            external_req_id=f"r{i}",
            additional_information=None,
            prompt_token_ids=[],
            num_computed_tokens=0,
        )
        host.register_chunk_recv(request)
    return host


def _queue(connector: _Connector, req_idx: int, depth: int) -> None:
    for chunk_id in range(depth):
        connector.put("0", "1", f"r{req_idx}_0_{chunk_id}", {"embed": {"decode": [chunk_id]}})


def _rows(host: _Host) -> int:
    with host._lock:
        return sum(
            len(payload.get("embed", {}).get("decode", ())) for payload in host._local_stage_payload_cache.values()
        )


def _legacy_until_idle(host: _Host) -> None:
    pending = [f"r{i}" for i in range(_N_REQS)]
    while host._poll_pending_requests_once(pending):
        pass


def _drain_until_idle(host: _Host) -> None:
    active = [f"r{i}" for i in range(_N_REQS)]
    while active:
        active = host._poll_pending_requests_once(active)


def test_queued_chunks_are_drained_without_rescanning_misses():
    """Eight requests already hold four chunks each. The other 24 have none.

    The old loop scans all 32 requests once per queued chunk, then once more
    to discover that nothing is left. The drain rescans only the requests
    that just returned a chunk.
    """
    legacy_connector = _Connector()
    legacy = _host(legacy_connector)
    drain_connector = _Connector()
    drain = _host(drain_connector)
    for req_idx in range(_QUEUED):
        _queue(legacy_connector, req_idx, _DEPTH)
        _queue(drain_connector, req_idx, _DEPTH)

    _legacy_until_idle(legacy)
    _drain_until_idle(drain)

    expected = _QUEUED * _DEPTH
    assert _rows(legacy) == expected
    assert _rows(drain) == expected
    print(
        f"queued drain: gets {legacy_connector.gets} -> {drain_connector.gets} "
        f"({legacy_connector.gets / drain_connector.gets:.2f}x fewer)"
    )
    assert drain_connector.gets * 2 <= legacy_connector.gets


def _legacy_loop(host: _Host) -> None:
    while not host._stop_event.is_set():
        snapshot = getattr(host._omni_connector, "get_wakeup_generation", None)
        generation = snapshot() if callable(snapshot) else None
        with host._lock:
            pending_ids = list(host._pending_load_reqs.keys())
        if not pending_ids:
            host._work_available.wait(timeout=0.01)
            host._work_available.clear()
            continue
        if not host._poll_pending_requests_once(pending_ids) and not host._stop_event.is_set():
            host._wait_for_next_payload(generation)


def _burst(loop) -> tuple[int, int, int]:
    connector = _Connector()
    host = _host(connector)
    stop = threading.Event()

    def produce() -> None:
        chunk = [0] * _N_REQS
        while not stop.is_set():
            for req_idx in range(_QUEUED):
                connector.put("0", "1", f"r{req_idx}_0_{chunk[req_idx]}", {"embed": {"decode": [chunk[req_idx]]}})
                chunk[req_idx] += 1
            time.sleep(0.002)

    host._recv_thread = threading.Thread(target=loop, args=(host,), daemon=True)
    host._recv_thread.start()
    producer = threading.Thread(target=produce, daemon=True)
    producer.start()
    iters = 0
    deadline = time.perf_counter() + 0.2
    try:
        while time.perf_counter() < deadline:
            iters += 1
        gets = connector.gets
        return gets, iters, _rows(host)
    finally:
        stop.set()
        producer.join(timeout=1)
        host.shutdown_omni_connectors()


def test_burst_receive_yields_the_model_thread():
    legacy_gets, legacy_iters, legacy_rows = _burst(_legacy_loop)
    drain_gets, drain_iters, drain_rows = _burst(_Host._recv_loop)
    print(
        f"burst: gets {legacy_gets} -> {drain_gets}, "
        f"model-thread iters {legacy_iters} -> {drain_iters}, "
        f"chunks {legacy_rows} -> {drain_rows}"
    )
    assert legacy_rows > 0 and drain_rows > 0
    # Poll count is the stable signal. The iteration count shows the model
    # thread got more of the GIL, but it moves with machine load.
    assert drain_gets < legacy_gets * 0.85
