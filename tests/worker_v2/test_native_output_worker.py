# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import gc
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace as N

import pytest

from vllm_omni.worker_v2.native_output_worker import NativeOutputWorker

pytestmark = [pytest.mark.cpu, pytest.mark.core_model]


class Output:
    copy_event = None

    def __init__(self, rid, gate=None, error=None):
        self.rid = rid
        self.gate = gate
        self.error = error

    def get_output(self):
        if self.gate is not None:
            assert self.gate.wait(3)
        if self.error is not None:
            raise self.error
        return N(req_ids=[self.rid], sampled_token_ids=[[1]], inter_stage_outputs=[{"value": 1}])


def test_publication_precedes_consumption_fifo_and_owner_signals():
    owner = threading.get_ident()
    gate = threading.Event()
    sent = threading.Event()
    calls = []

    def publish(**kwargs):
        calls.append(("publish", kwargs["req_ids"], threading.get_ident()))
        if kwargs["req_ids"] == ["b"]:
            sent.set()

    def signals():
        calls.append(("signals", threading.get_ident()))
        return "ready"

    plane = N(enqueue_outputs=publish, get_omni_connector_output=signals)
    worker = NativeOutputWorker(2)
    try:
        a = worker.submit(Output("a", gate), plane)
        b = worker.submit(Output("b"), plane)
        assert not sent.is_set()
        gate.set()
        assert sent.wait(3)
        assert [x[1] for x in calls] == [["a"], ["b"]]
        assert all(x[2] != owner for x in calls)
        result = a.get_output()
        assert a.get_output() is result
        assert result.inter_stage_outputs is None and result.omni_connector_output == "ready"
        b.get_output()
        assert calls[2:] == [("signals", owner), ("signals", owner)]
    finally:
        gate.set()
        worker.close()


@pytest.mark.parametrize("where", ["parse", "publish", "signals"])
def test_errors_are_preserved_and_close_rejects_submissions(where):
    error = ValueError(where)

    def fail(*args, **kwargs):
        raise error

    plane = N(
        enqueue_outputs=fail if where == "publish" else lambda **kwargs: None,
        get_omni_connector_output=fail if where == "signals" else lambda: None,
    )
    worker = NativeOutputWorker(1)
    output = worker.submit(Output("a", error=error if where == "parse" else None), plane)
    for _ in range(2):
        with pytest.raises(ValueError, match=where):
            output.get_output()
    worker.close()
    with pytest.raises(RuntimeError, match="closed"):
        worker.submit(Output("b"), plane)


def test_retained_wrapper_does_not_retain_raw_snapshot_or_cycle():
    worker = NativeOutputWorker(1)
    plane = N(enqueue_outputs=lambda **kw: None, get_omni_connector_output=lambda: None)
    enabled = gc.isenabled()
    gc.disable()
    try:
        raw = Output("a")
        raw_ref = weakref.ref(raw)
        wrapped = worker.submit(raw, plane)
        del raw
        result = wrapped.get_output()
        worker.close()
        assert raw_ref() is None
        ref = weakref.ref(wrapped)
        del wrapped
        assert ref() is None and result.req_ids == ["a"]
    finally:
        worker.close()
        if enabled:
            gc.enable()
        gc.collect()


def test_capacity_bounds_pending_materialization():
    worker = NativeOutputWorker(1)
    gate = threading.Event()
    attempted = threading.Event()
    plane = N(enqueue_outputs=lambda **kw: None, get_omni_connector_output=lambda: None)
    a = worker.submit(Output("a", gate), plane)

    def submit():
        attempted.set()
        return worker.submit(Output("b"), plane)

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(submit)
            assert attempted.wait(3)
            assert not future.done()
            gate.set()
            b = future.result(3)
            assert a.get_output().req_ids == ["a"] and b.get_output().req_ids == ["b"]
    finally:
        gate.set()
        worker.close()


@pytest.mark.parametrize("tp_size,enabled", [(1, True), (2, False)])
def test_multi_rank_preserves_existing_consumer(monkeypatch, tp_size, enabled):
    from vllm_omni.worker_v2.omni_model_runner import OmniGPUModelRunner

    monkeypatch.setenv("VLLM_OMNI_ASYNC_NATIVE_OUTPUT", "1")
    runner = N(_omni_data_plane=object(), vllm_config=N(parallel_config=N(tensor_parallel_size=tp_size)))
    assert OmniGPUModelRunner._uses_native_output_materializer(runner) is enabled


def test_worker_binds_device_before_materialization(monkeypatch):
    from vllm.platforms import current_platform

    owner = threading.get_ident()
    calls = []
    monkeypatch.setattr(current_platform, "set_device", lambda device: calls.append((device, threading.get_ident())))
    plane = N(enqueue_outputs=lambda **kwargs: None, get_omni_connector_output=lambda: None)
    worker = NativeOutputWorker(1, device="worker-device")
    try:
        worker.submit(Output("a"), plane).get_output()
        assert len(calls) == 1 and calls[0][0] == "worker-device" and calls[0][1] != owner
    finally:
        worker.close()
