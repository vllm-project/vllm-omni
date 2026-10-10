# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import ast
import concurrent.futures
import threading
import time
from pathlib import Path
from types import SimpleNamespace


def _payload_store_class(clock=time):
    path = Path(__file__).resolve().parents[2] / "examples/online_serving/text_to_speech/fish_speech/gradio_demo.py"
    module = ast.parse(path.read_text())
    node = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "PayloadStore")
    namespace = {"threading": __import__("threading"), "time": clock, "secrets": __import__("secrets")}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["PayloadStore"]


def test_payload_store_consumes_once_and_expires():
    store = _payload_store_class()(ttl=0.01, cap=2)
    req_id = store.store({"input": "test"})
    assert store.consume(req_id) == {"input": "test"}
    assert store.consume(req_id) is None

    expired = store.store({"input": "expired"})
    time.sleep(0.02)
    assert store.consume(expired) is None


def test_payload_store_caps_oldest_entry():
    store = _payload_store_class()(ttl=60, cap=2)
    first = store.store({"input": "first"})
    second = store.store({"input": "second"})
    third = store.store({"input": "third"})
    assert store.consume(first) is None
    assert store.consume(second) == {"input": "second"}
    assert store.consume(third) == {"input": "third"}


def test_payload_store_is_safe_for_concurrent_store_and_consume():
    store = _payload_store_class()(ttl=60, cap=128)

    def store_one(index):
        return store.store({"input": str(index)})

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        request_ids = list(executor.map(store_one, range(64)))
        results = list(executor.map(store.consume, request_ids))

    assert sorted(item["input"] for item in results if item is not None) == sorted(str(i) for i in range(64))
    assert store.consume("") is None
    assert store.consume(None) is None


def test_payload_store_interleaves_consumers_at_capacity():
    store = _payload_store_class()(ttl=60, cap=8)
    old_ids = [store.store({"input": f"old-{i}"}) for i in range(8)]
    start = threading.Barrier(9, timeout=5)

    def produce(index):
        start.wait()
        request_id = store.store({"input": f"new-{index}"})
        with store._lock:
            assert len(store._items) <= store.cap
        return request_id

    def consume_old():
        start.wait()
        return store.consume(old_ids[0])

    with concurrent.futures.ThreadPoolExecutor(max_workers=9) as executor:
        producers = [executor.submit(produce, i) for i in range(8)]
        consumer = executor.submit(consume_old)
        new_ids = [future.result(timeout=10) for future in producers]
        old_result = consumer.result(timeout=10)

    # Eight inserts and at most one removal force eviction whichever thread wins.
    assert old_result in (None, {"input": "old-0"})
    assert all(store.consume(request_id) is None for request_id in old_ids)
    results = [store.consume(request_id) for request_id in new_ids]
    assert sorted(item["input"] for item in results) == [f"new-{i}" for i in range(8)]
    assert all(store.consume(request_id) is None for request_id in new_ids)


def test_payload_store_samples_time_after_lock_acquisition():
    clock = SimpleNamespace(now=100.0)
    clock.monotonic = lambda: clock.now
    store = _payload_store_class(clock)(ttl=1, cap=8)
    expired_id = store.store({"input": "expired"})

    class DelayedLock:
        def __enter__(self):
            # Model the time spent waiting for another thread to release the lock.
            clock.now += 2

        def __exit__(self, *args):
            pass

    store._lock = DelayedLock()
    assert store.consume(expired_id) is None
    new_id = store.store({"input": "new"})
    assert store._items[new_id][0] == clock.now + store.ttl
    store._lock = threading.Lock()
    assert store.consume(new_id) == {"input": "new"}
