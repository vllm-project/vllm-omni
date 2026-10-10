# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import ast
import concurrent.futures
import time
from pathlib import Path


def _payload_store_class():
    path = Path(__file__).resolve().parents[2] / "examples/online_serving/text_to_speech/fish_speech/gradio_demo.py"
    module = ast.parse(path.read_text())
    node = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "PayloadStore")
    namespace = {"threading": __import__("threading"), "time": time, "secrets": __import__("secrets")}
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
    consumed = []

    def store_and_consume(index):
        request_id = store.store({"input": str(index)})
        value = store.consume(request_id)
        if value is not None:
            consumed.append(value["input"])

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(store_and_consume, range(64)))

    assert sorted(consumed) == sorted(str(i) for i in range(64))
