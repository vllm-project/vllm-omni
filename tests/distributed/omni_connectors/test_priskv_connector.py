# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Standalone tests for PrisKVConnector against a running PrisKV server.

Prerequisites:
    1. PrisKV server is running:
         export PRISKV_TRANSPORT=ucx UCX_TLS=tcp PRISKV_CLIENT_DIRECT_MODE=y PRISKV_USE_SHM=n
         ./server/priskv-server -a 127.0.0.1 -p 6379 --acl any

    2. pypriskv is installed:
         cd PrisKV/pypriskv && pip install -e .

Usage:
    python tests/distributed/omni_connectors/test_priskv_connector.py
"""

"""
uv venv --python 3.12 --seed
source .venv/bin/activate

# On CUDA
uv pip install vllm==0.19.0 --torch-backend=auto
git clone https://github.com/vllm-project/vllm-omni.git
cd vllm-omni
uv pip install -e .
after install the dependencies, install priskv
solve the problem of pybind11 version mismatch
uv pip install pybind11
uv pip install --no-build-isolation -e /workspace/priskv/PrisKV/pypriskv
"""


import os
import time

# PrisKV client reads transport config from env vars — must match the server.
os.environ.setdefault("PRISKV_TRANSPORT", "ucx")
os.environ.setdefault("UCX_TLS", "tcp")
os.environ.setdefault("PRISKV_CLIENT_DIRECT_MODE", "y")
os.environ.setdefault("PRISKV_USE_SHM", "n")

import sys
import types

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Lightweight package stubs: prevent heavy __init__.py files from executing.
# The connector modules under test have zero dependency on vLLM; the only
# reason imports fail is that various __init__.py files eagerly pull in the
# full model stack.  We pre-register thin namespace stubs for each ancestor
# package so that Python skips their __init__.py and resolves submodule
# files directly through __path__.
# ---------------------------------------------------------------------------
_OMNI_ROOT = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, "vllm_omni")
_OMNI_ROOT = os.path.normpath(_OMNI_ROOT)

_stub_packages = {
    "vllm_omni":                                                      _OMNI_ROOT,
    "vllm_omni.distributed":                                          os.path.join(_OMNI_ROOT, "distributed"),
    "vllm_omni.distributed.omni_connectors":                          os.path.join(_OMNI_ROOT, "distributed", "omni_connectors"),
    "vllm_omni.distributed.omni_connectors.connectors":               os.path.join(_OMNI_ROOT, "distributed", "omni_connectors", "connectors"),
    "vllm_omni.distributed.omni_connectors.utils":                    os.path.join(_OMNI_ROOT, "distributed", "omni_connectors", "utils"),
}

for fqn, path in _stub_packages.items():
    if fqn not in sys.modules:
        stub = types.ModuleType(fqn)
        stub.__path__ = [path]
        stub.__package__ = fqn
        sys.modules[fqn] = stub

from vllm_omni.distributed.omni_connectors.connectors.priskv_connector import (
    PrisKVConnector,
)
from vllm_omni.distributed.omni_connectors.factory import OmniConnectorFactory
from vllm_omni.distributed.omni_connectors.utils.config import ConnectorSpec

PRISKV_CONFIG = {
    "host": "127.0.0.1",
    "port": 6379,
    "password": "kvcache-redis",
    "get_retries": 10,
    "get_retry_interval": 0.05,
}


def _make_connector() -> PrisKVConnector:
    return PrisKVConnector(PRISKV_CONFIG)


# ---------- Factory ----------

def test_factory_creation():
    spec = ConnectorSpec(name="PrisKVConnector", extra=PRISKV_CONFIG)
    connector = OmniConnectorFactory.create_connector(spec)
    assert isinstance(connector, PrisKVConnector)
    connector.close()
    print("[PASS] test_factory_creation")


# ---------- put / get: basic types ----------

def test_put_get_dict():
    c = _make_connector()
    data = {"tokens": [1, 2, 3], "text": "hello priskv", "nested": {"a": 1}}

    ok, size, _ = c.put("stage_0", "stage_1", "req_dict", data)
    assert ok, "put failed"
    assert size > 0

    result = c.get("stage_0", "stage_1", "req_dict")
    assert result is not None, "get returned None"
    retrieved, ret_size = result
    assert retrieved == data, f"mismatch: {retrieved}"
    assert ret_size == size

    c.close()
    print(f"[PASS] test_put_get_dict  ({size} bytes)")


def test_put_get_bytes():
    c = _make_connector()
    data = b"\x00\x01\x02\xff" * 256  # 1 KB raw bytes

    ok, size, _ = c.put("stage_0", "stage_1", "req_bytes", data)
    assert ok

    result = c.get("stage_0", "stage_1", "req_bytes")
    assert result is not None
    retrieved, _ = result
    assert retrieved == data

    c.close()
    print(f"[PASS] test_put_get_bytes  ({size} bytes)")


def test_put_get_list():
    c = _make_connector()
    data = list(range(1000))

    ok, size, _ = c.put("stage_0", "stage_1", "req_list", data)
    assert ok

    result = c.get("stage_0", "stage_1", "req_list")
    assert result is not None
    retrieved, _ = result
    assert retrieved == data

    c.close()
    print(f"[PASS] test_put_get_list  ({size} bytes)")


# ---------- put / get: tensor & ndarray ----------

def test_put_get_tensor_small():
    c = _make_connector()
    tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    ok, size, _ = c.put("stage_0", "stage_1", "req_tensor_sm", tensor)
    assert ok

    result = c.get("stage_0", "stage_1", "req_tensor_sm")
    assert result is not None
    retrieved, _ = result
    assert torch.equal(tensor, retrieved)

    c.close()
    print(f"[PASS] test_put_get_tensor_small  ({size} bytes)")


def test_put_get_tensor_large():
    c = _make_connector()
    tensor = torch.randn(32, 4096)  # ~512 KB

    ok, size, _ = c.put("stage_0", "stage_1", "req_tensor_lg", tensor)
    assert ok

    result = c.get("stage_0", "stage_1", "req_tensor_lg")
    assert result is not None
    retrieved, _ = result
    assert torch.allclose(tensor, retrieved)

    c.close()
    print(f"[PASS] test_put_get_tensor_large  ({size / 1024:.1f} KB)")


def test_put_get_ndarray():
    c = _make_connector()
    arr = np.random.rand(64, 128).astype(np.float32)

    ok, size, _ = c.put("stage_0", "stage_1", "req_np", arr)
    assert ok

    result = c.get("stage_0", "stage_1", "req_np")
    assert result is not None
    retrieved, _ = result
    assert np.array_equal(arr, retrieved)

    c.close()
    print(f"[PASS] test_put_get_ndarray  ({size / 1024:.1f} KB)")


# ---------- put / get: composite (simulating real stage data) ----------

def test_put_get_composite():
    """Simulate real inter-stage payload: dict with tensors + metadata."""
    c = _make_connector()
    data = {
        "hidden_states": torch.randn(1, 128, 768),
        "attention_mask": torch.ones(1, 128, dtype=torch.bool),
        "metadata": {"request_id": "abc-123", "seq_len": 128},
    }

    ok, size, _ = c.put("stage_0", "stage_1", "req_composite", data)
    assert ok

    result = c.get("stage_0", "stage_1", "req_composite")
    assert result is not None
    retrieved, _ = result

    assert torch.allclose(data["hidden_states"], retrieved["hidden_states"])
    assert torch.equal(data["attention_mask"], retrieved["attention_mask"])
    assert data["metadata"] == retrieved["metadata"]

    c.close()
    print(f"[PASS] test_put_get_composite  ({size / 1024:.1f} KB)")


# ---------- get miss / timeout ----------

def test_get_nonexistent_key():
    c = _make_connector()
    c.get_retries = 2
    c.get_retry_interval = 0.01

    result = c.get("stage_0", "stage_1", "nonexistent_key_xyz")
    assert result is None

    c.close()
    print("[PASS] test_get_nonexistent_key  (correctly returned None)")


# ---------- overwrite ----------

def test_put_overwrite():
    c = _make_connector()

    ok1, _, _ = c.put("stage_0", "stage_1", "req_overwrite", {"v": 1})
    assert ok1

    ok2, _, _ = c.put("stage_0", "stage_1", "req_overwrite", {"v": 2})
    assert ok2

    result = c.get("stage_0", "stage_1", "req_overwrite")
    assert result is not None
    retrieved, _ = result
    assert retrieved["v"] == 2

    c.close()
    print("[PASS] test_put_overwrite")


# ---------- health ----------

def test_health():
    c = _make_connector()
    h = c.health()
    assert h["status"] == "healthy"
    assert h["host"] == "127.0.0.1"
    assert h["port"] == 6379
    c.close()
    print(f"[PASS] test_health  {h}")


# ---------- cleanup ----------

def test_cleanup():
    c = _make_connector()
    c.put("stage_0", "stage_1", "req_cleanup", {"temp": True})
    c.cleanup("req_cleanup")
    c.close()
    print("[PASS] test_cleanup")


# ---------- throughput micro-benchmark ----------

def test_throughput():
    c = _make_connector()
    payload = torch.randn(8, 4096)  # ~128 KB per item
    n_iters = 50

    t0 = time.perf_counter()
    for i in range(n_iters):
        key = f"bench_{i}"
        ok, sz, _ = c.put("stage_0", "stage_1", key, payload)
        assert ok
        result = c.get("stage_0", "stage_1", key)
        assert result is not None
    elapsed = time.perf_counter() - t0

    serialized_size = c.serialize_obj(payload)
    total_bytes = len(serialized_size) * n_iters * 2  # put + get
    throughput_mb = (total_bytes / 1024 / 1024) / elapsed

    c.close()
    print(
        f"[PASS] test_throughput  "
        f"{n_iters} round-trips in {elapsed:.2f}s, "
        f"{throughput_mb:.1f} MB/s, "
        f"{n_iters / elapsed:.0f} ops/s"
    )


# ---------- main ----------

ALL_TESTS = [
    test_factory_creation,
    test_put_get_dict,
    test_put_get_bytes,
    test_put_get_list,
    test_put_get_tensor_small,
    test_put_get_tensor_large,
    test_put_get_ndarray,
    test_put_get_composite,
    test_get_nonexistent_key,
    test_put_overwrite,
    test_health,
    test_cleanup,
    test_throughput,
]

if __name__ == "__main__":
    print(f"{'=' * 60}")
    print("PrisKVConnector Test Suite")
    print(f"Server: {PRISKV_CONFIG['host']}:{PRISKV_CONFIG['port']}")
    print(f"{'=' * 60}\n")

    passed, failed = 0, 0
    for test_fn in ALL_TESTS:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"[FAIL] {test_fn.__name__}: {e}")

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed, {len(ALL_TESTS)} total")
    print(f"{'=' * 60}")

    if failed > 0:
        exit(1)
