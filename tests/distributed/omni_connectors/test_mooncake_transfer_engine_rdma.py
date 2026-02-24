# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Integration tests for MooncakeTransferEngineConnector.

These tests require Mooncake TransferEngine and an RDMA environment.
Set RDMA_DEVICE_NAME to force a specific device for single-node testing.
"""

import hashlib
import os
import subprocess
import threading
import time
import unittest

import pytest
import torch

from vllm_omni.distributed.omni_connectors.connectors.mooncake_transfer_engine_connector import (
    ManagedBuffer,
    MooncakeTransferEngineConnector,
    TransferEngine,
)

# All tests in this file require Mooncake TransferEngine and an RDMA environment.
pytestmark = [pytest.mark.parallel, pytest.mark.gpu]

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def get_rdma_host() -> str:
    """Resolve RDMA-capable host IP (env > auto-detect IB > hostname > 127.0.0.1)."""
    import socket

    env_host = os.environ.get("RDMA_TEST_HOST")
    if env_host:
        return env_host

    for ip_cmd in ("ip", "/sbin/ip", "/usr/sbin/ip"):
        try:
            result = subprocess.run([ip_cmd, "addr", "show"], capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                continue
            lines = result.stdout.split("\n")
            rdma_patterns = ("ibp", "ib0", "ib1", "ib2", "mlx", "roce")
            for line in lines:
                if "inet " in line:
                    for pat in rdma_patterns:
                        if pat in line.lower():
                            ip = line.strip().split()[1].split("/")[0]
                            if not ip.startswith("127.") and not ip.startswith("169.254."):
                                return ip
            for line in lines:
                if "inet " in line and "scope global" in line:
                    ip = line.strip().split()[1].split("/")[0]
                    if not ip.startswith("127.") and not ip.startswith("172.17."):
                        return ip
            break
        except (FileNotFoundError, Exception):
            continue

    try:
        host_ip = socket.gethostbyname(socket.gethostname())
        if host_ip and not host_ip.startswith("127."):
            return host_ip
    except Exception:
        pass  # Hostname resolution failed; fall back to loopback below
    return "127.0.0.1"


def _detect_rdma_device() -> str:
    """Auto-detect a usable RDMA device for single-node loopback testing.

    On DGX machines with 12+ RDMA NICs, only RoCE NICs (with a bound network
    interface) can reliably do loopback.  IB-only NICs often fail cross-NIC
    QP handshakes, causing intermittent test failures.

    Priority:
      1. RDMA_DEVICE_NAME / RDMA_DEVICE environment variable
      2. ibdev2netdev: pick first RoCE device whose netdev is UP
      3. Mooncake TransferEngine topology (if available)
      4. Empty string (Mooncake uses all devices — risky on multi-NIC)
    """
    # 1. Explicit env override
    for env_key in ("RDMA_DEVICE_NAME", "RDMA_DEVICE"):
        env_val = os.environ.get(env_key, "").strip()
        if env_val:
            print(f"[_detect_rdma_device] Using {env_key}={env_val}")
            return env_val

    # 2. ibdev2netdev: most reliable, no Mooncake dependency
    #    Prefer RoCE (Ethernet-backed) over IB for loopback reliability.
    try:
        result = subprocess.run(["ibdev2netdev"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            up_lines = [ln.strip() for ln in result.stdout.strip().splitlines() if "(Up)" in ln]
            # Prefer Ethernet-backed (RoCE) — interface name is NOT ib*
            for line in up_lines:
                # Format: "mlx5_2 port 1 ==> enp75s0f0 (Up)"
                parts = line.split("==>")
                if len(parts) == 2:
                    iface = parts[1].strip().split()[0]  # e.g. "enp75s0f0"
                    if not iface.startswith("ib"):
                        dev = line.split()[0]
                        print(f"[_detect_rdma_device] ibdev2netdev selected RoCE: {dev} ({line})")
                        return dev
            # Fallback: any Up device (IB)
            if up_lines:
                dev = up_lines[0].split()[0]
                print(f"[_detect_rdma_device] ibdev2netdev selected IB: {dev} ({up_lines[0]})")
                return dev
            print("[_detect_rdma_device] ibdev2netdev: no device with (Up) link found")
    except FileNotFoundError:
        print("[_detect_rdma_device] ibdev2netdev not found, skipping")
    except Exception as exc:
        print(f"[_detect_rdma_device] ibdev2netdev failed: {exc}")

    # 3. Mooncake topology query
    if TransferEngine is not None:
        try:
            import json

            host = get_rdma_host()
            eng = TransferEngine()
            if eng.initialize(host, "P2PHANDSHAKE", "rdma", "") == 0:
                topo_str = getattr(eng, "get_local_topology", lambda: None)()
                if topo_str:
                    topo = json.loads(topo_str)
                    # Prefer RoCE (IPv4-mapped GID)
                    for name, info in topo.items():
                        if isinstance(info, dict):
                            gid = info.get("gid", "")
                            if gid.startswith("00:00:00:00:00:00:00:00:00:00:ff:ff"):
                                print(f"[_detect_rdma_device] Mooncake topology selected RoCE: {name}")
                                return name
                    # Prefer bonded device
                    for name in topo:
                        if "bond" in name:
                            print(f"[_detect_rdma_device] Mooncake topology selected bond: {name}")
                            return name
                    # First available
                    for name, info in topo.items():
                        if isinstance(info, dict):
                            print(f"[_detect_rdma_device] Mooncake topology fallback: {name}")
                            return name
                else:
                    print("[_detect_rdma_device] get_local_topology() returned None/empty")
        except Exception as exc:
            print(f"[_detect_rdma_device] Mooncake topology query failed: {exc}")

    print("[_detect_rdma_device] WARNING: no device detected, Mooncake will use ALL NICs")
    return ""


RDMA_HOST = get_rdma_host()
RDMA_DEVICE = _detect_rdma_device()


def _free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((RDMA_HOST, 0))
        return s.getsockname()[1]


def _connector_config(
    zmq_port: int,
    pool_size: int = 16 * 1024 * 1024,
    pool_device: str = "cpu",
) -> dict:
    cfg = {
        "host": RDMA_HOST,
        "zmq_port": zmq_port,
        "protocol": "rdma",
        "memory_pool_size": pool_size,
        "memory_pool_device": pool_device,
    }
    if RDMA_DEVICE:
        cfg["device_name"] = RDMA_DEVICE
    return cfg


def _md5(tensor: torch.Tensor) -> str:
    t = tensor.cpu() if tensor.is_cuda else tensor
    return hashlib.md5(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# 1. Basic functionality (single connector, no RDMA transfer)
# ---------------------------------------------------------------------------


@unittest.skipIf(TransferEngine is None, "Mooncake TransferEngine not available")
class TestBasicConnector(unittest.TestCase):
    """Verify connector initialization, put, cleanup, and health check."""

    def test_initialization(self):
        port = _free_port()
        with MooncakeTransferEngineConnector(_connector_config(port, pool_size=1024 * 1024)) as c:
            self.assertNotEqual(c.rpc_port, 0)
            self.assertEqual(c.pool_size, 1024 * 1024)
            self.assertTrue(c.pool.is_pinned())
            health = c.health()
            self.assertEqual(health["status"], "healthy")

    def test_put_tensor_bytes_object(self):
        """Put tensor / bytes / dict and verify metadata."""
        port = _free_port()
        with MooncakeTransferEngineConnector(_connector_config(port)) as c:
            ok, sz, meta = c.put("s0", "s1", "t", torch.randn(100))
            self.assertTrue(ok)
            self.assertTrue(meta["is_fast_path"])

            ok, sz, meta = c.put("s0", "s1", "b", b"hello" * 100)
            self.assertTrue(ok)
            self.assertTrue(meta["is_fast_path"])

            ok, sz, meta = c.put("s0", "s1", "d", {"k": [1, 2, 3]})
            self.assertTrue(ok)
            self.assertFalse(meta["is_fast_path"])

    def test_cleanup_releases_buffer(self):
        port = _free_port()
        with MooncakeTransferEngineConnector(_connector_config(port)) as c:
            c.put("s0", "s1", "r1", torch.randn(100))
            key = MooncakeTransferEngineConnector._make_key("r1", "s0", "s1")
            self.assertIn(key, c._local_buffers)
            c.cleanup("r1", from_stage="s0", to_stage="s1")
            self.assertNotIn(key, c._local_buffers)

    def test_pool_exhaustion_and_recovery(self):
        """Fill pool, verify failure, free, verify recovery."""
        port = _free_port()
        with MooncakeTransferEngineConnector(_connector_config(port, pool_size=64 * 1024)) as c:
            ids = []
            for i in range(10):
                ok, _, _ = c.put("s", "s", f"f{i}", torch.randn(1000))
                if ok:
                    ids.append(f"f{i}")
                else:
                    break
            for rid in ids:
                c.cleanup(rid, from_stage="s", to_stage="s")
            ok, _, _ = c.put("s", "s", "recovery", torch.randn(1000))
            self.assertTrue(ok, "Pool recovery failed after cleanup")


# ---------------------------------------------------------------------------
# 2. End-to-end RDMA transfer (producer + consumer, single node)
# ---------------------------------------------------------------------------


@unittest.skipIf(TransferEngine is None, "Mooncake TransferEngine not available")
class TestEndToEnd(unittest.TestCase):
    """E2E RDMA transfer: tensor, bytes, object, zero-copy, large payload, mixed types."""

    def _pair(self, pool_size=16 * 1024 * 1024):
        p = MooncakeTransferEngineConnector(_connector_config(_free_port(), pool_size))
        c = MooncakeTransferEngineConnector(_connector_config(_free_port(), pool_size))
        return p, c

    def test_tensor_e2e(self):
        p, c = self._pair()
        try:
            orig = torch.randn(1024, 1024, dtype=torch.float32)
            ok, sz, meta = p.put("s0", "s1", "t1", orig)
            self.assertTrue(ok)
            time.sleep(0.5)
            result = c.get("s0", "s1", "t1", meta)
            self.assertIsNotNone(result)
            buf, _ = result
            self.assertIsInstance(buf, ManagedBuffer)
            recon = buf.as_tensor(dtype=orig.dtype, shape=orig.shape)
            self.assertTrue(torch.equal(recon, orig))
            buf.release()
        finally:
            p.close()
            c.close()

    def test_bytes_e2e(self):
        p, c = self._pair()
        try:
            orig = b"Hello RDMA! " * 1000
            ok, _, meta = p.put("s0", "s1", "b1", orig)
            self.assertTrue(ok)
            time.sleep(0.5)
            result = c.get("s0", "s1", "b1", meta)
            self.assertIsNotNone(result)
            buf, _ = result
            recv = buf.to_bytes() if hasattr(buf, "to_bytes") else buf
            if hasattr(buf, "release"):
                buf.release()
            self.assertEqual(recv, orig)
        finally:
            p.close()
            c.close()

    def test_object_e2e(self):
        p, c = self._pair()
        try:
            orig = {"msg": "hello", "nums": [1, 2, 3], "nested": {"k": "v"}}
            ok, _, meta = p.put("s0", "s1", "o1", orig)
            self.assertTrue(ok)
            self.assertFalse(meta["is_fast_path"])
            time.sleep(1.0)
            result = c.get("s0", "s1", "o1", meta)
            self.assertIsNotNone(result)
            recv, _ = result
            self.assertEqual(recv, orig)
        finally:
            p.close()
            c.close()

    def test_zero_copy_e2e(self):
        p, c = self._pair()
        try:
            shape = (1024, 1024)
            nbytes = 1024 * 1024 * 4
            offset = p.allocator.alloc(nbytes)
            sbuf = ManagedBuffer(p.allocator, offset, nbytes, p.pool)
            ref = torch.randn(*shape, dtype=torch.float32)
            sbuf.as_tensor(dtype=torch.float32, shape=shape).copy_(ref)

            ok, sz, meta = p.put("s0", "s1", "zc", sbuf)
            self.assertTrue(ok)
            self.assertTrue(meta.get("is_fast_path"))
            time.sleep(0.5)

            result = c.get("s0", "s1", "zc", meta)
            self.assertIsNotNone(result)
            rbuf, _ = result
            recon = rbuf.as_tensor(dtype=torch.float32, shape=shape)
            self.assertTrue(torch.equal(recon, ref))
            rbuf.release()
        finally:
            p.close()
            c.close()

    def test_large_tensor_100mb(self):
        """Transfer ~100 MB tensor and verify MD5 integrity."""
        p, c = self._pair(pool_size=128 * 1024 * 1024)
        try:
            orig = torch.randn(5000, 5000, dtype=torch.float32)
            md5_send = _md5(orig)
            ok, _, meta = p.put("s0", "s1", "lg", orig)
            self.assertTrue(ok)
            time.sleep(1.0)
            result = c.get("s0", "s1", "lg", meta)
            self.assertIsNotNone(result)
            buf, _ = result
            recon = buf.as_tensor(dtype=orig.dtype, shape=orig.shape)
            self.assertEqual(_md5(recon), md5_send)
            buf.release()
        finally:
            p.close()
            c.close()

    def test_mixed_types_sequential(self):
        """Transfer different dtypes and data types in sequence."""
        p, c = self._pair(pool_size=32 * 1024 * 1024)
        try:
            cases = [
                ("f32", torch.randn(500, 500, dtype=torch.float32)),
                ("f16", torch.randn(500, 500, dtype=torch.float16)),
                ("i64", torch.randint(0, 100, (200, 200), dtype=torch.int64)),
                ("byte", b"X" * (1024 * 1024)),
                ("obj", {"key": "value", "list": [1, 2, 3]}),
            ]
            time.sleep(0.5)
            for name, data in cases:
                ok, _, meta = p.put("s0", "s1", name, data)
                self.assertTrue(ok, f"put failed: {name}")
                time.sleep(0.3)
                result = c.get("s0", "s1", name, meta)
                self.assertIsNotNone(result, f"get failed: {name}")
                recv, _ = result
                if isinstance(data, torch.Tensor):
                    self.assertIsInstance(recv, ManagedBuffer)
                    recon = recv.as_tensor(dtype=data.dtype, shape=data.shape)
                    self.assertTrue(torch.equal(recon, data), f"mismatch: {name}")
                    recv.release()
                elif isinstance(data, bytes):
                    got = recv.to_bytes() if hasattr(recv, "to_bytes") else recv
                    if hasattr(recv, "release"):
                        recv.release()
                    self.assertEqual(got, data)
                else:
                    self.assertEqual(recv, data)
        finally:
            p.close()
            c.close()

    def test_concurrent_put(self):
        """10 concurrent puts should all succeed."""
        port = _free_port()
        conn = MooncakeTransferEngineConnector(_connector_config(port, pool_size=64 * 1024 * 1024))
        errors: list[str] = []
        lock = threading.Lock()

        def worker(rid, data):
            try:
                ok, _, _ = conn.put("s0", "s1", rid, data)
                if not ok:
                    with lock:
                        errors.append(f"{rid}: put failed")
            except Exception as e:
                with lock:
                    errors.append(f"{rid}: {e}")

        with conn:
            threads = []
            for i in range(10):
                t = threading.Thread(target=worker, args=(f"r{i}", torch.randn(256, 256)))
                threads.append(t)
                t.start()
            for t in threads:
                t.join()
            self.assertEqual(len(errors), 0, f"errors: {errors}")

    def test_auto_cleanup(self):
        """Producer buffer should be released after consumer get."""
        p, c = self._pair()
        try:
            ok, _, meta = p.put("s0", "s1", "ac", torch.randn(100))
            self.assertTrue(ok)
            key = MooncakeTransferEngineConnector._make_key("ac", "s0", "s1")
            self.assertIn(key, p._local_buffers)
            time.sleep(0.5)
            result = c.get("s0", "s1", "ac", meta)
            self.assertIsNotNone(result)
            buf, _ = result
            if isinstance(buf, ManagedBuffer):
                buf.release()
            time.sleep(0.3)
            self.assertNotIn(key, p._local_buffers)
        finally:
            p.close()
            c.close()


# ---------------------------------------------------------------------------
# 3. Lifecycle & resource management
# ---------------------------------------------------------------------------


@unittest.skipIf(TransferEngine is None, "Mooncake TransferEngine not available")
class TestLifecycle(unittest.TestCase):
    """Close, context manager, double-close safety."""

    def test_close_releases_resources(self):
        c = MooncakeTransferEngineConnector(_connector_config(_free_port(), pool_size=1024 * 1024))
        c.put("s0", "s1", "x", torch.randn(100))
        c.close()
        self.assertTrue(c._stop_event.is_set())
        self.assertEqual(len(c._local_buffers), 0)

    def test_context_manager(self):
        with MooncakeTransferEngineConnector(_connector_config(_free_port())) as c:
            ok, _, _ = c.put("s0", "s1", "ctx", torch.randn(50))
            self.assertTrue(ok)
        self.assertTrue(c._stop_event.is_set())

    def test_double_close_safe(self):
        c = MooncakeTransferEngineConnector(_connector_config(_free_port()))
        c.close()
        c.close()


# ---------------------------------------------------------------------------
# 4. GPU memory pool (requires CUDA)
# ---------------------------------------------------------------------------


@pytest.mark.cuda
@unittest.skipIf(TransferEngine is None, "Mooncake TransferEngine not available")
@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
class TestGPUPool(unittest.TestCase):
    """GPU memory pool: initialization, put (CPU/GPU tensor), E2E transfer."""

    @staticmethod
    def _gpu_cfg(port, pool_size=32 * 1024 * 1024):
        return _connector_config(port, pool_size, pool_device="cuda:0")

    def test_gpu_pool_init(self):
        with MooncakeTransferEngineConnector(self._gpu_cfg(_free_port())) as c:
            self.assertEqual(c.pool_device, "cuda:0")
            self.assertTrue(c.pool.is_cuda)

    def test_gpu_pool_put_cpu_and_gpu_tensor(self):
        with MooncakeTransferEngineConnector(self._gpu_cfg(_free_port())) as c:
            ok, _, meta = c.put("s0", "s1", "h2d", torch.randn(256, 256))
            self.assertTrue(ok)
            self.assertTrue(meta["is_fast_path"])

            ok, _, meta = c.put("s0", "s1", "d2d", torch.randn(256, 256, device="cuda:0"))
            self.assertTrue(ok)
            self.assertTrue(meta["is_fast_path"])

    def test_gpu_e2e_transfer(self):
        p = MooncakeTransferEngineConnector(self._gpu_cfg(_free_port()))
        c = MooncakeTransferEngineConnector(self._gpu_cfg(_free_port()))
        try:
            orig = torch.randn(512, 512, dtype=torch.float32, device="cuda:0")
            ok, _, meta = p.put("s0", "s1", "ge", orig)
            self.assertTrue(ok)
            time.sleep(0.5)
            result = c.get("s0", "s1", "ge", meta)
            self.assertIsNotNone(result)
            buf, _ = result
            recon = buf.as_tensor(dtype=orig.dtype, shape=orig.shape)
            self.assertTrue(recon.is_cuda)
            self.assertTrue(torch.equal(recon.cpu(), orig.cpu()))
            buf.release()
        finally:
            p.close()
            c.close()


# ---------------------------------------------------------------------------
# 5. Stress / Correctness tests (marked slow, skipped in quick CI)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@unittest.skipIf(TransferEngine is None, "Mooncake TransferEngine not available")
class TestStressCorrectness(unittest.TestCase):
    """
    Slow but high-value regression tests: concurrent put+get with data
    integrity, large payload, edge cases, and sustained stress.
    """

    def _pair(self, pool_size=64 * 1024 * 1024):
        p = MooncakeTransferEngineConnector(_connector_config(_free_port(), pool_size))
        c = MooncakeTransferEngineConnector(_connector_config(_free_port(), pool_size))
        return p, c

    # -- Concurrent put + get with data integrity --

    def test_concurrent_put_get_integrity(self):
        """
        10 threads each put a unique tensor; a single consumer thread gets all
        of them and verifies MD5 matches.
        """
        p, c = self._pair(pool_size=128 * 1024 * 1024)
        num_workers = 10
        errors: list[str] = []
        lock = threading.Lock()
        md5_sent: dict[str, str] = {}
        meta_map: dict[str, dict] = {}

        def producer_worker(rid: str, tensor: torch.Tensor):
            try:
                md5_val = _md5(tensor)
                with lock:
                    md5_sent[rid] = md5_val
                ok, _, meta = p.put("s0", "s1", rid, tensor)
                if not ok:
                    with lock:
                        errors.append(f"{rid}: put failed")
                    return
                with lock:
                    meta_map[rid] = meta
            except Exception as e:
                with lock:
                    errors.append(f"{rid}: put exception {e}")

        try:
            threads = []
            for i in range(num_workers):
                t = threading.Thread(
                    target=producer_worker,
                    args=(f"cg{i}", torch.randn(512, 512, dtype=torch.float32)),
                )
                threads.append(t)
                t.start()
            for t in threads:
                t.join()

            self.assertEqual(len(errors), 0, f"Producer errors: {errors}")
            self.assertEqual(len(meta_map), num_workers)

            time.sleep(1.0)

            # Consumer: get each tensor and verify integrity
            for rid, meta in meta_map.items():
                result = c.get("s0", "s1", rid, meta)
                self.assertIsNotNone(result, f"get returned None for {rid}")
                buf, _ = result
                self.assertIsInstance(buf, ManagedBuffer, f"{rid} not ManagedBuffer")
                recon = buf.as_tensor(dtype=torch.float32, shape=(512, 512))
                recv_md5 = _md5(recon)
                self.assertEqual(recv_md5, md5_sent[rid], f"MD5 mismatch for {rid}")
                buf.release()
        finally:
            p.close()
            c.close()

    def test_concurrent_put_get_threaded_both_sides(self):
        """
        True overlapping concurrency: producer threads put items one by
        one and signal the consumer via a queue; the consumer thread
        starts getting as soon as each item is available, so put and get
        execute in overlapping time windows.
        """
        import queue

        p, c = self._pair(pool_size=128 * 1024 * 1024)
        num_items = 8
        errors: list[str] = []
        lock = threading.Lock()
        md5_sent: dict[str, str] = {}
        item_queue: queue.Queue[tuple[str, dict] | None] = queue.Queue()

        def producer_worker(rid, tensor):
            try:
                with lock:
                    md5_sent[rid] = _md5(tensor)
                ok, _, meta = p.put("s0", "s1", rid, tensor)
                if not ok:
                    with lock:
                        errors.append(f"{rid}: put failed")
                    return
                # Signal consumer immediately after put
                item_queue.put((rid, meta))
            except Exception as e:
                with lock:
                    errors.append(f"{rid}: {e}")

        consumed_count = [0]  # mutable container for thread access

        def consumer_thread():
            while consumed_count[0] < num_items:
                try:
                    item = item_queue.get(timeout=30)
                except queue.Empty:
                    with lock:
                        errors.append(f"consumer: queue.Empty after 30s, consumed {consumed_count[0]}/{num_items}")
                    break
                if item is None:
                    break
                rid, meta = item
                try:
                    # Small delay to let RDMA transfer initiate, but
                    # overlap with other producers that are still putting
                    time.sleep(0.1)
                    result = c.get("s0", "s1", rid, meta)
                    if result is None:
                        with lock:
                            errors.append(f"{rid}: get returned None")
                        continue
                    buf, _ = result
                    if isinstance(buf, ManagedBuffer):
                        recon = buf.as_tensor(dtype=torch.float32, shape=(256, 256))
                        if _md5(recon) != md5_sent.get(rid, ""):
                            with lock:
                                errors.append(f"{rid}: MD5 mismatch")
                        buf.release()
                except Exception as e:
                    with lock:
                        errors.append(f"{rid}: get exception {e}")
                finally:
                    consumed_count[0] += 1

        try:
            # Start consumer first so it is ready to get
            ct = threading.Thread(target=consumer_thread)
            ct.start()
            # Launch producers with staggered starts for overlap
            threads = []
            for i in range(num_items):
                t = threading.Thread(
                    target=producer_worker,
                    args=(f"bi{i}", torch.randn(256, 256, dtype=torch.float32)),
                )
                threads.append(t)
                t.start()
                time.sleep(0.05)  # slight stagger so puts overlap with gets
            for t in threads:
                t.join()
            ct.join(timeout=60)
            self.assertEqual(len(errors), 0, f"errors: {errors}")
            self.assertEqual(
                consumed_count[0],
                num_items,
                f"Consumer only processed {consumed_count[0]}/{num_items} items",
            )
        finally:
            p.close()
            c.close()

    # -- Edge cases --

    def test_small_tensor_1_element(self):
        """Transfer a single-element tensor."""
        p, c = self._pair()
        try:
            orig = torch.tensor([3.14], dtype=torch.float32)
            ok, _, meta = p.put("s0", "s1", "tiny", orig)
            self.assertTrue(ok)
            time.sleep(0.5)
            result = c.get("s0", "s1", "tiny", meta)
            self.assertIsNotNone(result)
            buf, _ = result
            recon = buf.as_tensor(dtype=torch.float32, shape=(1,))
            self.assertTrue(torch.equal(recon, orig))
            buf.release()
        finally:
            p.close()
            c.close()

    def test_empty_bytes_rejected(self):
        """Connector should gracefully reject empty bytes payload."""
        port = _free_port()
        with MooncakeTransferEngineConnector(_connector_config(port, pool_size=8 * 1024 * 1024)) as c:
            ok, sz, meta = c.put("s0", "s1", "empty_b", b"")
            self.assertFalse(ok, "Empty bytes should be rejected by connector")

    # -- Large payload stress (500 MB) --

    def test_large_tensor_500mb(self):
        """Transfer ~500 MB tensor and verify MD5 integrity."""
        p, c = self._pair(pool_size=600 * 1024 * 1024)
        try:
            # ~500 MB: 11180 * 11180 * 4 bytes ≈ 500 MB
            side = 11180
            orig = torch.randn(side, side, dtype=torch.float32)
            md5_send = _md5(orig)
            ok, _, meta = p.put("s0", "s1", "500mb", orig)
            self.assertTrue(ok)
            time.sleep(2.0)
            result = c.get("s0", "s1", "500mb", meta)
            self.assertIsNotNone(result)
            buf, _ = result
            recon = buf.as_tensor(dtype=orig.dtype, shape=orig.shape)
            self.assertEqual(_md5(recon), md5_send)
            buf.release()
        finally:
            p.close()
            c.close()

    # -- Allocator stress under connector --

    def test_rapid_alloc_free_cycle(self):
        """Put + cleanup in tight loop to stress allocator under real connector."""
        port = _free_port()
        with MooncakeTransferEngineConnector(_connector_config(port, pool_size=8 * 1024 * 1024)) as c:
            for i in range(50):
                rid = f"cycle_{i}"
                ok, _, _ = c.put("s0", "s1", rid, torch.randn(256, 256))
                self.assertTrue(ok, f"Put failed at iteration {i}")
                c.cleanup(rid, from_stage="s0", to_stage="s1")
            # After all cycles, pool should be fully recovered
            ok, _, _ = c.put("s0", "s1", "final", torch.randn(1024, 1024))
            self.assertTrue(ok, "Pool not fully recovered after rapid cycles")


if __name__ == "__main__":
    unittest.main()
