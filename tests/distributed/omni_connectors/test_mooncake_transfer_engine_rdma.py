# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

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

import pytest
import torch

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import get_open_port
from vllm_omni.distributed.omni_connectors.connectors.mooncake_transfer_engine_connector import (
    ManagedBuffer,
    MooncakeTransferEngineConnector,
    TransferEngine,
)

# All tests in this file require Mooncake TransferEngine and an RDMA environment.
pytestmark = [
    pytest.mark.parallel,
    pytest.mark.core_model,
    pytest.mark.skipif(TransferEngine is None, reason="Mooncake TransferEngine not installed"),
]

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


_SYSFS_INFINIBAND = "/sys/class/infiniband"

# Set by _detect_rdma_device() when RDMA devices are visible but none of them
# can carry a loopback QP from this process. Used to skip the transfer tests
# with a precise reason instead of an opaque ``get() returned None``.
RDMA_LOOPBACK_BLOCKER: str = ""


def _read_sysfs(path: str) -> str:
    try:
        with open(path) as fh:
            return fh.read().strip()
    except OSError:
        return ""


def _sysfs_rdma_devices() -> list[dict]:
    """Enumerate RDMA devices through sysfs (no ``ibdev2netdev`` needed).

    ``ibdev2netdev`` is an MLNX_OFED script that is not shipped by rdma-core,
    so plain rdma-core hosts (and most containers) do not have it. It only
    reads sysfs, so do the same here. Each entry:

        {"name", "link_layer" ("Ethernet"|"InfiniBand"), "active" (bool),
         "netdevs": [(ifname, operstate), ...]}

    ``netdevs`` lists the network interfaces bound to the HCA *as visible from
    the current network namespace*. Inside a container without hostNetwork
    the RoCE netdev lives in the host namespace, so the list is empty even
    though the ib device node exists -- and in that case the kernel rejects
    every RoCE QP INIT->RTR transition with ENODEV ("No such device").
    """
    devices: list[dict] = []
    try:
        names = sorted(os.listdir(_SYSFS_INFINIBAND))
    except OSError:
        return devices
    for name in names:
        dev_dir = os.path.join(_SYSFS_INFINIBAND, name)
        ports_dir = os.path.join(dev_dir, "ports")
        try:
            ports = sorted(os.listdir(ports_dir), key=lambda p: int(p) if p.isdigit() else 0)
        except OSError:
            ports = []
        port_dir = os.path.join(ports_dir, ports[0]) if ports else ""
        link_layer = _read_sysfs(os.path.join(port_dir, "link_layer")) if port_dir else ""
        state = _read_sysfs(os.path.join(port_dir, "state")) if port_dir else ""
        net_dir = os.path.join(dev_dir, "device", "net")
        try:
            ifnames = sorted(os.listdir(net_dir))
        except OSError:
            ifnames = []
        netdevs = [(ifname, _read_sysfs(f"/sys/class/net/{ifname}/operstate")) for ifname in ifnames]
        devices.append(
            {
                "name": name,
                "link_layer": link_layer or "unknown",
                "active": state.upper().endswith("ACTIVE"),
                "netdevs": netdevs,
            }
        )
    return devices


def _detect_rdma_device() -> str:
    """Auto-detect a usable RDMA device for single-node loopback testing.

    On DGX machines with 12+ RDMA NICs, only RoCE NICs (with a bound network
    interface) can reliably do loopback.  IB-only NICs often fail cross-NIC
    QP handshakes, causing intermittent test failures.

    Priority:
      1. RDMA_DEVICE_NAME / RDMA_DEVICE environment variable
      2. sysfs (/sys/class/infiniband): first RoCE device whose netdev is UP
         in this network namespace; sets RDMA_LOOPBACK_BLOCKER when RoCE
         devices exist but none has a netdev here (container without
         hostNetwork -> QP INIT->RTR fails with ENODEV on every GID)
      3. ibdev2netdev: pick first RoCE device whose netdev is UP
      4. Mooncake TransferEngine topology (if available)
      5. Empty string (Mooncake uses all devices — risky on multi-NIC)
    """
    global RDMA_LOOPBACK_BLOCKER

    # 1. Explicit env override
    for env_key in ("RDMA_DEVICE_NAME", "RDMA_DEVICE"):
        env_val = os.environ.get(env_key, "").strip()
        if env_val:
            print(f"[_detect_rdma_device] Using {env_key}={env_val}")
            return env_val

    # 2. sysfs: what ibdev2netdev reads, minus the MLNX_OFED dependency.
    sysfs_devs = _sysfs_rdma_devices()
    if sysfs_devs:
        roce_devs = [d for d in sysfs_devs if d["link_layer"] == "Ethernet"]
        for dev in roce_devs:
            if not dev["active"]:
                continue
            up = [ifname for ifname, oper in dev["netdevs"] if oper == "up"]
            if up:
                print(f"[_detect_rdma_device] sysfs selected RoCE: {dev['name']} (netdev {up[0]} up)")
                return dev["name"]
        if roce_devs and len(roce_devs) == len(sysfs_devs) and not any(d["netdevs"] for d in roce_devs):
            # Every HCA is RoCE and none exposes a netdev in this netns. The
            # GID table is bound to host-namespace interfaces, so the kernel
            # cannot resolve a source netdev for any QP: ibv_modify_qp(RTR)
            # returns ENODEV for every GID index, loopback included.
            RDMA_LOOPBACK_BLOCKER = (
                f"{len(roce_devs)} RoCE device(s) visible ({', '.join(d['name'] for d in roce_devs)}) "
                "but none has a network interface in this network namespace "
                "(container/pod without hostNetwork or an SR-IOV/macvlan RDMA netdev); "
                "RoCE QP INIT->RTR fails with ENODEV on every GID, so RDMA loopback is "
                "impossible from this process. Run with hostNetwork or pass the RoCE netdev "
                "into the namespace; set RDMA_DEVICE_NAME to force the tests anyway."
            )
            print(f"[_detect_rdma_device] {RDMA_LOOPBACK_BLOCKER}")
            return ""
        print(
            "[_detect_rdma_device] sysfs: no RoCE device with an UP netdev; devices="
            + ", ".join(
                f"{d['name']}({d['link_layer']},{'active' if d['active'] else 'down'},netdevs={d['netdevs']})"
                for d in sysfs_devs
            )
        )
    else:
        print(f"[_detect_rdma_device] sysfs: {_SYSFS_INFINIBAND} is empty or missing")

    # 3. ibdev2netdev (MLNX_OFED hosts).
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
                        device_name = line.split()[0]
                        print(f"[_detect_rdma_device] ibdev2netdev selected RoCE: {device_name} ({line})")
                        return device_name
            # Fallback: any Up device (IB)
            if up_lines:
                device_name = up_lines[0].split()[0]
                print(f"[_detect_rdma_device] ibdev2netdev selected IB: {device_name} ({up_lines[0]})")
                return device_name
            print("[_detect_rdma_device] ibdev2netdev: no device with (Up) link found")
    except FileNotFoundError:
        print("[_detect_rdma_device] ibdev2netdev not found, skipping")
    except Exception as exc:
        print(f"[_detect_rdma_device] ibdev2netdev failed: {exc}")

    # 4. Mooncake topology query
    if TransferEngine is not None:
        try:
            import json

            host = get_rdma_host()
            eng = TransferEngine()
            if eng.initialize(host, "P2PHANDSHAKE", "rdma", "") == 0:
                topo_str = getattr(eng, "get_local_topology", lambda: None)()
                if topo_str:
                    topo = json.loads(topo_str)
                    # Two formats exist:
                    #  (a) {hca_name: {"gid": ..., ...}, ...}           (old)
                    #  (b) {compute_unit: [[preferred_hcas], [avail]]} (>= 0.3.x)
                    hca_names: list[str] = []
                    for name, info in topo.items():
                        if isinstance(info, dict):
                            gid = info.get("gid", "")
                            if gid.startswith("00:00:00:00:00:00:00:00:00:00:ff:ff"):
                                print(f"[_detect_rdma_device] Mooncake topology selected RoCE: {name}")
                                return name
                            hca_names.append(name)
                        elif isinstance(info, list):
                            for group in info:
                                if isinstance(group, list):
                                    hca_names.extend(h for h in group if isinstance(h, str))
                    # Preserve order, drop duplicates.
                    hca_names = list(dict.fromkeys(hca_names))
                    # Prefer bonded device
                    for name in hca_names:
                        if "bond" in name:
                            print(f"[_detect_rdma_device] Mooncake topology selected bond: {name}")
                            return name
                    # First available
                    if hca_names:
                        print(f"[_detect_rdma_device] Mooncake topology fallback: {hca_names[0]}")
                        return hca_names[0]
                    print(f"[_detect_rdma_device] Mooncake topology has no HCA names: {topo_str[:200]!r}")
                else:
                    print("[_detect_rdma_device] get_local_topology() returned None/empty")
        except Exception as exc:
            print(f"[_detect_rdma_device] Mooncake topology query failed: {exc}")

    print("[_detect_rdma_device] WARNING: no device detected, Mooncake will use ALL NICs")
    return ""


RDMA_HOST = get_rdma_host()
RDMA_DEVICE = _detect_rdma_device()

# Tests that move data between two connectors need a working loopback QP.
# They are skipped -- with the kernel-level reason -- only when
# _detect_rdma_device() established that no visible RDMA device can reach
# RTR from this network namespace. Everything else in this file (engine init,
# put, allocator, lifecycle) never opens a QP and keeps running.
requires_rdma_loopback = pytest.mark.skipif(
    bool(RDMA_LOOPBACK_BLOCKER),
    reason=f"RDMA loopback unavailable: {RDMA_LOOPBACK_BLOCKER}",
)


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


@hardware_test(res={"cuda": "L4"}, num_cards=1)
class TestBasicConnector:
    """Verify connector initialization, put, cleanup, and health check."""

    def test_initialization(self):
        port = get_open_port(RDMA_HOST)
        with MooncakeTransferEngineConnector(_connector_config(port, pool_size=1024 * 1024)) as c:
            assert c.rpc_port != 0
            assert c.pool_size == 1024 * 1024
            assert c.pool.is_pinned()
            health = c.health()
            assert health["status"] == "healthy"

    def test_put_tensor_bytes_object(self):
        """Put tensor / bytes / dict and verify metadata."""
        port = get_open_port(RDMA_HOST)
        with MooncakeTransferEngineConnector(_connector_config(port)) as c:
            ok, sz, meta = c.put("s0", "s1", "t", torch.randn(100))
            assert ok
            assert meta["is_fast_path"]

            ok, sz, meta = c.put("s0", "s1", "b", b"hello" * 100)
            assert ok
            assert meta["is_fast_path"]

            ok, sz, meta = c.put("s0", "s1", "d", {"k": [1, 2, 3]})
            assert ok
            assert not meta["is_fast_path"]

    def test_cleanup_releases_buffer(self):
        port = get_open_port(RDMA_HOST)
        with MooncakeTransferEngineConnector(_connector_config(port)) as c:
            c.put("s0", "s1", "r1", torch.randn(100))
            key = MooncakeTransferEngineConnector._make_key("r1", "s0", "s1")
            assert key in c._local_buffers
            c.cleanup("r1", from_stage="s0", to_stage="s1")
            assert key not in c._local_buffers

    def test_pool_exhaustion_and_recovery(self):
        """Fill pool, verify failure, free, verify recovery."""
        port = get_open_port(RDMA_HOST)
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
            assert ok, "Pool recovery failed after cleanup"


# ---------------------------------------------------------------------------
# 2. End-to-end RDMA transfer (producer + consumer, single node)
# ---------------------------------------------------------------------------


@hardware_test(res={"cuda": "L4"}, num_cards=1)
class TestEndToEnd:
    """E2E RDMA transfer: tensor, bytes, object, zero-copy, large payload, mixed types."""

    def _pair(self, pool_size=16 * 1024 * 1024):
        p = MooncakeTransferEngineConnector(_connector_config(get_open_port(RDMA_HOST), pool_size))
        c = MooncakeTransferEngineConnector(_connector_config(get_open_port(RDMA_HOST), pool_size))
        return p, c

    @requires_rdma_loopback
    def test_tensor_e2e(self):
        p, c = self._pair()
        try:
            orig = torch.randn(1024, 1024, dtype=torch.float32)
            ok, sz, meta = p.put("s0", "s1", "t1", orig)
            assert ok
            time.sleep(0.5)
            result = c.get("s0", "s1", "t1", meta)
            assert result is not None
            buf, _ = result
            assert isinstance(buf, ManagedBuffer)
            recon = buf.as_tensor(dtype=orig.dtype, shape=orig.shape)
            assert torch.equal(recon, orig)
            buf.release()
        finally:
            p.close()
            c.close()

    @requires_rdma_loopback
    def test_bytes_e2e(self):
        p, c = self._pair()
        try:
            orig = b"Hello RDMA! " * 1000
            ok, _, meta = p.put("s0", "s1", "b1", orig)
            assert ok
            time.sleep(0.5)
            result = c.get("s0", "s1", "b1", meta)
            assert result is not None
            buf, _ = result
            recv = buf.to_bytes() if hasattr(buf, "to_bytes") else buf
            if hasattr(buf, "release"):
                buf.release()
            assert recv == orig
        finally:
            p.close()
            c.close()

    @requires_rdma_loopback
    def test_object_e2e(self):
        p, c = self._pair()
        try:
            orig = {"msg": "hello", "nums": [1, 2, 3], "nested": {"k": "v"}}
            ok, _, meta = p.put("s0", "s1", "o1", orig)
            assert ok
            assert not meta["is_fast_path"]
            time.sleep(1.0)
            result = c.get("s0", "s1", "o1", meta)
            assert result is not None
            recv, _ = result
            assert recv == orig
        finally:
            p.close()
            c.close()

    @requires_rdma_loopback
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
            assert ok
            assert meta.get("is_fast_path")
            time.sleep(0.5)

            result = c.get("s0", "s1", "zc", meta)
            assert result is not None
            rbuf, _ = result
            recon = rbuf.as_tensor(dtype=torch.float32, shape=shape)
            assert torch.equal(recon, ref)
            rbuf.release()
        finally:
            p.close()
            c.close()

    @requires_rdma_loopback
    def test_large_tensor_100mb(self):
        """Transfer ~100 MB tensor and verify MD5 integrity."""
        p, c = self._pair(pool_size=128 * 1024 * 1024)
        try:
            orig = torch.randn(5000, 5000, dtype=torch.float32)
            md5_send = _md5(orig)
            ok, _, meta = p.put("s0", "s1", "lg", orig)
            assert ok
            time.sleep(1.0)
            result = c.get("s0", "s1", "lg", meta)
            assert result is not None
            buf, _ = result
            recon = buf.as_tensor(dtype=orig.dtype, shape=orig.shape)
            assert _md5(recon) == md5_send
            buf.release()
        finally:
            p.close()
            c.close()

    @requires_rdma_loopback
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
                assert ok, f"put failed: {name}"
                time.sleep(0.3)
                result = c.get("s0", "s1", name, meta)
                assert result is not None, f"get failed: {name}"
                recv, _ = result
                if isinstance(data, torch.Tensor):
                    assert isinstance(recv, ManagedBuffer)
                    recon = recv.as_tensor(dtype=data.dtype, shape=data.shape)
                    assert torch.equal(recon, data), f"mismatch: {name}"
                    recv.release()
                elif isinstance(data, bytes):
                    got = recv.to_bytes() if hasattr(recv, "to_bytes") else recv
                    if hasattr(recv, "release"):
                        recv.release()
                    assert got == data
                else:
                    assert recv == data
        finally:
            p.close()
            c.close()

    def test_concurrent_put(self):
        """10 concurrent puts should all succeed."""
        port = get_open_port(RDMA_HOST)
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
            assert len(errors) == 0, f"errors: {errors}"

    @requires_rdma_loopback
    def test_auto_cleanup(self):
        """Producer buffer should be released after consumer get."""
        p, c = self._pair()
        try:
            ok, _, meta = p.put("s0", "s1", "ac", torch.randn(100))
            assert ok
            key = MooncakeTransferEngineConnector._make_key("ac", "s0", "s1")
            assert key in p._local_buffers
            time.sleep(0.5)
            result = c.get("s0", "s1", "ac", meta)
            assert result is not None
            buf, _ = result
            if isinstance(buf, ManagedBuffer):
                buf.release()
            time.sleep(0.3)
            assert key not in p._local_buffers
        finally:
            p.close()
            c.close()


# ---------------------------------------------------------------------------
# 3. Lifecycle & resource management
# ---------------------------------------------------------------------------


@hardware_test(res={"cuda": "L4"}, num_cards=1)
class TestLifecycle:
    """Close, context manager, double-close safety."""

    def test_close_releases_resources(self):
        c = MooncakeTransferEngineConnector(_connector_config(get_open_port(RDMA_HOST), pool_size=1024 * 1024))
        c.put("s0", "s1", "x", torch.randn(100))
        c.close()
        assert c._stop_event.is_set()
        assert len(c._local_buffers) == 0

    def test_context_manager(self):
        with MooncakeTransferEngineConnector(_connector_config(get_open_port(RDMA_HOST))) as c:
            ok, _, _ = c.put("s0", "s1", "ctx", torch.randn(50))
            assert ok
        assert c._stop_event.is_set()

    def test_double_close_safe(self):
        c = MooncakeTransferEngineConnector(_connector_config(get_open_port(RDMA_HOST)))
        c.close()
        c.close()


# ---------------------------------------------------------------------------
# 4. GPU memory pool (requires CUDA)
# ---------------------------------------------------------------------------


@hardware_test(res={"cuda": "L4"}, num_cards=1)
class TestGPUPool:
    """GPU memory pool: initialization, put (CPU/GPU tensor), E2E transfer."""

    @staticmethod
    def _gpu_cfg(port, pool_size=32 * 1024 * 1024):
        return _connector_config(port, pool_size, pool_device="cuda:0")

    def test_gpu_pool_init(self):
        with MooncakeTransferEngineConnector(self._gpu_cfg(get_open_port(RDMA_HOST))) as c:
            assert c.pool_device == "cuda:0"
            assert c.pool.is_cuda

    def test_gpu_pool_put_cpu_and_gpu_tensor(self):
        with MooncakeTransferEngineConnector(self._gpu_cfg(get_open_port(RDMA_HOST))) as c:
            ok, _, meta = c.put("s0", "s1", "h2d", torch.randn(256, 256))
            assert ok
            assert meta["is_fast_path"]

            ok, _, meta = c.put("s0", "s1", "d2d", torch.randn(256, 256, device="cuda:0"))
            assert ok
            assert meta["is_fast_path"]

    @requires_rdma_loopback
    def test_gpu_e2e_transfer(self):
        p = MooncakeTransferEngineConnector(self._gpu_cfg(get_open_port(RDMA_HOST)))
        c = MooncakeTransferEngineConnector(self._gpu_cfg(get_open_port(RDMA_HOST)))
        try:
            orig = torch.randn(512, 512, dtype=torch.float32, device="cuda:0")
            ok, _, meta = p.put("s0", "s1", "ge", orig)
            assert ok
            time.sleep(0.5)
            result = c.get("s0", "s1", "ge", meta)
            assert result is not None
            buf, _ = result
            recon = buf.as_tensor(dtype=orig.dtype, shape=orig.shape)
            assert recon.is_cuda
            assert torch.equal(recon.cpu(), orig.cpu())
            buf.release()
        finally:
            p.close()
            c.close()


# ---------------------------------------------------------------------------
# 5. Stress / Correctness tests, slow
# ---------------------------------------------------------------------------


@hardware_test(res={"cuda": "L4"}, num_cards=1)
class TestStressCorrectness:
    """
    Slow but high-value regression tests: concurrent put+get with data
    integrity, large payload, edge cases, and sustained stress.
    """

    def _pair(self, pool_size=64 * 1024 * 1024):
        p = MooncakeTransferEngineConnector(_connector_config(get_open_port(RDMA_HOST), pool_size))
        c = MooncakeTransferEngineConnector(_connector_config(get_open_port(RDMA_HOST), pool_size))
        return p, c

    # -- Concurrent put + get with data integrity --

    @requires_rdma_loopback
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

            assert len(errors) == 0, f"Producer errors: {errors}"
            assert len(meta_map) == num_workers

            time.sleep(1.0)

            # Consumer: get each tensor and verify integrity
            for rid, meta in meta_map.items():
                result = c.get("s0", "s1", rid, meta)
                assert result is not None, f"get returned None for {rid}"
                buf, _ = result
                assert isinstance(buf, ManagedBuffer), f"{rid} not ManagedBuffer"
                recon = buf.as_tensor(dtype=torch.float32, shape=(512, 512))
                recv_md5 = _md5(recon)
                assert recv_md5 == md5_sent[rid], f"MD5 mismatch for {rid}"
                buf.release()
        finally:
            p.close()
            c.close()

    @requires_rdma_loopback
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
            assert len(errors) == 0, f"errors: {errors}"
            assert consumed_count[0] == num_items, f"Consumer only processed {consumed_count[0]}/{num_items} items"
        finally:
            p.close()
            c.close()

    # -- Edge cases --

    @requires_rdma_loopback
    def test_small_tensor_1_element(self):
        """Transfer a single-element tensor."""
        p, c = self._pair()
        try:
            orig = torch.tensor([3.14], dtype=torch.float32)
            ok, _, meta = p.put("s0", "s1", "tiny", orig)
            assert ok
            time.sleep(0.5)
            result = c.get("s0", "s1", "tiny", meta)
            assert result is not None
            buf, _ = result
            recon = buf.as_tensor(dtype=torch.float32, shape=(1,))
            assert torch.equal(recon, orig)
            buf.release()
        finally:
            p.close()
            c.close()

    def test_empty_bytes_rejected(self):
        """Connector should gracefully reject empty bytes payload."""
        port = get_open_port(RDMA_HOST)
        with MooncakeTransferEngineConnector(_connector_config(port, pool_size=8 * 1024 * 1024)) as c:
            ok, sz, meta = c.put("s0", "s1", "empty_b", b"")
            assert not ok, "Empty bytes should be rejected by connector"

    # -- Large payload stress (500 MB) --

    @requires_rdma_loopback
    def test_large_tensor_500mb(self):
        """Transfer ~500 MB tensor and verify MD5 integrity."""
        p, c = self._pair(pool_size=600 * 1024 * 1024)
        try:
            # ~500 MB: 11180 * 11180 * 4 bytes ≈ 500 MB
            side = 11180
            orig = torch.randn(side, side, dtype=torch.float32)
            md5_send = _md5(orig)
            ok, _, meta = p.put("s0", "s1", "500mb", orig)
            assert ok
            time.sleep(2.0)
            result = c.get("s0", "s1", "500mb", meta)
            assert result is not None
            buf, _ = result
            recon = buf.as_tensor(dtype=orig.dtype, shape=orig.shape)
            assert _md5(recon) == md5_send
            buf.release()
        finally:
            p.close()
            c.close()

    # -- Allocator stress under connector --

    def test_rapid_alloc_free_cycle(self):
        """Put + cleanup in tight loop to stress allocator under real connector."""
        port = get_open_port(RDMA_HOST)
        with MooncakeTransferEngineConnector(_connector_config(port, pool_size=8 * 1024 * 1024)) as c:
            for i in range(50):
                rid = f"cycle_{i}"
                ok, _, _ = c.put("s0", "s1", rid, torch.randn(256, 256))
                assert ok, f"Put failed at iteration {i}"
                c.cleanup(rid, from_stage="s0", to_stage="s1")
            # After all cycles, pool should be fully recovered
            ok, _, _ = c.put("s0", "s1", "final", torch.randn(1024, 1024))
            assert ok, "Pool not fully recovered after rapid cycles"
