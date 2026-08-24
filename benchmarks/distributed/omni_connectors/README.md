# RDMA Test Configuration Guide

This document explains how to configure the RDMA environment and run tests for `MooncakeTransferEngineConnector`.

## Table of Contents

- [Docker Container Permissions](#docker-container-permissions)
- [Single-Node Testing](#single-node-testing)
- [Multi-Node Testing](#multi-node-testing)
- [Running Tests](#running-tests)
- [Cross-Node Testing](#cross-node-testing)
- [Troubleshooting](#troubleshooting)

---

## Docker Container Permissions

RDMA tests require access to InfiniBand/RoCE devices and system topology. Add the following permissions when running `docker run`.

### Option 1: Minimal Permissions (Recommended)

```bash
docker run -it \
    --cap-add=SYS_PTRACE \
    --cap-add=IPC_LOCK \
    --security-opt seccomp=unconfined \
    --network=host \
    --device=/dev/infiniband \
    -v /sys/class/infiniband:/sys/class/infiniband:ro \
    your-image:tag
```

Parameter explanation:
- `--cap-add=SYS_PTRACE`: Allow reading system topology information
- `--cap-add=IPC_LOCK`: Allow memory locking (required for RDMA memory registration)
- `--security-opt seccomp=unconfined`: Disable seccomp restrictions
- `--network=host`: Use host network (required for RDMA)
- `--device=/dev/infiniband`: Mount InfiniBand devices
- `-v /sys/class/infiniband`: Mount IB device info (read-only)

### Option 2: Full Permissions (Quick but not recommended for production)

```bash
docker run -it \
    --privileged \
    --network=host \
    your-image:tag
```

`--privileged` grants full host permissions. Suitable for quick testing but not recommended for production.

---

## Single-Node Testing

When running single-node tests (producer and consumer on the same machine), ensure they use the **same RDMA device**.

### Problem Background

InfiniBand devices use LID (Local Identifier) for routing. Different devices have different LIDs and cannot communicate directly. If no device is specified, Mooncake may assign different devices to connectors, causing handshake failures.

Common error:
```
[Handshake] Failed to modify QP to RTR, check mtu, gid, peer lid, peer qp num: Invalid argument [22]
```

### Solution

**Method 1: Set Environment Variable (Recommended)**

```bash
# List available RDMA devices
ibstat

# Select a device (e.g., mlx5_0)
export RDMA_DEVICE_NAME='mlx5_0'

# Run tests
pytest test_mooncake_transfer_engine_rdma.py -v -s
```

**Method 2: Use RoCE Devices**

If the system has RoCE devices (using IPv4 routing), the test code will automatically detect and prefer them. RoCE device GIDs start with `00:00:00:00:00:00:00:00:00:00:ff:ff` (IPv4-mapped).

**Method 3: Ensure MTU Consistency**

Make sure both endpoints use the same MTU:

```bash
# Check device MTU
ibstatus mlx5_0
```

---

## Multi-Node Testing

For multi-node tests, producer and consumer run on different machines connected via InfiniBand switch.

### Prerequisites

1. Both machines have Mooncake and RDMA drivers installed
2. Both machines are in the same InfiniBand subnet
3. Switch is properly configured

### Configuration

**Machine A (Producer):**

```bash
# Set RDMA host IP (InfiniBand interface IP)
export RDMA_TEST_HOST='10.0.0.1'

# Optional: Specify device
export RDMA_DEVICE_NAME='mlx5_0'
```

**Machine B (Consumer):**

```bash
# Set RDMA host IP
export RDMA_TEST_HOST='10.0.0.2'

# Optional: Specify device
export RDMA_DEVICE_NAME='mlx5_0'
```

### Verify Connectivity

```bash
# Ping IB interface
ping 10.0.0.2

# Test RDMA connectivity with ibping
# On Machine B (server)
ibping -S

# On Machine A (client)
ibping -G <Machine_B_GID>
```

---

## Running Tests

### Run All RDMA Tests (Single-Node, fast suite)

Slow tests (large payloads, stress, concurrency integrity) are marked `@pytest.mark.slow`. Use `-m "not slow"` to skip them in quick CI or local fast iteration.

```bash
cd tests/distributed/omni_connectors

# Fast suite only (excludes slow/stress tests)
pytest test_mooncake_transfer_engine_rdma.py test_mooncake_transfer_engine_buffer.py -v -s -m "not slow"
```

### Run Including Slow Tests

```bash
# Run ALL tests including slow/stress tests
pytest test_mooncake_transfer_engine_rdma.py test_mooncake_transfer_engine_buffer.py -v -s

# Run ONLY the slow/stress tests
pytest test_mooncake_transfer_engine_rdma.py test_mooncake_transfer_engine_buffer.py -v -s -m slow
```

### Run Buffer Management Tests

```bash
# Fast only
pytest test_mooncake_transfer_engine_buffer.py -v -s -m "not slow"

# Including allocator invariant tests (double-free, overlap, merge)
pytest test_mooncake_transfer_engine_buffer.py -v -s
```

### Run Specific Test Classes

```bash
# Basic connector tests
pytest test_mooncake_transfer_engine_rdma.py::TestBasicConnector -v -s

# End-to-end RDMA transfer tests
pytest test_mooncake_transfer_engine_rdma.py::TestEndToEnd -v -s

# Lifecycle & resource management tests
pytest test_mooncake_transfer_engine_rdma.py::TestLifecycle -v -s

# GPU memory pool tests (requires CUDA)
pytest test_mooncake_transfer_engine_rdma.py::TestGPUPool -v -s

# Stress / correctness tests (slow)
pytest test_mooncake_transfer_engine_rdma.py::TestStressCorrectness -v -s
```

### RDMA Environment Diagnostics

For quick diagnostics (device status, Mooncake availability, env vars, etc.),
see the [Troubleshooting section](../../../docs/design/feature/omni_connectors/mooncake_transfer_engine_connector.md#troubleshooting)
in the connector documentation.

---

## Cross-Node Testing

The `cross_node_mooncake_transfer_engine.py` script enables testing RDMA transfers between two separate physical machines. This script is **not** auto-discovered by `pytest` (it does not start with `test_`) — it must be run manually on each node.

### Prerequisites

1. Both machines have Mooncake installed
2. Both machines are connected via InfiniBand/RoCE switch
3. Firewall allows ZMQ ports (default: 15500, 15501)
4. Same RDMA device name on both nodes (if multiple devices exist)

### Running Cross-Node Tests

**On Machine A (Producer) — start first:**

```bash
cd benchmarks/distributed/omni_connectors/

# Optional: specify device if multiple exist
export RDMA_DEVICE_NAME='mlx5_0'

python cross_node_mooncake_transfer_engine.py \
    --role producer \
    --local-host <PRODUCER_IP> \
    --remote-host <CONSUMER_IP> \
    --tensor-size-mb 100 \
    --num-transfers 3
```

**On Machine B (Consumer) — start after producer:**

```bash
cd benchmarks/distributed/omni_connectors/

export RDMA_DEVICE_NAME='mlx5_0'

python cross_node_mooncake_transfer_engine.py \
    --role consumer \
    --local-host <CONSUMER_IP> \
    --remote-host <PRODUCER_IP> \
    --tensor-size-mb 100 \
    --num-transfers 3
```

### Transfer Modes

| Mode | Description | Example |
|------|-------------|---------|
| `copy` | Normal path — tensor copied to RDMA pool (default) | `--mode copy` |
| `zerocopy` | Zero-copy path — data created directly in RDMA pool | `--mode zerocopy` |
| `gpu` | GPU transfer — RDMA pool on GPU, uses GPUDirect | `--mode gpu --gpu-id 0` |

### Benchmark Mode

Skip MD5 verification and measure pure RDMA throughput:

```bash
# Producer
python cross_node_mooncake_transfer_engine.py \
    --role producer \
    --local-host <PRODUCER_IP> \
    --remote-host <CONSUMER_IP> \
    --tensor-size-mb 1024 \
    --num-transfers 20 \
    --benchmark

# Consumer
python cross_node_mooncake_transfer_engine.py \
    --role consumer \
    --local-host <CONSUMER_IP> \
    --remote-host <PRODUCER_IP> \
    --tensor-size-mb 1024 \
    --num-transfers 20 \
    --benchmark
```

### Cross-Node Test Options

| Option | Description | Default |
|--------|-------------|---------|
| `--role` | `producer` or `consumer` | Required |
| `--local-host` | Local RDMA IP address | Required |
| `--remote-host` | Remote RDMA IP address | Required |
| `--local-port` | Local ZMQ port for RDMA data | 15500 |
| `--remote-port` | Remote ZMQ port for RDMA data | 15500 |
| `--ctrl-port` | Control channel port | 15501 |
| `--tensor-size-mb` | Tensor size in MB | 100 |
| `--num-transfers` | Number of transfers | 3 |
| `--mode` | `copy`, `zerocopy`, or `gpu` | `copy` |
| `--gpu-id` | GPU ID for GPU mode | 0 |
| `--benchmark` | Skip MD5, pure performance test | off |

---

## Troubleshooting

### 1. "Failed to modify QP to RTR" Error

**Cause**: QP handshake failed, usually due to device configuration mismatch.

**Solution**:
```bash
# Force using the same device
export RDMA_DEVICE_NAME='mlx5_0'
```

### 2. "Mooncake TransferEngine is not available"

**Cause**: Mooncake not installed or import failed.

**Solution**:
```bash
# Check Mooncake installation
python -c "from mooncake.engine import TransferEngine; print('OK')"

# Reinstall if needed
pip install mooncake-transfer-engine
# Or using uv
uv pip install mooncake-transfer-engine

```

### 3. "Permission denied" accessing /dev/infiniband

**Cause**: Container lacks IB device access permissions.

**Solution**:
```bash
docker run --device=/dev/infiniband --cap-add=IPC_LOCK ...
```

### 4. Test Timeout

**Cause**: RDMA connection establishment failed or network latency.

**Solution**:
```bash
# Check network status
ibstat
ibstatus
```

### 5. GPU Test Failed "CUDA is not available"

**Cause**: CUDA environment not configured or GPU unavailable.

**Solution**:
```bash
# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Docker needs NVIDIA runtime
docker run --gpus all ...
```

---

## Environment Variables Reference

| Variable | Description | Example |
|----------|-------------|---------|
| `RDMA_DEVICE_NAME` | Specify RDMA device name | `mlx5_0` |
| `RDMA_TEST_HOST` | Specify test host IP | `10.0.0.1` |
| `MC_TE_METRIC` | Enable Mooncake metrics | `1` |
| `MC_IB_PCI_RELAXED_ORDERING` | Enable PCIe relaxed ordering | `1` |

---

## Test Files Overview

| File | Description | Auto-discovered by pytest |
|------|-------------|--------------------------|
| `test_mooncake_transfer_engine_rdma.py` | Integration tests for MooncakeTransferEngineConnector (basic, E2E, lifecycle, GPU) | Yes |
| `test_mooncake_transfer_engine_buffer.py` | Memory pool and buffer management unit tests | Yes |
| `cross_node_mooncake_transfer_engine.py` | Cross-node (multi-machine) testing script — run manually | No (filename does not start with `test_`) |

### test_mooncake_transfer_engine_rdma.py — Test Classes

| Test Class | Memory Pool | Marker | Description |
|------------|-------------|--------|-------------|
| `TestBasicConnector` | CPU | — | Initialization, put tensor/bytes/object, cleanup, pool exhaustion |
| `TestEndToEnd` | CPU | — | E2E RDMA transfer: tensor, bytes, object, zero-copy, large payload (100MB), mixed types, concurrency |
| `TestLifecycle` | CPU | — | Close, context manager, double-close safety |
| `TestGPUPool` | GPU | — | GPU pool init, put CPU/GPU tensor, GPU E2E transfer |
| `TestStressCorrectness` | CPU | `slow` | Concurrent put+get with MD5 integrity, bidirectional concurrency, edge cases (1-element tensor, empty bytes), 500MB payload, rapid alloc/free cycles |

### test_mooncake_transfer_engine_buffer.py — Test Classes

| Test Class | Marker | Description |
|------------|--------|-------------|
| `TestBufferAllocator` | — | Basic alloc/free, alignment, exhaustion/recovery, thread safety |
| `TestAllocatorInvariants` | `slow` | Double-free safety, overlap corruption detection, adjacent-block merging, fragmentation/defrag |
| `TestManagedBuffer` | — | Tensor views, context manager |
