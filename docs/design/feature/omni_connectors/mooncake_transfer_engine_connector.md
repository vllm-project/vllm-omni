# MooncakeTransferEngineConnector

## When to Use

Best for high-performance multi-node data transfer between stages using Mooncake
Transfer Engine. Supports both RDMA and TCP protocols with a managed memory pool,
zero-copy deserialization, and optional GPUDirect RDMA. Applicable to any
inter-stage data (KV caches, request payloads, etc.), not limited to KV cache transfer.

Compared to `MooncakeStoreConnector` (TCP key-value store), this connector
provides **~60x faster** data transfer via RDMA direct memory access.

## Installation

```bash
pip install mooncake-transfer-engine
```

Ensure RDMA drivers are installed on all nodes (e.g., Mellanox OFED for
InfiniBand/RoCE NICs).

## Configuration

Define the connector in runtime:

```yaml
runtime:
  connectors:
    rdma_connector:
      name: MooncakeTransferEngineConnector
      extra:
        host: "auto"                  # Auto-detect local RDMA IP
        zmq_port: 50051               # ZMQ base port (see "Port Offset Scheme" below)
        protocol: "rdma"              # "rdma" or "tcp"
        device_name: ""               # RDMA device (e.g., "mlx5_0"), empty for auto-detect
        memory_pool_size: 2147483648  # 2GB memory pool
        memory_pool_device: "cpu"     # "cpu" for pinned memory, "cuda" for GPUDirect RDMA
```

Wire stages to the connector:

```yaml
stage_args:
  - stage_id: 0
    output_connectors:
      to_stage_1: rdma_connector

  - stage_id: 1
    input_connectors:
      from_stage_0: rdma_connector
```

## Parameters

### Required

| Parameter | Description |
|---|---|
| `role` | **Internal, do not set manually.** Auto-injected by the orchestration layer (`"sender"` for `output_connectors`, `"receiver"` for `input_connectors`). Defaults to `"sender"` if omitted. |
| `host` | Local IP address for RDMA. `"auto"` detects from network interfaces. |
| `protocol` | Transport protocol: `"rdma"` (InfiniBand/RoCE) or `"tcp"`. |

### Memory Pool

| Parameter | Default | Description |
|---|---|---|
| `memory_pool_size` | 1 GB | Total size of the RDMA-registered memory pool in bytes. |
| `memory_pool_device` | `"cpu"` | `"cpu"`: pinned host memory (recommended). `"cuda"`: GPU VRAM for GPUDirect RDMA (requires NIC-GPU direct PCIe connectivity). |

### Networking

| Parameter | Default | Description |
|---|---|---|
| `zmq_port` | 50051 | ZMQ **base** port. The orchestration layer computes the actual port as `base + purpose_offset + stage_offset` (see table below). Users only set this base value. |
| `sender_host` | `None` | **Internal.** Receiver-side only — dynamically resolved via `update_sender_info()`. Not needed in YAML. |
| `sender_zmq_port` | `None` | **Internal.** Receiver-side only — defaults to the sender's adjusted port. Not needed in YAML. |
| `device_name` | `""` | RDMA device name (e.g., `"mlx5_0"`). Empty for auto-detect. Can also be set via `RDMA_DEVICE_NAME` env var. |

#### ZMQ Port Offset Scheme

To avoid port conflicts when multiple edges, purposes, DP replicas, or TP ranks share the same node, the actual ZMQ port is computed as:

```
side_channel_port = zmq_port + purpose_offset + stage_offset + dp_index * tp_size
sender_listen     = side_channel_port + tp_rank
receiver_connect  = remote_side_channel_port + tp_rank
```

| Component | Value | Description |
|---|---|---|
| `zmq_port` | 50051 (default) | Base port from YAML config |
| `purpose_offset` | `request_forwarding` = 0, `kv_transfer` = 100 | Separates control-plane vs KV-cache connections |
| `stage_offset` | `int(from_stage)` (0, 1, 2...) | Separates edges from different source stages |
| `dp_index * tp_size` | e.g., DP1 × TP2 = 2 | Each DP replica reserves a port range of size `tp_size` (following vLLM convention: `VLLM_MOONCAKE_BOOTSTRAP_PORT + dp_index * tp_size`) |
| `tp_rank` | 0, 1, 2... | Each TP rank within a DP replica uses its own port |
| orchestrator | +200 | Extra offset when caller is the orchestrator (avoids collision with stage workers on the same node) |

**Example** (base=50051, stage 0→1, DP=2, TP=2, kv_transfer):

| Caller | DP | TP rank | Port |
|---|---|---|---|
| Stage worker | DP0 | rank 0 | `50051 + 100 + 0 + 0×2 + 0 = 50151` |
| Stage worker | DP0 | rank 1 | `50051 + 100 + 0 + 0×2 + 1 = 50152` |
| Stage worker | DP1 | rank 0 | `50051 + 100 + 0 + 1×2 + 0 = 50153` |
| Stage worker | DP1 | rank 1 | `50051 + 100 + 0 + 1×2 + 1 = 50154` |
| Orchestrator | — | — | `50051 + 200 + 0 = 50251` |

## Memory Pool Modes

| Mode | Config | Data Flow | Best For |
|---|---|---|---|
| CPU Pinned | `memory_pool_device: "cpu"` | GPU → CPU pool → RDMA → CPU pool → GPU | Most hardware topologies (recommended) |
| GPUDirect | `memory_pool_device: "cuda"` | GPU → GPU pool → RDMA (NIC reads GPU BAR1) → GPU pool | NIC-GPU direct PCIe (PIX topology) |

> **Note**: GPUDirect RDMA requires the NIC and GPU to share a direct PCIe
> switch (PIX topology). On systems where they are connected via PXB or NODE,
> CPU pinned memory is faster due to GPU BAR1 bandwidth limitations.

## Environment Variables

| Variable | Description |
|---|---|
| `RDMA_DEVICE_NAME` | Override RDMA device name (e.g., `mlx5_0`). |
| `MC_IB_PCI_RELAXED_ORDERING` | Set to `1` to enable PCIe relaxed ordering for GPUDirect. |

## Docker / Container Setup

RDMA requires host-level device access:

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

## Performance

Benchmark results on H800 GPUs with mlx5_0 RDMA NIC (~186 MB KV cache):

| Metric | MooncakeStoreConnector | MooncakeTransferEngineConnector (CPU) |
|---|---|---|
| KV transfer wall time | ~810 ms | **~14 ms** |
| RDMA throughput | N/A (TCP) | ~22 GB/s |
| Speedup | 1x | **58x** |

## Troubleshooting

### Quick Diagnostics

```bash
# 1. Check RDMA devices and link status
ibdev2netdev
# Expected: "mlx5_X port 1 ==> <iface> (Up)"
# RoCE devices map to Ethernet interfaces (e.g., enp75s0f0)
# IB devices map to ib0, ib1, etc.

# 2. Check InfiniBand device details
ibstat

# 3. Verify /dev/infiniband is accessible (required in containers)
ls /dev/infiniband/

# 4. Check Mooncake installation
python -c "from mooncake.engine import TransferEngine; print('OK')"

# 5. Check environment variables
echo "RDMA_DEVICE_NAME=${RDMA_DEVICE_NAME:-<not set>}"
echo "MC_IB_PCI_RELAXED_ORDERING=${MC_IB_PCI_RELAXED_ORDERING:-<not set>}"
```

### Common Issues

| Symptom | Cause | Fix |
|---|---|---|
| `Failed to modify QP to RTR` | Cross-NIC QP handshake failure (multi-NIC DGX) | Set `device_name` to a single RoCE NIC (e.g., `mlx5_2`) or set `RDMA_DEVICE_NAME` env var |
| `transport retry counter exceeded` | RDMA path between incompatible NICs | Same as above — restrict to one NIC |
| `zmq.error.Again: Resource temporarily unavailable` | ZMQ recv timeout (transfer took too long) | Check NIC selection; increase data may need longer timeout |
| `Mooncake Engine initialization failed` | Missing RDMA drivers or `/dev/infiniband` | Install Mellanox OFED; in Docker add `--device=/dev/infiniband` |
| `MemoryError` in allocator | Memory pool too small for payload | Increase `memory_pool_size` |
| GPU transfer slower than CPU | GPU BAR1 bandwidth limitation (PXB/NODE topology) | Use `memory_pool_device: "cpu"` instead of `"cuda"` |

### Multi-NIC Environments (DGX)

On DGX machines with 12+ RDMA NICs, only RoCE NICs (with a bound network
interface) reliably support loopback. IB-only NICs may fail cross-NIC QP
handshakes. To identify RoCE NICs:

```bash
ibdev2netdev | grep -v "ib[0-9]"
# RoCE devices show Ethernet interface names like enp75s0f0
```

Then configure the connector:
```yaml
device_name: "mlx5_2"  # or set RDMA_DEVICE_NAME=mlx5_2
```

See the RDMA Test README in tests/distributed/omni_connectors/README.md
for test-specific setup instructions.

For more details on the underlying engine, refer to the
[Mooncake repository](https://github.com/kvcache-ai/Mooncake).
