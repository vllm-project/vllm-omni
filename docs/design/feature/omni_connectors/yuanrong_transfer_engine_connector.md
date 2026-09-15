# YuanrongTransferEngineConnector

## When to Use

Use `YuanrongTransferEngineConnector` for high-performance peer-to-peer KV cache
transfer on Ascend NPU. It is different from `YuanrongConnector`:

- `YuanrongConnector` uses Yuanrong Datasystem as a distributed KV store and
  requires Datasystem workers plus etcd.
- `YuanrongTransferEngineConnector` uses Yuanrong TransferEngine directly. It
  registers local memory in each stage worker and lets the receiver pull data
  from the sender with TransferEngine.

The implementation currently lives under
`vllm_omni/platforms/npu/omni_connectors/` because the supported backend is
Ascend NPU only. The connector is still registered with the generic
OmniConnector factory under the name `YuanrongTransferEngineConnector`.

For Ascend P2P transfer, the connector should use NPU memory as its transfer
pool. The Mooncake-style CPU pool is not supported for this connector.

## Prerequisites

Install Yuanrong Datasystem/TransferEngine Python bindings in the runtime
environment. The packaged install is:

```bash
pip install openyuanrong-datasystem
```

To build from source, follow the upstream project instructions:

- Source repository: [openeuler/yuanrong-datasystem](https://atomgit.com/openeuler/yuanrong-datasystem)
- Installation guide: [Yuanrong Datasystem Linux installation](https://pages.openeuler.openatom.cn/openyuanrong-datasystem/docs/zh-cn/latest/installation/installation_linux.html)

The Ascend driver, CANN runtime, HCCL/HCCP, and NPU device network must also be
configured on every machine participating in the transfer.

## Check NPU Device Network

Yuanrong TransferEngine with `protocol: "ascend"` uses the NPU device network,
not only the host management network. The connector uses the same
`protocol: "ascend"` configuration for both HCCS and RoCE; it does not select a
transport or require a specific IP address family. The actual transport is
determined by the Ascend hardware topology and the TransferEngine/CANN network
configuration.

For HCCS, verify that all participating NPUs are connected through the expected
HCCS topology and that the links are healthy.

For RoCE, check the device address, link, and health state:

```bash
for i in {0..7}; do
  echo "===== NPU $i ====="
  hccn_tool -i "$i" -ip -g
  hccn_tool -i "$i" -link -g
  hccn_tool -i "$i" -net_health -g
done
```

Expected output for each RoCE NPU should include a configured device address and
a healthy link:

```text
ipaddr:<device-address>
link status: UP
net health status: Success
```

An unconfigured RoCE device address or unhealthy link prevents RoCE transfer.
This check is not required for an HCCS path that does not use an IP-configured
device interface.

## Check Device-to-Device Connectivity

For RoCE, after collecting the target-stage NPU device addresses, ping them from
each source-stage NPU:

```bash
# Example: AR uses NPU 0..3 and DiT uses NPU 4..7.
# Replace these values with the device addresses reported by
# hccn_tool -i 4..7 -ip -g.
dst_addrs=(
  "<dit-rank-0-device-address>"
  "<dit-rank-1-device-address>"
  "<dit-rank-2-device-address>"
  "<dit-rank-3-device-address>"
)
for src in 0 1 2 3; do
  for dst_addr in "${dst_addrs[@]}"; do
    echo "===== NPU $src -> $dst_addr ====="
    hccn_tool -i "$src" -ping -g address "$dst_addr" || true
  done
done
```

The expected result is `0.00% packet loss` for every required source/target pair.
This verifies basic RoCE device connectivity. TransferEngine can still fail
later if RDMA/QP/HCCL setup is broken, but failed ping means the device network
must be fixed first.

For HCCS, validate connectivity through the platform HCCS topology and link
diagnostics instead of the IP-based ping check.

## YAML Configuration

For Ascend P2P, configure the connector with `protocol: "ascend"` and an NPU
memory pool:

```yaml
connectors:
  yuanrong_te_connector:
    name: YuanrongTransferEngineConnector
    extra:
      host: "auto"
      zmq_port: 50051
      rpc_port: "auto"
      protocol: "ascend"
      device_name: "auto"
      memory_pool_size: 1073741824
      memory_pool_device: "npu"

edges:
  - from: 0
    to: 1
    window_size: -1
    max_inflight: 1
```

Important fields:

| Parameter | Recommended Value | Notes |
|---|---|---|
| `host` | `"auto"` | Advertises a routable local host IP for ZMQ metadata exchange. |
| `zmq_port` | `50051` | Base ZMQ port. The runtime applies stage/rank offsets. |
| `rpc_port` | `"auto"` | Lets TransferEngine choose an available RPC port. |
| `protocol` | `"ascend"` | Uses the Ascend TransferEngine backend. |
| `device_name` | `"auto"` | Each TP worker resolves to its local logical NPU, for example `npu:0`, `npu:1`, etc. |
| `memory_pool_device` | `"npu"` | Required for the Ascend fast path. This registers NPU memory with TransferEngine. |
| `memory_pool_size` | `1073741824` | 1 GiB per worker is a practical starting point for KV-cache transfer validation. Increase if the transfer pool is exhausted. |

`memory_pool_device: "cpu"` is rejected for
`YuanrongTransferEngineConnector` with `protocol: "ascend"`. CPU pool registration
can lead to Ascend P2P failures such as HCCL RA init errors, QP timeouts, or
FFTS/SDMA runtime errors. `memory_pool_device: "cpu"` is common for
`MooncakeTransferEngineConnector`, but Yuanrong TE on Ascend only supports
`"npu"`.

## 4-Card AR-to-DiT TP Example

For AR 4-way TP sending KV cache to DiT 4-way TP, set the stage TP topology in
both stage `omni_kv_config` blocks:

```yaml
stages:
  - stage_id: 0
    devices: "0,1,2,3"
    tensor_parallel_size: 4
    omni_kv_config:
      need_send_cache: true
      rank_mapping:
        from_tp: 4
        to_tp: 4
    output_connectors:
      to_stage_1: yuanrong_te_connector

  - stage_id: 1
    devices: "4,5,6,7"
    parallel_config:
      tensor_parallel_size: 4
    omni_kv_config:
      need_recv_cache: true
      rank_mapping:
        from_tp: 4
        to_tp: 4
    input_connectors:
      from_stage_0: yuanrong_te_connector
```

In this topology each DiT rank receives the corresponding AR rank shard:

```text
DiT rank 0 <- AR rank 0
DiT rank 1 <- AR rank 1
DiT rank 2 <- AR rank 2
DiT rank 3 <- AR rank 3
```

## Validate a Successful Run

Search the inference log for Yuanrong TE receive-side messages:

```bash
grep -E "\[YR GET\]|Successfully received KV cache|transfer_engine get failed" inference.log
```

A successful Ascend fast-path transfer contains lines like:

```text
[YR GET] ... fast_path, zero-copy
Successfully received KV cache ... across 1 key(s)
Applied CFG KV caches
[KV Reuse] ...
```

Sender-side `KV transfer OK` only means that the sender registered the payload
and metadata. The actual P2P data transfer happens on the receiver side during
`[YR GET]`.

## Troubleshooting

| Symptom | Likely Cause | Action |
|---|---|---|
| RoCE device address is not configured | The NPU RoCE interface is not ready. | Run `hccn_tool -i <id> -ip -g` and configure the device address for every participating NPU. This does not apply to HCCS. |
| `P2PCommInitRootInfo failed`, `ra init failed`, `RA_QP_STATUS_TIMEOUT` | Ascend P2P transport setup failed. | For HCCS, verify the topology and link health. For RoCE, verify the device address, link health, and device-to-device ping. Then test a minimal Yuanrong TE pair outside vLLM. |
| `fftsplus sdma error`, `context is abort` | Runtime SDMA/FFTS failure, often after an invalid or failed P2P transfer. | Use `memory_pool_device: "npu"` for Ascend TE, reduce `memory_pool_size` if OOM occurs, and isolate the failing NPU pair with a Yuanrong TE demo. |
| Sender logs `KV transfer OK` but receiver never logs `[YR GET]` | Sender `put()` succeeded, receiver P2P read failed or timed out. | Check receiver logs and search for `transfer_engine get failed`. |
| Pool allocation fails or OOM | Transfer pool is too small or too large for available NPU memory. | Start with 1 GiB (`1073741824`), reduce to 512 MiB for validation, or increase for larger KV payloads. |
