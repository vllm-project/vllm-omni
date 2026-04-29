# MoriTransferEngineConnector

## When to Use

Currently supports intra-node deployment with Mori.

As noted in #1742, inter-node support will be added back in a future
refactor.

## Mechanism

Uses Mori's `IOEngine` / `MemoryDesc` API for zero-copy RDMA transfers.

- Data Plane: RDMA (InfiniBand/RoCE) with managed memory pool.
- Control Plane: ZMQ for pull-request handshake and async completion.

## Installation

See the [Mori repository](https://github.com/ROCm/mori) for installation instructions.

## Configuration

Mori is configured through the new deploy-config schema (see
[`docs/configuration/stage_configs.md`](../../../configuration/stage_configs.md)).
Define the connector at the top level of the deploy YAML and reference it
by name from each stage's `input_connectors` / `output_connectors`:

```yaml
connectors:
  mori_connector:
    name: MoriTransferEngineConnector
    extra:
      host: "auto"
      zmq_port: 50051
      device_name: ""
      memory_pool_size: 536870912
      memory_pool_device: "cuda"

stages:
  - stage_id: 0
    output_connectors:
      to_stage_1: mori_connector

  - stage_id: 1
    input_connectors:
      from_stage_0: mori_connector
```

A ready-to-run intra-node example for Qwen3-Omni-MoE on AMD MI300X lives
at
[`vllm_omni/deploy/qwen3_omni_moe_mori_intranode.yaml`](../../../../vllm_omni/deploy/qwen3_omni_moe_mori_intranode.yaml)
and can be loaded with:

```bash
vllm-omni serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --log-stats \
    --deploy-config vllm_omni/deploy/qwen3_omni_moe_mori_intranode.yaml
```

The yaml wires `MoriTransferEngineConnector` (with `backend_type: xgmi`) to
the chunk_transfer_adapter path (`async_chunk: true`) so stage-to-stage
hidden-state and codec-frame streams ship GPU-to-GPU over AMD Infinity
Fabric instead of SHM.  Qwen2.5-Omni + Mori is not yet functional on
the chunk path: it needs `thinker2talker_async_chunk` /
`talker2code2wav_async_chunk` input processors that do not exist yet
(the orchestrator-level path the upstream PR originally targeted was
lost during the entrypoints → engine refactor and removed in #1742).

Parameters:

- host: local RDMA IP (`"auto"` for auto-detect).
- zmq_port: ZMQ base port for control-plane communication.
- device_name: RDMA device (e.g., `"mlx5_0"`), empty for auto-detect.
- memory_pool_size: RDMA memory pool size in bytes.
- memory_pool_device: `"cpu"` (pinned) or `"cuda"` (GPUDirect / XGMI RDMA).

For more details, refer to the
[Mori repository](https://github.com/ROCm/mori).
