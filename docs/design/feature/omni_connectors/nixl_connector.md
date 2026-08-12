# NixlConnector

## When to Use

Multi-node or intra-node stage transfer over NIXL, which brokers RDMA/shared-memory
transports through a single API. Useful when the deployment already standardises on
NIXL for KV transfer, or when the target accelerator has no Mooncake/Mori backend --
NIXL reaches Intel XPU through UCX's Level Zero support.

## Mechanism

Uses vLLM's `NixlWrapper` (`vllm.distributed.nixl_utils`) to register the producer's
tensors and let the consumer pull them with a NIXL `READ`.

- Data Plane: NIXL agent-to-agent `READ`, GPU-to-GPU where the backend allows it.
- Control Plane: either the caller forwards the metadata returned by `put()`, or --
  when `zmq_port` is set -- a ZMQ ROUTER socket serves it to the consumer by key.

Payloads are not restricted to tensors. A single tensor or a list of tensors is
transferred as-is; a nested structure has its tensor leaves extracted and shipped
alongside a msgpack-encoded skeleton; anything else is msgpack-encoded into one
uint8 tensor. The consumer reassembles the original object.

## Installation

CUDA hosts can install the published wheel:

```bash
pip install nixl
```

Intel XPU needs NIXL built against a Level-Zero-enabled UCX. The image build does
this via `docker/build_ucx_wheel.sh` and `docker/build_nixl_wheels.sh`.

## Configuration

```yaml
connectors:
  nixl_connector:
    name: NixlConnector
    extra:
      host: "auto"
      zmq_port: 50061
      backends: ["UCX"]

stages:
  - stage_id: 0
    output_connectors:
      to_stage_1: nixl_connector

  - stage_id: 1
    input_connectors:
      from_stage_0: nixl_connector
```

Parameters:

- `host`: address the producer binds its handshake socket to (`"auto"` to detect).
- `zmq_port`: handshake port. Omit it when the pipeline forwards `put()`'s metadata
  itself, in which case no socket is opened. Stages colocated on one host each need
  a distinct port.
- `sender_host` / `sender_zmq_port`: consumer-side override naming the producer's
  handshake endpoint. Only needed when the consumer cannot learn it from metadata.
- `backends`: NIXL backends to register memory with. Defaults to `["UCX"]`.
- `receive_device`: forces where received tensors land. By default the consumer
  keeps the producer's device *type* but uses its own current device of that type.
- `memory_type`: overrides the NIXL memory type, otherwise `DRAM` for CPU tensors
  and `VRAM` for accelerator tensors.
- `lease_seconds`: how long a `put()` payload stays registered while waiting to be
  read (default 3600). The consumer reports completion, so this only bounds payloads
  nobody ever reads. `VLLM_OMNI_NIXL_LEASE_S` overrides it.
- `transfer_timeout_s`: how long a `get()` waits for its `READ` to complete
  (default 300). `VLLM_OMNI_NIXL_XFER_TIMEOUT_S` overrides it.

For more details, refer to the [NIXL repository](https://github.com/ai-dynamo/nixl).
