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
- Control Plane: a ZMQ ROUTER socket atomically claims the source allocation before
  returning descriptors. Forwarded `put()` metadata includes the generation and
  endpoint; the consumer claims that generation before submitting any READ.

Transfer metadata uses schema version 1. Tensor descriptors are grouped by NIXL
memory type (`DRAM`, `VRAM`, and so on), while each descriptor retains its global
tensor index so mixed CPU/accelerator payloads can be reconstructed in order.

Payloads are not restricted to tensors. A single tensor or a list of tensors is
transferred as-is; a nested structure has its tensor leaves extracted and shipped
alongside a msgpack-encoded skeleton; anything else is msgpack-encoded into one
uint8 tensor. Mapping and sequence structure is preserved. `msgspec.Struct` values,
including `OmniPayloadStruct`, are normalised to mappings so their tensor leaves use
the direct NIXL path rather than CPU object serialisation.

Each NIXL agent uses strict thread synchronization. A consumer performs one short
metadata query per scheduler poll (10ms by default), which prevents an unavailable
producer from blocking unrelated requests. The regular transfer-completion handshake
keeps the configured transfer timeout.

## Installation

CUDA hosts can install the published wheel:

```bash
pip install nixl
```

For Intel XPU, use `docker/Dockerfile.xpu`. Its upstream vLLM XPU base image
provides a matching torch-xpu, UCX with Level Zero support, and NIXL runtime.

## Configuration

MiniMax-H3's default disaggregated and Turbo deploy configs use shared memory
for same-host transfer. To opt into NIXL, select
`vllm_omni/deploy/minimax_h3_disaggregated_nixl.yaml` or
`vllm_omni/deploy/minimax_h3_disaggregated_turbo_nixl.yaml` with
`--deploy-config`. These overlays inherit the corresponding model and
parallel settings and switch both ends of the stage edge. They require NIXL
and a working UCX backend; initialization failures do not automatically fall
back to shared memory.

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
- `zmq_port`: handshake port. If a producer omits it, an ephemeral port is opened
  for ownership claims and returned in `put()` metadata. The value in the deploy YAML is a
  base port. Purpose, replica, rank, and producer-stage offsets are added centrally,
  so colocated connectors and tensor-parallel workers receive distinct endpoints
  without one config entry per rank.
- `sender_host` / `sender_zmq_port`: consumer-side override naming the producer's
  handshake endpoint. Only needed when the consumer cannot learn it from metadata.
  In replicated deployments, the producer endpoint is carried with each request and
  selected at `get()` time; it is not mutable connector-wide state.
  Explicit sender ports are preserved, not recomputed using the receiver replica.
  Intermediate NIXL stages retain their incoming receiver configuration and a
  separate outgoing bind configuration, so one worker can receive and publish.
- `backends`: NIXL backends to register memory with. Defaults to `["UCX"]`.
- `receive_device`: forces where received tensors land. By default the consumer
  keeps the producer's device *type* but uses its own current device of that type.
- `memory_type`: overrides the NIXL memory type, otherwise `DRAM` for CPU tensors
  and `VRAM` for accelerator tensors.
- `lease_seconds`: how long a `put()` payload stays registered while waiting to be
  read (default 3600). The consumer reports completion, so this only bounds payloads
  nobody has claimed. An internal reaper releases only unclaimed registrations.
  `VLLM_OMNI_NIXL_LEASE_S` overrides it.
- `transfer_timeout_s`: how long a `get()` waits for its `READ` to complete
  (default 300). NIXL 1.3 has no transfer cancellation API, so a timed-out transfer's
  buffers and registrations remain owned by the connector until NIXL reports a
  terminal state. `close()` polls remaining transfers once and returns without
  releasing active DMA resources. A serialized background closer automatically
  finishes cleanup after completion; no second call is required.
  `VLLM_OMNI_NIXL_XFER_TIMEOUT_S` overrides it.

### Diffusion worker integration

Diffusion initializes `OmniConnectorModelRunnerMixin` in synchronous mode,
borrowing its KV transfer manager's lazy connector. This mode creates no second
connector, starts no receive/save threads, and leaves KV key builders and
sharding callbacks unchanged. Connector lifetime remains with the manager.

The shared `recv_stage_payload` entry point and asynchronous `_poll_single_request`
use the same key and endpoint metadata builder. Diffusion passes its actual
incoming edge and external request ID; explicit transfer handles take precedence.
Sender endpoints are passed per request, without mutating KV sender state.
Diffusion-specific prompt merging, output handles and device placement live in
`diffusion/worker/stage_payload.py`.

Diffusion supplies ordered TP and SP groups to the shared payload fanout.
Only the rank leading both groups reads the connector. Each subsequent broadcast
uses its own group-local rank zero, including ranks that received the payload
from the preceding group. Tensor payloads use tensor-dictionary broadcasts rather
than serializing the full payload as a Python object. Receive misses still reach
the same collectives before falling back to the inline prompt.

`GroupCoordinator.broadcast_tensor_dict` preserves the group-local source for
`broadcast_object` while passing the mapped global source to PyTorch tensor
collectives. For subgroup `[1, 3]`, local source `0` maps to global source `1`;
passing `1` back to `broadcast_object` would incorrectly select global rank `3`.
The nonzero-subgroup regression covers both sender and receiver behavior.

### Source ownership and failure limits

Claims and expiry use the same lock. Completion identifies both a payload
generation and a unique READ claim; duplicate or stale ACKs cannot release a
different reader or replacement payload. Replacing a claimed key is rejected.
Terminal errors also release claims; timeout, unknown status, and active sibling
transfers retain the complete destination bundle and source claim until terminal.
Zero-element tensors retain their shape/dtype/skeleton positions but never create
DMA descriptors or registrations.

After a lost metadata response, retries on the same consumer thread reuse the
claim ID for that endpoint, key and requested generation until a reply arrives.
This makes claim acquisition idempotent; subsequent independent reads still get
distinct claims. Callers must retry on the same thread to recover that ownership.
A metadata query abandoned without retry, lost completion ACK, or abandoned
consumer can retain a claim indefinitely. NIXL 1.3 cannot prove remote cancellation, so neither TTL nor
`cleanup()` frees those allocations. Producer `close()` rejects new work and
retains its agent, listener and claimed allocations; the background closer finishes
teardown after claims drain. Permanently abandoned claims remain until process
exit. A consumer with unfinished local transfers likewise retains a strong reference
to its connector, agent, tensors and registrations after `close()` returns. Its
background closer polls and releases resources only after a terminal state;
permanently stuck transfers retain resources until process exit. Closing connectors
reject new work and report unhealthy. This avoids an unbounded transfer-polling loop
in shutdown without freeing DMA-owned memory. Only trusted
peers may access this unauthenticated control plane. Both endpoints must use the
claim-aware protocol; rolling compatibility with older GET_META clients is not
provided. Legacy externally-owned metadata without a generation is accepted only
under the caller's lifetime guarantee, not managed as a leased allocation here.

## Validation

The native CUDA smoke test requires NIXL and at least two visible CUDA devices:

```bash
UCX_RCACHE_MAX_UNRELEASED=1024 pytest -o addopts='' -q -s \
  tests/distributed/omni_connectors/test_nixl_connector_native.py
```

It starts independent producer and consumer processes, loads the real UCX backend,
performs the ZMQ `GET_META` handshake and a native NIXL `READ` from GPU 0 to GPU 1,
checks mixed CPU/GPU tensor leaves from an `OmniPayloadStruct`, sends `XFER_DONE`,
and verifies that producer registrations are released. This validates same-host,
cross-GPU operation. Inter-node UCX transport is supported by the design but was not
validated by this test.

For more details, refer to the [NIXL repository](https://github.com/ai-dynamo/nixl).
