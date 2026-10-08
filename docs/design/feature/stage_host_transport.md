# Local stage host transport

`SharedMemoryConnector` uses a persistent host ring for ordinary local stage
payloads by default. Models keep the existing `put`/`get` and typed payload
contracts; no model-specific sender, receiver or deployment change is required.
This applies to both V1 and MRV2 paths using this connector, including CPU codec
codes and CPU hidden states. Remote connectors use their existing transports.

Each producer process and destination stage lazily allocates a 4 MiB ring.
Small CPU tensor trees use a msgpack header and raw tensor regions. Plain
serialized payloads can use the same ring. Frames larger than a quarter of the
ring, and frames arriving while it is full, use the existing per-key SHM path.
The fallback preserves delivery without blocking a model step on ring capacity.
Large or unsupported tensor trees do not allocate an unused contiguous snapshot.

## Ownership and delivery

The producer copies the payload before `put` returns. Readers hold the ring's
file lock while copying into independently owned CPU storage, then mark the
frame consumed. Multiple receiving processes may compete for a key; one reader
claims it. Existing TP fanout remains responsible for distributing that owned
payload to peer ranks. This transport does not provide multicast.

Small tensor leaves share one owned allocation. Large multi-leaf messages use
separate allocations, so retaining small metadata does not retain a large
hidden state. No view of ring storage escapes the receive boundary.

Cancellation retires exact keys or tracked numeric chunk suffixes. Replacing
an unclaimed key retires its previous frame, including transitions between
the ring and per-key SHM. Descriptors include monotonic frame positions; an old
descriptor cannot read a newer publication after wraparound. Keys must have
one active publisher, as assumed by the existing stage request routing.

Only producers register ring allocations with Python's shared-memory resource
tracker. Readers map files without taking ownership. Producer process identity
includes its start time, and available Linux pidfds detect producer exit, so
old frames cannot be delivered after a producer restart. Normal close removes
owned allocations and discovery markers. Resource tracking covers allocation
cleanup when the owning application exits; an abnormal exit can leave small
discovery markers, which contain no payload data.
Live receivers release cached mappings after a producer exits and prune stale
markers during discovery within their deployment scope.

## Configuration and observation

Connector extras are shared by the endpoints of an edge. For example:

```yaml
connectors:
  shm:
    name: SharedMemoryConnector
    extra:
      host_ring_bytes: 4194304
```

Set `host_ring_bytes: 0` at both endpoints to use the per-key path, including
controlled comparisons or deployments with older receivers. Nonzero capacities
must be between 4 KiB and 64 MiB. All ring endpoints must support this protocol.
The launcher supplies a common deployment scope; discovery scans only that
scope's directory. Receive attempts do not wait for another process's file lock.
An unchanged empty channel skips its frame lock. Channel registration and
retirement advance a serialized discovery stamp, so empty polling does not
rescan the directory and new producers remain discoverable without a timer.
Ring endpoints must share a Linux PID namespace. Deployments that share SHM
across different PID namespaces must use `host_ring_bytes: 0`.

`health()` exposes `host_ring_puts`, `host_ring_gets` and `host_ring_fallbacks`
alongside the existing counters. Ring setup failure logs once per edge and uses
per-key SHM. CPU payload size and the cost of serialization determine the benefit;
reduced transfer latency does not imply higher model throughput.

## Scope

This mechanism reduces host serialization copies and per-message segment work.
It does not retain worker outputs on the GPU, allocate a CUDA arena, or replace
GPU stream synchronization. It also does not impose a global byte budget on
queued snapshots, downstream model caches or client audio delivery. The bounded
ring with a compatibility fallback is not end-to-end streaming backpressure.

Regression tests are in
`tests/distributed/omni_connectors/test_shm_ring.py`, collected by the existing
`core_model and cpu` CI lane. They cover ownership, tensor types, typed payloads,
pressure fallback, cancellation, overwrite, deadlines, competing processes and
producer restart.
