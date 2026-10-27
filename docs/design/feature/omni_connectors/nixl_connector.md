# NixlConnector

For the reusable model-runner API, illustrated control flow, deadline semantics,
and failure/ownership contracts, see [Cross-Stage Payload Transport](stage_payload.md).

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

Unclaimed payloads retain tensors without creating a native producer context.
The first claim registers storage before returning descriptors. Claimed numeric
payload keys of one request/stage prefix share a private producer agent; live
claims of different prefixes never share it. A connector may retain one fully
unregistered agent after normal completion; cancelled agents are retired.

CUDA source writes are synchronized before normalization, then any contiguous
copies finish before publication. Native READs do not inherit PyTorch stream
dependencies. By default, `put()` synchronizes each payload device. A caller
that knows the writer streams may pass a recorded `ready_event` covering all
source writes on the payload's single CUDA device. This waits for that event
instead of unrelated device work; storage must not be modified after handoff.

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
- `num_threads`: UCX worker-pool size. Defaults to `0`, NIXL's single-worker
  progress-thread mode; explicit positive values enable its worker pool.
- `receive_device`: forces where received tensors land. By default the consumer
  keeps the producer's device *type* but uses its own current device of that type.
- `memory_type`: overrides the NIXL memory type, otherwise `DRAM` for CPU tensors
  and `VRAM` for accelerator tensors.
- `lease_seconds`: how long a `put()` payload retains storage while waiting to be
  read (default 3600). The consumer reports completion, so this only bounds payloads
  nobody has claimed. An internal reaper releases unclaimed storage and any
  partial registrations left by a failed first claim.
  `VLLM_OMNI_NIXL_LEASE_S` overrides it.
- `transfer_timeout_s`: how long a `get()` waits for its `READ` to complete
  (default 300). A timeout does not prove DMA has stopped. Buffers and registrations
  remain owned until native handles can be released. `close()` polls remaining
  transfers once and returns without releasing active DMA resources. In-flight
  `get()` calls retain the completion-ACK channel until their READs and unresolved
  metadata claims have been cleaned up. A serialized background closer retries
  cleanup; no second call is required.
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
the same collectives. Inline fallback is allowed only when all required payload
keys remain present; otherwise each rank raises an explicit payload error.

The synchronous path passes one monotonic deadline to `get_with_deadline()` for
discovery and DMA completion waits, rather than calling the ordinary `get()`
with its potentially much longer transfer timeout. Native calls, serialization,
and distributed collectives are not preemptible by this deadline.

`GroupCoordinator.broadcast_tensor_dict` preserves the group-local source for
`broadcast_object` while passing the mapped global source to PyTorch tensor
collectives. For subgroup `[1, 3]`, local source `0` maps to global source `1`;
passing `1` back to `broadcast_object` would incorrectly select global rank `3`.
The nonzero-subgroup regression covers both sender and receiver behavior.

### Source ownership and failure limits

Claims and expiry use the same lock. Completion identifies both a payload
generation and a unique READ claim; duplicate or stale ACKs cannot release a
different reader or replacement payload. Replacing a claimed key is rejected.
The final ACK retains its claim and source ownership until deregistration succeeds.
Failed release unpublishes metadata; replaying the same ACK retries remaining
registrations without accepting another READ or releasing a replacement generation.
Terminal errors also release claims; timeout, unknown status, and active sibling
transfers retain the complete destination bundle and source claim until terminal.
Zero-element tensors retain their shape/dtype/skeleton positions but never create
DMA descriptors or registrations.

After a lost metadata response, retries on the same consumer thread reuse the
claim ID for that endpoint, key and requested generation until a reply arrives.
This makes claim acquisition idempotent; subsequent independent reads still get
distinct claims. Callers retry on the same thread and call `abandon_get()` when
giving up. Synchronous stage payload receive does this automatically. Background
recovery reacquires the same claim solely to acknowledge it, without issuing a
READ. Lost completion ACKs are retried against the exact payload generation.
Neither TTL nor ordinary `cleanup(key)` frees a claimed source allocation.
After publication has stopped, `cleanup_prefix()` removes that prefix's metadata
and destroys its private native contexts before releasing source tensors. This
also handles a lost remote READ owner without resetting another request's agent.
Retirement detaches the contexts under the state lock, destroys them outside that
lock while retaining source storage, then finalizes ownership under the lock.
Concurrent cleanup waits for the existing operation; key reuse is rejected until
retirement finishes. Independent publication, GET and ACK can continue.
Producer `close()` applies the same ordering to all pending contexts.
`wait_for_cleanup()` additionally waits for matching local READs and deferred
claims; a deadline reports failure while retaining resources still owned locally.
Native context destruction is synchronous and has no universal hard deadline.

After all stages finish, the orchestrator's existing best-effort release RPC
reaches the native MRv2 worker, rather than the scheduler's V1 chunk adapter.
The worker stops remaining publication under its output lock, then retires the
request's unconsumed prefixes outside that lock. Late output batches cannot
publish after request state is removed. The release is also enabled for native
full-payload stages; the V1 release path is unchanged.

Stage admission rejects reuse of a stage request ID with a request-scoped 409
error while its previous release
is queued, in flight or unconfirmed on any replica. A successful release permits
reuse. RPC failures and timeouts leave that ID fenced until the stage pool is
restarted: their outcome is unknown, so they are not automatically retried against
a potentially newer admission. Other request IDs can still enter and be released.
This applies to direct engine admissions; `AsyncOmni.generate()` already assigns
a unique stage request ID to each call, even when the user's ID is reused.

The lease reaper owns expiry cleanup. A failed expired release retains its source
ownership, keeps metadata unpublished and retries after a 100 ms backoff. Failure
of one payload does not stop the reaper or reject another request's publication.

Receive cleanup releases handles, descriptor lists, remote agents and registrations
in that order before acknowledging the source. Failed releases retain the remaining
bundle for retry. Overlapping GETs sharing a producer agent each hold a remote-agent
lease; only the final owner disconnects it, including when a READ is deferred.
A consumer with unfinished local transfers retains a strong
reference to its connector and resources after `close()` returns. Permanently
stuck local transfers can retain resources until process exit. Closing connectors
reject new work and report unhealthy. Once control threads and local READ ownership
are drained, `close()` releases its main native agent even if the connector object
remains referenced. Deferred close retains that agent until cleanup completes. Only trusted
peers may access this unauthenticated control plane. Both endpoints must use the
claim-aware protocol; rolling compatibility with older GET_META clients is not
provided. Legacy externally-owned metadata without a generation is accepted only
under the caller's lifetime guarantee, not managed as a leased allocation here.

Private contexts trade resource isolation for agent creation and backend thread
cost. Device synchronization can reduce overlap with other streams. The
[routing RFC's transport section](https://github.com/vllm-project/vllm-omni/issues/8740)
records measured costs and their limits; it does not establish general performance
parity. Native ownership validation covers NIXL 1.5 with UCX TCP/cuda_copy. Other
backend and SDK combinations require validation.

An [independent H100 review](https://github.com/vllm-project/vllm-omni/pull/8744#pullrequestreview-5483707044)
of commit `7ec25991` measured publication of ready 32 KiB tensors with 64 retained
prefixes: put P95 increased from 0.238 ms on the base to 1554.224 ms, with 315
additional threads. This fixture performs no payload READ or model inference.
Lazy registration avoids native contexts for unclaimed prefixes and permits
sequential claims to reuse the idle agent. Cold claims still pay context creation;
concurrent claims of different prefixes still require separate contexts. The
default CUDA fence remains conservative; the optional recorded-event contract
does not assume that the current stream owns all writes.

At the earlier `9ffc6136` candidate, four matched A6000 runs of 64 ready 32 KiB payloads include actual claim/READ/ACK
and exact-value checks. Against the previous request-isolated implementation,
put+READ+ACK P95 falls from 559.85–661.52 ms to 3.80–4.40 ms. Unclaimed producer
contexts fall from 64 to 0 and added native threads from 315 to 0. The event path
avoids waiting for unrelated stream work; callers without an event retain the
device fence. These warmed, sequential connector measurements do not qualify
cold/concurrent claims or full-serving performance. The last pair covers the
immutable publication-metadata snapshot implementation.

## Validation

The mixed-device smoke requires NIXL and two visible CUDA devices. Select
`-m 'cuda and cards_1'` to run only the single-GPU lifetime regressions.

```bash
UCX_RCACHE_MAX_UNRELEASED=1024 pytest -o addopts='' -q -s \
  -m cuda --run-level=advanced_model \
  tests/distributed/omni_connectors/test_nixl_connector_native.py
```

It starts independent producer and consumer processes, loads the real UCX backend,
performs the ZMQ `GET_META` handshake and a native NIXL `READ` from GPU 0 to GPU 1,
checks mixed CPU/GPU tensor leaves from an `OmniPayloadStruct`, sends `XFER_DONE`,
and verifies that producer registrations are released. This validates same-host,
cross-GPU operation. Inter-node UCX transport is supported by the design but was not
validated by this test.

One-GPU regressions inject transient/persistent deregistration failures into the
real SDK and gate native-owner destruction to check independent request progress.
The two-GPU live-READ test requires an actual `PROC` observation immediately before
prefix cleanup, then verifies survivor delivery, key reuse and resource release.
If the READ already completed, the test skips rather than claiming active-READ
cancellation coverage. These tests validate the selected local transport; cross-node
RDMA cancellation needs a separate run with the revised source.

Active READ retirement with NIXL 1.5 / UCX 1.23 over TCP has emitted UCX
`ucp_requests` pool warnings. Released connector records and successful survivor
READs do not establish complete native backend cleanup; this remains a validation gap.
The NIXL PEER force-close change alone does not close it: a 100 ms delayed,
still-`PROC` 512 MiB inter-node READ reproduces warnings on both peers. An isolated
NIXL error-request-release fix and UCX internal-request auto-release fix are
dependency candidates, not changes to the installed SDK. Local GPU READ/close
and live retirement checks pass with those candidates. A local two-GPU TCP run
also reaches `PROC` then `NIXL_ERR_REMOTE_DISCONNECT`: storage is released,
survivor delivery and key reuse pass, and neither peer logs pool warnings.
This covers one error-completion run with isolated native dependencies, not a
fix shipped by this PR. READs which finish by draining do not cover that path.
The failing inter-node case and RDMA still require separate qualification;
RDMA hardware is unavailable.

### Publication and synchronization fixture

Run on Linux from the repository root with an idle CUDA GPU and the same source, runtime
and UCX settings for each comparison:

```bash
: "${CUDA_VISIBLE_DEVICES:?Select an idle GPU}"
UCX_TLS=tcp,cuda_copy,self UCX_RCACHE_MAX_UNRELEASED=1024 \
  python benchmarks/distributed/omni_connectors/benchmark_nixl_publication.py \
  --prefixes 64 --output publication.json
```

This fixture publishes ready 32 KiB tensors with retained independent prefixes,
then performs sequential native READs and checks exact tensor values. It reports
put and claim/READ/ACK samples/P95, their summed latency, context count and
thread/RSS changes. The unrelated-stream probe measures publication only and
uses synthetic completion ACKs. Add `--ready-event` to exercise the explicit
readiness contract. This is not a serving benchmark; JSON records the connector
hash and runtime settings.
Use separate output paths; existing artifacts are not overwritten.

### Cross-node active-READ cancellation fixture

Use the same source and runtime on two nodes, with one idle GPU on each. Start
the producer with its reachable IPv4 address:

```bash
: "${CUDA_VISIBLE_DEVICES:?Select an idle GPU}"
: "${NIXL_PRODUCER_HOST:?Set the reachable producer address}"
python benchmarks/distributed/omni_connectors/cross_node_nixl_cancellation.py \
  producer --host "$NIXL_PRODUCER_HOST" --endpoint endpoint.json \
  --output producer.json
```

On the consumer, set the ports from the producer's `endpoint.json`:

```bash
: "${CUDA_VISIBLE_DEVICES:?Select an idle GPU}"
: "${NIXL_PRODUCER_HOST:?Set the reachable producer address}"
: "${NIXL_PORT:?Set the producer metadata port}"
: "${NIXL_CONTROL_PORT:?Set the producer control port}"
python benchmarks/distributed/omni_connectors/cross_node_nixl_cancellation.py \
  receiver --host "$NIXL_PRODUCER_HOST" --port "$NIXL_PORT" \
  --control-port "$NIXL_CONTROL_PORT" --output receiver.json
```

The consumer observes a real 512 MiB native READ in `PROC` before requesting
source retirement. It then checks destination cleanup, an independent request
and same-key/new-generation reuse with exact tensor values. A READ that already
completed cannot pass the active-cancellation check.

Set `UCX_TLS` and `UCX_NET_DEVICES` for the intended transport on both nodes and
capture their complete process logs with `UCX_LOG_LEVEL=info`. TCP/cuda_copy is
not RDMA evidence. An RDMA result also requires a usable RDMA link on both peers
and UCX logs showing the selected RMA lane; configured environment variables
alone do not establish the transport or GPUDirect support. A zero process exit
and empty connector records do not override UCX request-pool warnings.

These connector fixtures do not replace full-model serving qualification. Local
model checks cover normal completion, public cancellation after audio output,
sequential ID reuse and shutdown. Native error-cancel cleanup and real cross-node
RDMA cancellation still require separate validation; draining a READ is not proof
of error-cancel cleanup.

For more details, refer to the [NIXL repository](https://github.com/ai-dynamo/nixl).
