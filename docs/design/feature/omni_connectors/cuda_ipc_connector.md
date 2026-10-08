# CudaIpcConnector

## When to Use

Same-node stage pipelines whose payloads carry CUDA tensors at or above
`inline_tensor_bytes` (64 KiB by default), on one GPU or across peer-visible
GPUs. It behaves exactly like `SharedMemoryConnector` for everything else
(host tensors, small codec chunks), so it is a drop-in per-edge upgrade.

Not usable when the consumer cannot see the producer's GPU (e.g. the stage
processes were launched with disjoint `CUDA_VISIBLE_DEVICES`); set
`use_ipc: false` there to get framed host staging on the same code path.

## How It Works

The SHM control plane of `SharedMemoryConnector` is inherited unchanged
and the base class itself is untouched by this connector: every framing,
IPC, tensor-tree codec and transport-snapshot mechanism lives in
`cuda_ipc_connector.py`, and a payload without a large tensor is written
and read by the base class byte-for-byte.
`put()` still creates a `/dev/shm` segment, returns its handle in
metadata, and ownership is claim-by-unlink. What changes is the data
plane for large CUDA tensors: instead of copying their bytes into the
segment, the
producer exports each tensor as a CUDA IPC handle
(`torch.multiprocessing.reductions.reduce_tensor`) and the segment only
carries the tensor-free skeleton plus dtype/shape descriptors and the
handle. The consumer maps the handle with `rebuild_cuda_tensor`,
copies onto its own device and drops the mapping.

Producer side (`_export_tensor`):

1. Clone the tensor (`reduce_tensor` shares the whole allocation, and the
   source may be a CUDA-graph output or KV block reused as soon as `put()`
   returns; the clone also records the CUDA event the consumer waits on).
2. `reduce_tensor` → its args as plain msgpack values (no pickle: the
   class and dtype slots are restored by the consumer); the descriptor
   carries the producer's GPU UUID so the consumer can locate the device.
3. Failure to export (expandable segments, pluggable allocators) logs a
   warning once and falls back to copying the bytes into the segment.

Consumer side (`_import_tensors`):

1. Map the export's GPU UUID to a local device index via
   `CUDA_VISIBLE_DEVICES`; a missing GPU raises with the remedy (make it
   peer-visible or `use_ipc: false`).
2. `rebuild_cuda_tensor` on a side stream per source device, copy into a
   fresh tensor on the connector's receive device, synchronize, unmap.
3. A process that never initialized CUDA (TP>1 scheduler) receives host
   copies; device transfers need the consumer's CUDA context.

## Lifetime and Ownership

- One export maps to exactly one rebuild + release. Claiming the segment
  makes a reader the owner of every handle in it: a mapping returns its
  clone when dropped; the rest are released via
  `UntypedStorage._release_ipc_counter_cuda`.
- Unread payloads (`cleanup`, TTL reaping) release the counters on the
  producer, rate-limited `torch.cuda.ipc_collect()` (once per second)
  returns the clones.
- A consumer killed with SIGKILL after mapping: its context teardown
  drops the mapping, `ipc_collect` returns the clone.
- A producer killed before the consumer maps: the consumer's `get()`
  either succeeds off the still-live allocation or reports the payload
  gone; the consumer's CUDA context stays usable (hardware-tested).

## Configuration

```yaml
connectors:
  connector_of_shared_memory:
    name: CudaIpcConnector
    extra:
      inline_tensor_bytes: 65536  # tensors below stay inline (msgpack)
      use_ipc: true              # false = framed SHM host staging
      # MiniCPM-o 4.5 only, see the next section.
      thinker_talker_handoff: true
      thinker_talker_handoff_ttl_s: 600
```

`inline_tensor_bytes` defaults to 64 KiB here. `SharedMemoryConnector`
has no framing code at all and keeps serializing every payload with
msgpack, so edges that do not name `CudaIpcConnector` are unchanged;
sub-threshold payloads are byte-identical on both connectors.

## Thinker → Talker Handoff (MiniCPM-o 4.5)

The MiniCPM-o 4.5 stage 0→1 boundary is not a connector edge: `llm2tts`
runs in the orchestrator process and its output *is* the Talker's
engine-core request. The Thinker hidden states therefore used to be
converted to nested float lists and shipped inside the msgpack request,
which costs the orchestrator ~100 ms of GIL-bound time per reply.

`thinker_talker_handoff: true` on the Talker connector's `extra` (set by
`minicpmo_4_5_ipc.yaml`; for the two-GPU layout use the same overlay with
`base_config: minicpmo_4_5_2gpu.yaml`) moves those
tensors onto the connector instead. The same connector spec is
instantiated on both sides of the boundary and used as a key-addressed
mailbox (`vllm_omni/model_executor/models/minicpmo_4_5/talker_handoff.py`):

- **Producer** (`llm2tts`, orchestrator process): `put()` the float32
  hidden slice under `mcpo45_handoff_<request_id>_<n>` synchronously,
  then store `{"__minicpmo45_connector_handoff__": {"key", "metadata"}}`
  under `hidden_states.tts` in `model_intermediate_buffer`. `n` is a
  process-wide counter, so the duplex path, which re-puts once per Talker
  condition, never overwrites an unread payload. The put is ~0.2 ms and
  happens before the request is built, so the payload always exists
  before the Talker can ask for it.
- **Consumer** (Talker model, stage-1 worker): at the request's first
  prefill `get()` the key, claim the segment, and write the tensor back
  into the runner's per-request buffer so later prefill chunks reuse it.
  With TP>1 the local rank 0 claims and broadcasts to the other ranks.
- **Fallback**: without the option (every default deploy file), or when
  `put()` fails, the list handoff is produced exactly as before, and a
  Talker that receives a list keeps its existing path. `set_ref_audio`'s
  waveform is small and stays a list.

The option lives on the Talker's own connector spec rather than on a
declared `0->1` edge on purpose: a real edge would give the Talker a
`receiver` role (and the Thinker a `sender` role) in `stage_connector_config`,
which re-routes the existing Talker → Code2Wav transport and the
Thinker's output path.

Ownership is claim-by-unlink, as everywhere in the SHM family. The
producer unlinks by key on `cleanup(request_id)` / `close()`, and any
handoff still unread after `thinker_talker_handoff_ttl_s` (600 s by
default; keep it longer than the longest Talker admission wait you
tolerate) is unlinked on the producer's next `put()`, so an aborted
request cannot leak `/dev/shm`. Hidden states are host tensors in the
orchestrator, so no CUDA IPC export takes place on this edge: the
connector's framed host path is used (or plain msgpack when the edge
names `SharedMemoryConnector`) and the Talker copies to its GPU as it
did for lists.

## Notes

- Same-GPU handoff costs one on-device copy (the producer's clone → the
  consumer's tensor), ~0.03–0.1 ms at typical sizes; there is
  intentionally no zero-copy view, because it would tie consumer lifetime
  to the producer and a producer crash could poison the consumer context.
- Cross-GPU requires the producer's GPU to be visible to the consumer
  process (`same_gpu` or peer visibility); the copy then runs as a P2P
  transfer when the pair allows it.
- `health()` reports live `ipc_exports` and `ipc_export_bytes`.
- CPU tests cover handle ownership through injected
  `_reduce/_rebuild/_release/_uuid` seams (`test_cuda_ipc_connector.py`);
  two-process round trips, the 10k-transfer leak soak and the SIGKILL
  cases are `@hardware_test`s run on the cluster. The Thinker → Talker
  handoff (producer put, marker, consumer get, abort/TTL cleanup, list
  fallback) is covered on CPU in
  `tests/model_executor/models/minicpmo_4_5/test_talker_handoff.py`.
