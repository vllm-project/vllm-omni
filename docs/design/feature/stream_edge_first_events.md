# Request-owned stream-edge first events

This core candidate for RFC #6472 defines event ownership, segment fencing,
clock treatment, snapshots, and entrypoint cleanup. It includes a synthetic
two-stage caller. Production connector instrumentation and canonical benchmark
serialization are separate integration work; enabling the collector alone does
not make workers report events.

An instrumented boundary can opt in on the existing `ClientRequestState`:

```python
from vllm_omni.metrics.stream_edge import StreamEdgeEvent, StreamEdgeKey

events = request_state.enable_stream_edge_metrics()
edge = events.start_segment(StreamEdgeKey(from_stage=0, to_stage=1))
edge.record(
    StreamEdgeEvent.PRODUCER_EMIT,
    elapsed_ms=10.0,
    clock_domain="orchestrator.segment-0",
    meaningful=True,
    chunk_seq=0,
)
```

The owner is a request instance, rather than an external-ID-indexed global map.
An edge handle is valid only for its owner and its currently active segment.
Starting a later segment fences old handles while preserving their reached
evidence in snapshots. Request finalization closes the owner and releases all
event state, including cancellation and errors. Retained handles cannot
recreate state, and a new request with the same external ID starts empty.

| Event | Boundary obligation |
| --- | --- |
| `producer_first_nonempty_emit` | A payload that advances downstream work is ready |
| `edge_first_send_complete` | That payload was successfully committed |
| `consumer_first_accept` | Decode and admission successfully committed |
| `consumer_first_nonempty_output` | The consumer produced meaningful output |

The boundary classifies meaningful payloads. Heartbeats, duplicate retries,
metadata-only payloads, and empty terminal markers must not qualify. Record
send completion and consumer acceptance only after successful outcomes. Each
event records once; duplicate observations never overwrite the first value.

Each time is a finite, non-negative elapsed millisecond value from the
request/segment origin declared by `clock_domain`. The domain name identifies
both the clock and the origin. Independent worker clocks or different segment
origins must use different names. `elapsed_between` returns a delta only for
matching domains and non-negative order; missing or incomparable events return
`None`. No raw wall-clock timestamp is exposed.

`snapshot()` returns detached JSON-compatible records keyed by directed edge
and segment, such as `0->1:0`. Missing milestones remain `null`. The snapshot
includes per-event clock domains and first producer chunk sequence, and omits
internal and external request IDs.

The existing entrypoint response builder attaches non-empty snapshots under
`OmniRequestOutput.metrics["stream_edge_metrics"]`. The API's existing
`return_stage_metrics` behavior controls detailed response exposure. Collectors
are absent by default, and existing stage-only responses stay unchanged.
Snapshots taken before cleanup retain reached evidence after the owner closes.

The CPU contract covers synthetic handoff, failed outcomes, empty markers,
interleaved requests/edges/segments, reused external IDs, cross-clock deltas,
incremental responses, and canonical cleanup. Real worker event collection and
GPU reconciliation with client first output have not been validated. No
latency, throughput, or memory improvement is claimed.
