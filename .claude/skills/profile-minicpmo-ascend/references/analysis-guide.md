# MiniCPM-o Ascend Profile Interpretation

## Evidence to Hypothesis

| Signal | Likely mechanism | First code surface | Candidate class |
| --- | --- | --- | --- |
| Long gap before Stage 0 kernels | media preprocessing, scheduling, or graph miss | Thinker wrapper and NPU runner | preprocessing/configuration |
| Frequent synchronize APIs | host-device scalar reads, copies, or explicit waits | stage bridge and request loop | synchronization removal |
| High <=50 us kernel ratio | fragmented shapes or repeated small operations | owning stage model path | fusion, caching, batching |
| Stage 1 idle before Stage 2 | connector polling or codec chunk threshold | stage input processor and connector | queue/chunk scheduling |
| Stage 2 compute exceeds audio duration | Flow/CFM/HiFT throughput | batched Token2Wav | dtype/operator/graph |
| Stage 2 queue grows at concurrency | shape buckets or request-state fragmentation | Code2Wav batching | batch compatibility |
| High HBM with low utilization | oversized caches/buffers or placement pressure | deploy config and stage state | capacity/memory tuning |
| High Cube time/utilization | matrix compute bound | proven hot operator | supported kernel/precision |
| High MTE or memory traffic | layout conversion, copies, or bandwidth bound | NPU patch/model adapter | layout/cache changes |

## Interpretation Rules

- Aggregated kernel/operator time can exceed request wall time when streams overlap.
- Different profiler presets or stage selections are not timing-comparable.
- A profiler trace is not a performance benchmark; profiling overhead changes latency and scheduling.
- Operator names identify symptoms, not ownership. Confirm the stage and call path in the timeline.
- A high call count is actionable only when its cumulative time, queue effect, or launch overhead is material.
- Runtime API time may overlap device work. Inspect the timeline before adding API and kernel totals.
- Memory growth across increasing concurrency may be legitimate shape/cache warmup. Confirm a fixed-shape stability window and post-shutdown release before calling it a leak.

## First-Capture Order

For audio latency under concurrency:

1. Capture Stage 2 for one warmed text+audio request.
2. Inspect top kernels, synchronization APIs, small-kernel ratio, and first audio timing.
3. Capture Stage 1 only if Stage 2 begins late or lacks enough work.
4. Capture all stages only if the connector/queue gap remains ambiguous.
5. Choose one bounded candidate, then run unprofiled A/B/A before comparing traces.
