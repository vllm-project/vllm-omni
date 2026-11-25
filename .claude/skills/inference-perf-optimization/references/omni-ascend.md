# Multi-stage audio and Ascend

Use the relevant stage checks for streaming speech. The device-specific checks below apply when the deployment uses Ascend.

| Symptom | First evidence | Investigate |
|---|---|---|
| High TTFT | Per-request queueing, encoding, Thinker prefill, and first token | Encoder graph shape coverage, prefix cache, long prefill mixed with decode |
| Normal TTFT, late first nonempty PCM | First Talker codec chunk, connector handoff, first Code2Wav chunk, client arrival | First capture, batch formation, setup, host/device transfers |
| Good first audio, poor tail or continuity | PCM gaps, playback buffer underruns, audio duration, terminal events | Talker cadence, Code2Wav queueing, final chunk, network writes |
| Throughput plateaus or memory runs low with concurrency | Stage queues, effective batch width, utilization per device, HBM/KV, preemption and recomputation | Batching, graph pools, layout, placement, or replicas at the saturated stage |
| Isolated test improves, service stays flat | Deployed kernels and graphs, actual shapes, request critical path | Existing fusion, hidden overlap, padding and copies, device or NUMA changes |

## Reproduce the serving path

- Start from the deployment configuration and benchmark CLI in the checkout. When supported, use server `--log-stats` and client `--print-stage` for attribution and measure their overhead. Benchmark HTTP turns and native realtime duplex at their own endpoints. In duplex, correlate session, turn, response, input commit, PCM, and the single terminal event. Exercise trimming, cancellation, and the next turn to check EOS, sampling history, and isolation.
- Record physical devices, NUMA placement, and stage replicas. C1 and C64 denote concurrency. On Ascend record SoC, CANN, torch-npu, vLLM-Ascend, attention backend, and effective ACLGraph path. For routing or batching changes, distinguish client in-flight requests, ingress queues, replica reservations and active requests, batch width, and actual traffic split. Check per-device peak memory and playback cadence.
- Count startup capture, first capture during measured traffic, replay, and fallback separately. Validate cache or graph behavior against fixed-seed output and relevant KV, valid rows, and slot ownership. Test replacement kernels with real weights on the deployed backend and include transitions from short to long shapes within a request.
- Define audio RTF endpoints and aggregation (mean, median, or aggregate). Check duration distribution, successful full audio, repetitions, and truncation. Use first nonempty PCM for first audio. Measure chunk gaps or buffer underruns for continuity. Break down WER/CER by language and length. Add speaker similarity, naturalness scoring, or listening tests when appropriate.
- For context, KV, or routing changes, include near-limit inputs, overlapping requests, cancellation, and immediate reuse. Count preemptions and recomputations when investigating long-request timeouts. Qualify changes to precision, sampling, and conditional flow matching (CFM) separately.
- For reference prefetching or caching, use the real reference distribution. Measure cold misses, warm hits, eviction recovery, per-device residency, and replication cost. Match codec emission, receipt, execution, PCM readiness, and client arrival for the same chunk.
- Bind functional, performance, and quality evidence to the tested commit, effective configuration, import location, dependencies, and runtime path. On migration, distinguish a functional fix from a measured speedup. When revisiting a historical item, find its original A/B and quality results and verify whether its path exists on the current stack.

## Choose the smallest useful comparison

| Candidate | First test |
|---|---|
| Thinker prefill or near-limit KV | Run the same input with known expected output alone, in a prefill-only batch, and alongside active decode. For stopping differences compare pre-sampling logits and slots. Toggle prefill, decode, and draft independently. |
| Talker K-step, narrow/wide attention, fixed lanes, or device sampling | Compare valid-row logits, codec tokens, RNG, EOS, committed KV prefixes, and rollback on the target backend. Test early-finishing rows and short/long transitions. Check padded rows cannot write real state, then measure any shift in downstream queueing. |
| Code2Wav setup, encoder, CFM, or HiFT | Fix real codec input, reference, steps, and seed. Measure cold setup, warm continuation, and terminal chunks. Include padding, copying, unpacking, pool residency, and graph updates. Use A/A to establish natural PCM variation before comparing A/B state and final audio. |
| Scheduling, routing, batching, or placement | Separate first, continuation, and terminal chunks. Hold actual in-flight load fixed and compare routing on one deployment before changing replica layout. Join request, turn, and chunk identifiers to dispatch, execution, and delivery timestamps. Check activation, shifted waits, and peak memory. |
| Graph replacement or migration | Start with real-shape isolation, then serving shadow comparisons and uninstrumented measurements. Validate coverage on held-out inputs. Recheck platform and shape guards after migration. Preserve arm-specific warmup, failures, and quality evidence, and identify opt-in settings. |
