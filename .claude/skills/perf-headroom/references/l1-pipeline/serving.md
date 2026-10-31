# Serving, streaming and output boundaries

Use for loaded service, first-packet latency, multi-stage streaming or an output
tail that is not explained by GPU inference. Start from the selected native
benchmark in [the execution map](../omni-execution.md).

## Collection and diagnosis

1. State arrival process, offered load, concurrency cap, input distribution,
   warmup, run duration and cache policy. Distinguish open-loop arrivals from a
   closed-loop client that only sends another request after one finishes.
2. Record offered, admitted, completed and failed requests. Report queue time,
   service time and client completion separately when observable. A success-only
   latency improvement can hide rejected work.
3. Correlate per-request stage submissions/results and per-chunk readiness. For
   streaming retain first useful packet, inter-packet gaps and final completion;
   HTTP headers or a job ID are not necessarily generated-media availability.
4. Check output D2H, shared-memory packing, serialization, encoding and client
   fetch. Verify the benchmark actually receives the payload required by the task.
5. If stages are idle while ready queues grow, inspect routing/dispatch and CPU
   work. If queues grow behind a continuously busy stage, inspect its capacity.
   If no inputs arrive, this may be upstream dependency or light offered load.
6. Where authorized, use a bounded load sweep to distinguish capacity from
   latency. Do not stress a shared live deployment without that scope.

## Omni-specific checks

The orchestrator monitor is separate from worker GPU profiling. Its queue and
dispatch signals can locate backpressure missing from a worker trace. Inspect
`docs/contributing/profiling.md` for current flags. Loop iteration percentages
depend on the orchestration mode and are not portable CPU-utilization measures.
Queue snapshots may miss bursts; use high-water/time-series evidence when available.

Async output can overlap request N's materialization with N+1's computation.
That may improve throughput without reducing N's own output tail. Attribute
benefits to the correct request and objective. Inspect replica placement and
shared CPU/transport resources before recommending more replicas.

## Handoff

Give load-conditioned throughput/latency/error results, per-stage queue and
service distributions, output boundary and capacity hypotheses. Do not feed a
single-request latency stop argument into a whole-service SLO verdict. Missing
client or queue instrumentation stays UNKNOWN with a proposed collection step.
