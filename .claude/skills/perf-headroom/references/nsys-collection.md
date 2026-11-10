# Nsight Systems collection and timeline interpretation

## Native entry and scope

Inspect `docs/contributing/profiling.md` and the selected runner at the assessed
commit. Omni's diffusion workers control `start_profile()` / `stop_profile()`
through `profiler_config`; a trace of only the parent can miss actual GPU work.
For AR/multi-stage requests resolve stage-specific profiling configuration.

Start from a bounded warmed request and expand only for missing processes, ranks
or dependencies. Separate startup/capture overhead unless cold start is the goal.
Save exact tool/version, runner/config, capture range and stderr with the report.

## Command construction

Query installed help first. The repository's CUDA-profiler capture recipe uses:

```text
nsys profile
  --trace-fork-before-exec=true
  --cuda-graph-trace=node
  --capture-range=cudaProfilerApi
  --capture-range-end=repeat
  -o <new campaign report prefix>
  <approved runner with profiler_config.profiler=cuda>
```

This is a command sketch: resolve runner arguments, shell quoting, stage/rank
placement and output path before execution. Use `--trace=cuda,nvtx,osrt` and
targeted CPU sampling/context-switch collection when supported and needed for
host diagnosis. More tracing adds overhead and may require separate permissions.
Ensure start/stop reached the intended workers and artifacts finished flushing.

For a recorded report, inspect available stats reports and export options using
installed `nsys stats --help` / `nsys export --help`. Useful report families include
CUDA kernel, API, launch/exec and NVTX summaries. Preserve `.nsys-rep` as raw evidence
alongside CSV/SQLite exports. Re-exporting a trace is not a fresh capture.

## Coverage audit before arithmetic

Record all expected processes, ranks/devices, request/phase ranges, graph replays
and event classes. Compare expected versus observed step/node counts. Inspect
dropped-event warnings and truncation; a successful process exit is not enough.

If graph envelopes exist without node events, internal activity/gaps are unknown.
If a rank is missing, distributed completion is unknown. If the request tail is
outside capture, full E2E attribution is unknown. A perfectly closed ledger of
incomplete events does not resolve these gaps.

## Manual interpretation

1. Locate request/phase ranges and distinguish host range from device completion.
2. Correlate launches using process/thread/context/stream/correlation identity,
   not name or timestamp adjacency alone.
3. Inspect API time, queue time and device execution separately. A launch may
   execute before its host API returns; the phases are not always additive.
4. Union device activity for a chosen window using node events; preserve overlap,
   unclassified activity and head/tail separately from internal gaps.
5. For expensive regions, map shapes/call counts with a separate targeted torch
   trace or source assertions. Shape/stack instrumentation can perturb scheduling.
6. Follow exposed gaps into L2/L5 and high-work regions into L3/L4.

The bundled `trace` command accepts torch Chrome JSON, not raw Nsys SQLite. When
using SQLite, inspect that export's schema and units before building a run-local
extractor. Preserve graph-node and correlation identity; do not silently guess
table columns or mix timestamps from unrelated clocks. Version and test the
extractor with small known intervals and count checks before trusting totals.

## Output

Provide capture manifest, collection completeness audit, semantic ranges,
per-device ledger, kernel population and unresolved dependencies. Link the
normal baseline to describe perturbation, but do not force diagnostic totals
to match the unprofiled median.

[Official analysis guide](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html)
defines report timing columns. The [timing guide](l1-pipeline/timing-and-overlap.md)
explains why sums and overlap are not direct E2E savings.
