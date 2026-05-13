# Prometheus Metrics Design

This document describes how vLLM-Omni exposes Prometheus metrics for
multi-stage pipelines, the constraints that shaped the design, and how
the pipeline-level metrics coexist with upstream vLLM per-engine
metrics.

## Objectives

- Expose pipeline-level request and latency metrics that span the full
  multi-stage execution (orchestrator scope).
- Preserve all upstream vLLM per-engine metrics (`vllm:*`) for stages
  backed by an AR LLM engine, and reshape their `engine` label into
  `stage` + `replica` so multi-replica deployments gain per-replica
  visibility automatically.
- Expose per-stage diffusion timing breakdowns for pipelines that
  include a diffusion engine.
- Expose per-modality SLO metrics (audio TTFP / RTF / duration / frames,
  image TTFP / generation time / num, video generation time) that the
  upstream `vllm:*` families do not capture (e.g. `audio_ttfp` is the
  first audio packet, distinct from upstream's first audio token).
- Expose per-replica-edge cross-stage transfer metrics so the slack
  between E2E latency and the sum of per-stage `gen_time` (queueing,
  serialization, network) becomes attributable.
- Keep the metrics collection overhead low enough that it does not
  regress TTFA or throughput.

## Background

### Upstream vLLM Metrics

Upstream vLLM defines ~37 Prometheus metric families under the `vllm:`
prefix. These are registered by `PrometheusStatLogger` and cover
engine-level state: KV cache usage, running/waiting request counts,
token throughput, TTFT, inter-token latency, e2e latency, and so on.
They are served via the `/metrics` HTTP endpoint provided by
`prometheus_fastapi_instrumentator` and the default `prometheus_client`
WSGI handler.

vLLM's `unregister_vllm_metrics()` function strips every
`prometheus_client` collector whose `_name` attribute contains the
substring `"vllm"`. This runs during engine initialization to clean up
stale collectors from prior instantiations within the same process.

### The Problem

vLLM-Omni runs multiple engine instances (stages × replicas) within a
single process, coordinated by an Orchestrator. The pipeline needs its
own metrics — aggregate request counts, end-to-end latency across all
stages, diffusion timing breakdowns, per-modality SLO signals, and
cross-stage transfer attribution — that do not exist in upstream vLLM.
All pipeline-level metrics use the `vllm:omni_` prefix to distinguish
them from upstream per-engine metrics. The `unregister_vllm_metrics()`
function is monkey-patched to a no-op at import time (see
`vllm_omni/patch.py`) so that these metrics are not destroyed during
engine initialization.

Upstream per-engine metrics retain the `vllm:` prefix but are now
registered by `OmniPrometheusStatLogger`, a thin subclass of upstream's
`PrometheusStatLogger` that reshapes the single `engine` label into a
`stage` + `replica` pair (see "OmniPrometheusStatLogger wrap" below).

## Architecture

### Component Overview

```text
                       +------------------------+
                       |  API Server (FastAPI)  |
                       |   GET /metrics         |
                       +-----------+------------+
                                   |
                  prometheus_client default registry
                                   |
        +--------+--------+--------+--------+--------+
        |                                            |
   vllm:omni_*                                    vllm:*
   collectors                                  collectors
        |                                            |
   +----+--------+   +-----------+   +----------+   +-----------+
   | OmniPromet- |   | OmniMod-  |   | OmniTra- |   | OmniProm- |
   | heusMetrics |   | alityMet- |   | nsferMe- |   | etheusSt- |
   |  (PR#3362)  |   | rics (G1+ |   | trics    |   | atLogger  |
   |             |   |  G2)      |   | (G3)     |   | (G7 wrap) |
   +----+--------+   +-----+-----+   +----+-----+   +----+------+
        |                  |              |              |
     OmniBase           OmniBase     Orchestrator    Orchestrator
   (request life-     (finalize +   (record_trans-  (per-(stage,
    cycle, success/    streaming     fer_tx/rx        replica)
    fail counter,      hooks via     hooks via        scheduler/
    diffusion          observe_*     emit hook in     iteration
    timing)            APIs)         OrchestratorAg-  stats)
                                     gregator)
```

### Data Flow

There are four independent paths for metric collection.

**Path 1: Pipeline-level metrics (`vllm:omni_*`, PR #3362 + G6)**

`OmniPrometheusMetrics` registers Gauge, Counter, and Histogram
collectors at init time. It is instantiated once per entrypoint,
labeled with the model name. The entrypoint calls its methods as
requests progress:

- `set_running(n)` / `set_waiting(n)` — updated after each request
  completes. The running count comes from `OmniRequestCounter`, a
  simple counter incremented/decremented by the Orchestrator as it
  tracks requests. Waiting is derived as `total - running`.

- `request_succeeded(e2e_seconds, queue_seconds=None,
  finished_reason="stop")` — recorded when a request finishes at the
  final stage. `finished_reason` is extracted from
  `engine_outputs.outputs[0].finish_reason` (vLLM `CompletionOutput`
  convention) and increments
  `vllm:omni_requests_success_total{finished_reason}`.

- `request_failed()` — recorded by the cleanup path when a request
  exits without natural completion. Internally maps to
  `finished_reason="abort"` so a single Counter family covers both
  natural and aborted completion (G6).

- `observe_diffusion_metrics(stage_id, metrics)` — recorded when a
  diffusion stage finishes. The metrics dict contains timing
  breakdowns (preprocess, exec, postprocess, total step time)
  accumulated from engine output.

**Path 2: Modality metrics (`vllm:omni_audio_* / image_* / video_*`, G1 + G2)**

`OmniModalityMetrics` registers eight per-modality Histogram + Counter
families with `{model_name, stage, replica}` labels. Two observation
sites:

- `observe_modality_at_finalize(...)` — called from
  `omni_base._process_single_result` inside the existing `e2e_done`
  finalize guard. Routes by `final_output_type`:
  - `audio`: emits `audio_frames_total` (Counter), `audio_duration_seconds`,
    `audio_rtf` (Histograms). Sample rate is resolved from
    `engine_outputs.multimodal_output["audio_sample_rate"]` via
    `definitions.resolve_audio_sample_rate(...)` (fallback chain mirrors
    `serving_chat.py`).
  - `image`: emits `image_num_total`, `image_generation_time_seconds`,
    `image_ttfp_seconds`. (`image_ttfp` is observed at finalize because
    the diffusion path has no intermediate image streaming — first
    image equals final image.)
  - `video`: emits `video_generation_time_seconds`. Note that
    `video_duration_seconds` and `video_rtf` are deferred — diffusion
    video pipelines (i2v / t2v / cogvideo / hunyuan / wan) expose
    `num_frames` + `fps` in heterogeneous shapes and a clean abstraction
    is out of scope for this iteration.

- `observe_audio_first_packet(...)` — called from the OpenAI streaming
  paths (`serving_chat.py` HTTP-SSE audio branch and
  `serving_video_stream.py` WebSocket audio branch) on the first audio
  packet emerging for a request. The once-per-request guard is held by
  `ClientRequestState.first_audio_ts` (set on first emit). The
  `request_arrival_ts` anchor is also stored in `ClientRequestState`
  by `async_omni.generate()`, computed as the wall-clock time at
  request entry.

**Path 3: Cross-stage transfer metrics (`vllm:omni_transfer_*`, G3)**

`OmniTransferMetrics` registers four Histogram families with
`{model_name, from_stage, from_replica, to_stage, to_replica}` labels.
Each observation corresponds to one physical transfer hop (one chunk
between adjacent stages), not the per-request accumulated total — so
the histograms track per-transfer distribution.

The hook lives in `OrchestratorAggregator.record_transfer_tx` and
`record_transfer_rx`. After the existing `TransferEdgeStats`
accumulation, the aggregator calls `_emit_transfer_tx` /
`_emit_transfer_rx` which look up `from_replica` / `to_replica` via a
`replica_resolver` callback supplied by `async_omni.py`. The resolver
delegates to `stage_pool.get_bound_replica_id(request_id)` —
i.e. the orchestrator's existing sticky-routing binding (PR #2396) is
the source of truth for the per-edge replica labels. No plumbing
through `TransferEdgeStats`, `StageRequestStats`, or the connector
adapter is needed.

Defensive fail-safe: if `transfer_emitter` or `replica_resolver` is
missing, or the resolver returns `None` for either side, the emit is
skipped silently (the underlying `TransferEdgeStats` accumulation is
unaffected).

> The TX-side hook (`record_transfer_tx`) is wired up but only fires
> once `try_send_via_connector` is invoked from the main code path;
> until then only the RX-side families (`rx_decode_time_ms` +
> `in_flight_time_ms`) accumulate observations.

**Path 4: Per-engine metrics (`vllm:*`, G7 wrap)**

The Orchestrator instantiates `OmniPrometheusStatLogger` (a thin
subclass of upstream `vllm.v1.metrics.loggers.PrometheusStatLogger`)
and feeds it scheduler stats and iteration stats after processing
each batch of engine outputs. This populates the standard ~37 vLLM
metric families (TTFT, ITL, TPOT, KV cache usage, etc.) using the same
upstream code path — but with the `engine` label reshaped into
`stage` + `replica` so multi-replica deployments produce distinct
series per replica. See the next section for the wrap mechanics.

For diffusion-only pipelines that have no AR engine,
`SchedulerStats` is never produced and `vllm:*` metrics are absent.

### Shared State Between Threads

The Orchestrator runs in a background thread. The API server
(OmniBase) runs in the asyncio event loop thread.
`OmniRequestCounter` bridges them — a plain Python object with an
`int` field. The Orchestrator increments/decrements it; the
entrypoint reads it for gauge updates. No lock is needed because the
counter is advisory (a stale read by one Prometheus scrape interval
is acceptable). It is created by `AsyncOmniEngine.__init__()` and
passed to the Orchestrator at construction time.

### Metric Registration and Lifecycle

All `vllm:omni_*` collectors are registered once when their owning
class (`OmniPrometheusMetrics` / `OmniModalityMetrics` /
`OmniTransferMetrics`) is imported. Per-`(stage, replica)` labels are
bound lazily on first observation to avoid registering label sets for
combinations that never produce data (e.g. a diffusion pipeline has
no audio metrics).

The `prometheus_client` default registry holds all collectors.
FastAPI's `/metrics` endpoint serves the default registry, so
`vllm:omni_*` and the wrapped `vllm:*` metrics appear in the same
scrape response alongside `http_*` and `process_*` metrics from the
instrumentator and the Python client runtime.

## OmniPrometheusStatLogger Wrap (G7)

Upstream `PrometheusStatLogger.__init__` hard-codes
`labelnames = ["model_name", "engine"]` as a local variable, references
it across ~37 metric-family construction sites, and uses the `engine`
label value in five different `.labels()` call shapes (kwarg with int
engine, kwarg with str engine, positional with str engine in the
middle, plus a `metrics_info["engine"] = str(...)` dict pattern). To
reshape `engine` into `stage` + `replica` without forking the entire
upstream `__init__`, the wrap uses three coordinated mechanisms:

1. **Class-level metric class slot overrides.**
   `OmniPrometheusStatLogger` overrides `_gauge_cls`, `_counter_cls`,
   `_histogram_cls` (which upstream calls via `self._gauge_cls(...)`
   etc.) with `_RelabelGauge` / `_RelabelCounter` / `_RelabelHistogram`
   wrapper classes. These intercept the `labelnames` kwarg at metric
   family creation time and replace `engine` with `("stage", "replica")`.

2. **Property descriptor for `per_engine_labelvalues`.** Upstream
   builds `self.per_engine_labelvalues = {idx: [model_name, str(idx)]}`
   inside `__init__` and then captures it into a local variable for
   `create_metric_per_engine` calls. By making
   `per_engine_labelvalues` a Python property on the subclass, the
   setter intercepts upstream's assignment and rewrites each 2-tuple
   into a 3-tuple `[model_name, stage, replica]` using the
   `stage_replica_map` supplied at construction time. The captured
   local then sees the rewritten dict.

3. **Override of `.labels()` on the wrapper classes.** For the five
   call sites that pass `engine` directly (kwarg or positional, int or
   str), `_RelabelMixin.labels()` translates the engine value back to
   `(stage, replica)` via a process-level `_ENGINE_INDEX_MAP` populated
   by `OmniPrometheusStatLogger.__init__`. This handles
   `gauge_engine_sleep_state.labels(engine=idx, ...)`,
   `counter_request_success_base.labels(model_name, str(idx),
   str(reason))`, `info_gauge.labels(**metrics_info)`, etc.

The `Orchestrator` constructs `stage_replica_map` from the static
`stage_pools` configuration at startup:

```python
stage_replica_map = {
    flat_idx: (str(stage_id), str(replica_id))
    for flat_idx, (stage_id, replica_id) in enumerate(
        (s, r)
        for s, pool in enumerate(stage_pools)
        for r in range(pool.num_replicas)
    )
}
```

A reverse map `(stage_id, replica_id) -> flat_idx` is maintained on
the Orchestrator so the per-replica `record(engine_idx=...)` call site
can look up the right flat index.

> Dynamic add/remove of replicas at runtime is intentionally out of
> scope — the upstream `PrometheusStatLogger` materializes
> per-engine_idx child metrics at init time, and supporting hot-add
> would require non-trivial intervention into upstream's per-family
> child dictionaries.

## Throttling: `make_stats()` Override

Upstream vLLM's `Scheduler.make_stats()` runs on every AR generation step,
returning a SchedulerStats object for the orchestrator.
Under vLLM's architecture, this is fine. But since vLLM-Omni requires that the
object be serialized and transferred over ZMQ, receiving a SchedulerStats object on
every step can introduce unacceptable overhead to the system.

`OmniSchedulerMixin.make_stats()` (in
`vllm_omni/core/sched/omni_scheduler_mixin.py`) throttles stats
emission to at most once per second. Between intervals it returns
`None`, which the engine core skips serializing. This keeps gauges
fresh enough for Prometheus scrapes (typically 15-30s intervals) while
eliminating the per-step overhead.

## Metric Definitions

### Pipeline-Level

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `vllm:omni_num_requests_running` | Gauge | `model_name` | Requests currently executing across all stages |
| `vllm:omni_num_requests_waiting` | Gauge | `model_name` | Requests queued but not yet scheduled |
| `vllm:omni_requests_success_total` | Counter | `model_name`, `finished_reason` | Total requests by completion reason ({stop, length, abort, ...}); aborts include the previous "fail" path (G6) |
| `vllm:omni_e2e_request_latency_seconds` | Histogram | `model_name` | End-to-end request latency across all stages |
| `vllm:omni_request_queue_time_seconds` | Histogram | `model_name` | Time spent waiting in the request queue |

### Modality (G1 + G2)

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `vllm:omni_audio_ttfp_seconds` | Histogram | `model_name`, `stage`, `replica` | Time from request arrival to first audio packet (streaming hook) |
| `vllm:omni_audio_duration_seconds` | Histogram | same | Audio content duration (`audio_frames / sample_rate`) |
| `vllm:omni_audio_rtf` | Histogram | same | Real-time factor `stage_gen_time_s / audio_duration_s` (RFC SLO `< 1`) |
| `vllm:omni_audio_frames_total` | Counter | same | Cumulative audio frames generated |
| `vllm:omni_image_ttfp_seconds` | Histogram | same | Time from request arrival to image emission |
| `vllm:omni_image_num_total` | Counter | same | Cumulative images generated |
| `vllm:omni_image_generation_time_seconds` | Histogram | same | Per-request image stage generation time |
| `vllm:omni_video_generation_time_seconds` | Histogram | same | Per-request video stage generation time |

### Cross-Stage Transfer (G3)

Labels: `{model_name, from_stage, from_replica, to_stage, to_replica}`.

> `model_name` is included on the transfer family for consistency with
> the rest of the omni surface (audio_*, image_*, video_*, num_requests_*),
> even though RFC §3.2.6 originally listed only the four
> stage/replica labels. PromQL joins on `model_name` work uniformly
> across modality and transfer families.

| Metric | Type | Description |
|--------|------|-------------|
| `vllm:omni_transfer_size_bytes` | Histogram | Per-transfer payload size in bytes |
| `vllm:omni_transfer_tx_time_ms` | Histogram | Sender-side time (serialize + submit to connector) |
| `vllm:omni_transfer_rx_decode_time_ms` | Histogram | Receiver-side time (recv + deserialize) |
| `vllm:omni_transfer_in_flight_time_ms` | Histogram | Network in-flight time (TX done → RX recv start) |

### Diffusion Stage-Level

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `vllm:omni_diffusion_preprocess_time_ms` | Histogram | `model_name`, `engine` | Diffusion input preprocessing time |
| `vllm:omni_diffusion_exec_time_ms` | Histogram | `model_name`, `engine` | Diffusion model forward pass time |
| `vllm:omni_diffusion_postprocess_time_ms` | Histogram | `model_name`, `engine` | Diffusion output postprocessing time |
| `vllm:omni_diffusion_step_time_ms` | Histogram | `model_name`, `engine` | Total diffusion step time |

> The diffusion families bypass the `OmniPrometheusStatLogger` wrap, so
> their `engine` label is the diffusion stage_id (not relabelled to
> `stage` + `replica`).

### LLM Stage-Level (wrapped `vllm:*`)

After the G7 wrap, every upstream `vllm:*` family — TTFT, ITL, TPOT,
e2e latency, KV cache usage, scheduler running/waiting, request
success counts, etc. — carries `{model_name, stage, replica}` labels.
For the full upstream catalog see
[the vLLM docs](https://github.com/vllm-project/vllm/blob/main/docs/usage/metrics.md);
note that metrics depending on features unsupported in vLLM-Omni
(e.g. speculative decoding, LoRA) will not be available.

## Logging vs. Prometheus

`OrchestratorAggregator` (in `vllm_omni/metrics/stats.py`) is the
logging-oriented metrics path. It collects detailed per-request,
per-stage, and per-transfer statistics and prints formatted tables to
the `INFO` log. This is designed for development and debugging —
individual request traces, transfer bandwidth, inter-stage timing.

`OmniPrometheusMetrics` / `OmniModalityMetrics` / `OmniTransferMetrics`
form the Prometheus-oriented path. They record aggregate counters,
gauges, and histograms suitable for time-series monitoring and
alerting. Both paths share the same source data (`StageRequestStats`,
`TransferEdgeStats`) — `OrchestratorAggregator.record_transfer_tx/rx`
in particular calls both the existing accumulator code and the
Prometheus emit hook in the same method body. The two consumption
models can run simultaneously without coupling.

The separation follows upstream vLLM's pattern of `LoggingStatLogger`
vs. `PrometheusStatLogger` — same underlying data, different
consumption models.
