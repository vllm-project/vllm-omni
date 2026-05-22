# Prometheus Metrics Design

This document describes how vLLM-Omni exposes Prometheus metrics for
multi-stage pipelines, the constraints that shaped the design, and how
the pipeline-level metrics coexist with upstream vLLM per-engine
metrics.

The `vllm_omni:*` surface is locked to 23 families by
[RFC #3545](https://github.com/vllm-project/vllm-omni/issues/3545).

## Objectives

- Expose pipeline-level request and latency metrics that span the full
  multi-stage execution (orchestrator scope).
- Preserve all upstream vLLM per-engine metrics (`vllm:*`) for stages
  backed by an AR LLM engine, and reshape their `engine` label into
  `stage` + `replica` so multi-replica deployments gain per-replica
  visibility automatically.
- Expose per-stage diffusion-internal timing breakdowns (preprocess /
  exec / postprocess) for pipelines that include a diffusion engine.
- Expose per-modality SLO metrics that the upstream `vllm:*` families do
  not capture — audio TTFP / RTF / duration / frames / streaming
  continuity / silent-loss, image counts / generation time, video
  duration / RTF / generation time.
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
   |             |   | rics      |   | trics    |   | atLogger  |
   +----+--------+   +-----+-----+   +----+-----+   +----+------+
        |                  |              |              |
     OmniBase           OmniBase     Orchestrator    Orchestrator
   (request life-     (finalize +   (record_trans-  (per-(stage,
    cycle, success/    streaming     fer_tx/rx        replica)
    fail counter,      hooks via     hooks via        scheduler/
    diffusion-         observe_*     emit hook in     iteration
    internal           APIs)         OrchestratorAg-  stats)
    timing)                          gregator)
```

### Data Flow

There are four independent paths for metric collection.

**Path 1: Pipeline-level metrics (`vllm:omni_*`)**

`OmniPrometheusMetrics` registers the Gauge / Counter / Histogram
collectors at import time. It is instantiated once per entrypoint,
labeled with the model name. The entrypoint calls its methods as
requests progress:

- `set_running(n)` / `set_waiting(n)` — updated after each request
  completes. The running count comes from `OmniRequestCounter`, a
  simple counter incremented/decremented by the Orchestrator as it
  tracks requests. Waiting is derived as `total - running`.

- `request_succeeded(e2e_seconds, finished_reason="stop")` — recorded
  when a request finishes at the final stage. `finished_reason` is
  extracted from `engine_outputs.outputs[0].finish_reason` (vLLM
  `CompletionOutput` convention) and increments
  `vllm:omni_requests_success_total{finished_reason}`.

- `request_failed()` — recorded by the cleanup path when a request
  exits without natural completion. Internally maps to
  `finished_reason="abort"` so a single Counter family covers both
  natural and aborted completion (G6).

- `observe_diffusion_metrics(stage_id, replica_id, metrics)` — recorded
  when a diffusion stage finishes. The metrics dict carries the engine's
  legacy millisecond keys (`preprocess_time_ms` /
  `diffusion_engine_exec_time_ms` / `postprocess_time_ms`); the observe
  call converts them to seconds at the emit boundary and exposes them as
  the `_s`-suffixed families with `{model_name, stage, replica}` labels.

**Path 2: Audio modality metrics (`vllm:omni_audio_*`)**

`OmniModalityMetrics` registers seven audio families with
`{model_name, stage, replica}` (plus an extra `threshold_ms` /
`reason` label on the two extra-cardinality Counters). Three observation
sites:

- `observe_modality_at_finalize(...)` — called from
  `omni_base._process_single_result` inside the existing `e2e_done`
  finalize guard. For `output_type == "audio"` it emits
  `audio_frames_total`, `audio_duration_s`, `audio_rtf`. Sample rate is
  resolved from `engine_outputs.multimodal_output` via
  `definitions.resolve_audio_sample_rate(...)` (fallback chain mirrors
  `serving_chat.py`'s audio response path).

- `observe_audio_first_packet(...)` — called from the OpenAI SSE audio
  branch in `serving_chat.py` (and the WebSocket route in
  `serving_video_stream.py`) on the first audio packet for a request.
  The once-per-request guard is held by
  `ClientRequestState.first_audio_ts`. The `request_arrival_ts` anchor
  is stored in `ClientRequestState` by `async_omni.generate()`, computed
  at request entry.

- `observe_audio_streaming_finalize(...)` — called from `serving_chat.py`
  after the streaming chunk loop exhausts. It runs the per-chunk player
  simulation from `vllm_omni/benchmarks/audio_continuity.py` to compute
  the worst-case underrun and emits `audio_underrun_s` plus (when the
  request stayed below the threshold) `audio_continuity_ok_total{threshold_ms}`.
  Per-chunk PCM byte counts and arrival timestamps are recorded by the
  same audio branch that updates `first_audio_ts`.

The remaining audio family — `audio_skipped_requests_total{reason}` —
is wired through `OmniModalityMetrics.inc_audio_skipped(...)` for the
silent-loss path (e.g. code2wav rejecting malformed codec input and
returning `200 OK` with empty audio).

**Path 3: Visual modality metrics (`vllm:omni_image_*` / `video_*`)**

The same `OmniModalityMetrics` instance also serves the five visual
families. They are emitted from `observe_modality_at_finalize`:

- `output_type == "image"`: increments `image_num_total` by
  `len(engine_outputs.images)` and observes `image_generation_s` from
  the stage's `stage_gen_time_ms / 1000`.
- `output_type == "video"`: observes `video_generation_s`; also
  `video_duration_s` and `video_rtf` when `num_frames` and `fps` can be
  extracted from `multimodal_output["video"]` (or fallback attributes on
  `engine_outputs`). Heterogeneous video pipelines (i2v / t2v / cogvideo
  / hunyuan / wan) may surface those fields under different keys; the
  helper `_resolve_video_duration_seconds` walks a small fallback chain
  and skips the observation when neither shape applies.

**Path 4: Cross-stage transfer metrics (`vllm:omni_transfer_*`)**

`OmniTransferMetrics` registers four Histogram families with
`{model_name, from_stage, from_replica, to_stage, to_replica}` labels.
Each observation corresponds to one physical transfer hop (one chunk
between adjacent stages), not the per-request accumulated total — so
the histograms track per-transfer distribution.

The hook lives in `OrchestratorAggregator.record_transfer_tx` and
`record_transfer_rx`. After the existing `TransferEdgeStats`
accumulation, the aggregator calls `_emit_transfer_tx` /
`_emit_transfer_rx`. Those:

1. Resolve `from_replica` / `to_replica` via a `replica_resolver`
   callback supplied by `async_omni.py`. The resolver delegates to
   `stage_pool.get_bound_replica_id(request_id)` — i.e. the orchestrator's
   existing sticky-routing binding is the source of truth.
2. Convert the underlying `_ms` accumulators to seconds and call the
   `_s`-suffixed observe methods on `OmniTransferMetrics`.

Defensive fail-safe: if `transfer_emitter` or `replica_resolver` is
missing, or the resolver returns `None` for either side, the emit is
skipped silently (the underlying `TransferEdgeStats` accumulation is
unaffected).

> The TX-side hook is wired up but only fires once
> `try_send_via_connector` is invoked from the main code path; until
> then only the RX-side families (`transfer_rx_s` + `transfer_in_flight_s`)
> accumulate observations.

**Path 5: Per-engine metrics (`vllm:*`, G7 wrap)**

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

The three sub-helpers that upstream `PrometheusStatLogger.__init__`
constructs (`spec_decoding_prom` / `kv_connector_prom` /
`perf_metrics_prom`) use their own `_counter_cls` / `_gauge_cls` /
`_histogram_cls` slots and would otherwise build families with the raw
2-element labelnames. `_OmniPerfMetricsProm` / `_OmniSpecDecodingProm` /
`_OmniKVConnectorProm` subclass each helper to route the same relabel
mixin through their internal family construction.

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

## Metric Definitions (RFC-locked 23 families)

### Pipeline (4)

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `vllm:omni_num_requests_running` | Gauge | `model_name` | Requests currently executing across all stages |
| `vllm:omni_num_requests_waiting` | Gauge | `model_name` | Requests queued but not yet scheduled |
| `vllm:omni_requests_success_total` | Counter | `model_name`, `finished_reason` | Total requests by completion reason ({stop, length, abort, ...}); aborts include the previous "fail" path (G6) |
| `vllm:omni_e2e_request_latency_s` | Histogram | `model_name` | Pipeline-global end-to-end latency in seconds |

### Audio (7)

Labels: `{model_name, stage, replica}` plus the listed extra label.

| Metric | Type | Extra label | Description |
|--------|------|-------------|-------------|
| `vllm:omni_audio_ttfp_s` | Histogram | — | Time from request arrival to first audio packet/frame |
| `vllm:omni_audio_duration_s` | Histogram | — | Audio content duration (`audio_frames / sample_rate`) |
| `vllm:omni_audio_rtf` | Histogram | — | Real-time factor `stage_gen_time_s / audio_duration_s` (SLO `< 1`); uses `RTF_BUCKETS` |
| `vllm:omni_audio_frames_total` | Counter | — | Cumulative audio frames generated |
| `vllm:omni_audio_underrun_s` | Histogram | — | Per-request worst-case player deficit; `> 0` indicates listener heard silent gaps |
| `vllm:omni_audio_continuity_ok_total` | Counter | `threshold_ms` | Incremented when the request's worst underrun stayed below `threshold_ms` |
| `vllm:omni_audio_skipped_requests_total` | Counter | `reason` | Silent-loss counter — code2wav rejected malformed codec input and returned `200 OK` with empty audio |

### Visual — diffusion-internal (3)

Sourced from the diffusion engine's per-request metrics dict; the
emit-time observe converts the legacy millisecond keys to seconds.

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `vllm:omni_diffusion_preprocess_s` | Histogram | `model_name`, `stage`, `replica` | tokenizer + text_encoder + vae.encode |
| `vllm:omni_diffusion_exec_s` | Histogram | same | Per-request executor work time (sum of denoise steps) |
| `vllm:omni_diffusion_postprocess_s` | Histogram | same | vae.decode |

### Visual — business semantics (5)

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `vllm:omni_image_num_total` | Counter | `model_name`, `stage`, `replica` | Cumulative images generated |
| `vllm:omni_image_generation_s` | Histogram | same | Per-request image stage generation time (image has no RTF — no content duration) |
| `vllm:omni_video_duration_s` | Histogram | same | Video content duration (`num_frames / fps`) when extractable from `multimodal_output["video"]` |
| `vllm:omni_video_rtf` | Histogram | same | Real-time factor; uses `RTF_BUCKETS` |
| `vllm:omni_video_generation_s` | Histogram | same | Per-request video stage generation time |

### Cross-stage transfer (4)

Labels: `{model_name, from_stage, from_replica, to_stage, to_replica}`.

> `model_name` is included on the transfer family for consistency with
> the rest of the omni surface, even though RFC §3.2.6 originally listed
> only the four stage/replica labels. PromQL joins on `model_name` work
> uniformly across modality and transfer families.

| Metric | Type | Description |
|--------|------|-------------|
| `vllm:omni_transfer_size_bytes` | Histogram | Per-transfer payload size in bytes |
| `vllm:omni_transfer_tx_s` | Histogram | Sender-side time (serialize + submit to connector) |
| `vllm:omni_transfer_rx_s` | Histogram | Receiver-side time (recv + deserialize) |
| `vllm:omni_transfer_in_flight_s` | Histogram | Network in-flight time (TX done → RX recv start) |

### LLM stage-level (wrapped `vllm:*`)

After the G7 wrap, every upstream `vllm:*` family — TTFT, ITL, TPOT,
e2e latency, KV cache usage, scheduler running/waiting, request
success counts, etc. — carries `{model_name, stage, replica}` labels.
For the full upstream catalog see
[the vLLM docs](https://github.com/vllm-project/vllm/blob/main/docs/usage/metrics.md);
note that metrics depending on features unsupported in vLLM-Omni
(e.g. speculative decoding, LoRA) will not be available.

## Naming Convention

- All time-bearing metrics use the `_s` suffix (values in seconds).
  Two bucket families are used:
  - `SECONDS_BUCKETS` (0.05 s – 300 s) for e2e / generation / TTFP
    style values.
  - `SECONDS_FAST_BUCKETS` (0.001 s – 60 s) for fine-grained
    diffusion-internal and transfer values that need millisecond-level
    resolution.
- Counters use the `_total` suffix (auto-appended by `prometheus_client`).
- Sizes use the `_bytes` suffix.
- All omni-specific families are prefixed `vllm:omni_`. The upstream
  `unregister_vllm_metrics()` function is monkey-patched to a no-op so
  these are not destroyed during engine initialization.

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
