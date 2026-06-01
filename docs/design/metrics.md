# Prometheus Metrics Design

This document describes how vLLM-Omni exposes Prometheus metrics for multi-stage pipelines, the constraints that shaped the design, and how the pipeline-level metrics coexist with upstream vLLM per-engine metrics. This iteration covers the text and audio surface; diffusion / image / video families land in a follow-up.

## Objectives

- Expose pipeline-level request and latency metrics that span the full multi-stage execution (orchestrator scope).
- Preserve all upstream vLLM per-engine metrics (`vllm:*`) for stages backed by an AR LLM engine, and reshape their `engine` label into `stage` + `replica` so multi-replica deployments gain per-replica visibility automatically.
- Expose per-modality SLO metrics that the upstream `vllm:*` families do not capture — audio TTFP / RTF / duration / frames / streaming continuity / silent-loss.
- Expose per-replica-edge cross-stage transfer metrics so the slack between E2E latency and the sum of per-stage `gen_time` (queueing, serialization, network) becomes attributable.
- Keep the metrics collection overhead low enough that it does not regress TTFA or throughput.

## Background

### Upstream vLLM Metrics

Upstream vLLM defines ~37 Prometheus metric families under the `vllm:` prefix. These are registered by `PrometheusStatLogger` and cover engine-level state: KV cache usage, running/waiting request counts, token throughput, TTFT, inter-token latency, e2e latency, and so on. They are served via the `/metrics` HTTP endpoint provided by `prometheus_fastapi_instrumentator` and the default `prometheus_client` WSGI handler.

vLLM's `unregister_vllm_metrics()` function strips every `prometheus_client` collector whose `_name` attribute contains the substring `"vllm"`. This runs during engine initialization to clean up stale collectors from prior instantiations within the same process.

### The Problem

vLLM-Omni runs multiple engine instances (stages × replicas) within a single process, coordinated by an Orchestrator. The pipeline needs its own metrics — aggregate request counts, end-to-end latency across all stages, per-modality SLO signals, and cross-stage transfer attribution — that do not exist in upstream vLLM. All pipeline-level metrics use the `vllm:omni_` prefix to distinguish them from upstream per-engine metrics. At import time (see `vllm_omni/patch.py`) we replace `unregister_vllm_metrics()` with a scoped version that still strips upstream `vllm:*` collectors before each new `PrometheusStatLogger` registers (so multi-engine processes don't crash on duplicate registration), but preserves anything prefixed `vllm:omni_` / `vllm_omni`.

Upstream per-engine metrics retain the `vllm:` prefix but are now registered by `OmniPrometheusStatLogger`, a thin subclass of upstream's `PrometheusStatLogger` that reshapes the single `engine` label into a `stage` + `replica` pair (see "OmniPrometheusStatLogger wrap" below).

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
    fail counter)      hooks via     hooks via        scheduler/
                       observe_*     emit hook in     iteration
                       APIs)         OrchestratorAg-  stats)
                                     gregator)
```

### Data Flow

There are four independent paths for metric collection.

**Path 1: Pipeline-level metrics (`vllm:omni_*`)**

`OmniPrometheusMetrics` registers the Gauge / Counter / Histogram collectors at import time. It is instantiated once per entrypoint, labeled with the model name. The entrypoint calls its methods as requests progress:

- `set_running(n)` / `set_waiting(n)` — updated after each request completes. The running count comes from `OmniRequestCounter`, a simple counter incremented/decremented by the Orchestrator as it tracks requests. Waiting is derived as `total - running`.
- `request_succeeded(e2e_seconds, finished_reason="stop")` — recorded when a request finishes at the final stage. `finished_reason` is extracted from `engine_outputs.outputs[0].finish_reason` (vLLM `CompletionOutput` convention) and increments `vllm:omni_requests_success_total{finished_reason}`.
- `request_failed()` — recorded by the cleanup path when a request exits without natural completion. Internally maps to `finished_reason="abort"` so a single Counter family covers both natural and aborted completion.

**Path 2: Audio modality metrics (`vllm:omni_audio_*`)**

`OmniModalityMetrics` registers seven audio families with `{model_name, stage, replica}` (plus an extra `threshold_ms` / `reason` label on the two extra-cardinality Counters). Three observation sites:

- `observe_modality_at_finalize(...)` — called from `omni_base._process_single_result` inside the existing `e2e_done` finalize guard. For `output_type == "audio"` it emits `audio_frames_total`, `audio_duration_s`, `audio_rtf` (or `audio_skipped_requests_total{reason="no_audio_data"}` when no audio was produced). Sample rate is resolved from `engine_outputs.multimodal_output` via `definitions.resolve_audio_sample_rate(...)` (fallback chain mirrors `serving_chat.py`'s audio response path).
- `observe_audio_first_packet(...)` — called from the OpenAI SSE audio branch in `serving_chat.py` on the first audio packet for a request. The once-per-request guard is held by `ClientRequestState.first_audio_ts`. The `request_arrival_ts` anchor is stored in `ClientRequestState` by `async_omni.generate()`, computed at request entry.
- `observe_audio_streaming_finalize(...)` — called from `serving_chat.py` after the streaming chunk loop exhausts. It runs the per-chunk player simulation from `vllm_omni/benchmarks/audio_continuity.py` to compute the worst-case underrun and emits `audio_underrun_s` plus (when the request stayed below the threshold) `audio_continuity_ok_total{threshold_ms}`. Per-chunk PCM byte counts and arrival timestamps are recorded by the same audio branch that updates `first_audio_ts`.

**Path 3: Cross-stage transfer metrics (`vllm:omni_transfer_*`)**

`OmniTransferMetrics` registers four Histogram families with `{model_name, from_stage, from_replica, to_stage, to_replica}` labels. Each observation corresponds to one physical transfer hop (one chunk between adjacent stages), not the per-request accumulated total — so the histograms track per-transfer distribution.

The hook lives in `OrchestratorAggregator.record_transfer_tx` and `record_transfer_rx`. After the existing `TransferEdgeStats` accumulation, the aggregator calls `_emit_transfer_tx` / `_emit_transfer_rx`. Those:

1. Resolve `from_replica` / `to_replica` via a `replica_resolver` callback supplied by `async_omni.py`. The resolver delegates to `stage_pool.get_bound_replica_id(request_id)` — i.e. the orchestrator's existing sticky-routing binding is the source of truth.
2. Convert the underlying `_ms` accumulators to seconds and call the `_s`-suffixed observe methods on `OmniTransferMetrics`.

Defensive fail-safe: if `transfer_emitter` or `replica_resolver` is missing, or the resolver returns `None` for either side, the emit is skipped silently (the underlying `TransferEdgeStats` accumulation is unaffected).

**Path 4: Per-engine metrics (`vllm:*`, stage/replica wrap)**

The Orchestrator instantiates `OmniPrometheusStatLogger` (a thin subclass of upstream `vllm.v1.metrics.loggers.PrometheusStatLogger`) and feeds it scheduler stats and iteration stats after processing each batch of engine outputs. This populates the standard ~37 vLLM metric families (TTFT, ITL, TPOT, KV cache usage, etc.) using the same upstream code path — but with the `engine` label reshaped into `stage` + `replica` so multi-replica deployments produce distinct series per replica. See the next section for the wrap mechanics.

### Shared State Between Threads

The Orchestrator runs in a background thread. The API server (OmniBase) runs in the asyncio event loop thread. `OmniRequestCounter` bridges them — a plain Python object with an `int` field. The Orchestrator increments/decrements it; the entrypoint reads it for gauge updates. No lock is needed because the counter is advisory (a stale read by one Prometheus scrape interval is acceptable). It is created by `AsyncOmniEngine.__init__()` and passed to the Orchestrator at construction time.

### Metric Registration and Lifecycle

All `vllm:omni_*` collectors are registered once when their owning class (`OmniPrometheusMetrics` / `OmniModalityMetrics` / `OmniTransferMetrics`) is imported. Per-`(stage, replica)` labels are bound lazily on first observation to avoid registering label sets for combinations that never produce data.

The `prometheus_client` default registry holds all collectors. FastAPI's `/metrics` endpoint serves the default registry, so `vllm:omni_*` and the wrapped `vllm:*` metrics appear in the same scrape response alongside `http_*` and `process_*` metrics from the instrumentator and the Python client runtime.

## OmniPrometheusStatLogger Wrap

Upstream `PrometheusStatLogger.__init__` hard-codes `labelnames = ["model_name", "engine"]` as a local variable, references it across ~37 metric-family construction sites, and uses the `engine` label value in five different `.labels()` call shapes (kwarg with int engine, kwarg with str engine, positional with str engine in the middle, plus a `metrics_info["engine"] = str(...)` dict pattern). To reshape `engine` into `stage` + `replica` without forking the entire upstream `__init__`, the wrap uses three coordinated mechanisms:

1. **Class-level metric class slot overrides.** `OmniPrometheusStatLogger` overrides `_gauge_cls`, `_counter_cls`, `_histogram_cls` (which upstream calls via `self._gauge_cls(...)` etc.) with `_RelabelGauge` / `_RelabelCounter` / `_RelabelHistogram` wrapper classes. These intercept the `labelnames` kwarg at metric family creation time and replace `engine` with `("stage", "replica")`.
2. **Property descriptor for `per_engine_labelvalues`.** Upstream builds `self.per_engine_labelvalues = {idx: [model_name, str(idx)]}` inside `__init__` and then captures it into a local variable for `create_metric_per_engine` calls. By making `per_engine_labelvalues` a Python property on the subclass, the setter intercepts upstream's assignment and rewrites each 2-tuple into a 3-tuple `[model_name, stage, replica]` using the `stage_replica_map` supplied at construction time. The captured local then sees the rewritten dict.
3. **Override of `.labels()` on the wrapper classes.** For the five call sites that pass `engine` directly (kwarg or positional, int or str), `_RelabelMixin.labels()` translates the engine value back to `(stage, replica)` via a process-level `_ENGINE_INDEX_MAP` populated by `OmniPrometheusStatLogger.__init__`. This handles `gauge_engine_sleep_state.labels(engine=idx, ...)`, `counter_request_success_base.labels(model_name, str(idx), str(reason))`, `info_gauge.labels(**metrics_info)`, etc.

The three sub-helpers that upstream `PrometheusStatLogger.__init__` constructs (`spec_decoding_prom` / `kv_connector_prom` / `perf_metrics_prom`) use their own `_counter_cls` / `_gauge_cls` / `_histogram_cls` slots and would otherwise build families with the raw 2-element labelnames. `_OmniPerfMetricsProm` / `_OmniSpecDecodingProm` / `_OmniKVConnectorProm` subclass each helper to route the same relabel mixin through their internal family construction.

The `Orchestrator` constructs `stage_replica_map` from the static `stage_pools` configuration at startup:

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

A reverse map `(stage_id, replica_id) -> flat_idx` is maintained on the Orchestrator so the per-replica `record(engine_idx=...)` call site can look up the right flat index.

> Dynamic add/remove of replicas at runtime is intentionally out of scope — the upstream `PrometheusStatLogger` materializes per-engine_idx child metrics at init time, and supporting hot-add would require non-trivial intervention into upstream's per-family child dictionaries.

## Gating: `--log-stats`

All metrics — both the 15 `vllm:omni_*` families and the ~65 upstream `vllm:*` wrap families — are gated by the user's `--log-stats` CLI flag (default off). The flag is plumbed from `OmniBase.__init__(log_stats=...)` through `AsyncOmniEngine` to the stage-spawn helpers and to the three Prometheus metric classes (`OmniPrometheusMetrics` / `OmniModalityMetrics` / `OmniTransferMetrics`), and is forwarded to `Orchestrator._init_metrics_state(...)`.

Behavior with `--log-stats=off` (default):

- The three `Omni*Metrics` classes register their module-level Gauge / Counter / Histogram families at import time (prometheus_client requires up-front registration), but each `observe / inc / set` method early-returns. The per-label child series for `vllm:omni_*` stay materialized but never have data written to them.
- `OmniPrometheusStatLogger` is not constructed in `_init_metrics_state`, so the ~65 upstream `vllm:*` wrap families are not registered in the default registry at all.
- The engine core's `Scheduler.make_stats()` also short-circuits inside upstream (`if not self.log_stats: return None`), so no `SchedulerStats` is produced per step — the per-iteration cost is bounded by the existing upstream gate.

Behavior with `--log-stats=on`: all metric paths fire normally; the orchestrator's per-replica recording is bounded only by `OmniSchedulerMixin.make_stats()`'s per-scheduler 1 Hz throttle (see next section).

The overhead with the flag on is small enough that an A/B benchmark on Qwen3-Omni-30B single replica (30 sequential audio requests) showed a mean latency delta of +0.6% (Welch's t = 0.318, n=30, not statistically significant at α=0.05); the `/metrics` line count drops from 1358 to 124 lines when the flag is off.

## Throttling: `make_stats()` Override

Upstream vLLM's `Scheduler.make_stats()` runs on every AR generation step, returning a SchedulerStats object for the orchestrator. Under vLLM's architecture, this is fine. But since vLLM-Omni requires that the object be serialized and transferred over ZMQ, receiving a SchedulerStats object on every step can introduce unacceptable overhead to the system.

`OmniSchedulerMixin.make_stats()` (in `vllm_omni/core/sched/omni_scheduler_mixin.py`) throttles stats emission to at most once per second **per scheduler** — i.e. per `(stage, replica)` since each replica owns its own scheduler instance. Between intervals it returns `None`, which the engine core skips serializing. This keeps gauges fresh enough for Prometheus scrapes (typically 15-30s intervals) while eliminating the per-step overhead.

The orchestrator side does not add its own throttle on top: the per-replica recording loop gates only on `raw_outputs.scheduler_stats is not None` (i.e. this replica's scheduler passed its own 1 Hz gate). A previous global `_last_stats_ts` on the orchestrator was removed because it starved every replica other than the first to emit within each second.

## Metric Definitions

### Pipeline (4)

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `vllm:omni_num_requests_running` | Gauge | `model_name` | Requests currently executing across all stages |
| `vllm:omni_num_requests_waiting` | Gauge | `model_name` | Requests queued but not yet scheduled |
| `vllm:omni_requests_success_total` | Counter | `model_name`, `finished_reason` | Total requests by completion reason ({stop, length, abort, ...}); aborts cover client-disconnect / cancellation paths in addition to upstream `FinishReason.ABORT` |
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

### Cross-stage transfer (4)

Labels: `{model_name, from_stage, from_replica, to_stage, to_replica}`.

| Metric | Type | Description |
|--------|------|-------------|
| `vllm:omni_transfer_size_bytes` | Histogram | Per-transfer payload size in bytes |
| `vllm:omni_transfer_tx_s` | Histogram | Sender-side time (serialize + submit to connector) |
| `vllm:omni_transfer_rx_s` | Histogram | Receiver-side time (recv + deserialize) |
| `vllm:omni_transfer_in_flight_s` | Histogram | Network in-flight time (TX done → RX recv start) |

### LLM stage-level (wrapped `vllm:*`)

After the wrap, every upstream `vllm:*` family — TTFT, ITL, TPOT, e2e latency, KV cache usage, scheduler running/waiting, request success counts, etc. — carries `{model_name, stage, replica}` labels. For the full upstream catalog see [the vLLM docs](https://github.com/vllm-project/vllm/blob/main/docs/usage/metrics.md); note that metrics depending on features unsupported in vLLM-Omni (e.g. speculative decoding, LoRA) will not be available.

## Naming Convention

- All time-bearing metrics use the `_s` suffix (values in seconds). Two bucket families are used:
  - `SECONDS_BUCKETS` (0.05 s – 300 s) for e2e / generation / TTFP style values.
  - `SECONDS_FAST_BUCKETS` (0.001 s – 60 s) for fine-grained cross-stage transfer and audio-underrun values that need millisecond-level resolution.
- Counters use the `_total` suffix (auto-appended by `prometheus_client`).
- Sizes use the `_bytes` suffix.
- All omni-specific families are prefixed `vllm:omni_`. The upstream `unregister_vllm_metrics()` function is monkey-patched to a scoped version that still strips upstream `vllm:*` collectors (so multi-engine init within one process does not crash on duplicate registration) but preserves anything prefixed `vllm:omni_` / `vllm_omni`.

## Logging vs. Prometheus

`OrchestratorAggregator` (in `vllm_omni/metrics/stats.py`) is the logging-oriented metrics path. It collects detailed per-request, per-stage, and per-transfer statistics and prints formatted tables to the `INFO` log. This is designed for development and debugging — individual request traces, transfer bandwidth, inter-stage timing.

`OmniPrometheusMetrics` / `OmniModalityMetrics` / `OmniTransferMetrics` form the Prometheus-oriented path. They record aggregate counters, gauges, and histograms suitable for time-series monitoring and alerting. Both paths share the same source data (`StageRequestStats`, `TransferEdgeStats`) — `OrchestratorAggregator.record_transfer_tx/rx` in particular calls both the existing accumulator code and the Prometheus emit hook in the same method body. The two consumption models can run simultaneously without coupling.

The separation follows upstream vLLM's pattern of `LoggingStatLogger` vs. `PrometheusStatLogger` — same underlying data, different consumption models.
