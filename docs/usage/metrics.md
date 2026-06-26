# Production Metrics

vLLM-Omni exposes Prometheus metrics via the `/metrics` endpoint on the OpenAI-compatible API server.

```bash
vllm-omni serve Qwen/Qwen3-Omni-30B-A3B-Instruct --port 8000 --log-stats
curl http://localhost:8000/metrics
```

**`--log-stats` is required to populate metric data.** Without the flag, the endpoint still returns `200 OK` but the upstream `vllm:*` wrap is not registered at all, and the 15 `vllm:omni_*` families are registered as placeholders with no data written to them. This keeps the runtime cost essentially zero for deployments that don't need monitoring. With the flag, all ~80 families populate.

## Metric Namespaces

| Prefix | Source | Present when |
|--------|--------|--------------|
| `vllm:omni_` | vLLM-Omni orchestrator / audio modality / cross-stage transfer | Pipeline-dependent |
| `vllm:` | Upstream vLLM engine, wrapped by `OmniPrometheusStatLogger` to expose `{stage, replica}` | Pipeline includes an LLM (AR) stage |
| `http_` / `process_` | Uvicorn / Python runtime | Always |

## Pipeline-Level Metrics (`vllm:omni_`)

Defined in `vllm_omni/metrics/prometheus.py`. Track request lifecycle across the full multi-stage pipeline.

### Request counts

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `vllm:omni_num_requests_running` | Gauge | `model_name` | Pipeline-global in-flight requests (dispatched to engine, not yet finalized) |
| `vllm:omni_num_requests_waiting` | Gauge | `model_name` | Requests waiting in the Orchestrator queue |
| `vllm:omni_requests_success_total` | Counter | `model_name`, `finished_reason` | Total requests by completion reason. `finished_reason` ∈ {`stop`, `length`, `abort`, ...} mirroring upstream `vllm:request_success_total`; aborts cover client disconnect / cancellation paths in addition to upstream `FinishReason.ABORT` |

### Latency

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `vllm:omni_e2e_request_latency_s` | Histogram | `model_name` | Pipeline-global end-to-end request latency in seconds |

## Audio Modality Metrics (`vllm:omni_`)

Emitted at request finalize, except for `audio_ttfp_s` (streaming-hook at the first audio packet) and `audio_underrun_s` / `audio_continuity_ok_total` (streaming finalize, after the chunk stream is exhausted). All carry `{model_name, stage, replica}` plus the listed extra label.

| Metric | Type | Extra label | Description |
|--------|------|-------------|-------------|
| `vllm:omni_audio_ttfp_s` | Histogram | — | Time from request arrival to first audio packet/frame |
| `vllm:omni_audio_duration_s` | Histogram | — | Audio content duration (`audio_frames / sample_rate`) |
| `vllm:omni_audio_rtf` | Histogram | — | Real-time factor (`stage_gen_time_s / audio_duration_s`); streaming TTS SLO red line `< 1`; uses `RTF_BUCKETS` |
| `vllm:omni_audio_frames_total` | Counter | — | Cumulative audio frame count; throughput via `rate()` |
| `vllm:omni_audio_underrun_s` | Histogram | — | Per-request worst-case player deficit; `> 0` indicates listener heard silent gaps |
| `vllm:omni_audio_continuity_ok_total` | Counter | `threshold_ms` | Incremented when the request's worst underrun stayed below `threshold_ms` |
| `vllm:omni_audio_skipped_requests_total` | Counter | `reason` | Silent-loss counter — code2wav rejected malformed codec input and returned `200 OK` with empty audio |

The continuity math comes from `vllm_omni/benchmarks/audio_continuity.py::compute_continuity_stats` so the server-side observation aligns with the bench-side definition.

## Diffusion Metrics (`vllm:omni_`)

Per-request timing breakdowns for diffusion (image/video) stages. Emitted at request finalize when `stage_metrics.diffusion_metrics` is present. All carry `{model_name, stage, replica}`.

| Metric | Type | Description |
|--------|------|-------------|
| `vllm:omni_diffusion_exec_s` | Histogram | DiT forward pass execution time per request in seconds |
| `vllm:omni_diffusion_preprocess_s` | Histogram | Diffusion input preprocessing time per request in seconds |
| `vllm:omni_diffusion_postprocess_s` | Histogram | Diffusion output postprocessing (VAE decode) time per request in seconds |

All three use `SECONDS_FAST_BUCKETS` (1 ms → 60 s).

### Example PromQL

```promql
# P99 DiT exec time by stage
histogram_quantile(0.99, rate(vllm_omni:diffusion_exec_s_bucket[5m]))

# Ratio of preprocess to total exec wall time
rate(vllm_omni:diffusion_preprocess_s_sum[5m])
  / rate(vllm_omni:diffusion_exec_s_sum[5m])
```

## Cross-Stage Transfer Metrics (`vllm:omni_`)

Per-physical-transfer histograms tracking the data hop between adjacent stages. Labels `{model_name, from_stage, from_replica, to_stage, to_replica}` let dashboards attribute latency to specific replica edges. `from_replica` / `to_replica` are resolved from the orchestrator's sticky-routing binding (`stage_pool.get_bound_replica_id(request_id)`), so no extra plumbing through `TransferEdgeStats` is needed.

| Metric | Type | Description |
|--------|------|-------------|
| `vllm:omni_transfer_size_bytes` | Histogram | Per-transfer payload size in bytes |
| `vllm:omni_transfer_tx_s` | Histogram | Sender-side time (serialize + submit to connector) |
| `vllm:omni_transfer_rx_s` | Histogram | Receiver-side time (recv + deserialize) |
| `vllm:omni_transfer_in_flight_s` | Histogram | Network in-flight time (TX done → RX recv start) |

## vLLM Engine Metrics (`vllm:`)

When the pipeline includes an LLM stage, the upstream vLLM engine exposes its full set of ~37 metric families under the `vllm:` prefix.

vLLM-Omni wraps the upstream `vllm.v1.metrics.loggers.PrometheusStatLogger` with `OmniPrometheusStatLogger` so that the original `engine` single label is reshaped into `stage` + `replica`. Every `vllm:*` family — TTFT, ITL, TPOT, e2e latency, KV cache usage, scheduler running/waiting, request success counts, etc. — therefore gains per-`(stage, replica)` visibility automatically. No omni-side duplicate is needed for the text path.

```text
# Before wrap:
vllm:num_requests_running{model_name="...", engine="1"}              3.0

# After wrap:
vllm:num_requests_running{model_name="...", stage="1", replica="0"}  2.0
vllm:num_requests_running{model_name="...", stage="1", replica="1"}  1.0
```

For the full list of upstream metrics, see [the vLLM docs](https://github.com/vllm-project/vllm/blob/main/docs/usage/metrics.md).

## Metric Availability by Pipeline Type

| Metric group | Multi-stage LLM (Qwen3-Omni) |
|---|---|
| `vllm:omni_` request tracking + latency | With `--log-stats` |
| `vllm:omni_` audio modality | With `--log-stats`, if pipeline has a talker stage |
| `vllm:omni_` transfer | With `--log-stats`, if pipeline has ≥ 2 stages |
| `vllm:` engine metrics (per `(stage, replica)`) | With `--log-stats` |
| `vllm:` MFU metrics | With `--log-stats --enable-mfu-metrics` |

## Naming Convention

- All time-bearing metrics use the `_s` suffix (values in seconds). Buckets are `SECONDS_BUCKETS` for e2e / generation-style values and `SECONDS_FAST_BUCKETS` (1 ms → 60 s) for the fine-grained transfer and audio-underrun values.
- Counters use the `_total` suffix (auto-appended by `prometheus_client`).
- Sizes use the `_bytes` suffix.
- All omni-specific families are prefixed `vllm:omni_`. The upstream `unregister_vllm_metrics()` function is monkey-patched (see `vllm_omni/patch.py`) to a scoped version that still strips upstream `vllm:*` collectors so multi-engine init within one process does not crash on duplicate registration, but preserves anything prefixed `vllm:omni_` / `vllm_omni`.
- Text and audio first-output use distinct families (`vllm:time_to_first_token_seconds` reused from upstream for text; `vllm:omni_audio_ttfp_s` for audio) rather than a single metric with a `modality` label.
