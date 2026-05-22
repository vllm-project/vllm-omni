# Production Metrics

vLLM-Omni exposes Prometheus metrics via the `/metrics` endpoint on the
OpenAI-compatible API server. The metrics fall into three categories depending
on the pipeline type.

```bash
vllm-omni serve Qwen/Qwen3-Omni-30B-A3B-Instruct --port 8000
curl http://localhost:8000/metrics
```

The locked `vllm_omni:*` family set is 23 — see
[RFC #3545](https://github.com/vllm-project/vllm-omni/issues/3545).

## Metric Namespaces

| Prefix | Source | Present when |
|--------|--------|--------------|
| `vllm:omni_` | vLLM-Omni orchestrator / diffusion stages / modality / transfer | Always / pipeline-dependent |
| `vllm:` | Upstream vLLM engine, wrapped by `OmniPrometheusStatLogger` to expose `{stage, replica}` | Pipeline includes an LLM (AR) stage |
| `http_` / `process_` | Uvicorn / Python runtime | Always |

## Pipeline-Level Metrics (`vllm:omni_`)

Defined in `vllm_omni/metrics/prometheus.py`. Track request lifecycle across
the full multi-stage pipeline.

### Request counts

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `vllm:omni_num_requests_running` | Gauge | `model_name` | Pipeline-global in-flight requests (dispatched to engine, not yet finalized) |
| `vllm:omni_num_requests_waiting` | Gauge | `model_name` | Requests waiting in the Orchestrator queue |
| `vllm:omni_requests_success_total` | Counter | `model_name`, `finished_reason` | Total requests by completion reason. `finished_reason` ∈ {`stop`, `length`, `abort`, ...} mirroring upstream `vllm:request_success_total`; aborts include the previous "fail" path (G6) |

### Latency

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `vllm:omni_e2e_request_latency_s` | Histogram | `model_name` | Pipeline-global end-to-end request latency in seconds |

## Audio Modality Metrics (`vllm:omni_`)

Emitted at request finalize, except for `audio_ttfp_s` (streaming-hook at the
first audio packet) and `audio_underrun_s` / `audio_continuity_ok_total`
(streaming finalize, after the chunk stream is exhausted). All carry
`{model_name, stage, replica}` plus the listed extra label.

| Metric | Type | Extra label | Description |
|--------|------|-------------|-------------|
| `vllm:omni_audio_ttfp_s` | Histogram | — | Time from request arrival to first audio packet/frame |
| `vllm:omni_audio_duration_s` | Histogram | — | Audio content duration (`audio_frames / sample_rate`) |
| `vllm:omni_audio_rtf` | Histogram | — | Real-time factor (`stage_gen_time_s / audio_duration_s`); streaming TTS SLO red line `< 1`; uses `RTF_BUCKETS` |
| `vllm:omni_audio_frames_total` | Counter | — | Cumulative audio frame count; throughput via `rate()` |
| `vllm:omni_audio_underrun_s` | Histogram | — | Per-request worst-case player deficit; `> 0` indicates listener heard silent gaps |
| `vllm:omni_audio_continuity_ok_total` | Counter | `threshold_ms` | Incremented when the request's worst underrun stayed below `threshold_ms` |
| `vllm:omni_audio_skipped_requests_total` | Counter | `reason` | Silent-loss counter — code2wav rejected malformed codec input and returned `200 OK` with empty audio |

The continuity math comes from
`vllm_omni/benchmarks/audio_continuity.py::compute_continuity_stats` so the
server-side observation aligns with the bench-side definition (G4 single
source of truth).

## Visual Modality Metrics (`vllm:omni_`)

### Diffusion-internal (per-request decomposition)

`vllm_omni/metrics/prometheus.py::observe_diffusion_metrics` converts the
engine-emitted millisecond values to seconds at the emit boundary. PromQL
recipe for per-step latency:

```promql
rate(vllm:omni_diffusion_exec_s_sum[5m])
  / on(model_name, stage, replica) rate(num_inference_steps[5m])
```

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `vllm:omni_diffusion_preprocess_s` | Histogram | `model_name`, `stage`, `replica` | tokenizer + text_encoder + vae.encode |
| `vllm:omni_diffusion_exec_s` | Histogram | same | Per-request executor work time (sum of denoise steps) |
| `vllm:omni_diffusion_postprocess_s` | Histogram | same | vae.decode |

### Business semantics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `vllm:omni_image_num_total` | Counter | `model_name`, `stage`, `replica` | Cumulative image count; throughput via `rate()` |
| `vllm:omni_image_generation_s` | Histogram | same | Per-request total image generation stage time (image has no RTF — no content duration) |
| `vllm:omni_video_duration_s` | Histogram | same | Video content duration (`num_frames / fps`) |
| `vllm:omni_video_rtf` | Histogram | same | Real-time factor; uses `RTF_BUCKETS` |
| `vllm:omni_video_generation_s` | Histogram | same | Per-request total video generation stage time |

## Cross-Stage Transfer Metrics (`vllm:omni_`)

Per-physical-transfer histograms tracking the data hop between adjacent
stages. Labels `{model_name, from_stage, from_replica, to_stage, to_replica}`
let dashboards attribute latency to specific replica edges. `from_replica` /
`to_replica` are resolved from the orchestrator's sticky-routing binding
(`stage_pool.get_bound_replica_id(request_id)`), so no extra plumbing through
`TransferEdgeStats` is needed.

| Metric | Type | Description |
|--------|------|-------------|
| `vllm:omni_transfer_size_bytes` | Histogram | Per-transfer payload size in bytes |
| `vllm:omni_transfer_tx_s` | Histogram | Sender-side time (serialize + submit to connector) |
| `vllm:omni_transfer_rx_s` | Histogram | Receiver-side time (recv + deserialize) |
| `vllm:omni_transfer_in_flight_s` | Histogram | Network in-flight time (TX done → RX recv start) |

> The TX-side observe path (`record_transfer_tx`) is wired but only fires once
> the connector adapter (`try_send_via_connector`) is invoked from the main
> code path; until then only the RX-side families
> (`transfer_rx_s` + `transfer_in_flight_s`) are populated.

## vLLM Engine Metrics (`vllm:`)

When the pipeline includes an LLM stage, the upstream vLLM engine exposes its
full set of ~37 metric families under the `vllm:` prefix.

vLLM-Omni wraps the upstream `vllm.v1.metrics.loggers.PrometheusStatLogger`
with `OmniPrometheusStatLogger` so that the original `engine` single label
is reshaped into `stage` + `replica`. Every `vllm:*` family — TTFT, ITL,
TPOT, e2e latency, KV cache usage, scheduler running/waiting, request
success counts, etc. — therefore gains per-`(stage, replica)` visibility
automatically. No omni-side duplicate is needed for the text path.

```text
# Before wrap (PR #3362):
vllm:num_requests_running{model_name="...", engine="1"}              3.0

# After wrap (this branch):
vllm:num_requests_running{model_name="...", stage="1", replica="0"}  2.0
vllm:num_requests_running{model_name="...", stage="1", replica="1"}  1.0
```

For the full list of upstream metrics, see
[the vLLM docs](https://github.com/vllm-project/vllm/blob/main/docs/usage/metrics.md).

## Metric Availability by Pipeline Type

| Metric group | Multi-stage LLM (Qwen3-Omni) | Diffusion-only (Z-Image-Turbo) |
|---|---|---|
| `vllm:omni_` request tracking + latency | Yes | Yes |
| `vllm:omni_` audio modality | If pipeline has a talker stage | No |
| `vllm:omni_` image / video modality | If pipeline has a diffusion stage | Yes |
| `vllm:omni_` diffusion-internal timing | If pipeline has a diffusion stage | Yes |
| `vllm:omni_` transfer | If pipeline has ≥ 2 stages | No |
| `vllm:` engine metrics (per `(stage, replica)`) | Yes | No |
| `vllm:` MFU metrics | With `--enable-mfu-metrics` | No |

## Naming Convention

- All time-bearing metrics use the `_s` suffix (values in seconds).
  Buckets are `SECONDS_BUCKETS` for e2e / generation-style values and
  `SECONDS_FAST_BUCKETS` (1 ms → 60 s) for the fine-grained diffusion-internal
  and transfer values.
- Counters use the `_total` suffix (auto-appended by `prometheus_client`).
- Sizes use the `_bytes` suffix.
- All omni-specific families are prefixed `vllm:omni_`. The upstream
  `unregister_vllm_metrics()` function is monkey-patched to a no-op (see
  `vllm_omni/patch.py`) so these are not destroyed during engine initialization.
- For audio / image / video families, the RFC convention is "co-position,
  different name": each modality's time-to-first-output uses a distinct name
  (`vllm:time_to_first_token_seconds` for text — reused from upstream;
  `vllm:omni_audio_ttfp_s` for audio) rather than a single metric with a
  `modality` label. Image has no streaming first-packet equivalent, so
  `image_ttfp_*` is intentionally absent.
