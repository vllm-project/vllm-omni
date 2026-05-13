# Production Metrics

vLLM-Omni exposes Prometheus metrics via the `/metrics` endpoint on the
OpenAI-compatible API server. The metrics fall into three categories depending
on the pipeline type.

```bash
vllm-omni serve Qwen/Qwen3-Omni-30B-A3B-Instruct --port 8000
curl http://localhost:8000/metrics
```

## Metric Namespaces

| Prefix | Source | Present when |
|--------|--------|--------------|
| `vllm:omni_` | vLLM-Omni orchestrator / diffusion stages / modality / transfer | Always / pipeline-dependent |
| `vllm:` | Upstream vLLM engine, wrapped by `OmniPrometheusStatLogger` to expose `{stage, replica}` | Pipeline includes an LLM (AR) stage |
| `http_` / `process_` | Uvicorn / Python runtime | Always |

## Pipeline-Level Metrics (`vllm:omni_`)

These metrics are defined in `vllm_omni/metrics/prometheus.py` and track
request lifecycle across the full multi-stage pipeline.

### Request Tracking

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `vllm:omni_num_requests_running` | Gauge | `model_name` | Requests currently running across all pipeline stages |
| `vllm:omni_num_requests_waiting` | Gauge | `model_name` | Requests waiting to be scheduled |
| `vllm:omni_requests_success_total` | Counter | `model_name`, `finished_reason` | Total requests by completion reason. `finished_reason` ∈ {`stop`, `length`, `abort`, ...} mirroring upstream `vllm:request_success_total`; aborts include the previous "fail" path |

### Latency

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `vllm:omni_e2e_request_latency_seconds` | Histogram | `model_name` | End-to-end request latency in seconds |
| `vllm:omni_request_queue_time_seconds` | Histogram | `model_name` | Time spent waiting in the request queue |

## Modality Metrics (`vllm:omni_`)

Per-modality business-semantic histograms emitted at request finalize (or at
first-packet time for `audio_ttfp_seconds`). All carry
`{model_name, stage, replica}` labels.

### Audio (talker stage)

| Metric | Type | Description |
|--------|------|-------------|
| `vllm:omni_audio_ttfp_seconds` | Histogram | Time from request arrival to first audio packet (streaming hook) |
| `vllm:omni_audio_duration_seconds` | Histogram | Generated audio content duration (`audio_frames / sample_rate`) |
| `vllm:omni_audio_rtf` | Histogram | Real-time factor `stage_gen_time_s / audio_duration_s`; SLO red line `< 1` |
| `vllm:omni_audio_frames_total` | Counter | Cumulative audio frames generated; throughput via `rate()` |

### Image (diffusion stage)

| Metric | Type | Description |
|--------|------|-------------|
| `vllm:omni_image_ttfp_seconds` | Histogram | Time from request arrival to image emission (degenerates to `image_generation_time` when no intermediate streaming) |
| `vllm:omni_image_num_total` | Counter | Cumulative images generated |
| `vllm:omni_image_generation_time_seconds` | Histogram | Per-request image stage generation time (image has no RTF — no content duration) |

### Video (diffusion stage)

| Metric | Type | Description |
|--------|------|-------------|
| `vllm:omni_video_generation_time_seconds` | Histogram | Per-request video stage generation time |

> `video_duration_seconds` and `video_rtf` are deferred — diffusion video
> pipelines (i2v / t2v / cogvideo / hunyuan / wan) expose `num_frames` + `fps`
> in heterogeneous shapes and a clean abstraction is out of scope for this
> iteration.

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
| `vllm:omni_transfer_tx_time_ms` | Histogram | Sender-side time (serialize + submit to connector) |
| `vllm:omni_transfer_rx_decode_time_ms` | Histogram | Receiver-side time (recv + deserialize) |
| `vllm:omni_transfer_in_flight_time_ms` | Histogram | Network in-flight time (TX done → RX recv start) |

> The TX-side observe path (`record_transfer_tx`) is already wired but only
> fires once the connector adapter (`try_send_via_connector`) is invoked from
> the main code path; until then only the RX-side families
> (`rx_decode_time_ms` + `in_flight_time_ms`) are populated.

## Diffusion Engine Metrics (`vllm:omni_`)

These histograms are populated only when the pipeline includes a diffusion
stage. The `engine` label here is the diffusion stage_id (omni-side
families bypass the `OmniPrometheusStatLogger` wrap, so they retain the
original `engine` label rather than being relabelled to `stage` + `replica`).

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `vllm:omni_diffusion_preprocess_time_ms` | Histogram | `model_name`, `engine` | Input preprocessing time per request |
| `vllm:omni_diffusion_exec_time_ms` | Histogram | `model_name`, `engine` | DiT forward pass execution time per request |
| `vllm:omni_diffusion_postprocess_time_ms` | Histogram | `model_name`, `engine` | Output postprocessing time (VAE decode) per request |
| `vllm:omni_diffusion_step_time_ms` | Histogram | `model_name`, `engine` | Total diffusion step time per request |

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
| `vllm:omni_` transfer | If pipeline has ≥ 2 stages | No |
| `vllm:omni_` diffusion timing | If pipeline has a diffusion stage | Yes |
| `vllm:` engine metrics (per `(stage, replica)`) | Yes | No |
| `vllm:` MFU metrics | With `--enable-mfu-metrics` | No |

## Naming Convention

vLLM-Omni pipeline metrics use the `vllm:omni_` prefix to distinguish them
from upstream per-engine `vllm:` metrics. The upstream
`unregister_vllm_metrics()` function is monkey-patched to a no-op (see
`vllm_omni/patch.py`) so that these metrics are not destroyed during engine
initialization.

For the audio / image / video families, the RFC convention is "co-position,
different name": each modality's time-to-first-output uses a distinct name
(`vllm:time_to_first_token_seconds` for text — reused from upstream;
`vllm:omni_audio_ttfp_seconds` for audio; `vllm:omni_image_ttfp_seconds`
for image) rather than a single metric with a `modality` label. The three
modalities differ in unit semantics (text token vs. audio packet vs. image
emission) and typical latency magnitudes, so independent histogram buckets
fit each modality better.
