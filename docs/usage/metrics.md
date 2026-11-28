# Production Metrics

vLLM-Omni exposes Prometheus metrics via the `/metrics` endpoint on the OpenAI-compatible API server.

```bash
vllm-omni serve Qwen/Qwen3-Omni-30B-A3B-Instruct --port 8000 --log-stats
curl http://localhost:8000/metrics
```

**`--log-stats` is required to populate metric data.** Without the flag, the endpoint still returns `200 OK` but the upstream `vllm:*` wrap is not registered at all, and the 15 `vllm_omni:*` families are registered as placeholders with no data written to them. This keeps the runtime cost essentially zero for deployments that don't need monitoring. With the flag, all ~80 families populate.

## Metric Namespaces

| Prefix | Source | Present when |
| -------- | -------- | -------------- |
| `vllm_omni:` | vLLM-Omni orchestrator / audio modality / cross-stage transfer | Pipeline-dependent |
| `vllm:` | Upstream vLLM engine, wrapped by `OmniPrometheusStatLogger` to expose `{stage, replica}` | Pipeline includes an LLM (AR) stage |
| `http_` / `process_` | Uvicorn / Python runtime | Always |

## Pipeline-Level Metrics (`vllm_omni:`)

Defined in `vllm_omni/metrics/prometheus.py`. Track request lifecycle across the full multi-stage pipeline.

### Request counts

| Metric | Type | Labels | Description |
| -------- | ------ | -------- | ------------- |
| `vllm_omni:num_requests_running` | Gauge | `model_name` | Pipeline-global in-flight requests (dispatched to engine, not yet finalized) |
| `vllm_omni:num_requests_waiting` | Gauge | `model_name` | Requests waiting in the Orchestrator queue |
| `vllm_omni:requests_success_total` | Counter | `model_name`, `finished_reason` | Total requests by completion reason. `finished_reason` ∈ {`stop`, `length`, `abort`, ...} mirroring upstream `vllm:request_success_total`; aborts cover client disconnect / cancellation paths in addition to upstream `FinishReason.ABORT` |

### Latency

| Metric | Type | Labels | Description |
| -------- | ------ | -------- | ------------- |
| `vllm_omni:e2e_request_latency_s` | Histogram | `model_name` | Pipeline-global end-to-end request latency in seconds |

## Audio Modality Metrics (`vllm_omni:`)

Emitted at request finalize, except for `audio_ttfp_s` (the first audio packet/frame from streaming Chat or Speech APIs) and `audio_underrun_s` / `audio_continuity_ok_total` (Chat or Speech streaming finalize, after the chunk stream is exhausted). Non-streaming Speech latency is represented by `e2e_request_latency_s`, not TTFP. All audio metrics carry `{model_name, stage, replica}` plus the listed extra label.

| Metric | Type | Extra label | Description |
| -------- | ------ | ------------- | ------------- |
| `vllm_omni:audio_ttfp_s` | Histogram | — | Time from request arrival to first audio packet/frame |
| `vllm_omni:audio_duration_s` | Histogram | — | Audio content duration (`audio_frames / sample_rate`) |
| `vllm_omni:audio_rtf` | Histogram | — | Real-time factor (`stage_gen_time_s / audio_duration_s`); streaming TTS SLO red line `< 1`; uses `RTF_BUCKETS` |
| `vllm_omni:audio_frames_total` | Counter | — | Cumulative audio frame count; throughput via `rate()` |
| `vllm_omni:audio_underrun_s` | Histogram | — | Per-request worst-case player deficit; `> 0` indicates listener heard silent gaps |
| `vllm_omni:audio_continuity_ok_total` | Counter | `threshold_ms` | Incremented when the request's worst underrun stayed below `threshold_ms` |
| `vllm_omni:audio_skipped_requests_total` | Counter | `reason` | Silent-loss counter — code2wav rejected malformed codec input and returned `200 OK` with empty audio |

The continuity math comes from `vllm_omni/benchmarks/audio_continuity.py::compute_continuity_stats` so the server-side observation aligns with the bench-side definition.
For Speech, continuity applies to raw audio, SSE, and WebSocket streaming. WAV headers and empty chunks are excluded; non-streaming Speech requests do not emit continuity samples.

Continuity rate for the default 100 ms threshold can be derived from the successful-continuity counter and the underrun Histogram's request count:

```promql
sum by (model_name, stage, replica) (rate(vllm_omni:audio_continuity_ok_total{threshold_ms="100"}[5m]))
/
sum by (model_name, stage, replica) (rate(vllm_omni:audio_underrun_s_count[5m]))
```

## Diffusion Metrics (`vllm_omni:`)

Per-request timing breakdowns for diffusion (image/video) stages. Emitted at request finalize when `stage_metrics.diffusion_metrics` is present. All carry `{model_name, stage, replica}`.

| Metric | Type | Description |
| -------- | ------ | ------------- |
| `vllm_omni:diffusion_exec_s` | Histogram | DiT forward pass execution time per request in seconds |
| `vllm_omni:diffusion_exec_per_step_s` | Histogram | DiT forward pass execution time per denoising step in seconds (`exec_s / num_inference_steps`) |
| `vllm_omni:diffusion_preprocess_s` | Histogram | Diffusion input preprocessing time per request in seconds |
| `vllm_omni:diffusion_postprocess_s` | Histogram | Diffusion output postprocessing (VAE decode) time per request in seconds |

`diffusion_exec_s` uses `SECONDS_BUCKETS` (50 ms → 300 s); the other three use `SECONDS_FAST_BUCKETS` (1 ms → 60 s). `diffusion_exec_per_step_s` is skipped when `num_inference_steps` is not set.

### Example PromQL

```promql
# P99 DiT exec time by stage
histogram_quantile(0.99, rate(vllm_omni:diffusion_exec_s_bucket[5m]))

# Ratio of preprocess to total exec wall time
rate(vllm_omni:diffusion_preprocess_s_sum[5m])
  / rate(vllm_omni:diffusion_exec_s_sum[5m])
```

## Cross-Stage Transfer Metrics (`vllm_omni:`)

Per-physical-transfer histograms tracking the data hop between adjacent stages. Labels `{model_name, from_stage, from_replica, to_stage, to_replica}` let dashboards attribute latency to specific replica edges. `from_replica` / `to_replica` are resolved from the orchestrator's sticky-routing binding (`stage_pool.get_bound_replica_id(request_id)`), so no extra plumbing through `TransferEdgeStats` is needed.

| Metric | Type | Description |
| -------- | ------ | ------------- |
| `vllm_omni:transfer_size_bytes` | Histogram | Per-transfer payload size in bytes |
| `vllm_omni:transfer_tx_s` | Histogram | Sender-side time (serialize + submit to connector) |
| `vllm_omni:transfer_rx_s` | Histogram | Receiver-side time (recv + deserialize) |
| `vllm_omni:transfer_in_flight_s` | Histogram | Network in-flight time (TX done → RX recv start) |

## vLLM Engine Metrics (`vllm:`)

When the pipeline includes an LLM stage, the upstream vLLM engine exposes its full set of ~37 metric families under the `vllm:` prefix.

vLLM-Omni wraps the upstream `vllm.v1.metrics.loggers.PrometheusStatLogger` with `OmniPrometheusStatLogger` so that the original `engine` single label is reshaped into `stage` + `replica`. Every `vllm:*` family — TTFT, ITL, TPOT, e2e latency, KV cache usage, scheduler running/waiting, request success counts, etc. — therefore gains per-`(stage, replica)` visibility automatically. No omni-side duplicate is needed for the text path.

Final AR audio stages can coalesce token-only decode steps into the next audio output. Their coalesced intervals are excluded from both upstream ITL histograms and Omni `vllm_itls_ms`; ordinary step samples, completed-request token counts, TTFT, TPOT and completion statistics remain available. ITL coverage therefore excludes coalesced steps. An empty ITL list means no individual-step samples were available; the corresponding zero average does not mean zero decode latency.

`vllm:generation_tokens_total` counts tokens delivered to the output processor. Cancelling a request removes its processor state, so tokens still held by the scheduler at cancellation are not counted. Token-only step coalescing can therefore lower this counter for aborted audio streams compared with per-step delivery; it is not an exact measure of engine work for cancelled requests. Requests that complete normally flush their held tokens and retain their full generation counts.

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
| --- | --- |
| `vllm_omni:` request tracking + latency | With `--log-stats` |
| `vllm_omni:` audio modality | With `--log-stats`, if pipeline has a talker stage |
| `vllm_omni:` transfer | With `--log-stats`, if pipeline has ≥ 2 stages |
| `vllm:` engine metrics (per `(stage, replica)`) | With `--log-stats` |
| `vllm:` MFU metrics | With `--log-stats --enable-mfu-metrics` |

## Naming Convention

- All time-bearing metrics use the `_s` suffix (values in seconds). Buckets are `SECONDS_BUCKETS` for e2e / generation-style values and `SECONDS_FAST_BUCKETS` (1 ms → 60 s) for the fine-grained transfer and audio-underrun values.
- Counters use the `_total` suffix (auto-appended by `prometheus_client`).
- Sizes use the `_bytes` suffix.
- All omni-specific families are prefixed `vllm_omni:`. The upstream `unregister_vllm_metrics()` function is monkey-patched (see `vllm_omni/patch.py`) to a scoped version that still strips upstream `vllm:*` collectors so multi-engine init within one process does not crash on duplicate registration, but preserves anything prefixed `vllm_omni:`.
- Text and audio first-output use distinct families (`vllm:time_to_first_token_seconds` reused from upstream for text; `vllm_omni:audio_ttfp_s` for audio) rather than a single metric with a `modality` label.

## Distributed tracing

Install the `tracing` extra and pass `--otlp-traces-endpoint` to enable tracing:

```bash
pip install 'vllm-omni[tracing]'
vllm-omni serve <model> --otlp-traces-endpoint http://localhost:4317
```

The default transport is gRPC. For HTTP, set
`OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=http/protobuf` and pass an endpoint ending
in `/v1/traces`. Standard OTel service, resource and sampler environment variables
apply. Tracing is disabled when the flag is absent.

HTTP requests accept W3C `traceparent`, `tracestate` and `baggage`. Python callers
can use an active OTel span or pass `trace_headers` to `AsyncOmni.generate`.
A server span covers the HTTP response, including streaming; `omni.pipeline`
covers the admitted request in the orchestrator, with one `omni.stage` span per
submitted stage. These are orchestration lifetimes, including waits and transfers,
not GPU kernel timings. Stage spans are model independent and are ended on
completion, cancellation, failure or shutdown. `omni.stage.first_output` records
the first nonempty output observed by the orchestrator, not client receipt.

Realtime and duplex connections add `omni.session` and `omni.turn` spans, recording
response completion, failure, or cancellation. Realtime inference requests inherit
the active turn context.

No per-token or per-chunk spans, prompts, generated content or media are recorded
by the pipeline instrumentation.

## Execution summary

Pass `--enable-execution-summary` to log what each request actually ran, for
example to check a benchmark run against its declared workload. `[OmniTiming]`
reports how long a request took; `[OmniExecution]` reports what it ran: one JSON
line per finished request, logged next to its `[OmniTiming]` line, with
`schema_version` (currently `1`) and `event: "diffusion_request_completed"`. Each line stands on its
own; there are no separate startup lines to correlate.

| Field | Meaning |
| --- | --- |
| `request_id` | The id the client sees, to match client-side records: the `POST /v1/videos` job `id` or the `X-Request-Id` header. |
| `engine_request_id` | Internal id, as in `[OmniTiming] req=`. |
| `status`, `failure_reason` | `completed` or `failed`; the reason as in `vllm_omni:requests_failed` (e.g. `client_disconnect`, `stage_error`). |
| `stages` | One entry per stage: `stage_id`, `replica_id`, `output_type`, `batch_size`; `diffusion` and `runtime` for diffusion stages, `num_tokens_in` and `num_tokens_out` for the others, `audio` (`duration_s`, `sample_rate`) for stages that output audio samples. |

`diffusion` is what the request ran. Values the request left to the model
defaults report what the pipeline resolved where the pipeline reports it
(`DiffusionOutput.execution_facts`, e.g. MiniMax-H3's size and flow shifts):

| Field | Meaning |
| --- | --- |
| `width`, `height`, `size_source` | Output size and its source: `tensor` (typed video tensor), `image`, `pipeline` (reported by the pipeline, e.g. for video it encoded in the worker) or `request` (resolved sampling parameters). |
| `num_frames`, `frames_source` | Frame count before optional frame interpolation and its source: `tensor`, `encoder` (frame counts the encoder reported for video the worker encoded), `container` (frame count read from the encoded MP4), `stream` (sum of streamed chunks) or `request`. |
| `fps`, `seed` | Output rate; resolved seed, including a server-assigned random seed. |
| `num_inference_steps` | Requested denoising steps (`null` when the model default applied). |
| `completed_steps` | Denoising steps the pipeline completed, counted by the model runner from the pipeline's denoising-loop progress bar (`ProgressBarMixin`); step execution reports the step scheduler's index. `null` for pipelines without that progress bar. |
| `flow_shift` | Shift the pipeline reported, else the request's or server's value; `null` when unknown. |
| `guidance_scale` | Value set by the request; `null` when the request left it to the model default or the pipeline runs without CFG (`cfg: false`). |
| `num_outputs_per_prompt` | Outputs generated for the request. |
| `lora` | LoRA adapters in effect: the request's and the one the server loaded at startup (`lora_path`); `null` for none. |
| other keys | Further values the pipeline resolved, e.g. MiniMax-H3's `audio_flow_shift` and `cfg: false` (CFG-distilled, no guidance). |

`runtime` is how the stage ran it; the engine computes it once at startup and the
worker once after the model loads:

| Field | Meaning |
| --- | --- |
| `vllm_omni_version`, `vllm_version`, `torch_version` | Software versions (`dev` when vLLM-Omni runs from a source tree without version metadata). |
| `model`, `model_revision` | Model as served; a local directory that resolves into a Hugging Face hub snapshot reports the repo id and snapshot commit. |
| `model_class_name`, `configured_dtype`, `quantization` | Pipeline class, the stage's configured dtype (`--dtype`), quantization method (`null` for none). |
| `cache_backend`, `cache_config` | Step cache backend (`none` when off) and its settings (`null` when off). |
| `parallel_config`, `max_num_seqs` | Parallel degrees; maximum requests batched together. |
| `device`, `world_size` | Accelerator name and number of ranks. |
| `attention` | The backend each attention layer resolved, counted per role. |
| `weight_dtypes` | Weight elements per dtype of each pipeline component as loaded on the reporting rank (rank 0; a component sharded with tensor parallelism shows its shard) (parameters and persistent buffers, so packed quantized weights such as NVFP4 show), e.g. `{"transformer": {"bfloat16": …}, "video_vae": {"float32": …, "float16": …}}`. |
| other keys | Facts the pipeline reports through `execution_runtime_facts()`. MiniMax-H3 reports `component_quantization` (the method each component loaded with, including checkpoint-shipped FP8/NVFP4 that `--quantization` does not show), `adaln_cache` (whether its exact AdaLN projection cache is on; default on), `compute_dtypes` (`video_vae_decode: float16`) and `video_vae_decoder: nvfp4` when the NVFP4 decoder is installed. |

MiniMax-H3 on one B300; the request set `prompt`, `aspect_ratio`, `seed` and `num_inference_steps=49`:

```text
[OmniExecution] {"schema_version": 1, "event": "diffusion_request_completed", "request_id": "video_gen_<job id>", "engine_request_id": "video_gen_<job id>-<8 hex>", "status": "completed", "failure_reason": null, "stages": [{"stage_id": 0, "replica_id": 0, "output_type": "video", "batch_size": 1, "diffusion": {"width": 1344, "height": 768, "size_source": "pipeline", "num_frames": 209, "frames_source": "encoder", "fps": 24, "seed": 42, "num_inference_steps": 49, "completed_steps": 49, "guidance_scale": null, "flow_shift": 12.0, "num_outputs_per_prompt": 1, "lora": null, "audio_flow_shift": 3.0, "cfg": false}, "runtime": {"vllm_omni_version": "dev", "vllm_version": "0.34.0", "torch_version": "2.14.0+cu130", "model": "MiniMaxAI/MiniMax-H3", "model_revision": "42ed227ee7df40d41602854ae760620d6eb651fe", "model_class_name": "MiniMaxH3ModularPipeline", "configured_dtype": "bfloat16", "quantization": null, "cache_backend": "none", "cache_config": null, "parallel_config": {"tensor_parallel_size": 1, "pipeline_parallel_size": 1, "data_parallel_size": 1, "sequence_parallel_size": 1, "ulysses_degree": 1, "ring_degree": 1, "cfg_parallel_size": 1, "vae_patch_parallel_size": 1, "use_hsdp": false}, "max_num_seqs": 1, "device": "NVIDIA B300 SXM6 AC", "world_size": 1, "attention": {"minimax_h3.token_refiner": {"FLASHINFER_ATTN": 2}, "self": {"FLASHINFER_ATTN": 50}}, "weight_dtypes": {"transformer": {"float32": 17222160, "bfloat16": 33105770752}, "text_encoder": {"bfloat16": 25753095920}, "video_vae": {"float32": 186993464, "float16": 2416877568}, "audio_vae": {"float32": 151326585}}, "compute_dtypes": {"video_vae_decode": "float16"}}}]}
```

Failed requests whose diffusion stage did not return have no `diffusion` or
`runtime` entry. Nothing is logged per denoising step; counting adds one integer
update per step and only runs while the flag is set.
