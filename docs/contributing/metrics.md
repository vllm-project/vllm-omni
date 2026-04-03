
# Metrics

You can use these metrics in production to monitor the health and performance of the vLLM-omni system. Typical scenarios include:

- **Performance Monitoring**: Track throughput (e.g., `e2e_avg_tokens_per_s`), latency (e.g., `e2e_total_ms`), and resource utilization to verify that the system meets expected standards.

- **Debugging and Troubleshooting**: Use detailed per-request metrics to diagnose issues, such as high transfer times or unexpected token counts.

## Two Ways to Observe vLLM-Omni

`--log-stats` and `--enable-metrics` serve different purposes:

- `--log-stats` prints request-level summary tables to the server logs.
- `--enable-metrics` exports vLLM-Omni Prometheus metrics under the shared `/metrics` endpoint.

Use `--log-stats` when you want to inspect individual requests. Use `--enable-metrics` when you want long-running monitoring and dashboards. You can also enable both at the same time.

### Example Commands

Enable request-level log summaries only:

```bash
vllm serve /workspace/models/Qwen3-Omni-30B-A3B-Instruct --omni --port 8014 --log-stats
```

Enable both request-level log summaries and the Prometheus endpoint:

```bash
vllm serve /workspace/models/Qwen3-Omni-30B-A3B-Instruct \
  --omni \
  --port 8091 \
  --log-stats \
  --enable-metrics
```

If you only want the Prometheus endpoint, start the server with `--enable-metrics` and omit `--log-stats`.

## Request-Level Log Stats (`--log-stats`)

### Send a Request

```bash
python openai_chat_completion_client_for_multimodal_generation.py --query-type use_image
```

### Example Output

With `--log-stats` enabled, the server will output detailed metrics logs after each request. Example output:

#### Overall Summary

| Field                       | Value        |
|-----------------------------|--------------|
| e2e_requests                | 1            |
| e2e_wall_time_ms            | 41,299.190   |
| e2e_total_tokens            | 5,202        |
| e2e_avg_time_per_request_ms | 41,299.190   |
| e2e_avg_tokens_per_s        | 125.959      |
| e2e_stage_0_wall_time_ms    | 10,192.289   |
| e2e_stage_1_wall_time_ms    | 30,541.409   |
| e2e_stage_2_wall_time_ms    |    207.496   |

#### RequestE2EStats

| Field                   | Value      |
|-------------------------|------------|
| e2e_total_ms            | 41,299.133 |
| e2e_total_tokens        | 5,202      |
| transfers_total_time_ms | 245.895    |
| transfers_total_kbytes  | 138,089.939|

#### StageRequestStats

| Field                  | 0      | 1      | 2      |
|------------------------|--------|--------|--------|
| audio_generated_frames | 0      | 0      | 525,525|
| batch_id               | 38     | 274    | 0      |
| batch_size             | 1      | 1      | 1      |
| num_tokens_in          | 4,860  | 4,826  | 4,384  |
| num_tokens_out         | 67     | 275    | 0      |
| postprocess_time_ms    | 256.158| 0.491  | 0.000  |
| stage_gen_time_ms      | 9,910.007|30,379.198|160.745|

#### TransferEdgeStats

| Field               | 0->1        | 1->2       |
|---------------------|-------------|------------|
| size_kbytes         | 109,277.349 | 28,812.591 |
| tx_time_ms          | 78.701      | 18.790     |
| rx_decode_time_ms   | 111.865     | 31.706     |
| in_flight_time_ms   | 2.015       | 2.819      |

### How to Read the Tables

These logs include:

- **Overall summary**: total requests, wall time, average tokens/sec, etc.
- **E2E table**: per-request latency and token counts.
- **Stage table**: per-stage batch and timing details.
- **Transfer table**: data transfer and timing for each edge.

You can use these logs to monitor system health, debug performance, and analyze request-level metrics as described above.

### Metrics Scope: Offline vs Online Inference

For **offline inference** (batch mode), the summary includes both system-level metrics (aggregated across all requests) and per-request metrics. In this case, `e2e_requests` can be greater than 1, reflecting multiple completed requests in a batch.

For **online inference** (serving mode), the summary is always per-request. `e2e_requests` is always 1, and only request-level metrics are reported for each completion.

### Field Reference

#### Overall Summary

| Field                     | Meaning                                                                                       |
|---------------------------|-----------------------------------------------------------------------------------------------|
| `e2e_requests`            | Number of completed requests.                                                                 |
| `e2e_wall_time_ms`        | Wall-clock time span from run start to last completion, in ms.                                |
| `e2e_total_tokens`        | Total tokens counted across all completed requests (stage0 input + all stage outputs).        |
| `e2e_avg_time_per_request_ms` | Average wall time per request: `e2e_wall_time_ms / e2e_requests`.                        |
| `e2e_avg_tokens_per_s`    | Average token throughput over wall time: `e2e_total_tokens * 1000 / e2e_wall_time_ms`.       |
| `e2e_stage_{i}_wall_time_ms` | Wall-clock time span for stage `i`, in ms. Each stage's wall time is reported as a separate field, such as `e2e_stage_0_wall_time_ms` or `e2e_stage_1_wall_time_ms`. |

#### RequestE2EStats

| Field                     | Meaning                                                          |
|---------------------------|------------------------------------------------------------------|
| `e2e_total_ms`            | End-to-end latency in ms.                                        |
| `e2e_total_tokens`        | Total tokens for the request (stage0 input + all stage outputs). |
| `transfers_total_time_ms` | Sum of transfer edge `total_time_ms` for this request.           |
| `transfers_total_kbytes`  | Sum of transfer kbytes for this request.                         |

#### StageRequestStats

| Field                 | Meaning                                                                                                   |
|-----------------------|-----------------------------------------------------------------------------------------------------------|
| `batch_id`            | Batch index.                                                                                              |
| `batch_size`          | Batch size.                                                                                               |
| `num_tokens_in`       | Input tokens to the stage.                                                                                |
| `num_tokens_out`      | Output tokens from the stage.                                                                             |
| `stage_gen_time_ms`   | Stage compute time in ms, excluding post-processing time (reported separately as `postprocess_time_ms`). |
| `image_num`           | Number of images generated (for diffusion/image stages).                                                  |
| `resolution`          | Image resolution (for diffusion/image stages).                                                            |
| `postprocess_time_ms` | Diffusion/image: post-processing time in ms.                                                              |

#### TransferEdgeStats

| Field                | Meaning                                                                   |
|----------------------|---------------------------------------------------------------------------|
| `size_kbytes`        | Total kbytes transferred.                                                 |
| `tx_time_ms`         | Sender transfer time in ms.                                               |
| `rx_decode_time_ms`  | Receiver decode time in ms.                                               |
| `in_flight_time_ms`  | In-flight time in ms.                                                     |

### Sanity Checks

**Formulas:**

- `e2e_total_tokens = Stage0's num_tokens_in + sum(all stages' num_tokens_out)`

- `transfers_total_time_ms = sum(tx_time_ms + rx_decode_time_ms + in_flight_time_ms)` for every edge

**Using the example above:**

**e2e_total_tokens**

- Stage0's `num_tokens_in`: **4,860**
- Stage0's `num_tokens_out`: **67**
- Stage1's `num_tokens_out`: **275**
- Stage2's `num_tokens_out`: **0**

so `e2e_total_tokens = 4,860 + 67 + 275 + 0 = 5,202`, which matches the table value `e2e_total_tokens`.

**transfers_total_time_ms**

For each edge:

- 0->1: tx_time_ms (**78.701**) + rx_decode_time_ms (**111.865**) + in_flight_time_ms (**2.015**) = **192.581**

- 1->2: tx_time_ms (**18.790**) + rx_decode_time_ms (**31.706**) + in_flight_time_ms (**2.819**) = **53.315**

192.581 + 53.315 = **245.896** = transfers_total_time_ms, which matches the calculation (difference is due to rounding)

## Prometheus Metrics (`--enable-metrics`)

When `--enable-metrics` is set, vLLM-Omni registers its metrics into the shared Prometheus endpoint exposed by the serving process. If the server is started with `--port 8091`, you can inspect it with:

```bash
curl http://0.0.0.0:8091/metrics
```

The response contains:

- Default Python/runtime metrics such as `python_gc_*` and `process_*`.
- HTTP server metrics such as `http_requests_total` and `http_request_duration_seconds`.
- vLLM-Omni metrics prefixed with `vllm:omni_`.

### Example `/metrics` Output

The full endpoint can be long. A typical response looks like this:

```text
# HELP http_requests_total Total number of requests by method, status and handler.
# TYPE http_requests_total counter
http_requests_total{handler="/v1/chat/completions",method="POST",status="2xx"} 6.0
http_requests_total{handler="/v1/chat/completions",method="POST",status="4xx"} 1.0
http_requests_total{handler="/v1/images/generations",method="POST",status="5xx"} 1.0

# HELP vllm:omni_stage_generation_seconds Generation time for completed vLLM-Omni stage events.
# TYPE vllm:omni_stage_generation_seconds histogram
vllm:omni_stage_generation_seconds_bucket{final_output_type="text",le="0.5",model_name="Qwen3-Omni-30B-A3B-Instruct",stage_id="0"} 3.0
vllm:omni_stage_generation_seconds_bucket{final_output_type="text",le="1.0",model_name="Qwen3-Omni-30B-A3B-Instruct",stage_id="0"} 4.0
vllm:omni_stage_generation_seconds_count{final_output_type="text",model_name="Qwen3-Omni-30B-A3B-Instruct",stage_id="0"} 6.0
vllm:omni_stage_generation_seconds_sum{final_output_type="text",model_name="Qwen3-Omni-30B-A3B-Instruct",stage_id="0"} 11.19272518157959

# HELP vllm:omni_requests_total Total number of vLLM-Omni orchestrator requests.
# TYPE vllm:omni_requests_total counter
vllm:omni_requests_total{final_output_type="text",model_name="Qwen3-Omni-30B-A3B-Instruct"} 1.0
vllm:omni_requests_total{final_output_type="audio",model_name="Qwen3-Omni-30B-A3B-Instruct"} 3.0
vllm:omni_requests_total{final_output_type="image",model_name="Qwen3-Omni-30B-A3B-Instruct"} 2.0

# HELP vllm:omni_requests_aborted_total Total number of aborted vLLM-Omni orchestrator requests.
# TYPE vllm:omni_requests_aborted_total counter
vllm:omni_requests_aborted_total{final_output_type="text",model_name="Qwen3-Omni-30B-A3B-Instruct"} 2.0

# HELP vllm:omni_e2e_request_latency_seconds End-to-end latency of successful vLLM-Omni requests.
# TYPE vllm:omni_e2e_request_latency_seconds histogram
vllm:omni_e2e_request_latency_seconds_bucket{final_output_type="audio",le="10.0",model_name="Qwen3-Omni-30B-A3B-Instruct"} 1.0
vllm:omni_e2e_request_latency_seconds_bucket{final_output_type="audio",le="20.0",model_name="Qwen3-Omni-30B-A3B-Instruct"} 2.0
vllm:omni_e2e_request_latency_seconds_count{final_output_type="audio",model_name="Qwen3-Omni-30B-A3B-Instruct"} 3.0
vllm:omni_e2e_request_latency_seconds_sum{final_output_type="audio",model_name="Qwen3-Omni-30B-A3B-Instruct"} 72.84698152542114
```

### vLLM-Omni Metrics Exposed

The current vLLM-Omni Prometheus metrics are:

| Metric | Type | Labels | Meaning |
|--------|------|--------|---------|
| `vllm:omni_stage_generation_seconds` | Histogram | `model_name`, `stage_id`, `final_output_type` | Generation latency for completed stage events. |
| `vllm:omni_stage_postprocess_seconds` | Histogram | `model_name`, `stage_id`, `final_output_type` | Post-processing latency for completed stage events. |
| `vllm:omni_transfer_bytes_total` | Counter | `model_name`, `from_stage`, `to_stage`, `used_shm` | Total bytes transferred between stages. |
| `vllm:omni_requests_total` | Counter | `model_name`, `final_output_type` | Total number of orchestrator requests started. |
| `vllm:omni_requests_aborted_total` | Counter | `model_name`, `final_output_type` | Total number of aborted orchestrator requests. |
| `vllm:omni_e2e_request_latency_seconds` | Histogram | `model_name`, `final_output_type` | End-to-end latency for successful requests. |

!!! note
    Some metric families may appear without samples until the corresponding event happens. For example, `vllm:omni_stage_postprocess_seconds` is only populated after a stage records post-processing time, and `vllm:omni_transfer_bytes_total` only increases after inter-stage transfer traffic is observed.
