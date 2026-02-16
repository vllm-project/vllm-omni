
# Metrics vLLM-Omni:

You can use these metrics in production to monitor the health and performance of the vLLM-omni system. Typical scenarios include:
- **Performance Monitoring**: Track throughput (e.g., `e2e_avg_tokens_per_s`), latency (e.g., `e2e_total_ms`), and resource utilization to verify that the system meets expected standards.
- **Debugging and Troubleshooting**: Use detailed per-request metrics to diagnose issues, such as high transfer times or unexpected token counts.

## How to Enable and View Metrics

### 1. Start the Service with Metrics Logging

```bash
vllm serve /workspace/models/Qwen3-Omni-30B-A3B-Instruct --omni --port 8014 --log-stats
```

### 2. Send a Request

```bash
python openai_chat_completion_client_for_multimodal_generation.py --query-type use_image
```

### 3. What You Will See

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


These logs include:
- **Overall summary**: total requests, wall time, average tokens/sec, etc.
- **E2E table**: per-request latency and token counts.
- **Stage table**: per-stage batch and timing details.
- **Transfer table**: data transfer and timing for each edge.

You can use these logs to monitor system health, debug performance, and analyze request-level metrics as described above.


## Metrics Scope: Offline vs Online Inference

For **offline inference** (batch mode), the summary includes both system-level metrics (aggregated across all requests) and per-request metrics. In this case, `e2e_requests` can be greater than 1, reflecting multiple completed requests in a batch.

For **online inference** (serving mode), the summary is always per-request. `e2e_requests` is always 1, and only request-level metrics are reported for each completion.

---

## Parameter Details

| Field                     | Meaning                                                                                       |
|---------------------------|----------------------------------------------------------------------------------------------|
| `e2e_requests`            | Number of completed requests.                                                                |
| `e2e_wall_time_ms`        | Wall-clock time span from run start to last completion, in ms.                               |
| `e2e_total_tokens`        | Total tokens counted across all completed requests (stage0 input + all stage outputs).       |
| `e2e_avg_time_per_request_ms` | Average wall time per request: `e2e_wall_time_ms / e2e_requests`.                        |
| `e2e_avg_tokens_per_s`    | Average token throughput over wall time: `e2e_total_tokens * 1000 / e2e_wall_time_ms`.      |
| `e2e_stage_{i}_wall_time_ms` | Wall-clock time span for stage i, in ms. Each stage's wall time is reported as a separate field, e.g., `e2e_stage_0_wall_time_ms`, `e2e_stage_1_wall_time_ms`, etc. |

---

## E2E Table (per request)

| Field                     | Meaning                                                               |
|---------------------------|-----------------------------------------------------------------------|
| `e2e_total_ms`            | End-to-end latency in ms.                                             |
| `e2e_total_tokens`        | Total tokens for the request (stage0 input + all stage outputs).      |
| `transfers_total_time_ms` | Sum of transfer edge `total_time_ms` for this request.                |
| `transfers_total_kbytes`  | Sum of transfer kbytes for this request.                              |


---

## Stage Table (per stage event / request)

| Field                     | Meaning                                                                                         |
|---------------------------|-------------------------------------------------------------------------------------------------|
| `batch_id`                | Batch index.                                                                                    |
| `batch_size`              | Batch size.                                                                                     |
| `num_tokens_in`           | Input tokens to the stage.                                                                      |
| `num_tokens_out`          | Output tokens from the stage.                                                                   |
| `stage_gen_time_ms`       | Stage compute time in ms, excluding postprocessing time (reported separately as `postprocess_time_ms`). |
| `image_num`               | Number of images generated (for diffusion/image stages).                                        |
| `resolution`              | Image resolution (for diffusion/image stages).                                                                  |
| `postprocess_time_ms` | Diffusion/image: post-processing time in ms.                                                    |

---

## Transfer Table (per edge / request)

| Field                | Meaning                                                                   |
|----------------------|---------------------------------------------------------------------------|
| `size_kbytes`        | Total kbytes transferred.                                                 |
| `tx_time_ms`         | Sender transfer time in ms.                                               |
| `rx_decode_time_ms`  | Receiver decode time in ms.                                               |
| `in_flight_time_ms`  | In-flight time in ms.                                                     |


## Expectation of the Numbers (Verification)

**Formulas:**
- `e2e_total_tokens = Stage0's num_tokens_in + sum(all stages' num_tokens_out)`
- `transfers_total_time_ms = sum(tx_time_ms + rx_decode_time_ms + in_flight_time_ms)` for every edge

**Using the example above:**

### e2e_total_tokens
- Stage0's `num_tokens_in`: **4,860**
- Stage0's `num_tokens_out`: **67**
- Stage1's `num_tokens_out`: **275**
- Stage2's `num_tokens_out`: **0**

So,
```
e2e_total_tokens = 4,860 + 67 + 275 + 0 = 5,202
```
This matches the table value: `e2e_total_tokens = 5,202`.

### transfers_total_time_ms
For each edge:
- 0->1: tx_time_ms (**78.701**) + rx_decode_time_ms (**111.865**) + in_flight_time_ms (**2.015**) = **192.581**
- 1->2: tx_time_ms (**18.790**) + rx_decode_time_ms (**31.706**) + in_flight_time_ms (**2.819**) = **53.315**

Sum: 192.581 + 53.315 = **245.896**

The table shows `transfers_total_time_ms = 245.895`, which matches the calculation (difference is due to rounding).
