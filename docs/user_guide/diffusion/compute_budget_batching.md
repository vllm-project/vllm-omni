# Compute-Budget Diffusion Batching

Diffusion step execution can batch homogeneous requests together. A fixed
`max_num_seqs` works for one request shape, but different resolutions have
different per-step compute cost. Compute-budget batching keeps `max_num_seqs`
as a hard upper bound and derives the effective batch size from a profiled
compute-unit budget.

The built-in reference budget is only valid for the same profiling setup:

- reference request: 512x512 text-to-image
- model: Qwen-Image
- hardware: one Ascend 910B2 NPU
- sweet-spot batch size: 10
- compute-unit budget: `(512 / 16) * (512 / 16) * 10 = 10240`

If the model, hardware, backend, or profiling reference shape is different,
profile first and pass the generated `compute_unit_budget` explicitly.

For other homogeneous request types, the scheduler uses:

```text
effective_max_num_seqs = ceil(compute_unit_budget / request_compute_units)
```

`max_num_seqs` is still respected as the hard upper bound.

## First-Time Use on a New Setup

First run the profiler policy. Keep `max_num_seqs` large enough to cover the
batch sizes you want to sample. For diffusion batching, the recommended
reference traffic is high-concurrency 512x512 requests.

```bash
python3 -m vllm_omni.entrypoints.cli.main serve /data/Qwen-Image \
  --omni \
  --step-execution \
  --max-num-seqs 128 \
  --diffusion-batching-policy profile
```

Then send high-concurrency benchmark traffic using the 512x512 reference
request shape. The server log will contain `[DiffusionBatchProfiler]` records
with batch size, request compute units, total compute units, and denoise-step
time.

Generate a compute-budget config from the server log. The profiler groups the
512x512 records by batch size, filters outliers with IQR, uses median denoise
step time for each batch size, and applies a two-segment linear fit to locate
the 512x512 sweet-spot batch size.

```bash
python3 benchmarks/diffusion/diffusion_batching_profiler.py \
  --log-file /path/to/server.log \
  --output-json /tmp/diffusion_batching_config.json
```

Use the generated budget for later serving. On Ascend NPU, the serving command
is typically:

```bash
COMPUTE_UNIT_BUDGET=<generated_compute_unit_budget>

python3 -m vllm_omni.entrypoints.cli.main serve /data/Qwen-Image \
  --served-model-name Qwen/Qwen-Image \
  --omni \
  --port 8081 \
  --step-execution \
  --max-num-seqs 128 \
  --diffusion-batching-policy compute_budget \
  --diffusion-batching-config "{\"compute_unit_budget\":${COMPUTE_UNIT_BUDGET},\"drr_arrival_window_size\":256,\"log_stats\":true}" \
  --vae-use-tiling \
  --vae-use-slicing
```

The only value that normally needs to be carried from profiling to serving is
`compute_unit_budget`, which is computed as:

```text
compute_unit_budget = reference_compute_units * sweet_spot_batch_size
```

`drr_arrival_window_size` and `log_stats` are optional. The scheduler uses a
default sliding arrival window if `drr_arrival_window_size` is not set; `log_stats`
only enables verbose batching-policy logs.

## Use the Built-In Reference Profile

Use this only when the environment matches the built-in reference profile:
Qwen-Image on one Ascend 910B2 NPU, using 512x512 requests as the profiled
reference shape.

```bash
python3 -m vllm_omni.entrypoints.cli.main serve /data/Qwen-Image \
  --served-model-name Qwen/Qwen-Image \
  --omni \
  --port 8081 \
  --step-execution \
  --max-num-seqs 128 \
  --diffusion-batching-policy compute_budget \
  --diffusion-batching-config '{"compute_unit_budget":10240,"drr_arrival_window_size":256,"log_stats":true}' \
  --vae-use-tiling \
  --vae-use-slicing
```

For a 1024x1024 request, Qwen-Image packed latent compute units are
`(1024 / 16) * (1024 / 16) = 4096`, so the effective batch size is
`ceil(10240 / 4096) = 3`.

## Mixed-Shape Scheduling

Step-wise continuous batching still requires requests in the same running batch
to have compatible `SamplingParamsKey` values. A single FIFO waiting queue can
therefore suffer head-of-line blocking: a high-cost request at the front may
delay later low-cost requests even when those later requests could form an
efficient homogeneous batch.

The scheduler groups waiting requests by `SamplingParamsKey` and applies a
cost-aware deficit round-robin selection across non-empty queues:

- `cost_i` is estimated from the request shape as visual compute units.
- `prob_i` is estimated online from a sliding window of recent arrivals.
- Each non-empty queue accrues a normalized quantum proportional to
  `prob_i / cost_i`.
- The first selection starts from the lowest-cost queue. Later selections scan
  queues in ascending cost order from the DRR cursor and skip queues whose
  deficit is not yet sufficient.

Once a key is selected, normal homogeneous batching rules still apply. The
compute-budget policy then caps the batch size for that request type.

## Ascend 910B Result

On one Ascend 910B NPU with Qwen-Image, directly using fixed
`max_num_seqs=8` for mixed 512x512 and 1024x1024 requests took `830.84s` for
50 requests. With the profiled compute-budget policy and keyed DRR scheduling,
the same benchmark finished in `679.94s`, an 18.16% benchmark-duration speedup.
