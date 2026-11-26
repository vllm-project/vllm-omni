# Measurement contract

A performance number is only comparable if the question, workload, backend, and
generator inputs are fixed. Pin all of them before quoting a metric; an
optimization and its benchmark must move together.

## Fixed for one measured row

| Element | Why it must be pinned |
|---|---|
| Model + revision | Weights, config, and tokenizer determine equal work |
| Server flags (`serve_args`) | Parallelism, cache, offload, quant, compile all change the number |
| Workload (`extra_body`, `endpoint`, resolution, steps) | Different work is not a comparison |
| Input/output tokens (`random_input_len`/`random_output_len`), or dataset | Equal prompt length makes results comparable |
| Attention backend / `DIFFUSION_ATTENTION_BACKEND` | Different kernels give different numbers |
| Warmup count and sample count | Tail percentiles need enough samples |
| RNG/seed handling | Diffusion output depends on the generator |

## Command freezing

A comparable run holds all three of these constant and only varies the one
thing under test:

1. the server launch command (resolved from `server_params`),
2. the request shape (`benchmark_params`: endpoint, `extra_body`, tokens), and
3. the benchmark driver settings (runner, warmup, sweep, percentiles).

The unified result JSON records only part of this: the request count,
concurrency, input/output lengths, `tokenizer_id`, `model_id`, `Hardware`,
`endpoint_type`, and the resolved `baseline`. It does **not** record `extra_body`
(resolution, steps, guidance), `serve_args`, `num_warmups`, or
`percentile_metrics`, so the artifact alone cannot reconstruct the workload.
When a number is meant to be reused, keep the config row, the commit SHA and
model revision it was read at, the resolved server command, and the environment
pins with the result (in the PR/issue or a sidecar file). The checked-in config
can change under a row, so a row reference alone does not identify the workload.

## Warmup and sample count

Set an explicit `num_warmups` when the default is wrong; the runner defaults to
2 for a QPS sweep and `max(2, concurrency)` for a concurrency sweep. The first
requests carry lazy initialization and autotuning, so excluding them is part of
the definition of the metric, not a workaround. `num_prompts` counts measured
requests only: warmups are extra and are not included in `completed`.

Use a sample count that can shoulder the percentile you quote. A p99 needs on
the order of 100 samples to be a tail rather than the slowest observation; at
the shipped 10-prompt rows, `p99_e2el_ms` is effectively the maximum of ten
requests, so report the median and spread instead (or raise `num_prompts`).

## Profiler state

Keep `enable-diffusion-pipeline-profiler` **off** for numbers you intend to
compare or baseline; enabling it adds overhead and inflates latency. Turn it on
only when you need stage attribution (below) and label the output as
profiler-instrumented.

Caveat on the shipped reference: `test_qwen_image_21_vllm_omni.json` sets
`enable-diffusion-pipeline-profiler: true` on **every** row, including the
single-card rows. Treat those rows as profiler-instrumented. Their latency is not
directly comparable to a profiler-off row, and they are not a clean absolute
baseline; either strip the flag for a baseline row or keep the profiler state
consistent across everything you compare.

## Stage attribution

When latency must be split by stage, use the diffusion pipeline profiler and
attribute delta instead of guessing. A total-only number hides which stage
moved. The result's `stage_durations_mean/p50/p99` keys are
`<PipelineClass>.<profiler target>` plus `queue_wait_ms`; the default targets are
`vae.encode`, `vae.decode`, `diffuse`, `text_encoder.forward`, and
`tokenizer.forward`, and a pipeline may override them (for example
`transformer.forward`), so read the keys the result actually contains. A
profiled `denoise_step` is folded into `.diffuse`, not reported separately.
Values are in **seconds**, except `queue_wait_ms`, which is in milliseconds.
Attribute a claimed improvement to the capturing row, not to a different run's
row.

## Verify the effective workload

The runner forwards `extra_body` without validating it, and the result JSON does
not record it (see Command freezing). After a run, prove the row did the work it
claims before quoting a number:

- **Step count.** With the profiler on, `mean_denoise_step_latency_ms` is the
  measured denoise time divided by the **resolved** step count, so
  reverse-checking recovers the count the server recorded (it is not an
  independent count of executed iterations).
  The stage duration is in seconds and the per-step latency in milliseconds, so
  convert first:
  `stage_durations_mean["<PipelineClass>.diffuse"] * 1000 / mean_denoise_step_latency_ms`
  should equal the count the row intends to request. A mismatch means the value
  was clipped or replaced by a default; if the key itself is misspelled,
  `extra_body.num_inference_steps` is absent, so compare against the intended
  value from the config row. This check requires the pipeline profiler:
  `stage_durations_mean` is empty (`{}`) without it, and
  `mean_denoise_step_latency_ms` then falls back to
  `stage_gen_time_ms / num_inference_steps`, a stage-level average rather than a
  denoise-step figure — so apply the check only to profiler-instrumented rows.
- **Guidance.** If the row claims CFG coverage, confirm the log does not say
  `negative_prompt is passed but classifier-free guidance is not enabled since
  true_cfg_scale <= 1`; that message means the row measured the unguided path.
- **Backend.** Confirm the selector line (`Resolved diffusion attention backend
  ... via ...`) matches the pinned or expected backend.
- **Resolution.** The result does not record width/height; verify it from the
  config row or the request log.

Record the checks with the result. A row that cannot show them is unvalidated,
not passing.

## Units and fields

Report durations in milliseconds. Read the metric you actually claim; do not
convert units in prose.

Unified runner (`vllm bench serve --omni`):

| Field | Meaning |
|---|---|
| `mean_e2el_ms`, `median_e2el_ms`, `p99_e2el_ms` | End-to-end latency for the requested `percentile_metrics` (e.g. `e2el`); at concurrency > 1 it includes queueing |
| `image_throughput`, `request_throughput` | Images or requests per second; equal on image rows |
| `mean_denoise_step_latency_ms` | Denoise time / resolved step count (falls back to stage generation time without the profiler); pairs with the diffuse stage for the step-count check |
| `mean_peak_memory_mb`, `median_peak_memory_mb` | Observed peak HBM |
| `stage_durations_mean/p50/p99` | Per-stage attribution; keys are `<PipelineClass>.<stage>` plus `queue_wait_ms`; values in seconds (`queue_wait_ms` in ms) |
| `request_stage_metrics` | Per-request stage records; use for distributions, not just the mean |
| `completed`, `failed`, `num_prompts` | Request accounting; `completed` excludes warmups |
| `duration` | Benchmark wall time |
| `Hardware`, `model_id`, `tokenizer_id`, `endpoint_type` | Runtime metadata actually used |
| `backend` | Endpoint path (same value as `endpoint_type`); **not** the attention backend |
| `baseline` | Resolved baseline block, if configured (recorded, not enforced) |

The legacy runner emits an aggregate record with `throughput_qps`,
`latency_mean/median/p50/p95/p99`, `peak_memory_mb_*`, and
`stage_durations_*`.

## Hardware and parallelism metadata

Record the device and the parallelism actually used (cards, `ulysses-degree`,
CFG/HSDP) because the comparison bucket is the runtime hardware, not the
declared `mark`. Result filenames use the runtime label from
`get_runtime_resource_label()`; on the default H100 pool the filename omits it,
while the result's `Hardware` field still records `H100`. Record the attention
backend with the same care — see [Attention backend](#attention-backend).

## Run-to-run variance

One run is not a baseline. Repeat the row on the same hardware, quote the spread
(not a single sample), and only then treat a value as a baseline candidate. Keep
arrays aligned with the sweep length. Before attributing a delta between two
runs, check that neither run's generator strayed or fell back to a different
kernel.

## Memory

Peak HBM is part of the deployment row, not an afterthought. When resolution or
batch forces a memory trade-off, keep the trade-off explicit (for example
`vae-use-tiling` at 1536x1536 on a small-HBM card) and record the peak value
alongside the latency. A memory-rescuing flag that changes the number must be
part of the pinned contract for that row.

## Attention backend

Different kernels give different numbers. The result's `backend` field holds the
endpoint path, not the attention backend, so an env-pinned backend leaves no
trace in the artifact: the selector log line
(`Resolved diffusion attention backend ... via ...`) is the only evidence. Record
it with the number, and reuse the backend the row's CI step pins instead of
relying on auto-selection.


## What not to claim

- Do not compare across different hardware, backend, or resolution.
- Do not present an unprofiled and a profiled run as one series.
- Do not present a single run as a regression baseline.
- Do not attribute an improvement to a change that was not the only difference.
