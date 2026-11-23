# Choosing workloads and deployment rows

A benchmark row answers one question: how fast does deployment D serve workload
W. Add a row only when it adds a distinct D or W. More prompts, steps, or
repeats on the same D+W do not widen coverage.

## Input/output paths

Pick workloads that exercise the model's advertised paths, so the perf row and
the feature rows stay aligned.

| Model path | `endpoint` | `extra_body` highlights | `dataset_name` |
|---|---|---|---|
| Text → image | `/v1/images/generations` | `width`, `height`, `num_inference_steps`, `true_cfg_scale`, `negative_prompt` | `random` |
| Image edit / multi-image | `/v1/images/edits` | edit params + reference image count | `random-mm` |
| Image → video | `/v1/videos` | `num_frames`, `fps`, `seed` | `random-mm` |

`negative_prompt` is inert unless classifier-free guidance is enabled
(`true_cfg_scale > 1` on Qwen-Image-2.1). A row that claims CFG coverage must set
the guidance scale explicitly and confirm the server does not log
`negative_prompt is passed but classifier-free guidance is not enabled`. A row
that measures the unguided path should omit `negative_prompt` rather than imply
coverage it does not have. If the config has no CFG row at all, say so:
removing an inert parameter does not cover the two-forward-pass path.

Qwen-Image-2.1's single-card case is a representative image row:
`/v1/images/generations`, `512x512`, 20 steps, with
`enable-diffusion-pipeline-profiler` on for stage attribution — so it is
profiler-instrumented, not a profiler-off baseline (see
[measurement-contract.md](measurement-contract.md#profiler-state)). The
higher-resolution `1536x1536` rows exist because
resolution materially changes HBM and latency; on a small-HBM card they require
`vae-use-tiling`.

Prefer text-to-image as the default when the model supports it: it exercises the
text encoder, the diffusion loop, and the VAE decode with the fewest moving
inputs. Add an edit or multi-image row only when the model advertises it and its
cost profile differs.

## Deployment rows

Watch each deployment you actually ship, not an arbitrary best case. Typical
rows for a diffusion model:

| Row | `serve_args` / mark | Question |
|---|---|---|
| Single-card baseline | default, `num_cards: 1` | Nominal latency/throughput and HBM |
| High-concurrency | `step-execution`, `max-num-seqs`, sweep `max_concurrency` | How the model scales under in-flight load |
| Multi-card (Ulysses / SP) | `ulysses-degree: N`, `num_cards: N` | Whether sequence parallelism pays off, and its overhead |
| CFG-parallel | `cfg-parallel-size: N` | Whether splitting the CFG branches helps |
| Memory-constrained / DLO | offload/quant/tiling flags | Cost of fitting a bigger row on less HBM |

Cover at least the nominal single-card baseline before adding parallelism rows;
a parallel row without a same-hardware single-card reference cannot show a
speedup.

## Concurrency sweep vs QPS sweep

Use one of `max_concurrency` / `request_rate` per row. Supplying both makes the
runner run the QPS sweep and then the concurrency sweep back to back.

- **Concurrency sweep** (`max_concurrency: [1,2,4,8]`, `num_prompts: [20,40,80,160]`)
  drives load with `--request-rate inf` and is the standard scaling watch. It
  needs `step-execution` + a sufficient `max-num-seqs` server-side, otherwise
  requests queue instead of overlapping.
- **QPS sweep** (`request_rate: [0.5, 1, 2]`) drives an arrival rate and is used
  when the served traffic is rate-shaped rather than bursty.

Set an explicit `num_warmups` for a concurrency row; the default grows with
concurrency but tail percentiles still need enough retained samples.

## Workload sizing

Resolution and step count are the dominant cost levers. A row should reflect the
resolution and step count the model is actually used at, not the cheapest value
that passes. If two resolutions both matter, they are two rows (they differ in
HBM and in VAE decode cost). Keep the step count equal across rows you intend to
compare, and record it in `extra_body.num_inference_steps`.

For a video model, frame count and FPS drive cost more than resolution alone;
pin both in `extra_body`.

## Reference example

`tests/dfx/perf/tests/test_qwen_image_21_vllm_omni.json` shows the full set:
a single-card baseline with the pipeline profiler, a 1536x1536 row, a
`step-execution` high-concurrency sweep (`max_concurrency: [1,2,4,8]`), and a
`ulysses-degree: 4` multi-card row. Reuse its shape; replace the resolution,
steps, and parallelism with the target model's own advertised deployment.
