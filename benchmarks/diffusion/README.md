
# Diffusion Serving Benchmark (Image/Video)

This folder contains an online-serving benchmark script for diffusion models.
It sends requests to a vLLM OpenAI-compatible endpoint and reports throughput,
latency percentiles, and optional SLO attainment.

The main entrypoint is:

- `benchmarks/diffusion/diffusion_benchmark_serving.py`

## 1. Quick Start

1. Start the server:

```bash
vllm serve Qwen/Qwen-Image --omni --port 8099
```

1. Run a minimal benchmark:

```bash
python3 benchmarks/diffusion/diffusion_benchmark_serving.py \
 --base-url http://localhost:8099 \
 --model Qwen/Qwen-Image \
 --task t2i \
 --dataset vbench \
 --num-prompts 5
```

### Notes

- By default, image tasks talk to `http://<host>:<port>/v1/chat/completions`; video tasks talk to `/v1/videos`.
- If you run the server on another host or port, pass `--base-url` accordingly.

## 2. Supported Datasets

The benchmark supports three dataset modes via `--dataset`:

- `vbench`: Built-in prompt/data loader.
- `trace`: Heterogeneous request traces (each request can have different resolution/frames/steps).
- `random`: Synthetic prompts for quick smoke tests.

### VBench dataset

`vbench` only provides prompt data (and image paths for i2v/i2i); it does not carry
per-request generation fields. In this mode, all requests share CLI values:
`--width --height --num-frames --fps --num-inference-steps`
(pass `--width` and `--height` together).

Example (`t2v`):

```bash
python3 benchmarks/diffusion/diffusion_benchmark_serving.py \
 --base-url http://localhost:8099 \
 --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
 --task t2v \
 --dataset vbench \
 --num-prompts 50 \
 --width 640 --height 480 \
 --num-frames 81 --fps 16 \
 --num-inference-steps 40
```

Note: `vbench` can also be used for other tasks such as `t2i` / `i2v` (and `i2i`). For `t2i`, the loader reuses VBench t2v text prompts; for `i2v` / `i2i`, it loads the VBench i2v dataset (with image paths).

If you use i2v/i2i bench datasets and need auto-download support, you may need:

```bash
uv pip install gdown
```

### Trace dataset

Use `--dataset trace` to replay a trace file. The trace can specify per-request fields such as:

- `width`, `height`
- `num_frames` (video)
- `num_inference_steps`
- `seed`, `fps`
- optional `slo_ms` (per-request SLO target)

By default (when `--dataset-path` is not provided), the script downloads a default trace from
the HuggingFace dataset repo `asukaqaqzz/Dit_Trace`. The default filename can depend on `--task`
(e.g., `t2v` uses a video trace).

Current defaults:

- `--task t2i` -> `sd3_trace.txt`
- `--task t2v` -> `cogvideox_trace.txt`

You can point to your own trace using `--dataset-path`.

## 3. Benchmark Parameters

### Basic flags

- `--base-url`: Server address; `--endpoint` selects the path appended to this base URL.
- `--model`: The OpenAI-compatible `model` field.
- `--endpoint`: API endpoint path. Leading `/` is optional, e.g. `/v1/videos` or `v1/videos`.
- `--task`: Task type (e.g., `t2i`, `t2v`, `i2i`, `i2v`).
- `--dataset`: Dataset mode (`vbench` / `trace` / `random`).
- `--num-prompts`: Number of requests to send.

Common optional flags:

- `--output-file`: Write metrics to a JSON file.
- `--disable-tqdm`: Disable the progress bar.

### Resolution / frames / steps: CLI defaults vs dataset fields

Related flags: `--width`, `--height`, `--num-frames`, `--fps`, `--num-inference-steps`.

- For `vbench` / `random`: these CLI flags act as global defaults for all generated requests.
- For `trace`: requests can carry their own fields (e.g., `width/height/num_frames/num_inference_steps`), with overrides/fallbacks as below.

Precedence rules for `trace` (i.e., what actually gets sent):

- `width/height`: if either `--width` or `--height` is explicitly set, it overrides per-request values from the trace; otherwise per-request values are used when present.
- `num_frames`: per-request `num_frames` takes precedence; otherwise fall back to `--num-frames`.
- `num_inference_steps`: per-request `num_inference_steps` takes precedence; otherwise fall back to `--num-inference-steps`.

### SLO, warmup, and max concurrency

Enable SLO evaluation with `--slo`.

- If a request in the trace already has `slo_ms`, that value is used.
- Otherwise, the script runs warmup requests to infer a base unit time, estimates `expected_ms` by linearly scaling with area/frames/steps, and then sets `slo_ms = expected_ms * --slo-scale`.

Warmup flags:

- `--warmup-requests`: Number of warmup requests.
- `--warmup-num-inference-steps`: Steps used during warmup.
- `--warmup-concurrency`: Maximum concurrent warmup requests. Use this to warm
  the same batch shape as the measured run instead of warming only batch=`1`.
- For `--task t2v`: warmup requests are forced to use `num_frames=1` to make warmup faster and less noisy.

Traffic / concurrency flags:

- `--request-rate`: Target request rate (requests/second). If set to `inf`, the script sends all requests immediately.
- `--max-concurrency`: Max number of in-flight requests (default: `1`). This can hard-cap the achieved QPS: if it is too small, requests will queue behind the semaphore, and both achieved throughput and observed SLO attainment can be skewed.

### Video job timeout and failures

`--video-job-timeout` controls how long each `/v1/videos` job may be polled,
including time queued on the server (default: `900` seconds). For long video
jobs at high concurrency, use a larger budget, for example
`--video-job-timeout 1800`. This also applies to warmup jobs. The polling budget
starts after job creation; it excludes time waiting for the client concurrency
semaphore. Individual HTTP calls still use the aiohttp session timeout.

An expired job is counted as failed and deleted, which cancels queued or running
server generation. The benchmark continues processing the remaining requests.
The progress bar counts both successful and failed requests. The final report
shows the failure count and the first 10 errors; `--output-file` saves all errors
with their request IDs in `request_errors`.

### Batched warmup note

For batched serving runs, warm the same in-flight shape you plan to measure.
For example, a run with `--max-concurrency 8` should usually also use
`--warmup-requests 8 --warmup-concurrency 8`; otherwise the first measured
batch may still pay compile or CUDA-graph capture cost.

For a Qwen-Image continuous-batching replay example, see
[`performance_dashboard/qwen_image_serving_performance.md`](./performance_dashboard/qwen_image_serving_performance.md).

## HunyuanImage3 reference-prefix reuse

The prefix-cache benchmark uses the unified `vllm bench serve --omni` runner,
not this legacy diffusion benchmark. It reuses the two-image IT2I input on a
single DiT stage, comparing dense, paged without caching, and paged with caching.
See [Shared-reference benchmark](../../docs/design/feature/prefix_caching.md#shared-reference-benchmark)
for the configuration, commands, and separate partial-hit accuracy coverage.

The accuracy regression compares paged-no-cache, partial-hit and exact-repeat
outputs against the checked-in official-repository reference-image goldens at
**50 steps, CFG=2.5, seeds 43/45** (both CFG branches remain on CFGP1).
It uses the AR-to-DiT image criteria: CLIP ≥ 90, SSIM ≥ 0.26, PSNR ≥ 12.5 dB.
`HUNYUAN_IMAGE3_INCLUDE_DENSE=1` adds dense as another comparison to the goldens.
It also requires actual reference-image hits and model-side query slicing,
and checks bitwise repeatability for repeated requests at the same hit boundary.
Images, logs, deployment YAMLs, hit traces and quality metrics are saved under
pytest's temporary output directory. `HUNYUAN_IMAGE3_MODEL` selects a local model;
DFX uses the repository's normal model/cache resolution.

## MixFusion benchmarks

MixFusion packs a multi-request, mixed-resolution DiT batch into one flat varlen
attention call instead of padding every request to the longest sequence. It only
pays off when the per-request token counts share a large common divisor: the
chunk size is their GCD, so `1024x1024` + `1024x768` (4096 + 3072 tokens, GCD
1024) batch well, while `1024x1024` + `640x640` (4096 + 1600, GCD 64) fall below
the 256-token minimum and run independently instead.

Four entry points cover different levels of the stack.

### Synthetic DiT-like stack

`mixfusion_kernel_benchmark.py` builds a HunyuanImage-3.0-shaped transformer stack
(RMSNorm, fused QKV, SwiGLU MLP) at the requested hidden size and times three
layouts of the same sequences: independent, padded, and MixFusion. It never loads
a checkpoint, so it isolates the attention/layout effect and runs in seconds;
sweep resolutions here before spending GPU time on a real model.

```bash
python benchmarks/diffusion/mixfusion_kernel_benchmark.py \
  --image-sizes 1024x1024,512x512 --layers 4 --hidden-size 4096 \
  --heads 32 --dtype bfloat16 --iters 20
```

Its JSON reports `correctness` (max-abs diff against independent), `time_ms` per
strategy, `speedup`, `peak_memory_mb`, and `relative_work`.

### Real pipeline

`mixfusion_benefit.py` loads the real pipeline through `DiffusionEngine` and
compares one-request-at-a-time serving against the same prompts admitted
concurrently. `--family` selects the model family and its defaults:

| | `--family qwen` | `--family hunyuan` |
| --- | --- | --- |
| Default model | `Qwen/Qwen-Image` | `tencent/HunyuanImage-3.0-Instruct` |
| Resolution carried in | sampling params | prompt dict |
| Small-GCD pre-check | yes | n/a |

```bash
DIFFUSION_ATTENTION_BACKEND=FLASH_ATTN \
python benchmarks/diffusion/mixfusion_benefit.py \
  --family qwen --model Qwen/Qwen-Image \
  --image-sizes 1024x1024,1024x768 --steps 8 --iters 3

python benchmarks/diffusion/mixfusion_benefit.py \
  --family hunyuan --model tencent/HunyuanImage-3.0-Instruct \
  --image-sizes 1024x1024,512x512 --steps 20 --iters 3 \
  --tensor-parallel-size 4 --quantization fp8
```

Pass `--no-enable-mixfusion` for a batching-only control run, and
`--json-output <path>` to also write the result JSON to disk.

`--output-type latent --dump-outputs-dir <dir>` saves the final latents as one
`.pt` per request (used by the parity tooling below).

### Full-matrix A/B endorsement harness

For PR-scale A/B endorsements (e.g. main vs a MixFusion PR head), a set of
scripts drives `mixfusion_benefit.py` over two repo checkouts and aggregates
the evidence:

| script | role |
| --- | --- |
| `mixfusion_ab_bench.sh` | per-case driver: the 9-case timing matrix (A/B x W1/W2/W3 x mixfusion off/on) into `$RESULTS` |
| `mixfusion_ab_matrix.sh` | thin wrapper fixing `RESULTS=results_v3` |
| `mixfusion_ab_parity.sh` | latent dump matrix (both sides, `--output-type latent`, one `.pt` per request) into `$PARITY`, incl. A-vs-A noise-floor controls |
| `mixfusion_ab_summarize.py` | aggregates `$RESULTS` + `$PARITY` into a markdown report: timing, flat-varlen evidence, latent parity with an explicit noise-band tolerance, and an image-level PSNR/SSIM layer |
| `mixfusion_ab_image_parity.py` | decodes dumped latents through the same VAE and compares images pairwise (PSNR/SSIM/max pixel diff) |
| `mixfusion_ab_latent_diff.py` | quick pairwise diff of two dump dirs |

Workloads: W1 = single 1024x1024 request, W2 = 8x 1024x1024, W3 = 8 mixed
(1024x1024 x2, 1024x768 x2, 768x1024 x2, 512x512 x2).

Both sides run the *same* driver code: `VLLM_OMNI_ROOT` + `PYTHONPATH` point
the engine at the intended checkout (`A_WT` baseline, `B_WT` candidate), so the
comparison isolates the candidate's changes. The candidate should sit on top
of the measured baseline commit. If one side auto-compiles graphs while the
other does not, keep eager on both (the default, `EAGER=0` to disable).

```bash
export A_WT=/path/to/baseline B_WT=/path/to/candidate
export RESULTS=$PWD/results_v3 PARITY=$PWD/parity_v3
bash benchmarks/diffusion/mixfusion_ab_matrix.sh   # 9 timing cases, resumable
bash benchmarks/diffusion/mixfusion_ab_parity.sh   # 13 dump cases, resumable
python benchmarks/diffusion/mixfusion_ab_summarize.py \
  --a-worktree "$A_WT" --b-worktree "$B_WT" --report report.md
```

Each timing case is a fresh engine process; `result.json` files and `done`
markers make the runs resumable, `PILOT=1` switches to a steps=8 smoke run,
and `ONLY=<case>` re-runs a single case (`FORCE=1` to overwrite). The report's
acceptance bar: no B-off regression on any workload, W1 bit-identical to A,
and every A-vs-B-on image pair >= 40 dB PSNR / >= 0.98 SSIM after the
same-VAE decode.

Large batches need a normal-sized `/dev/shm` (the dump path copies result
tensors through shared memory); raise the container limit or the 8-request
cases can fail in the engine's output transfer.

### Online serving

To measure MixFusion through the OpenAI-compatible server, replay a trace with
mixed resolutions and attach the extra args to every request:

```bash
vllm serve Qwen/Qwen-Image --omni --port 8099
```

```bash
python3 benchmarks/diffusion/diffusion_benchmark_serving.py \
 --base-url http://localhost:8099 --model Qwen/Qwen-Image --task t2i \
 --dataset trace --num-prompts 20 --max-concurrency 4 \
 --warmup-requests 4 --warmup-concurrency 4 \
 --extra-body '{"extra_args": {"enable_mixfusion": true}}'
```

The trace supplies per-request `width`/`height`; `--extra-body` is merged into
every request body. Warm the same in-flight shape as the measured run, otherwise
the first batch still pays compile/CUDA-graph capture cost.

`OMNI_PROFILER_ACTIVITIES` narrows the pipeline profiler's activity list (default
`CPU,CUDA`), for example `OMNI_PROFILER_ACTIVITIES=CPU,CUDA,NPU`.
