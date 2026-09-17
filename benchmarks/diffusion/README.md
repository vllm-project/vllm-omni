
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

2. Run a minimal benchmark:

```bash
python3 benchmarks/diffusion/diffusion_benchmark_serving.py \
	--base-url http://localhost:8099 \
	--model Qwen/Qwen-Image \
	--task t2i \
	--dataset vbench \
	--num-prompts 5
```

**Notes**

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
- `--warmup-prompt`: Override warmup text only, keeping each request's reference
  image, seed, and other parameters. Use a disjoint prompt for partial-prefix
  benchmarks. A failed custom-prompt warmup aborts measurement.
- `--warmup-num-inference-steps`: Steps used during warmup.
- `--warmup-concurrency`: Maximum concurrent warmup requests. Use this to warm
  the same batch shape as the measured run instead of warming only batch=`1`.
- For `--task t2v`: warmup requests are forced to use `num_frames=1` to make warmup faster and less noisy.

Traffic / concurrency flags:

- `--request-rate`: Target request rate (requests/second). If set to `inf`, the script sends all requests immediately.
- `--max-concurrency`: Max number of in-flight requests (default: `1`). This can hard-cap the achieved QPS: if it is too small, requests will queue behind the semaphore, and both achieved throughput and observed SLO attainment can be skewed.

### Batched warmup note

For batched serving runs, warm the same in-flight shape you plan to measure.
For example, a run with `--max-concurrency 8` should usually also use
`--warmup-requests 8 --warmup-concurrency 8`; otherwise the first measured
batch may still pay compile or CUDA-graph capture cost.

For a Qwen-Image continuous-batching replay example, see
[`performance_dashboard/qwen_image_serving_performance.md`](./performance_dashboard/qwen_image_serving_performance.md).

## HunyuanImage3 reference-prefix reuse

The [DFX configuration](../../tests/dfx/perf/tests/test_hunyuan_image3_prefix_caching.json)
compares dense, paged without prefix caching, and paged with prefix caching on
the same DiT-only TP4 / CFGP1 deployment. It uses the checked-in reference image,
eight distinct editing instructions, a fixed seed per run, and concurrency 1.
Reference tokens precede the changed text, allowing partial-prefix reuse.
There is no AR-to-DiT KV transfer. The performance matrix covers guidance 1.0
(one row) and 2.5 (two CFG rows on CFGP1), each at 2 and 8 denoising steps.

Two warmups use a separate instruction and are excluded from latency metrics.
The four guidance/step combinations use separate seeds (42–45), **fixed across
all requests and modes within each run**. This prevents later combinations
from hitting complete prompts left by earlier ones. Do not increase `num-prompts`
beyond the eight dataset rows without adding distinct instructions: the custom
dataset cycles, which would change the workload to exact-request repetition.
Local image paths may be relative to the JSONL (existing cwd-relative paths
still take precedence).

From the repository root, with four GPUs allocated by your environment's GPU
scheduler:

```bash
python -m pytest tests/dfx/perf/scripts/run_diffusion_benchmark.py \
  --test-config-file tests/dfx/perf/tests/test_hunyuan_image3_prefix_caching.json -s
python -m pytest tests/e2e/accuracy/test_hunyuan_image3_prefix_cache_accuracy.py -s
```

The DFX runner writes latency/QPS metrics under `tests/dfx/perf/results` (override
with `DIFFUSION_BENCHMARK_DIR`). Compare each guidance/step combination separately:

- paged-no-cache vs dense isolates paged execution overhead;
- paged-prefix vs paged-no-cache isolates prefix reuse savings;
- paged-prefix vs dense measures the net user-visible benefit.

There are no prefilled performance baselines or assumed speedup thresholds.
This is a favorable low-step, repeated-reference workload, not a representative
50-step quality or maximum-batch-throughput benchmark. Prefix reuse saves
first-step Transformer work, not reference encoding or all denoising steps.
To audit hit lengths, run a separate diagnostic with `VLLM_LOGGING_LEVEL=DEBUG`
and inspect `Diffusion prefix prefill` worker lines; avoid mixing DEBUG timings
with normal performance runs.

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
