
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

- The benchmark talks to `http://<host>:<port>/v1/chat/completions`.
- If you run the server on another host or port, pass `--base-url` accordingly.

## 2. Supported Datasets

The benchmark supports three dataset modes via `--dataset`:

- `vbench`: Built-in prompt/data loader.
- `trace`: Heterogeneous request traces (each request can have different resolution/frames/steps).
- `random`: Synthetic prompts for quick smoke tests.

### VBench dataset

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

- `--base-url`: Server address (the script calls `.../v1/chat/completions`).
- `--model`: The OpenAI-compatible `model` field.
- `--task`: Task type (e.g., `t2i`, `t2v`, `i2i`, `i2v`).
- `--dataset`: Dataset mode (`vbench` / `trace` / `random`).
- `--num-prompts`: Number of requests to send.

Common optional flags:

- `--output-file`: Write metrics to a JSON file.
- `--disable-tqdm`: Disable the progress bar.

### Resolution / frames / steps: CLI defaults vs dataset fields

Related flags: `--width`, `--height`, `--num-frames`, `--fps`, `--num-inference-steps`.

- For `vbench` / `random`: these CLI flags act as global defaults for all generated requests.
- For `trace`: each request can carry its own fields (e.g., `width/height/num_frames/num_inference_steps`).

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
- For `--task t2v`: warmup requests are forced to use `num_frames=1` to make warmup faster and less noisy.

Traffic / concurrency flags:

- `--request-rate`: Target request rate (requests/second). If set to `inf`, the script sends all requests immediately.
- `--max-concurrency`: Max number of in-flight requests (default: `1`). This can hard-cap the achieved QPS: if it is too small, requests will queue behind the semaphore, and both achieved throughput and observed SLO attainment can be skewed.
