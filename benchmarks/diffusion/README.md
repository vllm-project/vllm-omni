
# Diffusion Serving Benchmark (Image/Video)

## Weight-loading microbenchmark

Use the CUDA-only microbenchmark to compare pageable H2D, pinned H2D,
dtype-converting H2D, and fused CPU cast plus pinned H2D without downloading a
model:

```bash
python benchmarks/diffusion/bench_weight_load_staging.py --size-mb 512 --repeats 5
```

### End-to-end startup timing

Startup paths emit parseable INFO records without an opt-in flag:

```text
[StartupTiming] phase=model.pipeline_construct duration_s=... status=ok device=cuda load_format=default
```

Collect process, snapshot-download, pipeline-construction, weight-loading,
worker, runtime-setup, warmup, and shutdown spans with:

```bash
python examples/offline_inference/text_to_video/text_to_video.py \
  --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --height 480 --width 832 --num-frames 9 --num-inference-steps 2 \
  2>&1 | tee startup.log

grep '\[StartupTiming\]' startup.log
```

Add `--enable-diffusion-pipeline-profiler` to report synchronized text-encode,
denoise, and VAE-decode durations, plus a JSON stage summary. GPU synchronization
makes that mode diagnostic; do not compare its end-to-end time against an
unprofiled run. The offline example also reports output processing separately
from generation.

### TP=2 cooperative-loading validation

A real TP=2 run must assign both devices in the deploy config **and** set the
nested tensor-parallel size. The CLI tensor-parallel flag alone does not assign
stage devices. The checked-in `wan2_2_tp2.yaml` captures the validated topology.
For a fixed checkpoint revision, run:

```bash
MODEL_PATH=$(python - <<'PY'
from huggingface_hub import snapshot_download

print(snapshot_download(
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    revision="5be7df9619b54f4e2667b2755bc6a756675b5cd7",
))
PY
)

CUDA_VISIBLE_DEVICES=0,1 python \
  examples/offline_inference/text_to_video/text_to_video.py \
  --model "$MODEL_PATH" \
  --deploy-config benchmarks/diffusion/wan2_2_tp2.yaml \
  --tensor-parallel-size 2 --enforce-eager \
  --prompt "A brown and white dog is running on the grass." \
  --negative-prompt "" --seed 42 \
  --height 480 --width 832 --num-frames 9 \
  --num-inference-steps 2 --guidance-scale 4.0 \
  --output wan22-tp2.mp4 2>&1 | tee wan22-tp2.log
```

Implementation commit `0a673b4b` (tree `7e504120`) was validated with a
zero-residency 117.512 GiB checkpoint on one AWS `g7e.12xlarge` with two RTX
PRO 6000 Blackwell GPUs. Both ranks cooperatively loaded each of the two Wan
DiTs. Three cold runs completed model loading in
27.381, 27.886, and 27.792 seconds; a confirmed 100%-resident run took 8.473
seconds. All runs produced raw frame-tensor SHA256
`546cbc02c686628e89c61d26daf6a2bb03c2796f7d5e5acd57d8e342060041de`
and shut down both workers without cooperative fallback or rank loss.

The transport-level regression test is
`tests/diffusion/model_loader/test_cooperative_staging_cuda.py`. It launches two
real CUDA processes, uses NCCL collectives with buckets owned by both ranks, and
checks every received tensor.

This folder also contains an online-serving benchmark script for diffusion models.
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
