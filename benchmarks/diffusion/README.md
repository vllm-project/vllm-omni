
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

## 2. Supported endpoints

The script drives the OpenAI-compatible endpoints of a running server, so it
measures the deployed service rather than an in-process pipeline.

| `--endpoint` | Tasks | Request shape | Notes |
| --- | --- | --- | --- |
| `/v1/videos` | `t2v`, `i2v`, `ti2v`, `v2v` | Multipart form upload, then poll | Asynchronous: the script creates the job and polls it until it finishes. |
| `/v1/chat/completions` | `t2i`, `i2i`, `ti2i`, `it2i` | OpenAI chat request | The image comes back in the chat response. Input images are sent as `image_url` content parts. |
| `/v1/images/edits` | `i2i`, `ti2i`, `it2i` | Multipart form upload | Needs at least one input image. Supports `--bot-task` for image-conditioned models. |
| `/v1/images/generations` | `t2i` | OpenAI image request | Text-to-image only: input images are not sent. |

`--endpoint` is optional. When it is omitted, the endpoint follows the task:

| Task | Endpoint |
| --- | --- |
| `t2v`, `i2v`, `ti2v`, `v2v` | `/v1/videos` |
| `i2i`, `ti2i`, `it2i` | `/v1/images/edits` |
| `t2i` | `/v1/chat/completions` |

Pass `--endpoint` explicitly to use an endpoint the task default does not
select, for example `--endpoint /v1/images/generations --task t2i`.

The endpoint has to match the task family: the image tasks (`t2i`, `i2i`,
`ti2i`, `it2i`) use the image endpoints, the video tasks (`t2v`, `i2v`, `ti2v`,
`v2v`) use `/v1/videos`. A mismatch fails with the list of endpoints that fit
the task. `/v1/images/edits` additionally needs an image-conditioned task,
because it uploads the input image as a file.

The legacy `--backend vllm-omni` and `--backend openai` values are aliases for
`/v1/chat/completions` and `/v1/images/generations`.

## 3. Supported Datasets

The benchmark supports four dataset modes via `--dataset`:

- `vbench`: Built-in prompt/data loader.
- `trace`: Heterogeneous request traces (each request can have different resolution/frames/steps).
- `random`: Synthetic requests, optionally mixed from several request profiles.
- `custom`: Requests replayed from a local JSONL file.

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

### Random dataset

`--dataset random` synthesizes requests from `--random-request-config`, a JSON
list of request profiles. Each profile may set `width`, `height`,
`num_inference_steps`, `num_frames`, `fps` and `prompt`; the `weight` field is
required for every profile and controls how often the profile is drawn, so one
run can mix several resolutions and step counts. The seed comes from `--seed`,
not from the profile.

```bash
python3 benchmarks/diffusion/diffusion_benchmark_serving.py \
    --endpoint /v1/videos --dataset random --task t2v --num-prompts 10 \
    --random-request-config '[
        {"width":854,"height":480,"num_inference_steps":18,"num_frames":120,"fps":24,"weight":1}
    ]'
```

Image-conditioned tasks (`i2v`, `ti2v`, `ti2i`, `i2i`, `it2i`) need input
media. With `--dataset random` the script synthesizes it, and
`--num-input-images` sets how many input images are attached per request
(default: 1). `--enable-negative-prompt` adds a negative prompt to the
generated requests.

### Custom dataset

`--dataset custom` replays requests from a JSONL file given by
`--dataset-path`, with one JSON object per line:

| Field | Required | Meaning |
| --- | --- | --- |
| `prompt` | yes | The text prompt. |
| `width`, `height` | no | Requested output size. |
| `num_inference_steps` | no | Diffusion steps. |
| `seed` | no | Random seed. |
| `image_paths` | no | Input image paths, for the image-conditioned tasks (`i2i` / `i2v` / `ti2i` / `it2i`). |
| `image_urls` | no | Input image URLs, as an alternative to `image_paths`. |
| `video_paths` | no | Input video paths, for `v2v`. |
| `video_urls` | no | Input video URLs, as an alternative to `video_paths`. |

```bash
python3 benchmarks/diffusion/diffusion_benchmark_serving.py \
    --endpoint /v1/images/edits --dataset custom \
    --dataset-path custom_requests.jsonl --task ti2i --bot-task think
```

## 4. Benchmark Parameters

### Basic flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `--base-url` | `http://<host>:<port>` | Server address; overrides `--host` and `--port`. |
| `--host` | `localhost` | Server host. |
| `--port` | `8091` | Server port. |
| `--model` | `default` | The OpenAI-compatible `model` field. |
| `--endpoint` | from `--task` | API endpoint path. Leading `/` is optional, e.g. `/v1/videos` or `v1/videos`. |
| `--task` | `t2v` | Task type (`t2v`, `i2v`, `ti2v`, `v2v`, `t2i`, `i2i`, `ti2i`, `it2i`). |
| `--dataset` | `vbench` | Dataset mode (`vbench` / `trace` / `random` / `custom`). |
| `--dataset-path` | none | Local dataset file (`custom`) or trace root (`trace`). |
| `--num-prompts` | `10` | Number of requests to send. |
| `--max-concurrency` | `1` | Maximum number of in-flight requests. |
| `--request-rate` | `inf` | Request rate. `inf` sends every request at time 0; otherwise arrivals follow a Poisson process. |
| `--output-file` | none | Write the metrics to a JSON file. |
| `--save-dir` | none | Save the generated outputs for inspection. Without it the outputs are discarded after the metrics are collected. |
| `--disable-tqdm` | off | Disable the progress bar. |

### Request shaping

- `--width`, `--height`: Output size. Pass them together; one of them alone
  cannot be resolved.
- `--num-frames`: Number of frames, for video tasks.
- `--num-inference-steps`: Number of diffusion steps (default: 50).
- `--fps`: Frames per second, for video tasks.
- `--seed`: Random seed, for diffusion models.
- `--extra-body`: JSON object merged into every generated request body or form.
- `--enable-negative-prompt`: Generate negative prompts, with the `random`
  dataset.
- `--random-request-config`: JSON list of request profiles, with the `random`
  dataset.
- `--num-input-images`: Number of synthetic input images per request, for
  image-conditioned tasks with the `random` dataset (default: 1).
- `--bot-task`: `bot_task` form field for `/v1/images/edits` (`think`,
  `recaption`, `think_recaption`, `vanilla`).
- `--return-stage-metrics`: Ask the server for per-stage durations. The flag only
  adds the request field on `/v1/chat/completions`. `/v1/videos` returns stage
  durations in its job-status response instead, and only when the server runs
  with `--enable-diffusion-pipeline-profiler`; the image endpoints do not report
  them.

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

`--video-poll-interval` sets the time between video status checks in seconds
(default: 2). Video request latency runs from job submission through receipt of
complete video bytes; job deletion is excluded.

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

## 5. Sample commands

One runnable command per supported endpoint. They assume a server started with
`vllm serve <model> --omni --port 8091`; adjust `--base-url` and `--model` for
your deployment.

`/v1/videos` (text-to-video):

```bash
python3 benchmarks/diffusion/diffusion_benchmark_serving.py \
    --base-url http://localhost:8091 --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
    --endpoint /v1/videos --dataset vbench --task t2v --num-prompts 10 \
    --width 640 --height 480 --fps 16 --num-frames 80 --num-inference-steps 40
```

`/v1/chat/completions` (text-to-image):

```bash
python3 benchmarks/diffusion/diffusion_benchmark_serving.py \
    --base-url http://localhost:8091 --model Qwen/Qwen-Image \
    --endpoint /v1/chat/completions --dataset vbench --task t2i --num-prompts 10 \
    --width 1024 --height 1024
```

`/v1/images/generations` (text-to-image):

```bash
python3 benchmarks/diffusion/diffusion_benchmark_serving.py \
    --base-url http://localhost:8091 --model Qwen/Qwen-Image \
    --endpoint /v1/images/generations --dataset vbench --task t2i --num-prompts 10 \
    --width 1024 --height 1024
```

`/v1/images/edits` (image-conditioned edit):

```bash
python3 benchmarks/diffusion/diffusion_benchmark_serving.py \
    --base-url http://localhost:8091 --model tencent/HunyuanImage-3.0 \
    --endpoint /v1/images/edits --dataset random --task ti2i --num-prompts 10 \
    --bot-task think --num-input-images 1
```

## 6. Using `vllm bench serve --omni` instead

The unified `vllm bench serve --omni` runner covers the same endpoints and
does not need this script. It registers an OpenAI-compatible client per
endpoint:

| Endpoint | How to select it |
| --- | --- |
| `/v1/videos` | `--endpoint /v1/videos` with `--dataset-name random-mm` |
| `/v1/images/generations` | `--endpoint /v1/images/generations` |
| `/v1/images/edits` | `--endpoint /v1/images/edits`, or `--backend openai-image-edits-omni` |
| `/v1/chat/completions` | `--backend openai-chat-omni` |

For example, the text-to-video case above becomes:

```bash
vllm bench serve --omni \
    --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
    --endpoint /v1/videos \
    --dataset-name random-mm \
    --tokenizer gpt2 \
    --num-prompts 10 \
    --max-concurrency 1 \
    --random-input-len 8 \
    --random-output-len 1 \
    --extra-body '{"width":640,"height":480,"num_inference_steps":40,"num_frames":80,"fps":16,"seed":42}'
```

The `random-mm` dataset builds synthetic prompts, so it needs a `--tokenizer`;
the `extra-body` values override the generated request's size, frame count and
seed.

The two runners report different metric sets, and the perf-test suites pick
between them by config schema: `benchmark_params[].dataset_name` selects
`vllm bench serve --omni`, while `benchmark_params[].dataset` without a
`dataset_name` selects this script. The configs live in
[tests/dfx/perf](../../tests/dfx/perf/), and the
[CLI guide](../../docs/cli/bench/serve.md) documents the
`vllm bench serve --omni` arguments.

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
