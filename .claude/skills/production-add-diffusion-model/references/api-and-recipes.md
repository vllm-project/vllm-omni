# API Contract and Hardware Recipes

## Contents

1. [Freeze the upstream contract](#freeze-the-upstream-contract)
2. [Validate before side effects](#validate-before-side-effects)
3. [Task request templates](#task-request-templates)
4. [Serving benchmark template](#serving-benchmark-template)
5. [Hardware recipe contract](#hardware-recipe-contract)
6. [Best deployment selection](#best-deployment-selection)

## Freeze the upstream contract

Pin three revisions independently:

- official source implementation commit or package version;
- checkpoint/model-card revision and partition;
- vLLM-Omni target commit.

Build a table before editing code. One row is one official task/partition.

| Field | Required content |
|---|---|
| Task and partition | Official task name, checkpoint subdirectory, task selection rules |
| Call contract | Constructor and call arguments, types, required/optional, official defaults |
| Prompt/input | Prompt count/length, modality, MIME, order, count, decoded shape |
| Size/time | Width/height alignment, pixels, frames, duration, FPS, start offset |
| Denoise | Scheduler, sigma schedule, steps, CFG/guidance, seed behavior |
| Output | Count, format, codec/sample rate, return type |
| API | Offline mapping, sync endpoint, async endpoint/job lifecycle |
| Rejection | Status code and stable error class/message for every invalid boundary |

Official API parity includes limits and defaults, not just names. Do not add a
generic default that changes a distilled checkpoint's fixed schedule, or accept
an input combination the official task rejects.

For every accepted boundary, add the neighboring rejected values. Examples:
`max_images` and `max_images + 1`, smallest/largest legal duration, aligned and
misaligned dimensions, omitted and illegal steps, one and too many outputs.

## Validate before side effects

Use one normalization/validation path for offline, sync, and async requests.
The ordering must be:

```text
parse -> normalize aliases -> validate task/source matrix -> validate declared
limits -> stream with byte caps -> inspect/decode with frame/pixel/duration caps
-> persist only if needed -> submit to engine
```

Fail before download, persistence, job creation, or engine submit whenever the
available metadata is sufficient. For streamed uploads, enforce:

- total reference count and per-modality count before reading bodies;
- per-file bytes while streaming;
- aggregate bytes across all files;
- decoded frames, duration, pixels, and media type before engine submission.

On rejection, close streams and remove every partial temporary resource. Use
HTTP 400 for invalid task/field combinations and 413 for payload limits. Test
both sync and async paths; an invalid async request must not create a job ID.

Do not trust client MIME alone. Sniff/validate content and keep ordered repeated
multipart fields ordered. Avoid logging prompts, raw references, signed URLs,
or base64 payloads.

## Task request templates

These are endpoint-shape templates, not model recipes. Replace placeholders
with the exact official defaults and validated assets for the target model.
Every production recipe must include all advertised tasks and an expected
output check.

### Text to image: JSON

```bash
curl --fail-with-body -sS -X POST http://127.0.0.1:8091/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "<model>",
    "prompt": "<validated prompt>",
    "size": "1024x1024",
    "num_inference_steps": 30,
    "seed": 42,
    "response_format": "b64_json"
  }' | jq -er '.data[0].b64_json' | base64 -d > output.png
file output.png
```

### Image edit: multipart

```bash
curl --fail-with-body -sS -X POST http://127.0.0.1:8091/v1/images/edits \
  -F 'model=<model>' \
  -F 'prompt=<validated edit prompt>' \
  -F 'image=@/path/to/reference.png;type=image/png' \
  -F 'size=1024x1024' \
  -F 'num_inference_steps=30' \
  -F 'seed=42' \
  -F 'output_format=png' \
  | jq -er '.data[0].b64_json' | base64 -d > edited.png
file edited.png
```

Use repeated `image` fields only if the model's official input matrix allows
multiple ordered sources. Never describe a file upload as JSON-only.

### Public request fields versus internal sampling fields

Use the field exposed by the target revision's endpoint schema; do not rename
it from the pipeline's internal attribute. For example, the current video JSON
and multipart contracts expose `extra_params`, while serving translates that
object into `DiffusionSamplingParams.extra_args`. Other request paths may
expose canonical `extra_args` or nest it under `extra_body`. Inspect the exact
Pydantic model/FastAPI form dependency and exercise the generated curl before
publishing a recipe. A deprecation warning in a different endpoint is not
evidence that an unrecognized multipart form field works here.

Current endpoint map (re-check it at the target commit):

| Endpoint | Public model-specific field |
|---|---|
| `/v1/videos`, `/v1/videos/sync` | Multipart `extra_params` JSON string |
| `/v1/images/generations` | JSON `extra_params` object |
| `/v1/chat/completions` diffusion | `extra_body.extra_args` |
| `/v1/images/edits` | Declared typed form fields only |
| `/v1/audio/generate` | Declared typed JSON fields only |

Prefer declared top-level fields. Use a generic model-specific envelope only
when the endpoint actually exposes it.

### Text to video: multipart

```bash
curl --fail-with-body -sS -X POST http://127.0.0.1:8091/v1/videos/sync \
  -H 'Accept: video/mp4' \
  -F 'model=<model>' \
  -F 'prompt=<validated prompt>' \
  -F 'size=1280x720' \
  -F 'num_frames=81' \
  -F 'fps=16' \
  -F 'num_inference_steps=30' \
  -F 'seed=42' \
  -F 'extra_params={"task":"<official-task>"}' \
  -o output.mp4
ffprobe -v error -show_entries stream=codec_name,width,height,r_frame_rate \
  -of json output.mp4
```

### Image/video to video: multipart

```bash
curl --fail-with-body -sS -X POST http://127.0.0.1:8091/v1/videos/sync \
  -H 'Accept: video/mp4' \
  -F 'model=<model>' \
  -F 'prompt=<validated prompt>' \
  -F 'num_frames=81' \
  -F 'fps=16' \
  -F 'num_inference_steps=30' \
  -F 'seed=42' \
  -F 'extra_params={"task":"<official-task>","start_time_seconds":0}' \
  -F 'input_reference=@/path/to/reference.png;type=image/png' \
  -o i2v.mp4

curl --fail-with-body -sS -X POST http://127.0.0.1:8091/v1/videos/sync \
  -H 'Accept: video/mp4' \
  -F 'model=<model>' \
  -F 'prompt=<validated prompt>' \
  -F 'num_frames=81' \
  -F 'num_inference_steps=30' \
  -F 'seed=43' \
  -F 'extra_params={"task":"<official-task>","start_time_seconds":0}' \
  -F 'input_reference=@/path/to/reference.mp4;type=video/mp4' \
  -o v2v.mp4
```

For multiple references, repeat the model's accepted multipart field in
official order. Include complete downloadable or repository fixtures plus
SHA256 hashes. A placeholder-only recipe is not runnable evidence.

### Text to audio: JSON

```bash
curl --fail-with-body -sS -X POST http://127.0.0.1:8091/v1/audio/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "<model>",
    "input": "<validated audio prompt>",
    "audio_length": 10.0,
    "num_inference_steps": 100,
    "seed": 42,
    "response_format": "wav"
  }' -o output.wav
ffprobe -v error -show_entries stream=codec_name,sample_rate,channels \
  -of json output.wav
```

Use `/v1/audio/generate` for general diffusion audio. Do not substitute the
speech API unless the model is actually served through the speech contract.

### Async video lifecycle

Alongside `/v1/videos/sync`, include `POST /v1/videos`, status polling,
content retrieval, cancellation/abort if public, and cleanup assertions. The
same request must normalize to the same engine sampling parameters in both
modes.

## Serving benchmark template

Run a smoke first, then fixed-concurrency and arrival-rate tests. Video task
names for the harness are `t2v`, `i2v`, `ti2v`, or `v2v`; model-specific task
selection belongs in `--extra-body` when required.

```bash
python benchmarks/diffusion/diffusion_benchmark_serving.py \
  --base-url http://127.0.0.1:8091 \
  --endpoint /v1/videos \
  --model '<model>' \
  --dataset random \
  --task t2v \
  --num-prompts 100 \
  --width 1280 --height 720 \
  --num-frames 81 --fps 16 \
  --num-inference-steps 30 \
  --request-rate 0.2 \
  --max-concurrency 4 \
  --warmup-requests 4 \
  --warmup-concurrency 4 \
  --extra-body '{"extra_params":"{\"task\":\"<official-task>\"}"}' \
  --output-file result.json
```

The video backend currently stringifies each multipart value. Pre-serialize
the nested `extra_params` object as above; passing it as a nested object would
produce Python dict syntax rather than valid JSON. If the target revision
changes the backend to JSON-encode dict/list values, update and smoke-test this
command against that revision.

For i2v/v2v, use the harness's validated dataset/trace path or provide the
documented media source; do not silently benchmark t2v and label it i2v. Keep
prompt, assets, seed, dimensions, frames, steps, guidance, warmup, concurrency,
and arrival process identical across A/B runs.

Keep three benchmark lanes distinct:

1. lossless runtime A/B with the released schedule/precision/attention policy;
2. accelerated-path A/B that changes one adapter, quantization, cache, or
   sparsity policy and adds same-seed quality evidence;
3. production-topology studies that vary placement or parallelism and report a
   latency/throughput/memory frontier rather than a kernel speedup.

For a fixed single-request A/B, retain every measured run and report median or
mean with range after an explicit warmup; do not infer p95/p99 from a few
repetitions. Tail latency requires a declared arrival process, request count,
concurrency, and enough successful samples. Reconcile client E2E with queue,
encoder, denoise, video/audio decode, D2H/IPC, codec, and residual timings.

For generated media, define `real-time` rather than relying on the label. A
complete-response claim uses `client E2E / validated output duration <= 1` and
does not imply streaming or low time-to-first-frame. Report first chunk/fragment
and cadence separately when a public streaming route exists. Never add gains
across benchmark lanes with different schedules, prompts, seeds, artifacts, or
topologies.

The diffusion serving harness currently covers image/video task families, not
`/v1/audio/generate`. For text-to-audio, use the target model's task-specific
audio benchmark if one exists; otherwise add a serving harness that records
request rate, concurrency, latency, generated-audio seconds per second, error
rate, and output validation. Do not relabel the TTS harness or a video workload
as diffusion-audio evidence.

## Hardware recipe contract

Create a separate row and runnable section for each vendor/card/topology:

| Evidence | Required detail |
|---|---|
| Identity | Vendor, exact SKU, device count, HBM, interconnect |
| Software | Driver, CUDA/ROCm/CANN/oneAPI, framework/container, dependency versions |
| Source | vLLM-Omni commit, official/checkpoint revision and hashes |
| Deployment | dtype, attention backend per role, TP/SP/CFG/HSDP/DP, cache, quant, offload |
| Commands | Complete deploy YAML/serve command, every task curl, benchmark command |
| Method | Warmup exclusion, repetitions or request count, concurrency, arrival distribution, duration, statistic definitions |
| Results | Success rate, p50/p95/p99, RPS/throughput per device, stage latency |
| Memory | Load/materialize, resident, encode, denoise, decode, transient per-rank HBM and host PSS |
| Quality | Metric/tolerance, artifact links/hashes, temporal/audio checks |
| Output | Raw/offline dtype, range, layout and ownership; online codec, payload bytes, transport route and client boundary |
| Limits | `validated`/`limited`/`unsupported`/`not tested`, fallback/rejection behavior |

Use current paths such as `recipes/<vendor>/`, `recipes/README.md`, supported
models, and feature compatibility tables. Top-level documentation must link to
the scoped matrix; it must not use an unqualified checkmark that hides a ROCm,
NPU, XPU, task, topology, or dtype limitation.

DLO evidence is per card, especially on small-HBM devices. A DLO result on one
CUDA card does not establish another CUDA SKU, ROCm, NPU, or XPU support.

## Best deployment selection

Benchmark a correctness-approved candidate matrix, not every theoretical
combination. Choose the recommended row by the production objective—latency,
throughput, cost, or small-HBM fit—and document that objective.

Do not recommend a row until it passes:

- all advertised task/API contracts;
- fixed-reference quality;
- request isolation and abort/error cleanup;
- sustained arrival-rate tests;
- per-rank HBM and host PSS limits;
- hardware-specific accuracy and performance CI.

Keep the most memory-constrained DLO row even when it is not the fastest; it is
a separate production recipe, not a failed latency candidate.
