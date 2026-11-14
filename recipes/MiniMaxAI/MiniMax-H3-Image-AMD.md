# MiniMax-H3-Image on AMD Instinct

This recipe serves the six-step
[`MiniMaxAI/MiniMax-H3-Image`](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image)
checkpoint through `POST /v1/images/generations` (text-to-image) and
`POST /v1/images/edits` (image editing) on AMD Instinct GPUs. Two sets of
results are documented:

- The original four-MI350X (`gfx950`) text-to-image profile, with its
  100-request benchmark ([Historical MI350X validation](#historical-mi350x-validation)).
- Text-to-image and editing at the release's 4K sizes on MI325X (`gfx942`)
  and MI350X (`gfx950`), on 1, 2, 4 and 8 GPUs with Ulysses or TP
  ([Measured profiles on MI325X and MI350X](#measured-profiles-on-mi325x-and-mi350x)).

MI300X shares MI325X's `gfx942` code path and MI355X shares MI350X's `gfx950`
path; neither was measured.

## Runtime and model preparation

Set `VLLM_OMNI_ROCM_NATIVE_DIFFUSION_OPS=1` to run the shared diffusion
`RMSNorm` and `SiluAndMul` in PyTorch on ROCm. This avoids precompiled vLLM
kernels that failed with `hipErrorInvalidImage` in the tested runtime. The
setting is off by default and applies to every diffusion model in the
process; other diffusion models were not qualified by the MiniMax-H3-Image
benchmark. A vLLM build with kernels for your GPU arch does not need it.

The tested runtime was the locally available image
`rocm/vllm-omni:438155621484248827a30256e679cb4c84f544fb`:

| Component | Recorded version |
| --- | --- |
| vLLM | `0.33.0+rocm100` |
| PyTorch | `2.13.0+rocm10.0.0` |
| Transformers | `5.19.0` |
| Diffusers | `0.40.0` |
| AMD AITER | `0.1.26` |

The older `vllm/vllm-omni-rocm:minimax-h3` video image contains vLLM 0.26 and
cannot import this checkout's vLLM 0.33 APIs. The commands below mount the
current source into the matching runtime. `PYTHONPATH` and the working
directory both select that source, avoiding the image's older Omni checkout.
An equivalent source-built ROCm runtime must be qualified separately; see the
[ROCm installation guide](../../docs/getting_started/installation/gpu/rocm.inc.md).

### Runtime per GPU architecture

The per-commit `rocm/vllm-omni:<commit>` CI images build vLLM with
`PYTORCH_ROCM_ARCH=gfx942`, so their vLLM extensions contain `gfx942` code
objects only. Check an image with:

```bash
docker run --rm --entrypoint bash "$SERVER_IMAGE" -c \
  'cd "$(python3 -c "import os, vllm; print(os.path.dirname(vllm.__file__))")" &&
   strings -a _C.abi3.so | grep -o "amdgcn-amd-amdhsa--gfx9[0-9a-f]*" | sort | uniq -c'
```

| GPU | Runtime | Validated |
| --- | --- | --- |
| MI300X, MI325X (`gfx942`) | A `rocm/vllm-omni:<commit>` CI image with this checkout mounted, as below | MI325X: `rocm/vllm-omni:90eeebbd173c2c3cceac75fe299fb98e06813500` |
| MI350X, MI355X (`gfx950`) | The CI image profile below, which works around its `gfx942`-only kernels with `VLLM_OMNI_ROCM_NATIVE_DIFFUSION_OPS=1` and `VLLM_ROCM_USE_SKINNY_GEMM=0` | MI350X, four GPUs, text-to-image |
| MI350X, MI355X (`gfx950`) | The multi-arch `vllm/vllm-openai-rocm:v0.33.0` image (`gfx90a`, `gfx942`, `gfx950`, the base used by `docker/Dockerfile.rocm`) with this checkout installed | MI350X, 1–8 GPUs, text-to-image and editing |

With a CI image on `gfx950`, startup fails with `hipErrorInvalidKernelFile` /
`hipErrorInvalidImage` in any vLLM kernel that has no `gfx950` code object.
The multi-arch image provides those kernels and needs neither
`VLLM_OMNI_ROCM_NATIVE_DIFFUSION_OPS` nor the skinny-GEMM override.
Install this checkout into it without replacing its PyTorch or vLLM:

```bash
docker run -d --name minimax-h3-image-amd \
  --device=/dev/kfd --device=/dev/dri \
  --security-opt seccomp=unconfined --group-add video \
  --ipc=host --network host --shm-size 64G \
  -v "$SOURCE_DIR:/workspace" \
  -v "$CACHE_ROOT:$CACHE_ROOT:ro" \
  -v "$RUNTIME_DIR/model_index.local.json:$MODEL_DIR/model_index.json:ro" \
  --entrypoint sleep vllm/vllm-openai-rocm:v0.33.0 infinity

docker exec minimax-h3-image-amd bash -c '
  pip freeze | grep -E "^(torch|torchvision|torchaudio|numpy)==" > /tmp/pins.txt
  grep -v -E "^(cosmos-guardrail|openai-whisper|s3tokenizer|kernels)" \
    /workspace/requirements/common.txt > /tmp/requirements.txt
  pip install -r /tmp/requirements.txt -c /tmp/pins.txt
  VLLM_OMNI_VERSION_OVERRIDE=0.0.0 pip install -e /workspace --no-deps --no-build-isolation'
```

The excluded packages are not used by MiniMax-H3-Image. Pip installs
Transformers 5.19.0, matching the recorded versions above. Then start the
server with `docker exec` and the `vllm serve` arguments shown below.

From the repository root, set the host paths:

```bash
export SOURCE_DIR="$PWD"
export CACHE_ROOT=/data/hf_hub_cache
export MODEL_DIR="$CACHE_ROOT/MiniMax-H3-Image"
export ENCODER_ROOT="$CACHE_ROOT/MiniMax-H3/FL2VA"
export SERVER_IMAGE=rocm/vllm-omni:438155621484248827a30256e679cb4c84f544fb
export RUNTIME_DIR="$(mktemp -d)"
```

The image checkpoint does not contain its external Qwen3-VL text encoder.
The historical run reused the cached `FL2VA/text_encoder` tensors from the
base H3 model. If those weights are absent, download them as well as the
image checkpoint using an authorized Hugging Face account:

```bash
hf download MiniMaxAI/MiniMax-H3-Image \
  --revision 7b3d20b4659f5d4b3ad7205daa008e47d6b1e868 \
  --local-dir "$MODEL_DIR"

hf download MiniMaxAI/MiniMax-H3 \
  --revision 42ed227ee7df40d41602854ae760620d6eb651fe \
  --include 'FL2VA/text_encoder/*' \
  --local-dir "$CACHE_ROOT/MiniMax-H3"

test -f "$ENCODER_ROOT/text_encoder/config.json"
```

Create a configuration overlay pointing the image model's external encoder
to the local base-model directory. The original model index and weight files
remain unchanged. Mount the cache at the same absolute path inside Docker so
cached weight symlinks resolve.

```bash
python3 - <<'PY'
import json
import os
from pathlib import Path

model = Path(os.environ["MODEL_DIR"])
index = json.loads((model / "model_index.json").read_text())
index["text_encoder"][2]["pretrained_model_name_or_path"] = os.environ["ENCODER_ROOT"]
index["text_encoder"][2]["revision"] = None
destination = Path(os.environ["RUNTIME_DIR"]) / "model_index.local.json"
destination.write_text(json.dumps(index, indent=2) + "\n")
PY
```

## Start the image server

The measured profile used physical GPUs 4–7 and port 8001, while a separate
video server used GPUs 0–3. Select available GPUs and an unused port on your
host. The four selected GPUs provide Ulysses sequence parallelism and
text-encoder tensor parallelism at degree four.

```bash
docker run -d --name minimax-h3-image-amd \
  --device=/dev/kfd --device=/dev/dri \
  --security-opt seccomp=unconfined --group-add video \
  --ipc=host --network host \
  -v "$SOURCE_DIR:/workspace:ro" \
  -v "$CACHE_ROOT:$CACHE_ROOT:ro" \
  -v "$RUNTIME_DIR/model_index.local.json:$MODEL_DIR/model_index.json:ro" \
  -w /workspace \
  -e PYTHONPATH=/workspace \
  -e HIP_VISIBLE_DEVICES=4,5,6,7 \
  -e PYTORCH_ROCM_ARCH=gfx950 \
  -e VLLM_OMNI_ROCM_NATIVE_DIFFUSION_OPS=1 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_ROCM_USE_AITER=1 \
  -e VLLM_ROCM_USE_SKINNY_GEMM=0 \
  -e HF_HUB_OFFLINE=1 \
  -e OMNI_DIFFUSION_PROMPT_EMBED_CACHE=0 \
  --entrypoint vllm "$SERVER_IMAGE" \
  serve "$MODEL_DIR" \
  --omni --trust-remote-code \
  --host 127.0.0.1 --port 8001 \
  --num-gpus 4 --usp 4 --ring 1 --text-encoder-tp-size 4 \
  --diffusion-attention-backend FLASH_ATTN \
  --additional-config '{"text_encoder_cudnn_sdp":false}'

docker logs --tail 100 minimax-h3-image-amd
curl --fail --silent --show-error http://127.0.0.1:8001/health
curl --fail --silent --show-error http://127.0.0.1:8001/v1/models
```

Important settings for this profile:

- `VLLM_ROCM_USE_SKINNY_GEMM=0` avoids an incompatible precompiled skinny
  GEMM selected by the small DiT conditioning projections. The same shape
  passed through the supported AITER/PyTorch dispatch with this setting.
- VAE tile decode uses CUDA graphs. On ROCm the pipeline replays a single
  graph replica: concurrent replay of several replicas could leave a request
  in VAE decode at 100% GPU utilization. With one replica, 4 × MI350X served
  84 of 84 requests (about 60 at 4K, including concurrent clients), with
  outputs bit-identical to eager decode and faster than eager (0.78 s vs
  0.87 s at 2048², 3.09 s vs 3.36 s at 4736 × 3552).
- `text_encoder_cudnn_sdp=false` records the selected encoder reproducibility
  policy. This is separate from the kernel compatibility fixes.
- Precision is BF16, without quantization or offload. The image pipeline
  keeps its DiT eager and uses the released six-step PDD schedule.

### Per-architecture settings and other layouts

- `--no-enable-cuda-graph-decode` is not needed on AMD GPUs. Earlier
  revisions replayed several VAE graph replicas, which could hang on both
  `gfx950` and `gfx942`; the pipeline now uses one replica on ROCm. Keep the
  flag only as a fallback if a request stays in VAE decode; eager decode is
  slower (at 4736 × 3552 on one GPU, 1.0 s with graphs on MI325X and 4.9 s
  eager on MI350X).
- For other GPU counts, set `--num-gpus N --usp N --ring 1` (Ulysses; the 56
  attention heads allow N = 2, 4 or 8) or `--num-gpus N
  --tensor-parallel-size N` (TP). Ulysses had the lowest measured latency
  at every GPU count; TP uses less memory per GPU. One GPU needs no parallel
  flags and holds the full model (about 112 GiB of weights).

### VAE graph decoding with independent SP1 workers

The image pipeline keeps the DiT eager and enables VAE tile graphs by
default. To select this explicitly, use
`--enable-cuda-graph-decode --no-enforce-eager`; compare with
`--no-enable-cuda-graph-decode --no-enforce-eager` for an eager VAE control.
ROCm uses one graph replica per tile shape, captured on its own replay
stream. Older images must include these runtime fixes before enabling graphs.

A 2026-11-13 experiment used four independent SP1/encoder-TP1 workers behind
an external router on 4 × MI350X (`gfx950`). The runtime was
`haihub007/minimax-h3-image:perf-1113`, image ID
`sha256:6f89ea36dd542234aa8b7118b8b7204232cd58dd355f17ab4a85fe9dd8be7869`,
with vLLM `0.33.0+rocm100`, PyTorch `2.13.0+rocm10.0.0`, AITER `0.1.26`,
Transformers `5.19.0` and Diffusers `0.40.0`. Its tuned AITER GEMM table
was held identical in both variants. Both used
the same VAE source overlay backporting the current single-replica and
capture-stream behavior; only the graph-decode flag changed.

| Concurrency | Eager mean HTTP s | Graph mean HTTP s | Eager images/s | Graph images/s | Throughput gain |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.5832 | 2.0752 | 0.3809 | 0.4723 | +24.02% |
| 8 | 4.7277 | 3.8305 | 1.5910 | 1.9883 | +24.97% |

Four separate server starts ran in eager → graph → graph → eager order,
with one ascending and one descending C1/C8 sweep per variant. Each phase
used four excluded warmups and 64 measured requests: 512 measured requests
passed. Requests used the same 20-prompt/seed corpus, six steps, BF16,
FLASH_ATTN, disabled prompt-embedding cache and encoder cuDNN SDP,
1344×768 output, and a 2720×1536 internal canvas. HTTP latency ends at full
body receipt; throughput includes client PNG validation and persistence.
The two graph blocks reached 1.9776 and 1.9990 images/s at C8.

Separately, 28 fixed-latent decoder comparisons at 2720×1536 and 4096²
matched eager decoded floats and quantized uint8 values exactly. This is a
decoder-equivalence check, not a full-model image-quality certification.
These historical graph-on/off results do not measure the recipe change
against current `main`, the default SP4 profile, native replica routing, or
native-4K HTTP serving. MI355X was not measured.

## Generate an image

Submit six inference steps or omit the step count to use the checkpoint's
schedule. This release accepts one output image per request.

```bash
curl --fail --silent --show-error \
  http://127.0.0.1:8001/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "A photorealistic California coastline in golden afternoon light",
    "size": "1344x768",
    "seed": 42,
    "generator_device": "cpu",
    "num_inference_steps": 6,
    "n": 1,
    "response_format": "b64_json",
    "output_format": "png"
  }' > image-response.json
```

The output PNG has the requested 1344×768 dimensions. The default
`min_generation_area=4194304` policy generates on a 2720×1536 canvas and
resizes the decoded image to the requested dimensions. Preserve both sizes
when comparing measurements.

## Edit an image

Editing uses the multipart endpoint. Repeat the `image` part for more
references; they are `<Picture 1>`, `<Picture 2>`, … in the prompt. The
release example (`examples/` in the model repository) is:

```bash
curl --fail --silent --show-error \
  http://127.0.0.1:8001/v1/images/edits \
  -F "image=@picture1.png" \
  -F "prompt=<prompt.edit.txt" \
  -F size=4096x4096 -F seed=1234 -F generator_device=cpu \
  -F response_format=b64_json -F output_format=png \
  > edit-response.json
```

## Benchmark

From the repository root, run the instrumented HTTP client after health
checks pass. Use a fresh output directory for each campaign:

```bash
python3 -m pip install 'aiohttp==3.13.5' 'Pillow==11.3.0'
export RUN_DIR="benchmark-results/h3-image-amd-$(date -u +%Y%m%dT%H%M%SZ)"

python3 -u benchmarks/diffusion/h3_image_bench.py \
  --base-url http://127.0.0.1:8001 \
  --model "$MODEL_DIR" \
  --corpus benchmarks/diffusion/h3-image-prompts.jsonl \
  --output-dir "$RUN_DIR" \
  --phases c1-b1 c1-b2 c1-b3 c2 c4 \
  --num-prompts 20 --warmups 2 --order-seed 20261105 \
  --width 1344 --height 768 --steps 6 \
  --server-min-generation-area 4194304 \
  --timeout 1800
```

Each phase has two excluded sequential warmups and 20 measured requests.
The three concurrency-one blocks combine into 60 measurements; concurrency
two and four have 20 each. The total is 100 measured requests and 10
excluded warmups, using the same 20 prompts and paired seeds 42–61. The
client validates full PNG decoding, dimensions, and nonconstant pixels.
It does not retry generation POSTs.

`--server-min-generation-area` records the server's existing policy for
canvas metadata; it does not override inference settings. Delivery latency
runs from POST submission through full response-body receipt, including
queueing, generation, encoding, and base64 transfer. Throughput includes
client validation and persistence. Warmups are excluded from both metrics.

## Measured profiles on MI325X and MI350X

Source `efecc3717` with `VLLM_OMNI_ROCM_NATIVE_DIFFUSION_OPS` unset (fused
vLLM kernels), BF16, no offload, default encoder TP 1, prompt-embedding cache
at its default. Image revision `7b3d20b4659f5d4b3ad7205daa008e47d6b1e868`; the
text encoder used the same cached `FL2VA/text_encoder` weights as above,
resolved offline from the local Hugging Face cache.

| GPU | Runtime | Extra server flags |
| --- | --- | --- |
| MI325X (`gfx942`) | `rocm/vllm-omni:90eeebbd173c2c3cceac75fe299fb98e06813500` | none (VAE graph decode on) |
| MI350X (`gfx950`) | `vllm/vllm-openai-rocm:v0.33.0` with this checkout installed | `--no-enable-cuda-graph-decode` (measured before the single-replica fix; no longer needed) |

Both used vLLM `0.33.0+rocm100`, PyTorch `2.13.0+rocm10.0.0`, Transformers
5.19.0 and Diffusers 0.40.0, with `--diffusion-attention-backend FLASH_ATTN`
and `--enable-diffusion-pipeline-profiler` for the stage split.

Mean delivery latency of warm requests, one request at a time (three per
cell, first excluded). Text-to-image uses the release prompt with seed 42;
editing uses the release prompt and reference with seed 1234; all use
`generator_device=cpu`. Delivery runs from POST to full response-body
receipt, including base64 transfer; client-side decoding is excluded.

| GPUs and layout | t2i 4736×3552 | edit 4096×4096 | t2i 6272×2688 | t2i 1024×1024 (2048² canvas) |
| --- | ---: | ---: | ---: | ---: |
| 1 × MI325X | 12.85 s | 15.46 s | 12.95 s | 2.66 s |
| 2 × MI325X, Ulysses 2 | 9.11 s | 10.79 s | 9.25 s | 2.06 s |
| 4 × MI325X, Ulysses 4 | 4.71 s | 5.45 s | 4.70 s | 1.12 s |
| 4 × MI325X, TP 4 | 5.42 s | 6.33 s | 5.41 s | 1.27 s |
| 8 × MI325X, Ulysses 8 (2174c09) | 2.32 s | 3.21 s | 2.31 s | 0.64 s |
| 8 × MI325X, TP 8 (2174c09) | 3.02 s | 4.01 s | 3.74 s | 0.80 s |
| 1 × MI350X | 12.81 s | 14.38 s | 12.71 s | 2.79 s |
| 2 × MI350X, Ulysses 2 | 8.13 s | 9.20 s | 8.09 s | 1.80 s |
| 4 × MI350X, Ulysses 4 | 3.91 s | 4.63 s | 3.90 s | 0.99 s |
| 8 × MI350X, Ulysses 8 | 2.15 s | 2.55 s | 2.16 s | 0.61 s |
| 8 × MI350X, TP 8 | 2.63 s | 3.04 s | 2.56 s | 0.79 s |

The 8-GPU MI325X rows were measured at `2174c09` (with a local guard that
skips the fused RMSNorm kernel for mixed weight/input dtypes) because GPUs
0–3 were in use during the `efecc3717` run; at 1, 2 and 4 GPUs the two
revisions agree within 2%.

Stage split for text-to-image at 4736 × 3552, from the pipeline profiler
(`_encode_text`, `_denoise`, `vae.decode_latent`; slowest rank).

| GPUs and layout | text encode | denoise (6 steps) | VAE decode | delivery |
| --- | ---: | ---: | ---: | ---: |
| 1 × MI325X | 0.06 s | 11.54 s | 1.06 s | 12.85 s |
| 2 × MI325X, Ulysses 2 | 0.06 s | 8.30 s | 0.55 s | 9.11 s |
| 4 × MI325X, Ulysses 4 | 0.06 s | 4.15 s | 0.29 s | 4.71 s |
| 4 × MI325X, TP 4 | 0.05 s | 4.87 s | 0.29 s | 5.42 s |
| 8 × MI325X, Ulysses 8 (2174c09) | 0.06 s | 1.90 s | 0.16 s | 2.32 s |
| 8 × MI325X, TP 8 (2174c09) | 0.06 s | 2.61 s | 0.16 s | 3.02 s |
| 1 × MI350X | 0.05 s | 7.56 s | 4.90 s | 12.81 s |
| 2 × MI350X, Ulysses 2 | 0.04 s | 5.47 s | 2.29 s | 8.13 s |
| 4 × MI350X, Ulysses 4 | 0.04 s | 2.33 s | 1.15 s | 3.91 s |
| 8 × MI350X, Ulysses 8 | 0.04 s | 1.16 s | 0.62 s | 2.15 s |
| 8 × MI350X, TP 8 | 0.04 s | 1.60 s | 0.67 s | 2.63 s |

Throughput for text-to-image at 4096 × 4096 with concurrent requests:

| GPUs and layout | Concurrency | Requests | Images/min | Mean delivery |
| --- | ---: | ---: | ---: | ---: |
| 1 × MI325X | 2 | 4 | 4.7 | 22.39 s |
| 1 × MI325X | 4 | 8 | 4.7 | 41.60 s |
| 4 × MI325X, Ulysses 4 | 2 | 4 | 13.2 | 7.98 s |
| 4 × MI325X, Ulysses 4 | 4 | 8 | 13.0 | 15.11 s |
| 8 × MI325X, Ulysses 8 (2174c09) | 2 | 4 | 27.6 | 3.81 s |
| 8 × MI325X, Ulysses 8 (2174c09) | 4 | 8 | 27.9 | 7.00 s |
| 1 × MI350X | 2 | 4 | 4.9 | 21.45 s |
| 1 × MI350X | 4 | 8 | 5.0 | 38.80 s |
| 8 × MI350X, Ulysses 8 | 2 | 4 | 31.4 | 3.37 s |
| 8 × MI350X, Ulysses 8 | 4 | 8 | 31.8 | 6.16 s |

A server processes one image at a time, so concurrency adds queueing rather
than throughput. For throughput, independent single-GPU servers are expected
to scale per GPU (for example, about 8 × 4.9 ≈ 39 images/min per MI350X
node); this was not measured. Every listed run completed with no failed
requests. The first request after startup was within 0.15 s of the warm
latency, because the server's startup dummy run absorbs compilation. Model
loading took 112 GiB per server on one GPU, in about 38–46 s.

Outputs were compared visually with the release examples and across GPUs
and layouts: composition and edited content match, but pixels are not
identical across hardware or layouts (edit PSNR 31–36 dB, text-to-image
17–20 dB between configurations). These are not image-quality scores.

## Historical MI350X validation

The following results were measured with source revision
`2174c09df5ec21bbf11f96becbcd2d6c7babcf75` plus the two Docker-only native
HIP overrides (now `VLLM_OMNI_ROCM_NATIVE_DIFFUSION_OPS=1`), using the runtime and flags above.
They document that tested deployment, rather than qualification of every
subsequent source revision.

| Concurrency | Successful measured images | Mean delivery | Images/hour |
| --- | --- | --- | --- |
| 1, three blocks combined | 60/60 | 0.958 s | 3560.21 |
| 2 | 20/20 | 1.742 s | 3910.26 |
| 4 | 20/20 | 3.335 s | 3931.84 |

All 100 measured requests succeeded; all 110 outputs including warmups
passed validation. Five-second GPU telemetry sampled memory peaks of
86.33–86.83 GiB per GPU, including warmups. These are delivery and throughput
measurements on a shared host, not pure GPU compute timings or image quality
scores. Concurrency two delivered nearly all observed throughput; four
increased queueing with little additional throughput.

The kernel fallbacks and this recipe do not establish cross-platform image
equivalence, editing quality, or performance on other AMD GPUs.
