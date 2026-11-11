# MiniMax-H3-Image on AMD Instinct

This recipe serves the six-step
[`MiniMaxAI/MiniMax-H3-Image`](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image)
checkpoint through `POST /v1/images/generations`. The validated task is
text-to-image on four AMD Instinct MI350X GPUs (`gfx950`). Image editing and
other hardware profiles were not validated by this run.

## Runtime and model preparation

Use a checkout containing the ROCm `RMSNorm.forward_hip` and
`SiluAndMul.forward_hip` native fallbacks. These avoid precompiled vLLM kernels
that failed with `hipErrorInvalidImage` in the tested runtime. The overrides
apply to these shared diffusion operators on ROCm; other diffusion models
were not qualified by the MiniMax-H3-Image benchmark.

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
  --no-enable-cuda-graph-decode \
  --additional-config '{"text_encoder_cudnn_sdp":false}'

docker logs --tail 100 minimax-h3-image-amd
curl --fail --silent --show-error http://127.0.0.1:8001/health
curl --fail --silent --show-error http://127.0.0.1:8001/v1/models
```

Important settings for this profile:

- `VLLM_ROCM_USE_SKINNY_GEMM=0` avoids an incompatible precompiled skinny
  GEMM selected by the small DiT conditioning projections. The same shape
  passed through the supported AITER/PyTorch dispatch with this setting.
- `--no-enable-cuda-graph-decode` disables VAE graph replay. A first image
  succeeded with graphs enabled, but a following request stalled. The
  graph-disabled profile passed consecutive-image tests and the full
  benchmark. Distributed VAE tile decoding remains enabled.
- `text_encoder_cudnn_sdp=false` records the selected encoder reproducibility
  policy. This is separate from the kernel compatibility fixes.
- Precision is BF16, without quantization or offload. The image pipeline
  keeps its DiT eager and uses the released six-step PDD schedule.

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

## Historical MI350X validation

The following results were measured with source revision
`2174c09df5ec21bbf11f96becbcd2d6c7babcf75` plus the two Docker-only native
HIP overrides now represented in source, using the runtime and flags above.
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
