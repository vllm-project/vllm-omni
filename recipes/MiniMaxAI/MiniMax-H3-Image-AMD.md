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

## Optional tuned BF16 DiT GEMMs

`--additional-config '{"text_encoder_cudnn_sdp":false,"minimax_h3_use_aiter_gemm":true}'`
opts the shared H3 DiT into AITER's existing `gemm_a16w16` tuned dispatcher.
The flag defaults to `false` and must be a JSON boolean. This integration
requires an AITER build providing `get_GEMM_A16W16_tuned_config` and
`gemm_a16w16`; these APIs are present in the recorded AITER 0.1.26 runtime.
An older build missing these APIs cannot use this opt-in. Image serving
also requires a checkout containing the H3-Image pipeline; the shared DiT
adapter does not supply that pipeline.

If the image-support checkout gates the native ROCm RMSNorm/Silu compatibility
fallbacks behind `VLLM_OMNI_ROCM_NATIVE_DIFFUSION_OPS`, set
`VLLM_OMNI_ROCM_NATIVE_DIFFUSION_OPS=1` in both control and tuned server runs
when using the recorded runtime. Keep this compatibility setting identical
across the comparison; it is separate from the GEMM opt-in.

Only the main DiT's unquantized QKV, MLP gate/up, and MLP down projections
are eligible. Tuned calls require inference mode, eager execution, TP1,
contiguous two-dimensional GPU BF16 inputs and native BF16 weights, and
no bias. Other dtypes, quantized/custom linear methods, tensor-parallel
projections, shuffled weights, batch-invariant execution, and shapes without
an AITER tuned entry retain the original linear method. Original weight
loading, layer forwards, LoRA composition, and offload hooks remain in use.
This does not change the six PDD steps, canvas policy, or precision.

Tune the shapes observed for your workload using the same AITER source,
ROCm/hipBLASLt build, and GPU model that will serve it. For the four-GPU,
1344×768 delivery profile above, the captured local shapes were:

| Projection | M | N | K |
| --- | --- | --- | --- |
| QKV | 1032 | 21504 | 5376 |
| MLP gate/up | 1032 | 28672 | 5376 |
| MLP down | 1032 | 5376 | 14336 |

These dimensions are tuning inputs, not runtime restrictions. Sequence
length, generation canvas, and parallel configuration can change them.
From the matching AITER source checkout, create an input CSV and run its
native BF16 tuner on an available GPU:

```bash
export TUNING_DIR=/tmp/minimax-h3-bf16-tuning
mkdir -p "$TUNING_DIR"
cat > "$TUNING_DIR/shapes.csv" <<'CSV'
M,N,K,bias,dtype,outdtype,scaleAB,bpreshuffle
1032,21504,5376,False,torch.bfloat16,torch.bfloat16,False,False
1032,28672,5376,False,torch.bfloat16,torch.bfloat16,False,False
1032,5376,14336,False,torch.bfloat16,torch.bfloat16,False,False
CSV

python csrc/gemm_a16w16/gemm_a16w16_tune.py \
  --input_file "$TUNING_DIR/shapes.csv" \
  --tuned_file "$TUNING_DIR/tuned.csv" \
  --libtype hipblaslt --with-hipblaslt

git rev-parse HEAD > "$TUNING_DIR/aiter-source-revision.txt"
python -m torch.utils.collect_env > "$TUNING_DIR/runtime.txt"
python -m pip freeze > "$TUNING_DIR/packages.txt"
sha256sum "$TUNING_DIR/tuned.csv" > "$TUNING_DIR/tuned.csv.sha256"
```

Record the serving image digest alongside these files. The tuner writes
AITER's standard columns, including GPU architecture, CU count, dimensions,
input/output dtypes, backend, and solution index. Numeric hipBLASLt solution
indices are specific to the compatible library build; retune after changing
the runtime. Do not copy indices from another machine's trace. No solution
indices or model-specific tuning table are shipped by this integration.

Add the following options to the server's `docker run`, before the image
argument, and use the `additional-config` flag above:

```bash
-v "$TUNING_DIR:/h3-gemm-tuning:ro" \
-e AITER_CONFIG_GEMM_BF16=/h3-gemm-tuning/tuned.csv \
```

Set the CSV before worker startup and restart the workers after changing
it: AITER and this adapter cache lookups for the worker lifetime. AITER
validates and selects entries through its own GPU/CU/dtype/shape policy;
unmatched entries fall back to the original path. Malformed CSVs and
incompatible native libraries must be corrected rather than treated as
performance results.

Warm the complete image pipeline before collecting profiler-disabled
latencies. Initial native-library/JIT setup took 14–98 seconds in the
exploratory runtime and is excluded from steady-state timings. Keep the
same prompts, paired seeds, warmups, canvas, six-step schedule, precision,
concurrency, and source revision for comparisons with the flag disabled.
Check output validity and image quality separately; a tuned BF16 GEMM
can have different floating-point accumulation order. This opt-in alone
does not establish an end-to-end speedup or image-quality equivalence.

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
