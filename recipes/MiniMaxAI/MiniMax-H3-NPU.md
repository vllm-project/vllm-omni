# MiniMax H3 on Ascend NPU

> Joint video and audio generation with text, first/last keyframes, and mixed
> image/video/audio references — Ascend NPU deployment guide for Atlas 800I A2
> and Atlas 800I A3

## Summary

- Vendor: MiniMaxAI
- Model: [`MiniMaxAI/MiniMax-H3`](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- Tasks: T2VA, FL2VA, and Ref2VA
- Mode: OpenAI-compatible `/v1/videos` HTTP serving
- Hardware: Atlas 800I A2 / Atlas 800I A3, 8x NPU
- Maintainer: Community

This recipe adapts [MiniMax-H3.md](MiniMax-H3.md) for Ascend NPU
environments. Differences from the GPU path:

- Runs on a CPU-only (aarch64) PyTorch build with `torch_npu`; no CUDA
  runtime is present.
- Audio loading does **not** require TorchCodec (whose aarch64 wheels are
  built against CUDA torch and fail to load on CPU-only builds). vLLM-Omni
  automatically falls back to soundfile / ffmpeg for wav/mp3/m4a/mp4 audio
  inputs.

For the four-card Ascend 950PR / 950DT route, see
[MiniMax-H3-NPU-950PR.md](MiniMax-H3-NPU-950PR.md).

## Prerequisites

### Checkpoint

Download the weights once; `vllm serve` accepts the repository ID or a local
path and resolves the nested components automatically:

```bash
export MODEL=/path/to/MiniMax-H3
hf download MiniMaxAI/MiniMax-H3 --local-dir "${MODEL}"
```

The checkpoint directory layout is:

```text
MiniMax-H3/
├── FL2VA/          # shared by T2VA/FL2VA tasks
├── Ref2VA/         # Ref2VA task weights
└── model_index.json
```

### Environment

The recommended deployment uses the official vLLM-Omni NPU images, which ship
the matching vLLM / vLLM-Ascend / vLLM-Omni stack (see
[NPU installation](../../docs/getting_started/installation/npu.md)). Pick the
image matching the hardware — `quay.io/ascend/vllm-omni:v0.29.0` for
Atlas 800I A2, `quay.io/ascend/vllm-omni:v0.29.0-a3` for Atlas 800I A3 — and
check the
[tag list](https://quay.io/repository/ascend/vllm-omni?tab=tags) for newer
releases:

```bash
export IMAGE=quay.io/ascend/vllm-omni:v0.29.0-a3
export CONTAINER_NAME=h3

docker run -it -u root --name ${CONTAINER_NAME} \
  --privileged=true \
  --shm-size=2000g \
  --net=host \
  --device /dev/davinci0 \
  --device /dev/davinci1 \
  --device /dev/davinci2 \
  --device /dev/davinci3 \
  --device /dev/davinci4 \
  --device /dev/davinci5 \
  --device /dev/davinci6 \
  --device /dev/davinci7 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  -v /home:/home \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /root/.cache:/root/.cache \
  ${IMAGE} \
  /bin/bash
```

Inside the container:

- **MindIE-SD** provides the Ascend-optimized fused operators
  (`adalayernorm`, etc.), the LaserAttention kernel, and the RainFusion
  block-sparse attention kernel. If the image does not ship it, build it from
  source:

```bash
git clone https://gitcode.com/Ascend/MindIE-SD.git
cd MindIE-SD
python setup.py bdist_wheel
pip install dist/mindiesd-*.whl
```

- To track the latest vLLM-Omni, install it from a checkout:

```bash
git clone https://github.com/vllm-project/vllm-omni.git
cd vllm-omni
VLLM_OMNI_TARGET_DEVICE=npu pip install -e . --no-build-isolation
```

- `ffmpeg` and `ffprobe` must be available on `PATH` (reference-video
  preparation and MP4 output). Reference-video decoding uses `decord` when
  available and falls back to PyAV otherwise; `decord` is required only for
  the Ref2VA workflow and can be built with:

```bash
apt update
apt install -y ffmpeg pkg-config libavcodec-dev libavformat-dev \
  libavutil-dev libswscale-dev libavfilter-dev libavdevice-dev
git clone --depth 1 --recursive https://github.com/dmlc/decord.git
cd decord
mkdir -p build && cd build
cmake .. -DUSE_CUDA=0 -DDECODE_FFMPEG=1 -DCMAKE_BUILD_TYPE=Release
make
cd ../python
pip3 install . --no-build-isolation
```

## Start a server

T2VA and FL2VA share one set of weights; the Ref2VA weights differ. Select
the task partition at startup with **`--task-type`** (`fl2va` or `ref2va`).
When unset, only the T2VA/FL2VA weights are loaded. All remaining flags are
task-agnostic.

### Recommended lossless configuration (Atlas 800I A2 / A3)

Lossless means fused operators plus the high-performance FLASH_ATTN backend
only. The configuration uses 8-card USP, 8-card text-encoder TP, distributed
layerwise offload (DLO), and VAE `tile` parallelism:

```bash
export PORT=9098
export MODEL=/path/to/MiniMax-H3
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=4000
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export MINDIE_SD_FA_TYPE="ascend_laser_attention"
export HCCL_NPU_SOCKET_PORT_RANGE="auto"

vllm serve "${MODEL}" \
  --omni \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --trust-remote-code \
  --task-type fl2va \
  --num-gpus 8 \
  --usp 8 \
  --ring 1 \
  --text-encoder-tp-size 8 \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --vae-patch-parallel-size 8 \
  --enable-distributed-layerwise-offload \
  --enable-diffusion-pipeline-profiler \
  --diffusion-attention-backend FLASH_ATTN
```

Notes:

- Do not add `--enforce-eager`. The first request includes regional
  compilation; warm the server once before measuring steady-state latency.
- H3 is CFG-distilled, so `--cfg-parallel-size` must remain 1.
- Keep `--ring 1` when switching to RainFusion attention (below): the
  block-sparse kernel ranks key blocks over the whole sequence, so ring
  parallelism would split away the keys it needs. Scale with `--usp` instead.
- DLO offloads DiT layers with parameters gathered across the parallel group
  instead of replicated per rank, which is what fits the combined 768P
  service into 64 GB HBM per device (Atlas 800I A3). Host (CPU) memory usage
  is high with this option.

### Recommended lossy configuration (Atlas 800I A2 / A3)

Adds RainFusion block-sparse attention and INT8 online quantization on top of
the lossless configuration. Validated for T2VA and Ref2VA; see
[Benchmarks](#benchmarks-atlas-800i-a2--a3):

```bash
vllm serve "${MODEL}" \
  --omni \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --trust-remote-code \
  --task-type fl2va \
  --num-gpus 8 \
  --usp 8 \
  --ring 1 \
  --text-encoder-tp-size 8 \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --vae-patch-parallel-size 8 \
  --enable-distributed-layerwise-offload \
  --enable-diffusion-pipeline-profiler \
  --diffusion-quantization-config '{"transformer":{"method":"int8"}}' \
  --diffusion-attention-config '{"default": {"backend": "RAINFUSION_ATTN",
      "block_sparse": {"sparsity": 0.8, "start_step": 12}}}'
```

Online INT8 quantization can be combined with distributed layerwise offload
while AllGather is enabled. Keep AllGather enabled for SP>1; use
`--dlo-no-use-allgather` only when collective communication is unavailable.

### Optional optimizations

The following sections list only the **increment or replacement** relative to
the recommended lossless configuration; environment variables and the
remaining flags stay unchanged.

#### Cache-DiT

Append to enable DiT block caching with TaylorSeer extrapolation — see the
[Cache-DiT guide](../../docs/user_guide/diffusion/cache_acceleration/cache_dit.md):

```bash
  --cache-backend cache_dit \
  --enable-cache-dit-summary \
  --cache-config '{"Fn_compute_blocks":2,"Bn_compute_blocks":1,"max_warmup_steps":4,"residual_diff_threshold":0.4,"max_continuous_cached_steps":4,"enable_taylorseer":true,"taylorseer_order":2}'
```

#### EQBSA sparse attention

EQBSA applies Q/K INT8 with V FP8 mixed quantization on top of block-sparse
attention. Replace `--diffusion-attention-backend FLASH_ATTN` with:

```bash
  --diffusion-attention-config '{"default":{"backend":"RAINFUSION_ATTN","block_sparse":{"sparsity":0.8,"start_step":8,"end_step":13,"precision":"mix"}}}'
```

`precision` defaults to `bf16`; enabling EQBSA requires setting it to `mix`
explicitly. `start_step` and `end_step` are the numbers of leading/trailing
steps that fall back to dense attention — not step indices. For 1344x768,
50-step T2VA:

| Configuration | `start_step` | `end_step` | Notes |
| --- | ---: | ---: | --- |
| Quality first (recommended) | 8 | 13 | General and complex-motion scenes |
| Balanced | 0 | 13 | Balanced |
| Speed first | 0 | 0 | Simple, low-motion scenes |

When frames look discontinuous or details unstable, increase `end_step`
first, then `start_step`; do not compensate by lowering `sparsity` alone.
These pairings were validated at 1344x768 / 50 steps only; re-evaluate for
other resolutions or step counts.

## HTTP API examples

The request contract is identical to the GPU recipe; see
[MiniMax-H3.md § HTTP API examples](MiniMax-H3.md#http-api-examples). Use the
validated 768P shapes (e.g. `width=1344 height=768`) on NPU. Quick start
against the synchronous endpoint:

```bash
export API_URL="http://127.0.0.1:${PORT}/v1/videos/sync"

curl -sS -X POST "${API_URL}" \
  -F 'prompt=In a snowy blue-purple forest, Ori carefully walks past a sleeping giant; footsteps crunch in the snow while the creature breathes and softly snorts.' \
  -F 'width=1344' \
  -F 'height=768' \
  -F 'aspect_ratio=16:9' \
  -F 'fps=24' \
  -F 'num_inference_steps=50' \
  -F 'flow_shift=12.0' \
  -F 'audio_flow_shift=3.0' \
  -F 'seed=1101' \
  -F 'extra_params={"task":"t2va","duration":5.0}' \
  -o t2va.mp4
```

This requires the FLASH_ATTN backend and MindIE-SD. H3 automatically applies
exact power-of-two input pre-scaling (`laser_input_scale=256`) so the
kernel's fp16 workspace cannot overflow on outlier activations. Measured on
the FastH3 four-step Dense configuration below, T2VA 15 s at 1344x768, this
kernel reduced end-to-end latency from ~73 s to ~57 s (about 22%).


## FastH3 four-step on A3

[FastH3](https://haoailab.com/blogs/fasth3-preview/) is FastVideo's four-step
DMD2 student of H3-Base; see
[MiniMax-H3.md § FastH3 adapter](MiniMax-H3.md#fasth3-adapter) for the adapter
contract. On A3 the adapter cannot be fused the GPU way. The GPU recipe fuses
it at load time from `--lora-path`, but that path replicates the full model per
rank and does not fit in 64 GB HBM. A3 needs distributed layerwise offload, and
offload is refused with `--lora-path` because it streams weights in without
going through the fusion (see the GPU recipe's note on why
`--enable-distributed-layerwise-offload` fails fast with a FastH3 adapter).

The A3 path therefore fuses the adapter **offline**, once, into a native-layout
checkpoint that distributed layerwise offload can memory-map directly. This
also lets the offloaded server start without `--lora-path`, so the fusion
contract check that rejects offload never fires.

### Prepare the fused checkpoint (one-time, offline)

The fusion runs in the vLLM-Omni native namespace
(`blocks.N.attn.qkv_proj.weight`), matching the base H3 `FL2VA` transformer
layout. It reads the base transformer and the Dense adapter, adds the adapter's
low-rank and full-rank deltas per shard, symlinks the unchanged components
(text encoder, VAEs, tokenizer, processor), and writes the four-step sigma
ladder into `model_index.json` so the pipeline samples on the release's rungs
rather than a uniform schedule.

Download the base checkpoint's `FL2VA` partition (native layout) and the Dense
adapter:

```bash
export BASE_DIR=/path/to/MiniMax-H3
hf download MiniMaxAI/MiniMax-H3 --include "FL2VA/*" --local-dir "${BASE_DIR}"

export FASTH3_DIR=/path/to/fasth3
hf download FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA \
  dense-datafree/adapter_model.safetensors --local-dir "${FASTH3_DIR}"
```

> The published full checkpoint
> `FastVideo/FastVideo-FastH3-4-step-Preview-v1-Dense-DataFree` is stored in the
> diffusers layout (`transformer_blocks.N.attn.to_q/to_k/to_v`, split QKV,
> value-first MLP). Distributed layerwise offload memory-maps by exact runtime
> parameter name and has no diffusers-to-native remap for H3, so that artifact
> cannot be served directly. Fuse from the base `FL2VA` partition instead — its
> keys already match the runtime.

Run the fusion:

```python
# prepare_fasth3_dense.py — fuse base H3 FL2VA + FastH3 Dense adapter (native layout)
import json, os, shutil, sys, time
import torch
from safetensors.torch import load_file, save_file
from vllm_omni.diffusion.models.minimax_h3.fasth3 import FastH3WeightFusion

BASE = os.environ["BASE_DIR"] + "/FL2VA"          # base H3 FL2VA partition (native layout)
ADAPTER = os.environ["FASTH3_DIR"] + "/dense-datafree/adapter_model.safetensors"
OUT = os.environ.get("FUSED_DIR", "/path/to/FastH3-Dense-Fused/FL2VA")

# MiniMax-H3 architecture
HEAD_DIM, NUM_BLOCKS, NUM_REFINER_BLOCKS = 128, 50, 2
# FastH3 four-step sigma positions; the pipeline adds per-modality shift on top.
BASE_SCHEDULE = [0.999, 0.749, 0.5, 0.25, 0.0]
COMPONENTS = ["audio_vae", "video_vae", "text_encoder", "tokenizer", "processor"]

fusion = FastH3WeightFusion.from_path(
    ADAPTER, head_dim=HEAD_DIM, num_blocks=NUM_BLOCKS,
    num_refiner_blocks=NUM_REFINER_BLOCKS)
assert fusion is not None, "adapter not recognized as FastH3"

os.makedirs(OUT + "/transformer", exist_ok=True)
src_t = BASE + "/transformer"
for shard in sorted(f for f in os.listdir(src_t)
                    if f.endswith(".safetensors") and "index" not in f):
    data = load_file(os.path.join(src_t, shard), device="cpu")
    fused = {}
    for k, v in data.items():
        fv = fusion.fuse(k, v)
        fused[k] = fv.to("cpu").to(torch.bfloat16) if fv is not v else v
    save_file(fused, os.path.join(OUT, "transformer", shard))
fusion.validate_fully_applied()   # every delta must have met its parameter

for j in os.listdir(src_t):       # copy transformer index/config
    if j.endswith(".json"):
        shutil.copy2(os.path.join(src_t, j), os.path.join(OUT, "transformer", j))

for c in COMPONENTS:              # symlink unchanged components
    s = os.path.join(BASE, c)
    if os.path.exists(s):
        os.symlink(os.path.realpath(s), os.path.join(OUT, c))

idx = json.loads(open(BASE + "/model_index.json").read())   # inject sigma ladder
idx.setdefault("_minimax_h3", {})["base_schedule"] = BASE_SCHEDULE
open(OUT + "/model_index.json", "w").write(json.dumps(idx, indent=4) + "\n")
print("fused ->", OUT)
```

```bash
export FUSED_DIR=/path/to/FastH3-Dense-Fused/FL2VA
python prepare_fasth3_dense.py
```

### Serve the fused checkpoint

Serve the fused directory with the same multi-NPU flags as the base recipe, but
**without** `--lora-path` — the adapter is already in the weights. Keep
distributed layerwise offload and, optionally, LaserAttention:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PORT=9098
export MODEL=/path/to/FastH3-Dense-Fused/FL2VA
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800
export MINDIE_SD_FA_TYPE="ascend_laser_attention"   # optional

vllm serve "${MODEL}" \
  --omni \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --trust-remote-code \
  --task-type t2va \
  --init-timeout 1800 \
  --stage-init-timeout 1800 \
  --num-gpus 8 \
  --usp 8 \
  --ring 1 \
  --text-encoder-tp-size 8 \
  --enable-distributed-layerwise-offload \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --vae-patch-parallel-size 8 \
  --diffusion-attention-backend FLASH_ATTN
```

Requests must ask for `num_inference_steps=4` and `task=t2va`; FastH3 preview v1
distills the T2VA path only. The first request includes regional compilation
(~50 s warmup); exclude it from steady-state numbers.

```bash
curl -s -X POST "http://localhost:${PORT}/v1/videos/sync" \
  -F 'prompt=A golden retriever running through a sunflower field at sunset' \
  -F 'seconds=5' \
  -F 'aspect_ratio=16:9' \
  -F 'num_inference_steps=4' \
  -o out.mp4
```

### Measured FastH3 four-step evidence

Measured on an Atlas 800I A3 server (8x NPU) with CANN 9.0.1, PyTorch
2.10.0+cpu, torch_npu 2.10.0.post2, the multi-NPU configuration above, T2VA at
1344x768, one warmup excluded:

| Duration | LaserAttention | End-to-end |
| ---: | --- | ---: |
| 4 s | on | ~12 s (768x768), ~15 s (1344x768) |
| 8 s | on | ~26 s |
| 15 s | on | ~57 s |
| 5 s | off | ~19 s |
| 10 s | off | ~41 s |
| 15 s | off | ~73 s |

These describe the validated shapes rather than a general throughput guarantee.

> A native `--lora-path + --enable-distributed-layerwise-offload` path that
> fuses during the offload memory-map (avoiding the offline step) is possible
> through the offload backend's per-tensor transform hook, but is left as future
> work pending upstream design discussion.


## HTTP API examples
Switch tasks via `extra_params.task` (`t2va` / `fl2va` / `ref2va`); FL2VA and
Ref2VA additionally pass `input_references` (first-frame image, reference
video/audio). The full parameter table lives in the GPU recipe.

## FlashGen 4-step online LoRA (T2VA)

The FlashGen 4-step weights are a native-layout LoRA single file
(`key_format=minimax-h3-native`) loaded at runtime — no merge into the base
checkpoint. Download them from
[FlashGen/Minimax-H3-4step-lora-flashgen](https://modelscope.cn/models/FlashGen/Minimax-H3-4step-lora-flashgen),
then on top of the recommended configuration above:

- append `--lora-backend peft --lora-path "${FLASHGEN_LORA}"`,
- use `--diffusion-attention-backend FLASH_ATTN` (dense; drop the RainFusion
  attention config),
- issue T2VA requests with `num_inference_steps=4` and the `lora` request
  field (`name` / `path` / `scale`) — see
  [MiniMax-H3.md § FlashGen native LoRA](MiniMax-H3.md#flashgen-native-lora).

T2VA only; FL2VA and Ref2VA are not supported, and neither
`--enable-cpu-offload` nor `--enable-layerwise-offload` may be combined with
it.

## CPU MP4 response encoding (Atlas A2)

Non-streaming MP4 responses use one public automatic encoder. It checks the
runtime frame shape, common dtype, and RGB channel-plane contiguity for every
request; compatible inputs use direct planar PyAV frames, while unsupported
inputs fall back to the legacy muxer before opening the PyAV container. No CLI
flag, model declaration, or user configuration is required. Streaming fMP4
output remains on its existing incremental path.

## Ascend-optimized components

The optimizations stacked by the recommended configurations above:

1. Ascend-friendly attention: the MindIE-SD `ascend_laser_attention` fused
   kernel selected through `MINDIE_SD_FA_TYPE`, carried by the FLASH_ATTN
   backend with a mask-free packed varlen path and K/V prefix slicing driven
   by the packed `cu_seqlens` metadata.
2. Ascend-friendly fused operators: RMSNorm, AddRMSNorm, GQA, SwiGLU, and
   rotary position embedding from MindIE-SD.
3. RainFusion block-sparse attention, optionally with EQBSA mixed INT8/FP8
   precision — see
   [RainFusion](../../docs/user_guide/diffusion/attention_backends/rainfusion.md).
4. Online quantization: INT8, MXFP8, and MXFP4 — see
   [online quantization](../../docs/user_guide/quantization/online.md).
5. Parallelism: USP, text-encoder TP, and native VAE `tile` patch
   parallelism, plus distributed layerwise offload.
6. Post-training: few-step distillation via the FlashGen online LoRA.

## Benchmarks (Atlas 800I A2 / A3)

Measured with `--enable-diffusion-pipeline-profiler` on vLLM-Omni 0.28.0:
lossless rows use the recommended lossless configuration, lossy rows the
recommended lossy configuration. Memory figures are per device.

### Atlas 800I A2

| Task | Resolution | Frames | Duration (s) | Config | E2E (s) | DiT total (s) | DiT per-step (s) | DiT steps | Text encode (s) | Ref video encode (s) | Ref audio encode (s) | VAE decode (s) | Ref preprocess (s) | Post-process (s) | CPU MP4 (s) | Resident weights (GB) | Peak memory (GB) |
| ---- | ---- | ---- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| t2va | 1344x768 | 124 | 5 | lossless | 150.06 | 139.9 | 2.85 | 49 | 0.83 | NA | NA | 6.76 | NA | 0.462 | 1.18 | 2.97 | 24.7 |
| t2va | 1344x768 | 362 | 15 | lossless | 786.24 | 761.36 | 15.53 | 49 | 1.20 | NA | NA | 15.87 | NA | 1.98 | 2.95 | 2.97 | 36.21 |
| t2va | 1344x768 | 124 | 5 | lossy | 108.1 | 95.85 | 1.95 | 49 | 0.9 | NA | NA | 6.73 | NA | 0.745 | 1.04 | 2.55 | 22.47 |
| t2va | 1344x768 | 362 | 15 | lossy | 552.51 | 526.04 | 10.73 | 49 | 1.28 | NA | NA | 16.07 | NA | 1.90 | 3.76 | 2.55 | 35.11 |
| ref2va | 1344x768 | 124 | 5 | lossless | 474.89 | 421.68 | 8.6 | 49 | 2.42 | 13.31 | 0.38 | 6.47 | 3.94 | 0.489 | 1.16 | 3.4 | 23.45 |
| ref2va | 1344x768 | 362 | 15 | lossless | 3414.18 | 3336.98 | 68.08 | 49 | 6.48 | 30.85 | 0.59 | 15.8 | 12.59 | 1.99 | 3.08 | 3.4 | 36.28 |
| ref2va | 1344x768 | 124 | 5 | lossy | 380.92 | 326.50 | 6.66 | 49 | 2.47 | 13.48 | 0.38 | 6.36 | 4.68 | 0.591 | 1.22 | 2.55 | 22.57 |
| ref2va | 1344x768 | 362 | 15 | lossy | 2332.04 | 2249.63 | 45.89 | 49 | 6.33 | 35.1 | 0.59 | 15.72 | 11.07 | 1.35 | 2.3 | 2.55 | 35.58 |

### Atlas 800I A3

| Task | Resolution | Frames | Duration (s) | Config | E2E (s) | DiT total (s) | DiT per-step (s) | DiT steps | Text encode (s) | Ref video encode (s) | Ref audio encode (s) | VAE decode (s) | Ref preprocess (s) | Post-process (s) | CPU MP4 (s) | Resident weights (GB) | Peak memory (GB) |
| ---- | ---- | ---- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| t2va | 1344x768 | 124 | 5 | lossless | 114.19 | 105.87 | 2.16 | 49 | 0.48 | NA | NA | 6.56 | NA | 0.001 | 0.46 | 3.49 | 24.7 |
| t2va | 1344x768 | 362 | 15 | lossless | 599.29 | 580.38 | 11.84 | 49 | 1.83 | NA | NA | 14.95 | NA | 0.001 | 1.02 | 3.49 | 36.21 |
| t2va | 1344x768 | 124 | 5 | lossy | 84.82 | 77.47 | 1.58 | 49 | 0.57 | NA | NA | 6.23 | NA | 0.001 | 0.61 | 2.63 | 22.47 |
| t2va | 1344x768 | 362 | 15 | lossy | 401.43 | 383.56 | 7.82 | 49 | 1.72 | NA | NA | 14.83 | NA | 0.001 | 1.14 | 2.63 | 35.11 |
| ref2va | 1344x768 | 124 | 5 | lossless | 355.53 | 311.86 | 6.36 | 49 | 1.89 | 12.37 | 0.24 | 5.55 | 2.49 | 0.001 | 0.43 | 3.49 | 23.45 |
| ref2va | 1344x768 | 362 | 15 | lossless | 2339.56 | 2277.61 | 46.47 | 49 | 5.61 | 34.46 | 0.36 | 13.61 | 6.83 | 0.001 | 1.11 | 3.49 | 36.28 |
| ref2va | 1344x768 | 124 | 5 | lossy | 306.82 | 247.72 | 5.05 | 49 | 1.88 | 12.71 | 0.24 | 5.24 | 2.45 | 0.001 | 0.54 | 2.63 | 22.57 |
| ref2va | 1344x768 | 362 | 15 | lossy | 1695.26 | 1634.58 | 33.35 | 49 | 5.57 | 33.27 | 0.36 | 13.51 | 6.96 | 0.001 | 1.22 | 2.63 | 35.58 |

These numbers describe the validated shapes rather than a general throughput
guarantee.

## Known limitations

- Task serving is partitioned by `--task-type`: T2VA/FL2VA and Ref2VA load
  different DiTs, so switching between them requires a restart with the other
  partition.
- H3 currently executes one generation request per diffusion batch.
- The first regional-compile request is a warmup and should not be included
  in steady-state performance measurements.
- The official H3 input matrix and media limits are documented in the [GPU
  recipe](MiniMax-H3.md#official-input-matrix-and-limits); this NPU path uses
  the same HTTP request contract.
- VAE patch parallelism requires size 1 or the full DiT group size and
  supports the H3 native `tile` mode only.
- The lossy configuration (RainFusion + INT8) is validated for T2VA and
  Ref2VA; use the lossless configuration for FL2VA or re-validate first.

## Additional resources

- [MiniMax-H3.md](MiniMax-H3.md) — full GPU guide
- [MiniMax-H3-NPU-950PR.md](MiniMax-H3-NPU-950PR.md) — four-card
  Ascend 950PR / 950DT guide
- [NPU installation](../../docs/getting_started/installation/npu.md)
- [RainFusion attention](../../docs/user_guide/diffusion/attention_backends/rainfusion.md)
  — block-sparse knobs and tuning
- [Online quantization](../../docs/user_guide/quantization/online.md)
  — INT8 / MXFP8 / MXFP4
- [Cache-DiT guide](../../docs/user_guide/diffusion/cache_acceleration/cache_dit.md)
- [Supported models](../../docs/models/supported_models.md)
- [Video API](../../docs/serving/videos_api.md)
