# MiniMax H3 on Ascend NPU

> Joint video and audio generation with text, first/last keyframes, and mixed
> image/video/audio references — Ascend NPU deployment guide

## Summary

- Vendor: MiniMaxAI
- Model: [`MiniMaxAI/MiniMax-H3`](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- Tasks: T2VA, FL2VA, and Ref2VA
- Mode: OpenAI-compatible `/v1/videos` HTTP serving
- Hardware: Atlas 800I A3
- Maintainer: Community

This recipe adapts [MiniMax-H3.md](MiniMax-H3.md) for Ascend NPU
environments. Differences from the GPU path:

- Runs on a CPU-only (aarch64) PyTorch build with `torch_npu`; no CUDA
  runtime is present.
- Audio loading does **not** require TorchCodec (whose aarch64 wheels are
  built against CUDA torch and fail to load on CPU-only builds). vLLM-Omni
  automatically falls back to soundfile / ffmpeg for wav/mp3/m4a/mp4 audio
  inputs.

## Prerequisites

### Checkpoint

Same as the GPU recipe — Hugging Face access approval is required. Authenticate
once; `vllm serve` downloads the required nested components automatically:

```bash
hf auth login
export MODEL=MiniMaxAI/MiniMax-H3
```

### Environment

- Ascend driver & firmware: 25.5.1
- CANN toolkit: 9.0.1
- Python: 3.12
- PyTorch: 2.10.0+cpu
- torch_npu: 2.10.0.post2
- Install vLLM-Omni from a checkout with MiniMax H3 support:

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

Install the **mindie-sd** operator library to enable Ascend-optimized fused
operators (`adalayernorm`, etc.) and the RainFusion `rf_v2` block-sparse
attention kernel:

```bash
git clone https://gitcode.com/Ascend/MindIE-SD.git && cd MindIE-SD

# Comment out the tik_ops build step (not needed for this use case)
sed -i 's|^\(\s*\)source ${current_script_dir}/build_tik_ops.sh|\1# source ${current_script_dir}/build_tik_ops.sh|' build/build_ops.sh

python setup.py bdist_wheel
cd dist
pip install mindiesd-*.whl
```

- `ffmpeg` and `ffprobe` must be available on `PATH`. They are used for
  reference-video preparation and MP4 output.
- Reference-video decoding uses `decord` when available and falls back to
  PyAV otherwise.
- Audio inputs do not require TorchCodec: wav/mp3/m4a/mp4 files are loaded
  at native sample rate through the soundfile / ffmpeg fallback.

## Start a server

Pass the repository ID directly. The pipeline loads the two nested DiTs while
sharing the tokenizer, processor, text encoder, and VAEs from `FL2VA`.

### Multi-NPU: 768P combined-service configuration

Use Ulysses sequence parallelism degree 8, text-encoder tensor parallelism
degree 8, native tiled VAE patch parallelism degree 8, and distributed
layerwise offload:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PORT=9098
export MODEL=MiniMaxAI/MiniMax-H3
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800
export PYTHONDONTWRITEBYTECODE=1

vllm serve "${MODEL}" \
  --omni \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --trust-remote-code \
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

Do not add `--enforce-eager`. The first request includes regional
compilation; warm the server once before measuring steady-state latency.
H3 is CFG-distilled, so `--cfg-parallel-size` must remain 1.

The same endpoint accepts `task=t2va`, `task=fl2va`, and `task=ref2va`; no
partition restart is required. Layerwise offload applies to both DiTs.

On Atlas 800I A3 (64 GB HBM per device) the combined service does not fit at
768P without offloading or sharding: use distributed layerwise offload (as
above) or HSDP — see
[§ Memory and attention optimizations](#memory-and-attention-optimizations-a3).

### CPU MP4 response encoding (Atlas A2)

Non-streaming MP4 responses use one public automatic encoder. It checks the
runtime frame shape, common dtype, and RGB channel-plane contiguity for every
request; compatible inputs use direct planar PyAV frames, while unsupported
inputs fall back to the legacy muxer before opening the PyAV container. No CLI
flag, model declaration, or user configuration is required. Streaming fMP4
output remains on its existing incremental path.

Current performance and correctness validation is limited to one Atlas A2 host
with 8x Ascend 910B4-1 NPUs, the `FL2VA`/`t2va` partition, one request at a
time, 1344x768 at 24 fps, and 5, 8.7, and 15 second requests. This optimization
only changes CPU MP4 response encoding; it does not change DiT execution or
stage 0.

### Optional optimizations

Two independent optimizations may be enabled on top of the configuration
above. Both are validated for T2VA only.

**RainFusion block-sparse attention** — switch the attention backend, keeping
every other flag unchanged:

```bash
  --diffusion-attention-backend RAINFUSION_ATTN
```

**INT8 online quantization** — add one flag to the server command above:

```bash
  --quantization int8
```

Keep `--ring 1` when using RainFusion: the `rf_v2` kernel ranks key blocks
over the whole sequence, so ring parallelism would split away the keys it
needs. Scale with `--usp` instead.

## Memory and attention optimizations (A3)

### Fitting 768P into 64 GB HBM: distributed layerwise offload or HSDP

Each Atlas 800I A3 NPU has 64 GB of HBM, which is not enough for the
combined MiniMax-H3 service at 768P. Enable **one** of the following at
server startup:

**Distributed layerwise offload** — DiT layers are offloaded with parameters
gathered across the parallel group instead of replicated per rank:

```bash
  --enable-distributed-layerwise-offload
```

Measured peak NPU memory is about 45 GB per device for Ref2VA with a 13.88 s
input video generating a 15 s 768P (1344x768) output video. Host (CPU) memory
usage is high with this option.

**HSDP** — hybrid sharded data parallelism for the DiT parameters. The
multi-stream memory-reuse knob is mandatory with this flag:

```bash
export MULTI_STREAM_MEMORY_REUSE=2
```

```bash
  --use-hsdp
```

Host memory usage is small, but large shapes may still OOM; HBM usage
optimizations for this path are ongoing.

### FLASH_ATTN backend with MindIE-SD

Keep `--diffusion-attention-backend FLASH_ATTN` and install MindIE-SD (see
[§ Environment](#environment)). On NPU this backend carries most of the
memory-reduction and performance work: a mask-free packed varlen path that
never materializes the quadratic `full_qk` padding mask, and K/V prefix
slicing, both driven by the packed `cu_seqlens` metadata emitted by the H3
transformer.

### Optional: LaserAttention fused kernel

For an additional attention speedup, select the Ascend LaserAttention fused
kernel before starting the server:

```bash
export MINDIE_SD_FA_TYPE="ascend_laser_attention"
```

This requires the FLASH_ATTN backend and MindIE-SD. H3 automatically applies
exact power-of-two input pre-scaling (`laser_input_scale=256`) so the
kernel's fp16 workspace cannot overflow on outlier activations. Measured on
the FastH3 four-step Dense configuration below, T2VA 15 s at 1344x768, this
kernel reduced end-to-end latency from ~73 s to ~57 s (about 28%).


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

The request format is identical to the GPU recipe; see
[MiniMax-H3.md § HTTP API examples](MiniMax-H3.md#http-api-examples).
Use the validated 768P shapes (e.g. `width=1344 height=768`) on NPU.

## Key parameters

Same as the GPU recipe; see
[MiniMax-H3.md § Key parameters](MiniMax-H3.md#key-parameters).
The validated resolution on NPU is 768P (e.g. 1344x768).

## Validated NPU evidence

Measured on an Atlas 800I A3 server (8x NPU) with CANN 9.0.1,
PyTorch 2.10.0+cpu, and torch_npu 2.10.0.post2, using the multi-NPU
configuration above:

| Workload | Configuration |
|----------|---------------|
| T2VA, 209 frames, 1344x768 | TE TP8, distributed layerwise offload, Ulysses 8, VPP8 tile, regional compile |
| Ref2VA (prompt + video), 124 frames, 1344x768 | TE TP8, distributed layerwise offload, Ulysses 8, VPP8 tile, regional compile |

These measurements describe the validated shapes rather than a general
throughput guarantee.

## Known limitations

- Combined serving requires sibling `FL2VA` and `Ref2VA` directories, loads
  both task-specific DiTs, and loads one copy of every shared component.
- H3 currently executes one generation request per diffusion batch.
- The first regional-compile request is a warmup and should not be included
  in steady-state performance measurements.
- The official H3 input matrix and media limits are documented in the [GPU
  recipe](MiniMax-H3.md#official-input-matrix-and-limits); this NPU path uses
  the same HTTP request contract.
- VAE patch parallelism requires size 1 or the full DiT group size and
  supports the H3 native `tile` mode only.
- RainFusion block-sparse attention and INT8 quantization are validated for
  T2VA only; use the BF16 dense configuration for FL2VA and Ref2VA.
- Online quantization cannot be combined with distributed layerwise offload
  while AllGather is enabled; pass `--dlo-no-use-allgather` in that case.
- FastH3 on A3 requires the offline fusion step above: the load-time
  `--lora-path` fusion used on GPU is incompatible with the distributed
  layerwise offload that A3's 64 GB HBM needs. FastH3 preview v1 distills T2VA
  only, and requests must use `num_inference_steps=4`.

## Additional resources

- [MiniMax-H3.md](MiniMax-H3.md) — full GPU guide
- [Attention backends § RAINFUSION_ATTN](../../docs/user_guide/diffusion/attention_backends.md#rainfusion_attn-backend-and-block-sparse-video-attention)
  — RainFusion knobs and tuning
- [Int8 quantization](../../docs/user_guide/quantization/int8.md)
- [Supported models](../../docs/models/supported_models.md)
- [Video API](../../docs/serving/videos_api.md)
