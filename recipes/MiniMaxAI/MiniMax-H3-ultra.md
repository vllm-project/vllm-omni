# MiniMax-H3 acceleration

The [MiniMax-H3 recipe](MiniMax-H3.md) covers checkpoints and standard serving.
This recipe configures the optional FastH3 execution path: cached AdaLN,
Sage block-sparse attention, MXFP8 DiT and VAE projections, and overlapped
attention/output and video/audio decoding.

## Requirements

- Use a vLLM version compatible with the vLLM-Omni checkout.
- MXFP8 requires CUDA Blackwell hardware and PyTorch exposing
  `torch.nn.functional.scaled_mm`, `ScalingType.BlockWise1x32`, and
  `SwizzleType.SWIZZLE_32_4_4`.
- Sage requires FlashInfer's SM120 block-sparse Sage provider.
- The optional RDMA adapter requires `flashinfer.comm.UlyssesCommunicator`
  with the PCIe backend and registered input/output buffers, supplied by
  [FlashInfer #4876](https://github.com/flashinfer-ai/flashinfer/pull/4876).
  This dependency is not yet available in a released FlashInfer wheel.
- VAE quantization uses `comfy-kitchen==0.2.33`. MP4 output uses PyAV/libx264.

These dependencies are loaded only when the corresponding path is enabled.
The complete preset currently requires TP1, Ulysses8, ring1, eight SM120 GPUs,
the VSA/Data-Free adapter, four denoiser evaluations, and top-k 162. Its
communication schedule requires 11,992 local rows, prefix segments `(558, 1206)`,
and a target token grid of `(107, 22, 40)`. It rejects other layouts.
The paired VAE schedule covers 21 temporal windows and 28 spatial tiles per
window. Standard serving remains available for other request geometries.

## Prepare the model

Download the original H3 FL2VA checkpoint and the FastH3 VSA/Data-Free adapter
following the base recipe. Set `MODEL_DIR` to the local H3 repository root and
`FASTH3_LORA` to the adapter file. Build the exact, adapter-bound AdaLN cache:

```bash
export ADALN_CACHE="$PWD/minimax-h3-ultra-adaln.safetensors"
python tools/minimax_h3/build_adaln_cache.py \
  --transformer-path "${MODEL_DIR}/FL2VA/transformer" \
  --output "${ADALN_CACHE}" \
  --model-variant fl2va --mode t2va \
  --num-inference-steps 4 --flow-shift 12 --audio-flow-shift 3 \
  --fasth3-adapter "${FASTH3_LORA}" \
  --base-schedule 0.999 0.749 0.5 0.25 0.0
```

The sidecar is bound to the adapter fingerprint, task, sigma schedule, and
modality shifts. Startup checks that identity before omitting the cached
AdaLN projections. MXFP8 conversion follows student fusion; it does not
quantize the teacher and then apply a student delta.

## Configure serving

`VLLM_OMNI_H3_ULTRA=1` selects the complete model schedule. Conflicting explicit
settings are rejected before the preset modifies the environment. Backend
selection, parallelism, and the cache use the existing Omni interfaces:

```bash
export VLLM_OMNI_H3_ULTRA=1
export OMP_NUM_THREADS=28
export PYTORCH_ALLOC_CONF=expandable_segments:False
export VLLM_WORKER_MULTIPROC_METHOD=spawn

vllm serve "${MODEL_DIR}/FL2VA" --omni --trust-remote-code \
  --host 127.0.0.1 --port 8093 \
  --task-type fl2va --lora-path "${FASTH3_LORA}" \
  --num-gpus 8 --tensor-parallel-size 1 --usp 8 --ring 1 \
  --ulysses-mode strict --ulysses-a2a-permute \
  --text-encoder-tp-size 8 \
  --vae-patch-parallel-size 8 --vae-parallel-mode tile --vae-use-tiling \
  --cache-backend none \
  --cache-config "{\"minimax_h3_adaln_cache_path\":\"${ADALN_CACHE}\"}" \
  --diffusion-attention-config '{"default":{"backend":"FASTVIDEO_VSA","fastvideo_vsa_topk":162},"per_role":{"minimax_h3.token_refiner":{"backend":"CUDNN_ATTN"}}}'
```

Configure GPU/NIC affinity for the host topology. Warm up a resident server
before measuring request latency. This integration has no E2E latency claim
until the combined dependency build and PR commit pass model qualification.

FastH3/VSA, Sage and MXFP8 change numerical precision or model computation;
the full configuration is not bitwise equivalent to dense BF16 H3.

## Implementation boundaries

Device detection, memory queries and synchronization use the existing platform
interfaces. Shared layers provide indexed modulation and MXFP8 producers;
shared attention ops provide Sage quantization and sparse attention. Ulysses
owns the reusable communication and buffer lifecycle.

`models/minimax_h3/attention/` owns H3 layout metadata, QKV/gate scheduling,
and output projection overlap. `models/minimax_h3/ops/attention/` owns its tile
maps and chunk kernels. VAE schedules, adapter/cache identity and quantized
layer selection also remain in the model directory.
