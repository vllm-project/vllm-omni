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
- Sage requires FlashInfer's SM120 block-sparse Sage provider with the
  by-value descriptor behavior from
  [FlashInfer #5127](https://github.com/flashinfer-ai/flashinfer/pull/5127).
  The reproducible software version remains to be pinned and qualified.
- The current `FASTVIDEO_VSA` selector also requires `fastvideo-kernel`, even
  for the FlashInfer compute path.
- The optional RDMA adapter requires `flashinfer.comm.UlyssesCommunicator`
  with the PCIe backend and registered input/output buffers, supplied by
  [FlashInfer #4876](https://github.com/flashinfer-ai/flashinfer/pull/4876).
  This dependency is not yet available in a released FlashInfer wheel.
- VAE quantization uses `comfy-kitchen==0.2.33`. MP4 output uses PyAV/libx264.

The optional compute providers are imported on their execution paths; the
selector still checks `fastvideo-kernel` availability before provider selection.
The complete preset currently requires TP1, Ulysses8, ring1, eight SM120 GPUs,
the VSA/Data-Free adapter, four denoiser evaluations, and top-k 162. Its
communication schedule requires 11,992 local rows, prefix segments `(558, 1206)`,
and a target token grid of `(107, 22, 40)`. It rejects other layouts.
The paired VAE schedule covers 21 temporal windows and 28 spatial tiles per
window. Standard serving remains available for other request geometries.

## Attention provider on SM120

Keep `FASTVIDEO_VSA` in `--diffusion-attention-config`: it selects the VSA
routing contract. This draft's complete preset chooses **FlashInfer/CAKE Sage**
for the fine attention kernel, with INT8 Q/K, FP8 V and BF16 output. It does not
use the default FastVideo BF16 kernel for that computation. Changing the public
backend to `FLASHINFER_ATTN`, `SAGE_ATTN` or `SAGE_ATTN_3` does not reproduce this
configuration.

The preset selects this compute path automatically. The FlashInfer BF16
provider is a separate numerical configuration. Neither provider choice implies
FlashInfer Ulysses/RDMA is enabled. Users should not need to set per-kernel
implementation flags to reproduce the final recipe.

The current public integration has not completed full E2E qualification with a
clean, pinned dependency installation. Historical sub-15-second results do not
qualify this PR revision or a different model, layout, provider or NIC topology.

## Prepare the model

This FastH3 adapter supports **T2VA only**. It loads the original H3 `FL2VA`
weight partition; the partition name does not enable first/last-frame tasks
with this adapter. In the server command, `--task-type fl2va` selects those
weights. Requests must set `extra_params.task` to `t2va` and
`num_inference_steps` to `4`.

Follow the base recipe to download H3 and the FastH3 adapter. Select
`vsa-datafree/adapter_model.safetensors`; the base recipe's `dense-datafree`
example is a different adapter. Set `MODEL_DIR` to the local H3 repository
root and `FASTH3_LORA` to the VSA/Data-Free adapter file. Build the exact,
adapter-bound AdaLN cache:

```bash
export ADALN_CACHE="$PWD/minimax-h3-t2va-adaln.safetensors"
python tools/minimax_h3/build_adaln_cache.py \
  --transformer-path "${MODEL_DIR}/FL2VA/transformer" \
  --output "${ADALN_CACHE}" \
  --model-variant fl2va --mode t2va \
  --num-inference-steps 4 --flow-shift 12 --audio-flow-shift 3 \
  --fasth3-adapter "${FASTH3_LORA}" \
  --base-schedule 0.999 0.749 0.5 0.25 0.0
```

The cache filename is arbitrary; it does not select an optimization mode.
The sidecar is bound to the adapter fingerprint, task, sigma schedule, and
modality shifts. Startup checks that identity before omitting the cached
AdaLN projections. MXFP8 conversion follows student fusion; it does not
quantize the teacher and then apply a student delta.

## Configure serving

The current draft integration still requires `VLLM_OMNI_H3_ULTRA=1` to select
the complete schedule. The split PRs will replace this preset with automatic
selection of validated exact optimizations and explicit backend, precision,
and transport settings. Until that implementation lands, omitting the switch
does not run the same accelerated configuration. Conflicting explicit settings
are rejected. Backend selection, parallelism, and the cache use the existing
Omni interfaces:

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
