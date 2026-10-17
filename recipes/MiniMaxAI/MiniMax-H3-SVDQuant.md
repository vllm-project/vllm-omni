# MiniMax-H3 rank-32 SVDQuant low-memory candidate

This recipe implements the V2/V3 checkpoint and residency contract from
[issue #6493](https://github.com/vllm-project/vllm-omni/issues/6493).
The supplied exporter is a **data-free SVD/round-to-nearest candidate**.
It has not been calibrated and is not an official MiniMax checkpoint release.
Loading successfully does not establish acceptable output quality or a 32 GB
deployment result.

## Offline export

Prepare a standard, undistilled `MiniMaxAI/MiniMax-H3` partition locally.
Run one partition at a time; the source and destination must be different
directories. The exporter refuses to overwrite an existing destination.
Reserve at least 50 GiB of free disk in addition to the source for FL2VA;
video VAE conversion needs temporary storage beside the packed output.

```bash
python -m vllm_omni.quantization.tools.export_minimax_h3_low_memory_checkpoint \
  /path/to/MiniMax-H3/FL2VA /path/to/MiniMax-H3-low-memory/FL2VA \
  --source-revision 42ed227ee7df40d41602854ae760620d6eb651fe \
  --steps 2 5 50 --video-shift 12 --audio-shift 3 --device cuda:0
```

Repeat with `Ref2VA` as both partition names. No download, calibration,
conversion or publication occurs during model loading.

The export stores:

- rank-32 NVFP4 W4A4 DiT linears with a BF16 correction;
- BF16 AdaLN output tables at exact FP32 timesteps for the exported schedules;
- a rank-32 NVFP4 W4A16 text encoder, with BF16 vision, embeddings and norms;
- FP16 video decoder weights, retaining FP32 keyframe encoding; and
- the original audio VAE precision, tokenizer, processor and partition metadata.

W4A16 uses upstream NVFP4 dequantization followed by BF16 GEMM. Only the current
linear's dense weight is temporary; all packed encoder weights stay packed.
This is a correctness reference path, not a fused weight-only kernel.

`export_manifest.json` records the source revision supplied by the operator,
exporter hash, seeds, schedules, software version and output shard hashes.
Verify that the supplied revision matches the source files; it is provenance
metadata, not a source-content integrity check.

## Single-GPU sequential residency

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve /path/to/MiniMax-H3-low-memory/FL2VA \
  --omni --trust-remote-code --host 127.0.0.1 --port 8080 \
  --num-gpus 1 --tensor-parallel-size 1 --text-encoder-tp-size 1 \
  --usp 1 --ring 1 --vae-patch-parallel-size 1 \
  --vae-parallel-mode tile --vae-use-tiling \
  --enable-cpu-offload --enforce-eager \
  --diffusion-attention-backend CUDNN_ATTN
```

The compatibility CPU-offload lifecycle includes the VAEs. The compact
`diffusion_offload_config` component selector keeps VAEs resident, so its memory
result is a different configuration. Serialized quantization is loaded on CPU;
accelerator-specific layout processing stages each linear separately.

The AdaLN checkpoint requires eager execution and an exported timestep plan.
Unlisted steps, flow shifts or condition timesteps raise an error; no nearest
row lookup or interpolation is used. Text-encoder TP is currently limited to
one for the serialized W4A16 format. Stop the server before switching partitions.

## Validation requirements

Use matched BF16 and candidate prompts, media, seeds, dimensions, duration,
sampler, flow shifts, requested steps, attention backend and execution mode.
Keep generated video and audio, and compare both modalities to BF16. A
short or reduced-resolution smoke test is only local evidence.

For the 32 GB claim, run the complete target request on an actual 32 GB card,
including startup, text/reference encoding, denoising and both decoders.
Sample device-wide memory externally and report worker allocator peaks as
separate measurements. A 96 GB card with a measured peak below 32 GiB is
preliminary capacity evidence; it does not prove execution on the 32 GB card.

Native fusion of the NVFP4 GEMM and rank correction remains owned by the
independent FlashInfer optimization. This recipe retains the compatibility
kernel registry path until that implementation has independent correctness
and production-shape evidence.
