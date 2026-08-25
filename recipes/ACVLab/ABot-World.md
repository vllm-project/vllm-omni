# ABot-World

> Experimental: offline image-conditioned generation and in-process realtime interactive generation

## Summary

- Vendor: ACVLab (Amap CV Lab)
- Model: `acvlab/ABot-World-0-5B-LF`
- Task: image-conditioned interactive world generation
- Modes: offline image-conditioned generation and experimental in-process realtime AR-Diffusion ticks
- Hardware validated: NVIDIA A100 40GB
- Maintainer: Community

The checkpoint is Apache-2.0 licensed. The vLLM-Omni integration code is also Apache-2.0.

## Realtime architecture

PR #5491 established a model-neutral tick/session orchestration layer for
autoregressive diffusion world models on top of the existing AR-Diffusion
engine and paged-KV session capabilities. ABot-World reuses its typed tick
protocol, session manager, tick consumer, worker lifecycle, request
orchestration, output metadata, and session-owned paged-KV cache.

Only the model-specific pieces live in the ABot integration: the causal
transformer and pipeline, official checkpoint loader, UMT5 and Wan2.2 VAE
conversion, action reducer, and ABot tensor/KV geometry. It does not reuse
LingBot weights or LingBot-specific conditioning logic.

The official ABot checkpoint uses its native single-directory layout and does
not need a Diffusers `model_index.json`.

## Offline generation

Create `AsyncOmni` with the checkpoint path and
`model_class_name="ABotWorldCausalPipeline"`. Submit an image-conditioned
diffusion request with `height=512`, `width=832`, `num_frames=9`,
`num_inference_steps=4`, `max_sequence_length=512`, and
`extra_args={"flow_shift": 5.0}`.

The bundled Wan2.2 VAE compresses space by 16 and the DiT applies a 2x2
spatial patch, so 512x832 produces 16x26 = 416 tokens per latent frame. The
FlashAttention paged-KV kernel requires this page size to be a multiple of 16.
480x832 (390 tokens) and 448x832 (364 tokens) are therefore rejected before
model execution.

Raw frame counts must be `9 + 12k` (9, 21, 33, ..., 117), up to 117 frames.

Camera-action trajectories are not currently supported through
`Omni.generate()`. Camera actions are supported only by the experimental
in-process realtime Session/Tick API, where controls are submitted
incrementally through `ARDiffusionControlInput`. Offline trajectory input
is planned as a follow-up.

## Realtime in-process generation

Each JSONL line describes the prompt and/or three latent-frame camera
actions (W/A/S/D/I/J/K/L) applied at the next chunk boundary:

```json
{"event_id":1,"prompt":"A road through a forest","frames":[["j"],[],[]]}
{"event_id":2,"frames":[["w"],["w"],["w"]]}
{"event_id":3,"prompt":"The road enters a snowy valley","frames":[[],[],[]]}
```

Use `ARDiffusionSessionManager` with `ARDiffusionOmniTickConsumer`,
`ARDiffusionWorkerLifecycle`, and `ABotCameraControlReducer`. Configure
`ARDiffusionEngine`, one replica, `max_num_seqs=1`, latent output, and the same
512x832 four-step sampling contract as offline generation. The current control
plane is an internal Python API; no public server transport is exposed yet.

## Current limitations

- Only the 0.5B-LF causal student checkpoint is supported.
- The realtime control plane is internal; there is no public server transport yet.
- AR-Diffusion stages require one replica.
- One AR block is generated per request and `max_num_seqs` must be one.
- SP/USP, pipeline/CFG parallelism, HSDP, VAE parallelism, quantization, Cache-DiT, and TeaCache are not supported.
- No AMD GPU, Ascend NPU, or Intel GPU support is claimed.
- TAE (tiny autoencoder) fast decoding is not yet integrated; standard VAE decode is used.
- Reference images (5-view surround) are not yet integrated.

## References

- Checkpoint: <https://huggingface.co/acvlab/ABot-World-0-5B-LF>
- Official implementation: <https://github.com/amap-cvlab/ABot-World>
- Realtime design: [`docs/design/feature/realtime_ar_diffusion.md`](../../docs/design/feature/realtime_ar_diffusion.md)
