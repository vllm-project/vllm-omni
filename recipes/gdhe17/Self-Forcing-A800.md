# Self-Forcing — A800

> Few-step causal text-to-video, offline and streaming, on one A800

## Summary

- Vendor: Self-Forcing authors (Hugging Face `gdhe17`)
- Model: `gdhe17/Self-Forcing` (`checkpoints/self_forcing_dmd.pt`) on the
  `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` text encoder, tokenizer and VAE
- Task: text-to-video
- Modes: request mode (`Omni`, `POST /v1/videos`); stepwise streaming with one
  AR block per chunk (`AsyncOmni`, `WS /v1/realtime/video`)
- Hardware: 1x NVIDIA A800-SXM4-80GB
- Maintainer: Community

## When to use this recipe

Self-Forcing distills Wan2.1-T2V-1.3B into a block-causal generator that
produces each three-latent-frame block in four DMD steps and caches the clean
block's K/V for the next one. Use it for fast 480x832 text-to-video, or stream
blocks as they finish when latency to the first frames matters.

## Supported model contract

| Item | Contract |
| --- | --- |
| Input | One text prompt (UMT5, up to 512 tokens). No negative prompt, no image. |
| Guidance | None: the generator is distilled without classifier-free guidance; `guidance_scale > 1` is rejected. |
| Steps | Fixed four DMD steps at warped timesteps 1000, 937.5, 833.3, 625 (shift 5). |
| Geometry | 480x832 by default; request mode accepts any multiple of 16 up to the same pixel area. Streaming serves the geometry its deploy config declares. |
| Frames | `num_frames = 12k - 3` (9, 21, ..., 81), i.e. a whole number of three-latent-frame blocks. 81 frames at 16 fps is the trained horizon. |
| Longer clips | Supported with a 21-latent-frame sliding window; frames beyond 81 run past the trained horizon. |
| Prompt updates | Streaming only. An update applies from the block after the one being generated when it arrives. |

| Profile | Devices | Precision | Path |
| --- | --- | --- | --- |
| Request mode | 1x A800 80GB | bf16 | `Omni`, shared `text_to_video.py`, or `vllm serve --omni` with `POST /v1/videos` |
| Streaming | 1x A800 80GB | bf16 | `self_forcing_stepwise.yaml` (`ARDiffusionEngine` step execution) behind `AsyncOmni` or `WS /v1/realtime/video` |

## References

- Paper and code: [Self Forcing](https://github.com/guandeh17/Self-Forcing) (reference revision `33593df`)
- Weights: [gdhe17/Self-Forcing](https://huggingface.co/gdhe17/Self-Forcing), Apache-2.0
- Shared examples: [`text_to_video.py`](../../examples/offline_inference/text_to_video/text_to_video.py),
  [`streaming_video_client.py`](../../examples/online_serving/streaming_video_generation/streaming_video_client.py)
- Model directory assembly: [`tools/prepare_self_forcing_checkpoint.py`](../../tools/prepare_self_forcing_checkpoint.py)
- Streaming protocol: [`docs/serving/streaming_video_output_api.md`](../../docs/serving/streaming_video_output_api.md)
- Supported models: [`docs/models/supported_models.md`](../../docs/models/supported_models.md)

## Checkpoint setup

The release ships only the distilled generator, so assemble a local model
directory once. Base components are symlinked; pass `--copy` to copy them.

```bash
python tools/prepare_self_forcing_checkpoint.py \
  --output-dir ./self-forcing-dmd
```

## Hardware

- Accelerator model and per-device memory: NVIDIA A800-SXM4-80GB (SM80)
- Number of devices: 1
- Device interconnect: not used (single device)
- Qualification scope: runtime-qualified on a shared host; timings below are
  bounded local measurements, not isolated benchmarks.

## Software environment

- OS: Linux x86_64
- Python: 3.12
- Driver / runtime: NVIDIA driver 590.48.01, PyTorch 2.13.0+cu132
- vLLM version: 0.31.0
- vLLM-Omni version or commit: `main` at `194b9366` plus this integration

## Command

Offline generation through the shared example:

```bash
python examples/offline_inference/text_to_video/text_to_video.py \
  --model ./self-forcing-dmd \
  --prompt "A red fox trots through fresh snow in a quiet birch forest at sunrise." \
  --seed 0 --enforce-eager \
  --output self_forcing.mp4
```

Height, width, frame count, step count, guidance and fps default to the
model's contract through `vllm_omni.model_extras`.

Online request mode:

```bash
vllm serve ./self-forcing-dmd --omni --enforce-eager --port 8091

curl -s http://localhost:8091/v1/videos/sync \
  -F "prompt=A red fox trots through fresh snow in a quiet birch forest at sunrise." \
  -F "width=832" -F "height=480" -F "num_frames=81" -F "fps=16" -F "seed=0" \
  -o self_forcing.mp4
```

Streaming serving, one chunk per AR block:

```bash
vllm serve ./self-forcing-dmd --omni \
  --deploy-config vllm_omni/deploy/self_forcing_stepwise.yaml \
  --port 8091

python examples/online_serving/streaming_video_generation/streaming_video_client.py \
  --port 8091 --model ./self-forcing-dmd \
  --prompt "A red fox trots through fresh snow in a quiet birch forest at sunrise." \
  --width 832 --height 480 --num-frames 81 --seed 0 \
  --no-helios-distilled-preset \
  --output self_forcing_stream.mp4
```

`--deploy-config` is what selects streaming: the `self_forcing` pipeline
registers no default deploy config, so a plain `vllm serve` stays in request
mode. Add `--prompt-updates '[{"at": 0, "prompt": "...", "transition_chunks": 0}]'`
to the client to switch prompts mid-stream.

In-process streaming uses the same deploy config:

```python
from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

engine = AsyncOmni(model="./self-forcing-dmd", deploy_config="vllm_omni/deploy/self_forcing_stepwise.yaml")
# The step scheduler sizes the request from num_inference_steps, so pass it explicitly.
sampling = OmniDiffusionSamplingParams(height=480, width=832, num_frames=81, num_inference_steps=4, seed=0)
async for chunk in engine.generate({"prompt": "A red fox trots through fresh snow."}, sampling):
    ...  # one decoded 3-latent-frame block per chunk
```

## Verification

- The offline command and `/v1/videos/sync` write an 81-frame 480x832 MP4 at
  16 fps.
- The streaming client reports seven media chunks (9 frames, then six of 12)
  followed by a trailer.
- [`tests/e2e/offline_inference/test_self_forcing.py`](../../tests/e2e/offline_inference/test_self_forcing.py)
  checks that streamed chunks reproduce the request-mode latents for the same
  seed; set `VLLM_OMNI_SELF_FORCING_MODEL_PATH` to the assembled directory.

## Validation

Measured on one A800 against the reference at `33593df` (PyTorch
2.9.1+cu128, flash-attn 2.8.3) with the `generator_ema` weights, bf16,
480x832, 81 frames. Latency is one warmup plus five measured runs of the same
prompt; the reference is timed around `CausalInferencePipeline.inference`,
vLLM-Omni around `Omni.generate`.

| Check | vLLM-Omni | Reference |
| --- | --- | --- |
| Per-call DiT output on identical inputs, 35 calls | rel-L2 0.8-3.3%, cosine >= 0.9994 | - |
| Error against an fp32 oracle, median call | 1.38% | 1.59% |
| CLIP score, ViT-H-14, 32 prompts, seed 0 | 0.368 ± 0.049 | 0.364 ± 0.046 |
| End-to-end latency, median (min-max) | 10.20 s (10.20-10.20) | 14.93 s (14.92-14.94) |
| Text encode / DiT rollout / VAE decode | 0.06 s / 7.67 s / 2.21 s | 0.06 s / 7.93 s / 6.94 s |
| Peak reserved memory | 26.0 GiB | 35.1 GiB |

Free-running rollouts are chaotic at bf16 precision: the reference drifts to
24% latent rel-L2 against itself when FlashAttention is swapped for SDPA, so
pixel-exact agreement with the reference is not a meaningful bar.

Streaming over `WS /v1/realtime/video` delivers the first 9 frames after
1.8 s and all 81 after 10.6 s (about 0.48x realtime at 16 fps, eager). Through
`AsyncOmni` the decoded chunks are bitwise equal to the request-mode frames
for the same seed. A 153-frame stream, past the trained horizon, takes 1.75 s
per chunk once the 21-frame window is full, and device memory stays at
36.5 GiB from its second chunk to its last. Layerwise CPU offload in request
mode leaves the latents bitwise unchanged at the same latency.

## Notes

- Memory usage: 26 GiB peak reserved for an 81-frame request, about 2.6 GiB
  less with `--enable-layerwise-offload`; a streaming server holds 36.5 GiB of
  device memory while a session runs.
- Key flags: `--enforce-eager` is the validated path. Without it the DiT is
  compiled on the first request, since the engine's generic warmup probe (two
  steps at 512x512) is not a valid DMD rollout and is skipped; that request
  takes about 60 s, later ones 8.6 s instead of 10.2 s of engine time. The
  quality of compiled output has not been evaluated.
- Prompt inertia: a prompt update changes what new blocks draw, but the K/V
  cached from earlier frames hold the scene. A stream switched from a snowy
  forest to a beach after its first block still shows the forest 72 frames
  later. LongLive ([arXiv 2509.22622](https://arxiv.org/abs/2509.22622),
  Section 3.1) reports the same for kept caches and recomputes the cache under
  the new prompt; that re-cache is not implemented here.
- Known limitations: one GPU only (no TP, SP, CFG parallelism or HSDP); no
  Cache-DiT or TeaCache, since the four DMD steps are all distinct; the
  deprecated AR-Diffusion tick control plane is not supported; streaming
  decode requires the untiled VAE path.

## Supported features

| Feature | Status |
| --- | --- |
| Offline text-to-video | Validated |
| `POST /v1/videos` | Validated |
| Stepwise streaming (`AsyncOmni`, `WS /v1/realtime/video`) | Validated: chunks bitwise-equal request mode (latents and decoded frames) |
| Mid-stream prompt updates | Validated: applied from the next block, earlier chunks unchanged; scene changes are limited by prompt inertia (see Notes) |
| CPU offload (layerwise) | Validated in request mode |
| Tensor / sequence / CFG parallelism | Not supported |
| Cache-DiT / TeaCache | Not applicable |
| Quantization | Not validated |
