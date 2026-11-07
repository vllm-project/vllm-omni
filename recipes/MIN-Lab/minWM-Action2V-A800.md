# minWM Action2V — A800

> Camera-controlled causal text-to-video, offline and streaming, on one A800

## Summary

- Vendor: MIN-Lab (minWM)
- Model: `MIN-Lab/minWM` (`Wan21/Action2V/dmd/model.pt`) on the
  `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` text encoder, tokenizer and VAE
- Task: text-to-video along a camera trajectory
- Modes: request mode (`Omni`, `POST /v1/videos`); stepwise streaming with one
  AR block per chunk (`AsyncOmni`, `WS /v1/realtime/video`)
- Hardware: 1x NVIDIA A800-SXM4-80GB
- Maintainer: Community

## When to use this recipe

minWM's Wan2.1 Action2V generator is the Self-Forcing causal
Wan2.1-T2V-1.3B backbone with a PRoPE camera branch in every self-attention
layer, distilled to four DMD steps per four-latent-frame block. Use it to
render a scene along a scripted camera path, offline or block by block as a
stream.

## Supported model contract

| Item | Contract |
| --- | --- |
| Input | One text prompt (UMT5, up to 512 tokens) and an optional camera trajectory in `extra_params.trajectory`. No negative prompt, no image. |
| Trajectory | Comma-separated `<token>*<steps>` segments, one step per latent frame (0.08 units or 3 degrees): `w`/`s` forward/backward, `a`/`d` left/right, `u`/`dn` up/down, `j`/`l` yaw left/right, `i`/`k` pitch up/down. The first latent frame is the start pose, so a clip of `N` latent frames needs `N - 1` steps: 77 frames are 20 latent frames, e.g. `a*4,w*8,s*7`. |
| No trajectory | The camera branch is skipped, as in the reference. |
| Guidance | None: the generator is distilled without classifier-free guidance; `guidance_scale > 1` is rejected. |
| Steps | Fixed four DMD steps at warped timesteps 1000, 937.5, 833.3, 625 (shift 5). |
| Geometry | 480x832 by default; request mode accepts any multiple of 16 up to the same pixel area. Streaming serves the geometry its deploy config declares. |
| Frames | `num_frames = 16k - 3` (13, 29, ..., 77), i.e. a whole number of four-latent-frame blocks. 77 frames at 16 fps is the distilled length. |
| Attention window | 20 latent frames, including the block being generated. |
| Prompt updates | Streaming only. An update applies from the block after the one being generated when it arrives. |
| Live camera control | Not supported yet; the camera follows the request's trajectory. |

| Profile | Devices | Precision | Path |
| --- | --- | --- | --- |
| Request mode | 1x A800 80GB | bf16 | `Omni`, shared `text_to_video.py`, or `vllm serve --omni` with `POST /v1/videos` |
| Streaming | 1x A800 80GB | bf16 | `minwm_action2v_stepwise.yaml` (`ARDiffusionEngine` step execution) behind `AsyncOmni` or `WS /v1/realtime/video` |

## References

- Code: [minWM](https://github.com/shengshu-ai/minWM) (reference revision `2a54f4d`)
- Weights: [MIN-Lab/minWM](https://huggingface.co/MIN-Lab/minWM) (`Wan21/Action2V/dmd`)
- Backbone recipe: [`gdhe17/Self-Forcing-A800.md`](../gdhe17/Self-Forcing-A800.md)
- Shared examples: [`text_to_video.py`](../../examples/offline_inference/text_to_video/text_to_video.py),
  [`streaming_video_client.py`](../../examples/online_serving/streaming_video_generation/streaming_video_client.py)
- Model directory assembly: [`tools/prepare_self_forcing_checkpoint.py`](../../tools/prepare_self_forcing_checkpoint.py)

## Checkpoint setup

The release ships only the distilled generator, so assemble a local model
directory once. Base components are symlinked; pass `--copy` to copy them.

```bash
python tools/prepare_self_forcing_checkpoint.py --release minwm-action2v \
  --output-dir ./minwm-action2v-dmd
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
- vLLM-Omni version or commit: `main` plus this integration

## Command

Offline generation through the shared example:

```bash
python examples/offline_inference/text_to_video/text_to_video.py \
  --model ./minwm-action2v-dmd \
  --prompt "A brick pathway winds through manicured bushes beside a dark building with butterfly displays." \
  --extra-body '{"trajectory": "a*4,w*8,s*7"}' \
  --seed 0 --enforce-eager \
  --output minwm_action2v.mp4
```

Height, width, frame count, step count, guidance and fps default to the
model's contract through `vllm_omni.model_extras`.

Online request mode:

```bash
vllm serve ./minwm-action2v-dmd --omni --enforce-eager --port 8091

curl -s http://localhost:8091/v1/videos/sync \
  -F "prompt=A brick pathway winds through manicured bushes beside a dark building with butterfly displays." \
  -F "width=832" -F "height=480" -F "num_frames=77" -F "fps=16" -F "seed=0" \
  -F 'extra_params={"trajectory": "a*4,w*8,s*7"}' \
  -o minwm_action2v.mp4
```

Streaming serving, one chunk per AR block:

```bash
vllm serve ./minwm-action2v-dmd --omni \
  --deploy-config vllm_omni/deploy/minwm_action2v_stepwise.yaml \
  --port 8091

python examples/online_serving/streaming_video_generation/streaming_video_client.py \
  --port 8091 --model ./minwm-action2v-dmd \
  --prompt "A brick pathway winds through manicured bushes beside a dark building with butterfly displays." \
  --width 832 --height 480 --num-frames 77 --seed 0 \
  --extra-params '{"trajectory": "a*4,w*8,s*7"}' \
  --no-helios-distilled-preset \
  --output minwm_action2v_stream.mp4
```

`--deploy-config` is what selects streaming: the `minwm` pipeline registers no
default deploy config, so a plain `vllm serve` stays in request mode.

## Verification

- The offline command and `/v1/videos/sync` write a 77-frame 480x832 MP4 at
  16 fps.
- The streaming client reports five media chunks (13 frames, then four of 16)
  followed by a trailer.
- [`tests/e2e/offline_inference/test_self_forcing.py`](../../tests/e2e/offline_inference/test_self_forcing.py)
  checks that streamed chunks reproduce the request-mode latents for the same
  seed and trajectory; set `VLLM_OMNI_MINWM_ACTION2V_MODEL_PATH` to the
  assembled directory.

## Validation

Measured on one A800 against the reference at `2a54f4d` (PyTorch
2.9.1+cu128, flash-attn 2.8.3) with the `Wan21/Action2V/dmd` weights, bf16,
480x832, 77 frames, trajectory `a*4,w*8,s*7`. Latency is one warmup plus five
measured runs of the same prompt; the reference is timed around its
`ARGenerationLoop.generate`, vLLM-Omni around `Omni.generate`.

| Check | vLLM-Omni | Reference |
| --- | --- | --- |
| Per-call DiT output on identical inputs and cameras, 20 DMD probes | rel-L2 0.72-1.5%, cosine >= 0.9998 | - |
| Same 20 probes and 5 commits with the camera branch disabled | worst rel-L2 24% | - |
| CLIP score (ViT-H-14), minWM's 18-item benchmark with its trajectories | 0.352 ± 0.041 | 0.347 ± 0.038 |
| End-to-end latency, median (min-max) | 12.86 s (12.85-12.90) | 17.44 s (17.44-17.48) |
| Text encode / DiT rollout / VAE decode | 0.06 s / 10.48 s / 2.11 s | 0.06 s / 10.78 s / 6.61 s |
| Peak reserved memory | 31.6 GiB | 40.6 GiB |

The camera-disabled row shows that the comparison exercises the PRoPE branch:
without cameras the same calls drift far from the reference, with them they
match it to bf16 precision.

Streaming over `WS /v1/realtime/video` delivers the first 13 frames after
2.6 s and all 77 after 13.6 s (about 0.36x realtime at 16 fps, eager); the
camera branch doubles self-attention, so each four-frame block costs more than
a Self-Forcing three-frame block.

## Notes

- Memory usage: 31.6 GiB peak reserved for a 77-frame request. With
  `use_prope` set, the paged KV cache reserves the `prope` branch for every
  request, with or without a trajectory, which doubles the self-attention KV
  pages per request; size the KV pool for that.
- Key flags: `--enforce-eager` is the validated path.
- Known limitations: one GPU only (no TP, SP, CFG parallelism or HSDP); PRoPE
  self-attention rejects Ulysses sequence parallelism when the model loads, so
  `--usp` above 1 fails at startup even for requests without a trajectory; no
  Cache-DiT or TeaCache, since the four DMD steps are all distinct; no live
  camera control; streaming decode requires the untiled VAE path, so the
  stepwise deploy refuses `--vae-use-tiling` at startup; request mode can still
  tile.

## Supported features

| Feature | Status |
| --- | --- |
| Offline text-to-video with a trajectory | Validated |
| `POST /v1/videos` | Validated |
| Stepwise streaming (`AsyncOmni`, `WS /v1/realtime/video`) | Validated: chunks reproduce the request-mode latents |
| Mid-stream prompt updates | Inherited from Self-Forcing, not separately validated |
| Live camera control | Not supported |
| Tensor / sequence / CFG parallelism | Not supported |
| Cache-DiT / TeaCache | Not applicable |
| CPU offload, quantization | Not validated |
